"""Filesystem and Git archive helpers for MCP Agent Mail."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ctypes
import hashlib
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterable, Sequence, TypeVar

from git import Actor, Repo
from git.objects.tree import Tree
from PIL import Image

from .config import Settings

if TYPE_CHECKING:
    pass


# Windows Named Mutex constants
WAIT_OBJECT_0 = 0x00000000
WAIT_ABANDONED = 0x00000080
WAIT_TIMEOUT = 0x00000102
WAIT_FAILED = 0xFFFFFFFF
INFINITE = 0xFFFFFFFF


class CrossProcessLock:
    """Cross-process lock using Windows Named Mutex or Unix file locks.

    On Windows: Uses Named Mutex which auto-releases when process dies/crashes.
    On Unix: Falls back to fcntl file locking (advisory locks).

    This solves the Windows SoftFileLock problem where file handles remain
    locked even after process death, causing WinError 32.
    """

    def __init__(self, name: str, timeout_seconds: float = 60.0) -> None:
        self._name = name
        self._timeout = timeout_seconds
        self._handle: int | None = None
        self._held = False
        self._abandoned = False
        self._is_windows = sys.platform == "win32"
        self._lock_file: Any = None
        self._lock_fd: int | None = None

    @staticmethod
    def name_from_path(path: str | Path) -> str:
        """Create a valid mutex name from a file path.

        Mutex names on Windows:
        - Can contain any character except backslash
        - Max 260 chars
        - Use Local\\ prefix for session-local (default)
        """
        path_str = str(Path(path).resolve())
        path_hash = hashlib.sha256(path_str.encode()).hexdigest()[:32]
        return f"Local\\mcp_agent_mail_{path_hash}"

    def acquire(self, timeout_ms: int | None = None) -> bool:
        """Acquire the lock. Returns True if acquired, False on timeout.

        Raises RuntimeError on failure.
        Sets self._abandoned = True if the lock was abandoned by a dead process.
        """
        if timeout_ms is None:
            timeout_ms = int(self._timeout * 1000)

        if self._is_windows:
            return self._acquire_windows(timeout_ms)
        else:
            return self._acquire_unix(timeout_ms)

    def _acquire_windows(self, timeout_ms: int) -> bool:
        """Acquire using Windows Named Mutex."""
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        self._handle = kernel32.CreateMutexW(
            None,  # default security
            False,  # not initially owned
            self._name  # mutex name
        )

        if not self._handle:
            error = ctypes.get_last_error()
            raise RuntimeError(f"Failed to create mutex '{self._name}': error {error}")

        result = kernel32.WaitForSingleObject(self._handle, timeout_ms)

        if result == WAIT_OBJECT_0:
            self._held = True
            self._abandoned = False
            return True
        elif result == WAIT_ABANDONED:
            self._held = True
            self._abandoned = True
            logging.warning(
                f"Mutex '{self._name}' was abandoned by a dead process. "
                "Previous holder may have crashed mid-operation."
            )
            return True
        elif result == WAIT_TIMEOUT:
            kernel32.CloseHandle(self._handle)
            self._handle = None
            return False
        else:
            error = ctypes.get_last_error()
            kernel32.CloseHandle(self._handle)
            self._handle = None
            raise RuntimeError(f"WaitForSingleObject failed: error {error}")

    def _acquire_unix(self, timeout_ms: int) -> bool:
        """Acquire using Unix fcntl file locking."""
        import errno
        import fcntl

        lock_path = Path(f"/tmp/{self._name.replace('/', '_').replace('\\', '_')}.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock_file = lock_path.open("w")
        self._lock_fd = self._lock_file.fileno()

        start = time.monotonic()
        timeout_sec = timeout_ms / 1000.0

        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[attr-defined]
                self._held = True
                self._abandoned = False
                return True
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise
                if time.monotonic() - start >= timeout_sec:
                    self._lock_file.close()
                    self._lock_file = None
                    self._lock_fd = None
                    return False
                time.sleep(0.01)

    def release(self) -> None:
        """Release the lock."""
        if not self._held:
            return

        if self._is_windows:
            self._release_windows()
        else:
            self._release_unix()

        self._held = False

    def _release_windows(self) -> None:
        """Release Windows mutex."""
        if self._handle:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.ReleaseMutex(self._handle)
            kernel32.CloseHandle(self._handle)
            self._handle = None

    def _release_unix(self) -> None:
        """Release Unix file lock."""
        import fcntl

        if self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)  # type: ignore[attr-defined]
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None
            self._lock_fd = None

    @property
    def was_abandoned(self) -> bool:
        """Returns True if the lock was acquired from an abandoned state.

        This indicates the previous holder crashed and recovery may be needed.
        """
        return self._abandoned

    def __enter__(self) -> "CrossProcessLock":
        if not self.acquire():
            raise TimeoutError(f"Timed out acquiring lock '{self._name}'")
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

_IMAGE_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def _pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is alive (cross-platform)."""
    if pid <= 0:
        return False

    try:
        import psutil  # type: ignore[import-not-found]

        return bool(psutil.pid_exists(pid))
    except ImportError:
        pass

    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True


@dataclass(slots=True)
class ProjectArchive:
    settings: Settings
    slug: str
    # Project-specific root inside the single global archive repo
    root: Path
    # The single Git repo object rooted at settings.storage.root
    repo: Repo
    # Path used for advisory file lock during archive writes
    lock_path: Path
    # Filesystem path to the Git repo working directory (archive root)
    repo_root: Path

    @property
    def attachments_dir(self) -> Path:
        return self.root / "attachments"

_PROCESS_LOCKS: dict[tuple[int, str], asyncio.Lock] = {}
_PROCESS_LOCK_OWNERS: dict[tuple[int, str], int] = {}
# Track when each lock was acquired and by which task name
_PROCESS_LOCK_INFO: dict[tuple[int, str], tuple[float, str]] = {}  # (timestamp, task_name)
# Store references to held CrossProcessLock instances so we can force-release them
_HELD_CROSS_PROCESS_LOCKS: dict[tuple[int, str], "CrossProcessLock"] = {}
# Store references to the actual task objects so we can cancel them
_LOCK_HOLDER_TASKS: dict[tuple[int, str], asyncio.Task[Any]] = {}

# Maximum time a lock can be held before it's considered stuck (in seconds)
_MAX_LOCK_HOLD_TIME: float = 120.0  # 2 minutes - most operations should complete much faster


def force_release_stale_in_process_locks(max_age_seconds: float = _MAX_LOCK_HOLD_TIME) -> list[str]:
    """Force-release in-process locks that have been held too long.

    This is a recovery mechanism for stuck tasks that hold locks indefinitely.
    It cancels the stuck task and releases both the CrossProcessLock and asyncio.Lock.

    Returns a list of lock paths that were force-released.
    """
    now = time.time()
    released: list[str] = []

    for loop_key, (acquired_at, task_name) in list(_PROCESS_LOCK_INFO.items()):
        age = now - acquired_at
        if age < max_age_seconds:
            continue

        lock_path = loop_key[1]  # loop_key is (loop_id, lock_path_str)
        logging.warning(
            f"Force-releasing stale in-process lock at {lock_path} "
            f"(held by '{task_name}' for {age:.1f}s, max={max_age_seconds}s)"
        )

        # Cancel the stuck task if we have a reference
        stuck_task = _LOCK_HOLDER_TASKS.pop(loop_key, None)
        if stuck_task is not None and not stuck_task.done():
            logging.warning(f"Cancelling stuck task '{task_name}' that held lock for {age:.1f}s")
            stuck_task.cancel(f"Lock held too long ({age:.1f}s > {max_age_seconds}s)")

        # Release the CrossProcessLock (Windows mutex auto-releases on process death anyway)
        cross_lock = _HELD_CROSS_PROCESS_LOCKS.pop(loop_key, None)
        if cross_lock is not None:
            try:
                cross_lock.release()
                logging.info(f"Force-released CrossProcessLock for {lock_path}")
            except Exception as e:
                logging.warning(f"Error force-releasing CrossProcessLock for {lock_path}: {e}")

        # Clean up tracking dicts
        _PROCESS_LOCK_INFO.pop(loop_key, None)
        _PROCESS_LOCK_OWNERS.pop(loop_key, None)

        # Release the asyncio.Lock if it's held
        process_lock = _PROCESS_LOCKS.get(loop_key)
        if process_lock is not None and process_lock.locked():
            try:
                process_lock.release()
                logging.info(f"Force-released asyncio.Lock for {lock_path}")
            except RuntimeError as e:
                logging.warning(f"Could not release asyncio.Lock for {lock_path}: {e}")

        released.append(lock_path)

    return released


async def async_force_release_stale_locks(max_age_seconds: float = _MAX_LOCK_HOLD_TIME) -> list[str]:
    """Async version of force_release_stale_in_process_locks.

    This version can properly await the cancelled task to ensure it has a chance
    to clean up before we forcibly release its locks.
    """
    now = time.time()
    released: list[str] = []

    for loop_key, (acquired_at, task_name) in list(_PROCESS_LOCK_INFO.items()):
        age = now - acquired_at
        if age < max_age_seconds:
            continue

        lock_path = loop_key[1]
        logging.warning(
            f"Async force-releasing stale in-process lock at {lock_path} "
            f"(held by '{task_name}' for {age:.1f}s, max={max_age_seconds}s)"
        )

        # Cancel the stuck task and wait briefly for it to process
        stuck_task = _LOCK_HOLDER_TASKS.pop(loop_key, None)
        if stuck_task is not None and not stuck_task.done():
            logging.warning(f"Cancelling stuck task '{task_name}' that held lock for {age:.1f}s")
            stuck_task.cancel(f"Lock held too long ({age:.1f}s > {max_age_seconds}s)")
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError, Exception):
                await asyncio.wait_for(asyncio.shield(stuck_task), timeout=1.0)

        # Release the CrossProcessLock
        cross_lock = _HELD_CROSS_PROCESS_LOCKS.pop(loop_key, None)
        if cross_lock is not None:
            try:
                cross_lock.release()
                logging.info(f"Force-released CrossProcessLock for {lock_path}")
            except Exception as e:
                logging.warning(f"Error force-releasing CrossProcessLock for {lock_path}: {e}")

        # Clean up tracking dicts
        _PROCESS_LOCK_INFO.pop(loop_key, None)
        _PROCESS_LOCK_OWNERS.pop(loop_key, None)

        # Release the asyncio.Lock
        process_lock = _PROCESS_LOCKS.get(loop_key)
        if process_lock is not None and process_lock.locked():
            try:
                process_lock.release()
                logging.info(f"Force-released asyncio.Lock for {lock_path}")
            except RuntimeError as e:
                logging.warning(f"Could not release asyncio.Lock for {lock_path}: {e}")

        released.append(lock_path)

    return released


class _LRURepoCache:
    """LRU cache for Git Repo objects with size limit.

    This prevents file descriptor leaks by:
    1. Limiting the number of cached repos (default: 8)
    2. Evicting oldest repos when at capacity (they will be GC'd when no longer referenced)

    IMPORTANT: Evicted repos are NOT closed immediately because they may still be in use
    by other coroutines. They will be closed when garbage collected or when clear() is called.
    """

    def __init__(self, maxsize: int = 8) -> None:
        self._maxsize = max(1, maxsize)
        self._cache: dict[str, Repo] = {}
        self._order: list[str] = []  # LRU order: oldest first
        self._evicted: list[Repo] = []  # Evicted repos pending close

    def peek(self, key: str) -> Repo | None:
        """Check if key exists and return value WITHOUT updating LRU order.

        Safe to call without holding the external lock for a fast-path check.
        """
        return self._cache.get(key)

    def get(self, key: str) -> Repo | None:
        """Get a repo from cache, updating LRU order.

        Should only be called while holding the external lock.
        """
        if key in self._cache:
            # Move to end (most recently used)
            with contextlib.suppress(ValueError):
                self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, repo: Repo) -> None:
        """Add a repo to cache, evicting oldest if at capacity.

        Should only be called while holding the external lock.
        Evicted repos are added to a pending list for later cleanup.
        """
        if key in self._cache:
            # Already exists, just update LRU order
            with contextlib.suppress(ValueError):
                self._order.remove(key)
            self._order.append(key)
            return

        # Evict oldest entries if at capacity
        while len(self._cache) >= self._maxsize and self._order:
            oldest_key = self._order.pop(0)
            old_repo = self._cache.pop(oldest_key, None)
            if old_repo is not None:
                # Don't close immediately - repo may still be in use by another coroutine
                # Add to evicted list for later cleanup
                self._evicted.append(old_repo)

        self._cache[key] = repo
        self._order.append(key)

        # Opportunistically try to close evicted repos that are no longer referenced
        self._cleanup_evicted()

    def _cleanup_evicted(self) -> int:
        """Try to close evicted repos that have only one reference (ours).

        Returns count of repos closed.
        """
        import sys
        still_in_use: list[Repo] = []
        closed = 0
        for repo in self._evicted:
            # If refcount is 2 (our list + the getrefcount call), it's safe to close
            # In practice, we use 3 as threshold to be safe (list + local var + getrefcount)
            if sys.getrefcount(repo) <= 3:
                with contextlib.suppress(Exception):
                    repo.close()
                    closed += 1
            else:
                still_in_use.append(repo)
        self._evicted = still_in_use
        return closed

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)

    def clear(self) -> int:
        """Close all cached and evicted repos and clear the cache. Returns count closed."""
        count = 0
        # Close cached repos
        for repo in self._cache.values():
            with contextlib.suppress(Exception):
                repo.close()
                count += 1
        self._cache.clear()
        self._order.clear()
        # Also close any evicted repos still in pending list
        for repo in self._evicted:
            with contextlib.suppress(Exception):
                repo.close()
                count += 1
        self._evicted.clear()
        return count

    def values(self) -> list[Repo]:
        """Return list of cached repos (for iteration)."""
        return list(self._cache.values())


# LRU cache for Repo objects with automatic cleanup
# Limits to 8 concurrent repos to prevent file handle exhaustion
_REPO_CACHE: _LRURepoCache = _LRURepoCache(maxsize=8)
_REPO_CACHE_LOCK: asyncio.Lock | None = None


def _get_repo_cache_lock() -> asyncio.Lock:
    """Get or create the repo cache lock (must be called from async context)."""
    global _REPO_CACHE_LOCK
    if _REPO_CACHE_LOCK is None:
        _REPO_CACHE_LOCK = asyncio.Lock()
    return _REPO_CACHE_LOCK


def clear_repo_cache() -> int:
    """Close all cached Repo objects and clear the cache.

    Returns the number of repos that were closed.
    Should be called during shutdown or between tests.
    """
    return _REPO_CACHE.clear()


# =============================================================================
# Git Commit Queue - Serializes all Git commits to prevent lock contention
# =============================================================================

@dataclass
class _CommitRequest:
    """A request to commit changes to a Git repository."""
    repo_path: str  # Working tree path for the repo
    message: str
    rel_paths: Sequence[str]
    author_name: str
    author_email: str
    future: asyncio.Future[None]  # Caller awaits this


class _GitCommitQueue:
    """A queue that serializes Git commit operations to prevent lock contention.

    All Git commits go through a single worker, ensuring only one commit
    happens at a time across the entire server. This prevents:
    1. Git index.lock contention
    2. Deadlocks between concurrent writers
    3. Race conditions in the Git working tree

    Usage:
        queue = get_commit_queue()
        await queue.enqueue(repo_path, message, rel_paths, author_name, author_email)
    """

    def __init__(self, max_queue_size: int = 1000) -> None:
        self._queue: asyncio.Queue[_CommitRequest | None] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._started = False
        self._shutdown = False

    async def start(self) -> None:
        """Start the commit queue worker."""
        if self._started:
            return
        self._started = True
        self._shutdown = False
        self._worker_task = asyncio.create_task(self._worker(), name="git-commit-worker")

    async def stop(self, stop_timeout: float = 30.0) -> None:
        """Stop the commit queue worker gracefully."""
        if not self._started or self._worker_task is None:
            return
        self._shutdown = True
        # Signal worker to exit
        await self._queue.put(None)
        try:
            await asyncio.wait_for(self._worker_task, timeout=stop_timeout)
        except asyncio.TimeoutError:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
        self._started = False
        self._worker_task = None

    async def enqueue(
        self,
        repo_path: str,
        message: str,
        rel_paths: Sequence[str],
        author_name: str,
        author_email: str,
        wait_timeout: float = 120.0,
    ) -> None:
        """Enqueue a commit request and wait for it to complete.

        Args:
            repo_path: Working tree path of the Git repo
            message: Commit message
            rel_paths: Relative paths to add and commit
            author_name: Git author name
            author_email: Git author email
            wait_timeout: Max time to wait for commit to complete (default 120s)

        Raises:
            GitOperationTimeout: If the commit times out
            RuntimeError: If the queue is not started
            Exception: Any error from the actual Git commit
        """
        if not self._started:
            await self.start()

        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        request = _CommitRequest(
            repo_path=repo_path,
            message=message,
            rel_paths=rel_paths,
            author_name=author_name,
            author_email=author_email,
            future=future,
        )

        try:
            # Put request in queue (with timeout to avoid blocking forever)
            await asyncio.wait_for(self._queue.put(request), timeout=10.0)
        except asyncio.TimeoutError:
            raise GitOperationTimeout(
                "Commit queue is full - too many pending commits. Try again later."
            ) from None

        # Wait for the worker to process our request
        try:
            await asyncio.wait_for(future, timeout=wait_timeout)
        except asyncio.TimeoutError:
            raise GitOperationTimeout(
                f"Commit operation timed out after {wait_timeout}s. "
                f"The commit queue may be backed up."
            ) from None

    async def _worker(self) -> None:
        """Worker task that processes commit requests one at a time."""
        while not self._shutdown:
            try:
                request = await self._queue.get()

                # None signals shutdown
                if request is None:
                    break

                try:
                    await self._perform_commit(request)
                    request.future.set_result(None)
                except Exception as e:
                    if not request.future.done():
                        request.future.set_exception(e)
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception:
                # Log but don't crash the worker
                continue

    async def _perform_commit(self, request: _CommitRequest) -> None:
        """Execute a single commit operation."""
        if not request.rel_paths:
            return

        def _do_commit() -> None:
            repo = Repo(request.repo_path)
            actor = Actor(request.author_name, request.author_email)

            repo.index.add(request.rel_paths)
            if repo.is_dirty(index=True, working_tree=True):
                # Build trailers
                trailers: list[str] = []
                message = request.message
                try:
                    lower_msg = message.lower()
                    have_agent_line = "\nagent:" in lower_msg
                    if message.startswith("mail: ") and not have_agent_line:
                        head = message[len("mail: "):]
                        agent_part = head.split("->", 1)[0].strip()
                        if agent_part:
                            trailers.append(f"Agent: {agent_part}")
                    elif message.startswith("file_reservation: ") and not have_agent_line:
                        head = message[len("file_reservation: "):]
                        agent_part = head.split(" ", 1)[0].strip()
                        if agent_part:
                            trailers.append(f"Agent: {agent_part}")
                except Exception:
                    pass

                final_message = message
                if trailers:
                    final_message = message + "\n\n" + "\n".join(trailers) + "\n"
                repo.index.commit(final_message, author=actor, committer=actor)

        # Run the actual commit in a thread with timeout
        await _to_thread(_do_commit, _timeout=60.0)


# Global commit queue instance
_COMMIT_QUEUE: _GitCommitQueue | None = None
_COMMIT_QUEUE_LOCK: asyncio.Lock | None = None


def _get_commit_queue_lock() -> asyncio.Lock:
    """Get or create the commit queue lock."""
    global _COMMIT_QUEUE_LOCK
    if _COMMIT_QUEUE_LOCK is None:
        _COMMIT_QUEUE_LOCK = asyncio.Lock()
    return _COMMIT_QUEUE_LOCK


async def get_commit_queue() -> _GitCommitQueue:
    """Get the global commit queue, creating and starting it if needed."""
    global _COMMIT_QUEUE

    # Fast path - already initialized
    if _COMMIT_QUEUE is not None and _COMMIT_QUEUE._started:
        return _COMMIT_QUEUE

    async with _get_commit_queue_lock():
        if _COMMIT_QUEUE is None:
            _COMMIT_QUEUE = _GitCommitQueue()
        if not _COMMIT_QUEUE._started:
            await _COMMIT_QUEUE.start()
        return _COMMIT_QUEUE


async def stop_commit_queue() -> None:
    """Stop the global commit queue gracefully."""
    global _COMMIT_QUEUE
    if _COMMIT_QUEUE is not None:
        await _COMMIT_QUEUE.stop()


class AsyncFileLock:
    """Async-friendly cross-process lock using Windows Named Mutex.

    On Windows: Uses Named Mutex which auto-releases when process dies.
    On Unix: Falls back to fcntl advisory locks.

    This replaces the previous SoftFileLock approach which had issues with
    file handles remaining locked after process death on Windows.
    """

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = 60.0,
        stale_timeout_seconds: float = 180.0,  # kept for API compatibility, not used
    ) -> None:
        self._path = Path(path)
        self._lock_key = str(self._path.resolve())
        self._mutex_name = CrossProcessLock.name_from_path(self._lock_key)
        self._cross_process_lock: CrossProcessLock | None = None
        self._timeout = float(timeout_seconds)
        self._held = False
        self._loop_key: tuple[int, str] | None = None
        self._process_lock: asyncio.Lock | None = None
        self._process_lock_held = False
        self._was_abandoned = False

    async def __aenter__(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop_key = (id(loop), self._lock_key)

        # FIRST: Force-release any stale in-process locks before attempting to acquire
        released = await async_force_release_stale_locks()
        if released:
            logging.info(f"Pre-acquire: force-released {len(released)} stale in-process locks")

        # Acquire in-process asyncio.Lock first
        process_lock = _PROCESS_LOCKS.get(self._loop_key)
        if process_lock is None:
            process_lock = asyncio.Lock()
            _PROCESS_LOCKS[self._loop_key] = process_lock

        current_task = asyncio.current_task()
        owner_id = _PROCESS_LOCK_OWNERS.get(self._loop_key)
        current_task_id = id(current_task) if current_task else id(self)

        if owner_id == current_task_id:
            raise RuntimeError(f"Re-entrant AsyncFileLock acquisition detected for {self._path}")

        self._process_lock = process_lock
        logging.debug(f"AsyncFileLock: waiting for process lock {self._lock_key}")

        try:
            await asyncio.wait_for(self._process_lock.acquire(), timeout=self._timeout)
        except asyncio.TimeoutError:
            # Try force-releasing stale locks one more time
            released = await async_force_release_stale_locks()
            if released and self._lock_key in released:
                try:
                    await asyncio.wait_for(self._process_lock.acquire(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                else:
                    logging.info(f"Acquired lock after force-release: {self._lock_key}")
                    self._process_lock_held = True
                    _PROCESS_LOCK_OWNERS[self._loop_key] = current_task_id
                    current = asyncio.current_task()
                    task_name = current.get_name() if current else "unknown"
                    _PROCESS_LOCK_INFO[self._loop_key] = (time.time(), task_name)
                    if current is not None:
                        _LOCK_HOLDER_TASKS[self._loop_key] = current
                    return await self._acquire_cross_process_lock()

            lock_info = _PROCESS_LOCK_INFO.get(self._loop_key)
            if lock_info:
                acquired_at, task_name = lock_info
                held_for = time.time() - acquired_at
                raise TimeoutError(
                    f"Timed out waiting for in-process lock {self._path} after {self._timeout:.2f}s. "
                    f"Lock held by '{task_name}' for {held_for:.1f}s."
                ) from None
            else:
                raise TimeoutError(
                    f"Timed out waiting for in-process lock {self._path} after {self._timeout:.2f}s. "
                    "Another task in this process may be holding the lock."
                ) from None

        logging.debug(f"AsyncFileLock: acquired process lock {self._lock_key}")
        self._process_lock_held = True
        _PROCESS_LOCK_OWNERS[self._loop_key] = current_task_id

        current = asyncio.current_task()
        task_name = current.get_name() if current else "unknown"
        _PROCESS_LOCK_INFO[self._loop_key] = (time.time(), task_name)
        if current is not None:
            _LOCK_HOLDER_TASKS[self._loop_key] = current

        return await self._acquire_cross_process_lock()

    async def _acquire_cross_process_lock(self) -> None:
        """Acquire the cross-process lock (Windows mutex or Unix flock)."""
        try:
            self._cross_process_lock = CrossProcessLock(
                self._mutex_name,
                timeout_seconds=self._timeout
            )

            logging.debug(f"AsyncFileLock: acquiring cross-process lock {self._mutex_name}")

            # Run blocking acquire in thread pool
            acquired = await _to_thread(self._cross_process_lock.acquire)

            if not acquired:
                raise TimeoutError(
                    f"Timed out acquiring cross-process lock for {self._path} "
                    f"after {self._timeout:.2f}s"
                )

            self._held = True
            self._was_abandoned = self._cross_process_lock.was_abandoned

            if self._was_abandoned:
                logging.warning(
                    f"Lock for {self._path} was abandoned by a dead process. "
                    "Previous operation may have been interrupted."
                )

            # Store reference for force-release
            if self._loop_key is not None:
                _HELD_CROSS_PROCESS_LOCKS[self._loop_key] = self._cross_process_lock

            logging.debug(f"AsyncFileLock: acquired cross-process lock {self._mutex_name}")

        except Exception:
            # Clean up on failure
            if self._loop_key is not None:
                _PROCESS_LOCK_OWNERS.pop(self._loop_key, None)
                _PROCESS_LOCK_INFO.pop(self._loop_key, None)
                _HELD_CROSS_PROCESS_LOCKS.pop(self._loop_key, None)
                _LOCK_HOLDER_TASKS.pop(self._loop_key, None)
            if self._process_lock_held and self._process_lock:
                self._process_lock.release()
                self._process_lock_held = False
            if (
                self._loop_key is not None
                and self._process_lock
                and not self._process_lock.locked()
            ):
                _PROCESS_LOCKS.pop(self._loop_key, None)
            self._process_lock = None
            raise

    @property
    def was_abandoned(self) -> bool:
        """Returns True if lock was acquired from abandoned state (previous holder crashed)."""
        return self._was_abandoned

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        if self._held and self._cross_process_lock is not None:
            # Release the cross-process lock synchronously - it's very fast
            # and we don't want thread pool issues to delay lock release
            with contextlib.suppress(Exception):
                self._cross_process_lock.release()
            self._held = False

        # Clean up process-level locks
        if self._loop_key is not None:
            _PROCESS_LOCK_OWNERS.pop(self._loop_key, None)
            _PROCESS_LOCK_INFO.pop(self._loop_key, None)
            _HELD_CROSS_PROCESS_LOCKS.pop(self._loop_key, None)
            _LOCK_HOLDER_TASKS.pop(self._loop_key, None)

        if self._process_lock_held and self._process_lock:
            self._process_lock.release()
            self._process_lock_held = False

        if (
            self._loop_key is not None
            and self._process_lock
            and not self._process_lock.locked()
        ):
            _PROCESS_LOCKS.pop(self._loop_key, None)

        self._process_lock = None
        self._loop_key = None
        self._cross_process_lock = None


@asynccontextmanager
async def archive_write_lock(archive: ProjectArchive, *, timeout_seconds: float = 60.0) -> AsyncIterator[None]:
    """Context manager for safely mutating archive surfaces."""

    lock = AsyncFileLock(archive.lock_path, timeout_seconds=timeout_seconds)
    await lock.__aenter__()
    try:
        yield
    except Exception as exc:
        await lock.__aexit__(type(exc), exc, exc.__traceback__)
        raise
    else:
        await lock.__aexit__(None, None, None)


T = TypeVar('T')

# Default timeout for thread operations (especially Git ops that can hang on lock contention)
_DEFAULT_THREAD_TIMEOUT: float = 60.0


class GitOperationTimeout(Exception):
    """Raised when a Git operation times out waiting for locks."""
    pass


async def _to_thread(
    func: Any,
    /,
    *args: Any,
    _timeout: float | None = None,
    **kwargs: Any,
) -> Any:
    """Run a blocking function in a thread with optional timeout.

    Args:
        func: The blocking function to run
        *args: Positional arguments for func
        _timeout: Timeout in seconds. None means use default (60s).
                  Use 0 or negative to disable timeout (infinite wait).
        **kwargs: Keyword arguments for func

    Raises:
        GitOperationTimeout: If the operation times out
    """
    timeout = _timeout if _timeout is not None else _DEFAULT_THREAD_TIMEOUT

    if timeout <= 0:
        # No timeout - original behavior
        return await asyncio.to_thread(func, *args, **kwargs)

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        func_name = getattr(func, '__name__', str(func))
        raise GitOperationTimeout(
            f"Operation '{func_name}' timed out after {timeout}s. "
            f"This may indicate Git lock contention. Try again or check for stale lock files."
        ) from None


def _ensure_str(value: str | bytes) -> str:
    """Ensure a value is a string, decoding bytes if necessary."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def collect_lock_status(settings: Settings) -> dict[str, Any]:
    """Return structured metadata about active archive locks."""

    root = Path(settings.storage.root).expanduser().resolve()
    locks: list[dict[str, Any]] = []
    summary = {"total": 0, "active": 0, "stale": 0, "metadata_missing": 0}

    if root.exists():
        now = time.time()
        for lock_path in sorted(root.rglob("*.lock"), key=lambda p: str(p)):
            metadata_path = lock_path.parent / f"{lock_path.name}.owner.json"
            if not lock_path.exists():
                continue
            metadata_present = metadata_path.exists()
            if lock_path.name != ".archive.lock" and not metadata_present:
                continue

            info: dict[str, Any] = {
                "path": str(lock_path),
                "metadata_path": str(metadata_path) if metadata_present else None,
                "status": "held",
                "metadata_present": metadata_present,
                "category": "archive" if lock_path.name == ".archive.lock" else "custom",
            }

            with contextlib.suppress(Exception):
                stat = lock_path.stat()
                info["size"] = stat.st_size
                info["modified_ts"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

            metadata: dict[str, Any] = {}
            if metadata_present:
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except Exception:
                    metadata = {}
            info["metadata"] = metadata

            pid_val = metadata.get("pid")
            pid_int: int | None = None
            if pid_val is not None:
                with contextlib.suppress(Exception):
                    pid_int = int(pid_val)
            info["owner_pid"] = pid_int
            info["owner_alive"] = _pid_alive(pid_int) if pid_int else False

            created_ts = metadata.get("created_ts") if isinstance(metadata, dict) else None
            if isinstance(created_ts, (int, float)):
                info["created_ts"] = datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat()
                info["age_seconds"] = max(0.0, now - float(created_ts))
            else:
                info["created_ts"] = None
                info["age_seconds"] = None

            stale_threshold = 180.0  # Default stale timeout
            info["stale_timeout_seconds"] = stale_threshold
            age_val = info.get("age_seconds")
            # Lock is stale if EITHER the owner is dead OR the age exceeds timeout
            # Special case: if stale_timeout is 0, only check owner liveness (ignore age)
            is_stale = False
            if bool(metadata):
                if not info["owner_alive"]:
                    # Owner process is dead - lock is stale
                    is_stale = True
                elif stale_threshold > 0 and isinstance(age_val, (int, float)) and age_val >= stale_threshold:
                    # Lock is too old - stale
                    # (only if stale_timeout > 0, otherwise age check is disabled)
                    is_stale = True
            info["stale_suspected"] = is_stale

            summary["total"] += 1

            if is_stale:
                summary["stale"] += 1
            elif info["owner_alive"]:
                summary["active"] += 1
            if not metadata_present:
                summary["metadata_missing"] += 1

            locks.append(info)

    return {"locks": locks, "summary": summary}


async def ensure_archive_root(settings: Settings) -> tuple[Path, Repo]:
    repo_root = Path(settings.storage.root).expanduser().resolve()
    await _to_thread(repo_root.mkdir, parents=True, exist_ok=True)
    repo = await _ensure_repo(repo_root, settings)
    return repo_root, repo


async def ensure_archive(settings: Settings, slug: str) -> ProjectArchive:
    repo_root, repo = await ensure_archive_root(settings)
    project_root = repo_root / "projects" / slug
    await _to_thread(project_root.mkdir, parents=True, exist_ok=True)
    return ProjectArchive(
        settings=settings,
        slug=slug,
        root=project_root,
        repo=repo,
        # Use a per-project advisory lock to avoid cross-project contention
        lock_path=project_root / ".archive.lock",
        repo_root=repo_root,
    )


async def _ensure_repo(root: Path, settings: Settings) -> Repo:
    """Get or create a Repo for the given root, with caching to prevent file handle leaks."""
    cache_key = str(root.resolve())

    # Fast path: check cache without lock using peek() which doesn't modify LRU order
    cached = _REPO_CACHE.peek(cache_key)
    if cached is not None:
        return cached

    # Slow path: acquire lock and check/create
    async with _get_repo_cache_lock():
        # Double-check after acquiring lock, use get() to update LRU order
        cached = _REPO_CACHE.get(cache_key)
        if cached is not None:
            return cached

        git_dir = root / ".git"
        if git_dir.exists():
            repo = Repo(str(root))
            _REPO_CACHE.put(cache_key, repo)
            return repo

        repo_result = await _to_thread(Repo.init, str(root))
        repo = Repo(repo_result.working_dir) if hasattr(repo_result, 'working_dir') else repo_result
        # Ensure deterministic, non-interactive commits (disable GPG signing)
        try:
            def _configure_repo() -> None:
                with repo.config_writer() as cw:
                    cw.set_value("commit", "gpgsign", "false")
            await _to_thread(_configure_repo)
        except Exception:
            pass
        attributes_path = root / ".gitattributes"
        if not attributes_path.exists():
            await _write_text(attributes_path, "*.json text\n*.md text\n")
        await _commit(repo, settings, "chore: initialize archive", [".gitattributes"])
        _REPO_CACHE.put(cache_key, repo)
        return repo


async def write_agent_profile(archive: ProjectArchive, agent: dict[str, object]) -> None:
    profile_path = archive.root / "agents" / agent["name"].__str__() / "profile.json"
    await _write_json(profile_path, agent)
    rel = profile_path.relative_to(archive.repo_root).as_posix()
    await _commit(archive.repo, archive.settings, f"agent: profile {agent['name']}", [rel])


async def write_file_reservation_record(archive: ProjectArchive, file_reservation: dict[str, object]) -> None:
    path_pattern = str(file_reservation.get("path_pattern") or file_reservation.get("path") or "").strip()
    if not path_pattern:
        raise ValueError("File reservation record must include 'path_pattern'.")
    normalized_file_reservation = dict(file_reservation)
    normalized_file_reservation["path_pattern"] = path_pattern
    normalized_file_reservation.pop("path", None)
    digest = hashlib.sha1(path_pattern.encode("utf-8")).hexdigest()
    file_reservation_path = archive.root / "file_reservations" / f"{digest}.json"
    await _write_json(file_reservation_path, normalized_file_reservation)
    agent_name = str(normalized_file_reservation.get("agent", "unknown"))
    await _commit(
        archive.repo,
        archive.settings,
        f"file_reservation: {agent_name} {path_pattern}",
        [file_reservation_path.relative_to(archive.repo_root).as_posix()],
    )


async def write_message_bundle(
    archive: ProjectArchive,
    message: dict[str, object],
    body_md: str,
    sender: str,
    recipients: Sequence[str],
    extra_paths: Sequence[str] | None = None,
    commit_text: str | None = None,
) -> None:
    timestamp_obj: Any = message.get("created") or message.get("created_ts")
    timestamp_str = timestamp_obj if isinstance(timestamp_obj, str) else datetime.now(timezone.utc).isoformat()
    now = datetime.fromisoformat(timestamp_str)
    y_dir = now.strftime("%Y")
    m_dir = now.strftime("%m")

    canonical_dir = archive.root / "messages" / y_dir / m_dir
    outbox_dir = archive.root / "agents" / sender / "outbox" / y_dir / m_dir
    inbox_dirs = [archive.root / "agents" / r / "inbox" / y_dir / m_dir for r in recipients]

    rel_paths: list[str] = []

    await _to_thread(canonical_dir.mkdir, parents=True, exist_ok=True)
    await _to_thread(outbox_dir.mkdir, parents=True, exist_ok=True)
    for path in inbox_dirs:
        await _to_thread(path.mkdir, parents=True, exist_ok=True)

    frontmatter = json.dumps(message, indent=2, sort_keys=True)
    content = f"---json\n{frontmatter}\n---\n\n{body_md.strip()}\n"

    # Descriptive, ISO-prefixed filename: <ISO>__<subject-slug>__<id>.md
    created_iso = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    subject_value = str(message.get("subject", "")).strip() or "message"
    subject_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", subject_value).strip("-_").lower()[:80] or "message"
    id_suffix = str(message.get("id", ""))
    filename = (
        f"{created_iso}__{subject_slug}__{id_suffix}.md"
        if id_suffix
        else f"{created_iso}__{subject_slug}.md"
    )
    canonical_path = canonical_dir / filename
    await _write_text(canonical_path, content)
    rel_paths.append(canonical_path.relative_to(archive.repo_root).as_posix())

    outbox_path = outbox_dir / filename
    await _write_text(outbox_path, content)
    rel_paths.append(outbox_path.relative_to(archive.repo_root).as_posix())

    for inbox_dir in inbox_dirs:
        inbox_path = inbox_dir / filename
        await _write_text(inbox_path, content)
        rel_paths.append(inbox_path.relative_to(archive.repo_root).as_posix())

    # Update thread-level digest for human review if thread_id present
    thread_id_obj = message.get("thread_id")
    if isinstance(thread_id_obj, str) and thread_id_obj.strip():
        canonical_rel = canonical_path.relative_to(archive.repo_root).as_posix()
        digest_rel = await _update_thread_digest(
            archive,
            thread_id_obj.strip(),
            {
                "from": sender,
                "to": list(recipients),
                "subject": message.get("subject", "") or "",
                "created": timestamp_str,
            },
            body_md,
            canonical_rel,
        )
        if digest_rel:
            rel_paths.append(digest_rel)

    if extra_paths:
        rel_paths.extend(extra_paths)
    thread_key = message.get("thread_id") or message.get("id")
    if commit_text:
        commit_message = commit_text if commit_text.endswith("\n") else f"{commit_text}\n"
    else:
        commit_subject = f"mail: {sender} -> {', '.join(recipients)} | {message.get('subject', '')}"
        # Enriched commit body mirroring console logs
        commit_body_lines = [
            "TOOL: send_message",
            f"Agent: {sender}",
            f"Project: {message.get('project', '')}",
            f"Started: {timestamp_str}",
            "Status: SUCCESS",
            f"Thread: {thread_key}",
        ]
        commit_message = commit_subject + "\n\n" + "\n".join(commit_body_lines) + "\n"
    await _commit(archive.repo, archive.settings, commit_message, rel_paths)


async def _update_thread_digest(
    archive: ProjectArchive,
    thread_id: str,
    meta: dict[str, object],
    body_md: str,
    canonical_rel_path: str,
) -> str | None:
    """
    Append a compact entry to a thread-level digest file for human review.

    The digest lives at messages/threads/{thread_id}.md and contains an
    append-only sequence of sections linking to canonical messages.
    """
    digest_dir = archive.root / "messages" / "threads"
    await _to_thread(digest_dir.mkdir, parents=True, exist_ok=True)
    digest_path = digest_dir / f"{thread_id}.md"

    # Ensure recipients list is typed as list[str] for join()
    to_value = meta.get("to")
    if isinstance(to_value, (list, tuple)):
        recipients_list: list[str] = [str(v) for v in to_value]
    elif isinstance(to_value, str):
        recipients_list = [to_value]
    else:
        recipients_list = []
    header = (
        f"## {meta.get('created', '')} — {meta.get('from', '')} → {', '.join(recipients_list)}\n\n"
    )
    link_line = f"[View canonical]({canonical_rel_path})\n\n"
    subject = str(meta.get("subject", "")).strip()
    subject_line = f"### {subject}\n\n" if subject else ""

    # Truncate body to a preview to keep digest readable
    preview = body_md.strip()
    if len(preview) > 1200:
        preview = preview[:1200].rstrip() + "\n..."

    entry = subject_line + header + link_line + preview + "\n\n---\n\n"

    # Append atomically
    def _append() -> None:
        mode = "a" if digest_path.exists() else "w"
        with digest_path.open(mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(f"# Thread {thread_id}\n\n")
            f.write(entry)

    await _to_thread(_append)
    return digest_path.relative_to(archive.repo_root).as_posix()


async def process_attachments(
    archive: ProjectArchive,
    body_md: str,
    attachment_paths: Iterable[str] | None,
    convert_markdown: bool,
    *,
    embed_policy: str = "auto",
) -> tuple[str, list[dict[str, object]], list[str]]:
    attachments_meta: list[dict[str, object]] = []
    commit_paths: list[str] = []
    updated_body = body_md
    if convert_markdown and archive.settings.storage.convert_images:
        updated_body = await _convert_markdown_images(
            archive, body_md, attachments_meta, commit_paths, embed_policy=embed_policy
        )
    else:
        # Even when not converting, surface inline data-uri images in attachments meta for visibility
        if "data:image" in body_md:
            for m in _IMAGE_PATTERN.finditer(body_md):
                raw_path = m.group("path")
                if raw_path.startswith("data:"):
                    try:
                        header = raw_path.split(",", 1)[0]
                        media_type = "image/webp"
                        if ";" in header:
                            mt = header[5:].split(";", 1)[0]
                            if mt:
                                media_type = mt
                        attachments_meta.append({"type": "inline", "media_type": media_type})
                    except Exception:
                        attachments_meta.append({"type": "inline"})
    if attachment_paths:
        for path in attachment_paths:
            p = Path(path)
            if not p.is_absolute():
                p = (archive.root / path).resolve()
            meta, rel_path = await _store_image(archive, p, embed_policy=embed_policy)
            attachments_meta.append(meta)
            if rel_path:
                commit_paths.append(rel_path)
    return updated_body, attachments_meta, commit_paths


async def _convert_markdown_images(
    archive: ProjectArchive,
    body_md: str,
    meta: list[dict[str, object]],
    commit_paths: list[str],
    *,
    embed_policy: str = "auto",
) -> str:
    matches = list(_IMAGE_PATTERN.finditer(body_md))
    if not matches:
        return body_md
    result_parts: list[str] = []
    last_idx = 0
    for match in matches:
        path_start, path_end = match.span("path")
        result_parts.append(body_md[last_idx:path_start])
        raw_path = match.group("path")
        normalized_path = raw_path.strip()
        if raw_path.startswith("data:"):
            # Preserve inline data URI and record minimal metadata so callers can assert inline behavior
            try:
                header = normalized_path.split(",", 1)[0]
                media_type = "image/webp"
                if ";" in header:
                    mt = header[5:].split(";", 1)[0]
                    if mt:
                        media_type = mt
                meta.append({
                    "type": "inline",
                    "media_type": media_type,
                })
            except Exception:
                meta.append({"type": "inline"})
            result_parts.append(raw_path)
            last_idx = path_end
            continue
        file_path = Path(normalized_path)
        if not file_path.is_absolute():
            file_path = (archive.root / raw_path).resolve()
        if not file_path.is_file():
            result_parts.append(raw_path)
            last_idx = path_end
            continue
        attachment_meta, rel_path = await _store_image(archive, file_path, embed_policy=embed_policy)
        replacement_value: str
        if attachment_meta["type"] == "inline":
            replacement_value = f"data:image/webp;base64,{attachment_meta['data_base64']}"
        else:
            replacement_value = str(attachment_meta["path"])
        leading_ws_len = len(raw_path) - len(raw_path.lstrip())
        trailing_ws_len = len(raw_path) - len(raw_path.rstrip())
        leading_ws = raw_path[:leading_ws_len] if leading_ws_len else ""
        trailing_ws = raw_path[len(raw_path) - trailing_ws_len :] if trailing_ws_len else ""
        result_parts.append(f"{leading_ws}{replacement_value}{trailing_ws}")
        meta.append(attachment_meta)
        if rel_path:
            commit_paths.append(rel_path)
        last_idx = path_end
    result_parts.append(body_md[last_idx:])
    return "".join(result_parts)


async def _store_image(archive: ProjectArchive, path: Path, *, embed_policy: str = "auto") -> tuple[dict[str, object], str | None]:
    data = await _to_thread(path.read_bytes)

    # Open image and convert, properly closing the original to prevent file handle leaks
    def _open_and_convert(p: Path) -> Image.Image:
        with Image.open(p) as pil:
            return pil.convert("RGBA" if pil.mode in ("LA", "RGBA") else "RGB")  # type: ignore[no-any-return]

    img = await _to_thread(_open_and_convert, path)
    try:
        width, height = img.size
        buffer_path = archive.attachments_dir
        await _to_thread(buffer_path.mkdir, parents=True, exist_ok=True)
        digest = hashlib.sha1(data).hexdigest()
        target_dir = buffer_path / digest[:2]
        await _to_thread(target_dir.mkdir, parents=True, exist_ok=True)
        target_path = target_dir / f"{digest}.webp"
        # Optionally store original alongside (in originals/)
        original_rel: str | None = None
        if archive.settings.storage.keep_original_images:
            originals_dir = archive.root / "attachments" / "originals" / digest[:2]
            await _to_thread(originals_dir.mkdir, parents=True, exist_ok=True)
            orig_ext = path.suffix.lower().lstrip(".") or "bin"
            orig_path = originals_dir / f"{digest}.{orig_ext}"
            if not orig_path.exists():
                await _to_thread(orig_path.write_bytes, data)
            original_rel = orig_path.relative_to(archive.repo_root).as_posix()
        if not target_path.exists():
            await _save_webp(img, target_path)
        new_bytes = await _to_thread(target_path.read_bytes)
        rel_path = target_path.relative_to(archive.repo_root).as_posix()
        # Update per-attachment manifest with metadata
        try:
            manifest_dir = archive.root / "attachments" / "_manifests"
            await _to_thread(manifest_dir.mkdir, parents=True, exist_ok=True)
            manifest_path = manifest_dir / f"{digest}.json"
            manifest_payload = {
                "sha1": digest,
                "webp_path": rel_path,
                "bytes_webp": len(new_bytes),
                "width": width,
                "height": height,
                "original_path": original_rel,
                "bytes_original": len(data),
                "original_ext": path.suffix.lower(),
            }
            await _write_json(manifest_path, manifest_payload)
            await _append_attachment_audit(
                archive,
                digest,
                {
                    "event": "stored",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "webp_path": rel_path,
                    "bytes_webp": len(new_bytes),
                    "original_path": original_rel,
                    "bytes_original": len(data),
                    "ext": path.suffix.lower(),
                },
            )
        except Exception:
            pass

        should_inline = False
        if embed_policy == "inline":
            should_inline = True
        elif embed_policy == "file":
            should_inline = False
        else:
            should_inline = len(new_bytes) <= archive.settings.storage.inline_image_max_bytes
        if should_inline:
            encoded = base64.b64encode(new_bytes).decode("ascii")
            return {
                "type": "inline",
                "media_type": "image/webp",
                "bytes": len(new_bytes),
                "width": width,
                "height": height,
                "sha1": digest,
                "data_base64": encoded,
            }, rel_path
        meta: dict[str, object] = {
            "type": "file",
            "media_type": "image/webp",
            "bytes": len(new_bytes),
            "path": rel_path,
            "width": width,
            "height": height,
            "sha1": digest,
        }
        if original_rel:
            meta["original_path"] = original_rel
        return meta, rel_path
    finally:
        # Close the converted image to prevent file handle leaks
        img.close()


async def _save_webp(img: Image.Image, path: Path) -> None:
    await _to_thread(img.save, path, format="WEBP", method=6, quality=80)


async def _write_text(path: Path, content: str) -> None:
    await _to_thread(path.parent.mkdir, parents=True, exist_ok=True)
    await _to_thread(path.write_text, content, encoding="utf-8")


async def _write_json(path: Path, payload: dict[str, object]) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True)
    await _write_text(path, content + "\n")


async def _append_attachment_audit(archive: ProjectArchive, sha1: str, event: dict[str, object]) -> None:
    """Append a single JSON line audit record for an attachment digest.

    Creates attachments/_audit/<sha1>.log if missing. Best-effort; failures are ignored.
    """
    try:
        audit_dir = archive.root / "attachments" / "_audit"
        await _to_thread(audit_dir.mkdir, parents=True, exist_ok=True)
        audit_path = audit_dir / f"{sha1}.log"

        def _append_line() -> None:
            line = json.dumps(event, sort_keys=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        await _to_thread(_append_line)
    except Exception:
        pass


async def _commit(repo: Repo, settings: Settings, message: str, rel_paths: Sequence[str]) -> None:
    """Commit changes to the Git repository via the global commit queue.

    This ensures all commits are serialized to prevent Git lock contention.
    """
    if not rel_paths:
        return

    working_tree = repo.working_tree_dir
    if working_tree is None:
        raise ValueError("Repository has no working tree directory")

    # Use the global commit queue to serialize all Git commits
    queue = await get_commit_queue()
    await queue.enqueue(
        repo_path=str(working_tree),
        message=message,
        rel_paths=list(rel_paths),
        author_name=settings.storage.git_author_name,
        author_email=settings.storage.git_author_email,
    )


async def heal_archive_locks(settings: Settings) -> dict[str, Any]:
    """Scan the archive root for stale lock artifacts and clean them.

    This includes:
    1. Application-level locks (*.lock files with owner metadata)
    2. Git internal locks (.git/index.lock, .git/HEAD.lock, etc.) that may be
       left behind after crashes or interrupted operations
    """

    root = Path(settings.storage.root).expanduser().resolve()
    await _to_thread(root.mkdir, parents=True, exist_ok=True, _timeout=10.0)
    summary: dict[str, Any] = {
        "locks_scanned": 0,
        "locks_removed": [],
        "metadata_removed": [],
        "git_locks_removed": [],
    }
    if not root.exists():
        return summary

    # --- Phase 1: Clean stale application-level lock files ---
    # Note: With Named Mutex, these files should no longer be created,
    # but we clean up any legacy files for backward compatibility
    for lock_path_raw in sorted(root.rglob("*.lock"), key=str):
        lock_path = Path(str(lock_path_raw))
        # Skip Git internal locks - handled separately in Phase 2
        if ".git" in lock_path.parts:
            continue
        summary["locks_scanned"] += 1
        # Check if this is a stale lock file (no owner or dead owner)
        metadata_path = lock_path.parent / f"{lock_path.name}.owner.json"
        try:
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                pid = metadata.get("pid")
                if pid is not None and _pid_alive(int(pid)):
                    continue  # Lock is still active
            # Remove stale lock file
            lock_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            summary["locks_removed"].append(str(lock_path))
        except (FileNotFoundError, PermissionError, json.JSONDecodeError):
            continue

    # --- Phase 2: Clean orphaned metadata files ---
    for metadata_path_raw in sorted(root.rglob("*.lock.owner.json"), key=str):
        metadata_path = Path(str(metadata_path_raw))
        name = metadata_path.name
        if not name.endswith(".owner.json"):
            continue
        lock_candidate = metadata_path.parent / name[: -len(".owner.json")]
        if lock_candidate.exists():
            continue
        try:
            await _to_thread(metadata_path.unlink, _timeout=5.0)
            summary["metadata_removed"].append(str(metadata_path))
        except FileNotFoundError:
            continue
        except PermissionError:
            continue

    # --- Phase 3: Clean stale Git internal locks ---
    # These are left behind when Git operations are interrupted (crash, timeout, etc.)
    # Git lock files: index.lock, HEAD.lock, config.lock, refs/heads/*.lock, etc.
    git_lock_patterns = [
        "index.lock",
        "HEAD.lock",
        "config.lock",
        "COMMIT_EDITMSG.lock",
        "MERGE_HEAD.lock",
        "FETCH_HEAD.lock",
        "ORIG_HEAD.lock",
    ]

    # Find all .git directories in the archive
    for git_dir in root.rglob(".git"):
        if not git_dir.is_dir():
            continue

        # Clean known Git lock files
        for lock_name in git_lock_patterns:
            lock_file = git_dir / lock_name
            if lock_file.exists():
                try:
                    # Check if lock is stale (file older than 5 minutes = likely orphaned)
                    mtime = lock_file.stat().st_mtime
                    age_seconds = time.time() - mtime
                    if age_seconds > 300:  # 5 minutes
                        await _to_thread(lock_file.unlink, _timeout=5.0)
                        summary["git_locks_removed"].append(str(lock_file))
                except (FileNotFoundError, PermissionError, OSError):
                    continue

        # Also check refs/heads/*.lock for stale branch locks
        refs_heads = git_dir / "refs" / "heads"
        if refs_heads.exists():
            for ref_lock in refs_heads.glob("*.lock"):
                try:
                    mtime = ref_lock.stat().st_mtime
                    age_seconds = time.time() - mtime
                    if age_seconds > 300:  # 5 minutes
                        await _to_thread(ref_lock.unlink, _timeout=5.0)
                        summary["git_locks_removed"].append(str(ref_lock))
                except (FileNotFoundError, PermissionError, OSError):
                    continue

    return summary


# ==================================================================================
# Git Archive Visualization & Analysis Helpers
# ==================================================================================


async def get_recent_commits(
    repo: Repo,
    limit: int = 50,
    project_slug: str | None = None,
    path_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Get recent commits from the Git repository.

    Args:
        repo: GitPython Repo object
        limit: Maximum number of commits to return
        project_slug: Optional slug to filter commits for specific project
        path_filter: Optional path pattern to filter commits

    Returns:
        List of commit dicts with keys: sha, short_sha, author, email, date,
        relative_date, subject, body, files_changed, insertions, deletions
    """
    def _get_commits() -> list[dict[str, Any]]:
        commits = []
        path_spec = None

        if project_slug:
            path_spec = f"projects/{project_slug}"
        elif path_filter:
            path_spec = path_filter

        # Get commits, optionally filtered by path (explicit kwargs for better typing)
        if path_spec:
            iterator = repo.iter_commits(paths=[path_spec], max_count=limit)
        else:
            iterator = repo.iter_commits(max_count=limit)

        for commit in iterator:
            # Parse commit stats
            files_changed = len(commit.stats.files)
            insertions = commit.stats.total["insertions"]
            deletions = commit.stats.total["deletions"]

            # Calculate relative date
            commit_time = datetime.fromtimestamp(commit.authored_date, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            delta = now - commit_time

            if delta.days > 30:
                relative_date = commit_time.strftime("%b %d, %Y")
            elif delta.days > 0:
                relative_date = f"{delta.days} day{'s' if delta.days != 1 else ''} ago"
            elif delta.seconds > 3600:
                hours = delta.seconds // 3600
                relative_date = f"{hours} hour{'s' if hours != 1 else ''} ago"
            elif delta.seconds > 60:
                minutes = delta.seconds // 60
                relative_date = f"{minutes} minute{'s' if minutes != 1 else ''} ago"
            else:
                relative_date = "just now"

            message_str = _ensure_str(commit.message)
            commits.append({
                "sha": commit.hexsha,
                "short_sha": commit.hexsha[:8],
                "author": commit.author.name,
                "email": commit.author.email,
                "date": commit_time.isoformat(),
                "relative_date": relative_date,
                "subject": message_str.split("\n")[0],
                "body": message_str,
                "files_changed": files_changed,
                "insertions": insertions,
                "deletions": deletions,
            })

        return commits

    result: list[dict[str, Any]] = await _to_thread(_get_commits)
    return result


async def get_commit_detail(
    repo: Repo, sha: str, max_diff_size: int = 5 * 1024 * 1024
) -> dict[str, Any]:
    """
    Get detailed information about a specific commit including full diff.

    Args:
        repo: GitPython Repo object
        sha: Commit SHA (full or abbreviated)
        max_diff_size: Maximum diff size in bytes (default 5MB)

    Returns:
        Dict with commit metadata and diff information
    """
    def _get_detail() -> dict[str, Any]:
        # Validate SHA format (basic check)
        if not sha or not (7 <= len(sha) <= 40) or not all(c in "0123456789abcdef" for c in sha.lower()):
            raise ValueError("Invalid commit SHA format")

        commit = repo.commit(sha)

        # Get parent for diff (use empty tree if initial commit)
        if commit.parents:
            parent = commit.parents[0]
            diffs = parent.diff(commit, create_patch=True)
        else:
            # Initial commit - diff against empty tree
            diffs = commit.diff(None, create_patch=True)

        # Build unified diff string
        diff_text = ""
        changed_files = []

        for diff in diffs:
            # File metadata
            a_path = diff.a_path or "/dev/null"
            b_path = diff.b_path or "/dev/null"

            # Change type
            if diff.new_file:
                change_type = "added"
            elif diff.deleted_file:
                change_type = "deleted"
            elif diff.renamed_file:
                change_type = "renamed"
            else:
                change_type = "modified"

            changed_files.append({
                "path": b_path if b_path != "/dev/null" else a_path,
                "change_type": change_type,
                "a_path": a_path,
                "b_path": b_path,
            })

            # Get diff text with size limit
            if diff.diff:
                diff_bytes = diff.diff
                if isinstance(diff_bytes, bytes):
                    decoded_diff = diff_bytes.decode("utf-8", errors="replace")
                else:
                    decoded_diff = str(diff_bytes)
                if len(diff_text) + len(decoded_diff) > max_diff_size:
                    diff_text += "\n\n[... Diff truncated - exceeds size limit ...]\n"
                    break
                diff_text += decoded_diff

        # Parse commit body into message and trailers
        message_str = _ensure_str(commit.message)
        lines = message_str.split("\n")
        subject = lines[0] if lines else ""

        # Find where trailers start (Git trailers are at end after blank line)
        # We scan backwards to find the trailer block
        body_lines: list[str] = []
        trailer_lines: list[str] = []

        rest_lines = lines[1:] if len(lines) > 1 else []
        if not rest_lines:
            body = ""
            body_lines = []
            trailer_lines = []
        else:
            # Find trailer block by scanning from end
            # Git trailers must be consecutive lines at the end
            # First, skip trailing blank lines to find last content
            end_idx = len(rest_lines) - 1
            while end_idx >= 0 and not rest_lines[end_idx].strip():
                end_idx -= 1

            # Now scan backwards collecting consecutive trailer-looking lines
            trailer_start_idx = end_idx + 1  # Default: no trailers
            for i in range(end_idx, -1, -1):
                line = rest_lines[i]
                # Trailers have format "Key: Value" with specific pattern
                if line.strip() and ": " in line and not line.startswith(" "):
                    # This looks like a trailer, keep going
                    trailer_start_idx = i
                else:
                    # Not a trailer (blank or other content), stop
                    break

            # Git spec: trailers must be separated from body by blank line
            # If we found trailers, verify there's a blank line before them
            if trailer_start_idx <= end_idx:  # We found some trailers
                if trailer_start_idx > 0:
                    # Check if line before trailers is blank
                    if rest_lines[trailer_start_idx - 1].strip():
                        # No blank line separator - these aren't trailers!
                        trailer_start_idx = end_idx + 1
                        trailer_lines = []
                    else:
                        # Valid trailers with blank separator
                        trailer_lines = rest_lines[trailer_start_idx:end_idx + 1]
                else:
                    # Trailers start at beginning (no body) - this is valid
                    trailer_lines = rest_lines[trailer_start_idx:end_idx + 1]
            else:
                trailer_lines = []

            # Split body and trailers
            body_lines = rest_lines[:trailer_start_idx]

        body = "\n".join(body_lines).strip()

        # Parse trailers into dict (only first occurrence of ": " to handle multiple colons)
        trailers = {}
        for line in trailer_lines:
            if ": " in line:
                parts = line.split(": ", 1)  # Split on first ": " only
                if len(parts) == 2:
                    trailers[parts[0].strip()] = parts[1].strip()

        commit_time = datetime.fromtimestamp(commit.authored_date, tz=timezone.utc)

        return {
            "sha": commit.hexsha,
            "short_sha": commit.hexsha[:8],
            "author": commit.author.name,
            "email": commit.author.email,
            "date": commit_time.isoformat(),
            "subject": subject,
            "body": body,
            "trailers": trailers,
            "files_changed": changed_files,
            "diff": diff_text,
            "stats": {
                "files": len(commit.stats.files),
                "insertions": commit.stats.total["insertions"],
                "deletions": commit.stats.total["deletions"],
            },
        }

    result: dict[str, Any] = await _to_thread(_get_detail)
    return result


async def get_message_commit_sha(archive: ProjectArchive, message_id: int) -> str | None:
    """
    Find the commit SHA that created a specific message.

    Args:
        archive: ProjectArchive instance
        message_id: Message ID to look up

    Returns:
        Commit SHA string or None if not found
    """
    def _find_commit() -> str | None:
        # Find message file in archive
        messages_dir = archive.root / "messages"

        if not messages_dir.exists():
            return None

        # Search for file ending with __{message_id}.md (limit search depth for performance)
        pattern = f"__{message_id}.md"

        # Use iterdir with depth limit instead of rglob for better performance
        for year_dir in messages_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                for md_file in month_dir.iterdir():
                    if md_file.is_file() and md_file.name.endswith(pattern):
                        try:
                            # Get relative path from repo root
                            rel_path = md_file.relative_to(archive.repo_root)

                            # Get FIRST commit that created this file (oldest, not most recent)
                            # iter_commits returns newest first, so we need to get all and take the last
                            # Limit to 1000 commits to prevent performance issues
                            commits_list = list(archive.repo.iter_commits(paths=[str(rel_path)], max_count=1000))
                            if commits_list:
                                # The last commit in the list is the oldest (first commit)
                                return commits_list[-1].hexsha
                        except (ValueError, StopIteration, FileNotFoundError, OSError):
                            # File may have been deleted or moved during iteration
                            continue

        return None

    result: str | None = await _to_thread(_find_commit)
    return result


async def get_archive_tree(
    archive: ProjectArchive,
    path: str = "",
    commit_sha: str | None = None,
) -> list[dict[str, Any]]:
    """
    Get directory tree structure from the Git archive.

    Args:
        archive: ProjectArchive instance
        path: Relative path within the project archive (e.g., "messages/2025")
        commit_sha: Optional commit SHA to view historical tree

    Returns:
        List of tree entries with keys: name, path, type (file/dir), size, mode
    """
    def _get_tree() -> list[dict[str, Any]]:
        # Sanitize path to prevent directory traversal
        if path:
            # Normalize path separators to forward slash
            normalized = path.replace("\\", "/")
            # Reject any path traversal patterns
            if (
                normalized.startswith("/")
                or normalized.startswith("..")
                or "/../" in normalized
                or normalized.endswith("/..")
                or normalized == ".."
            ):
                raise ValueError("Invalid path: directory traversal not allowed")
            safe_path = normalized.lstrip("/")
        else:
            safe_path = ""

        # Get commit (HEAD if not specified)
        if commit_sha:
            # Validate SHA format
            if not (7 <= len(commit_sha) <= 40) or not all(c in "0123456789abcdef" for c in commit_sha.lower()):
                raise ValueError("Invalid commit SHA format")
            commit = archive.repo.commit(commit_sha)
        else:
            commit = archive.repo.head.commit

        # Navigate to the requested path within project root
        project_rel = f"projects/{archive.slug}"
        tree_path = f"{project_rel}/{safe_path}" if safe_path else project_rel

        # Get tree object at path
        try:
            tree_obj = commit.tree / tree_path
        except KeyError:
            # Path doesn't exist
            return []

        # Ensure we have a tree object (not a blob)
        if not isinstance(tree_obj, Tree):
            return []

        entries = []
        for item in tree_obj:
            entry_type = "dir" if item.type == "tree" else "file"
            size = item.size if hasattr(item, "size") else 0

            entries.append({
                "name": item.name,
                "path": f"{path}/{item.name}" if path else item.name,
                "type": entry_type,
                "size": size,
                "mode": item.mode,
            })

        # Sort: directories first, then files, both alphabetically
        entries.sort(key=lambda x: (x["type"] != "dir", str(x["name"]).lower()))

        return entries

    result: list[dict[str, Any]] = await _to_thread(_get_tree)
    return result


async def get_file_content(
    archive: ProjectArchive,
    path: str,
    commit_sha: str | None = None,
    max_size_bytes: int = 10 * 1024 * 1024,  # 10MB default limit
) -> str | None:
    """
    Get file content from the Git archive.

    Args:
        archive: ProjectArchive instance
        path: Relative path within the project archive
        commit_sha: Optional commit SHA to view historical content
        max_size_bytes: Maximum file size to read (prevents DoS)

    Returns:
        File content as string, or None if not found
    """
    def _get_content() -> str | None:
        # Sanitize path to prevent directory traversal
        if path:
            # Normalize path separators to forward slash
            normalized = path.replace("\\", "/")
            # Reject any path traversal patterns
            if (
                normalized.startswith("/")
                or normalized.startswith("..")
                or "/../" in normalized
                or normalized.endswith("/..")
                or normalized == ".."
            ):
                raise ValueError("Invalid path: directory traversal not allowed")
            safe_path = normalized.lstrip("/")
        else:
            return None

        if commit_sha:
            # Validate SHA format
            if not (7 <= len(commit_sha) <= 40) or not all(c in "0123456789abcdef" for c in commit_sha.lower()):
                raise ValueError("Invalid commit SHA format")
            commit = archive.repo.commit(commit_sha)
        else:
            commit = archive.repo.head.commit

        project_rel = f"projects/{archive.slug}/{safe_path}"

        try:
            obj = commit.tree / project_rel
            # Check if it's a file (blob), not a directory (tree)
            if obj.type != "blob":
                raise ValueError("Path is a directory, not a file")
            # Check size before reading
            if obj.size > max_size_bytes:
                raise ValueError(f"File too large: {obj.size} bytes (max {max_size_bytes})")
            return str(obj.data_stream.read().decode("utf-8", errors="replace"))
        except KeyError:
            return None

    result: str | None = await _to_thread(_get_content)
    return result


async def get_agent_communication_graph(
    repo: Repo,
    project_slug: str,
    limit: int = 200,
) -> dict[str, Any]:
    """
    Analyze commit history to build an agent communication network graph.

    Args:
        repo: GitPython Repo object
        project_slug: Project slug to analyze
        limit: Maximum number of commits to analyze

    Returns:
        Dict with keys: nodes (list of agent dicts), edges (list of connection dicts)
    """
    def _analyze_graph() -> dict[str, Any]:
        path_spec = f"projects/{project_slug}/messages"

        # Track agent message counts and connections
        agent_stats: dict[str, dict[str, int]] = {}
        connections: dict[tuple[str, str], int] = {}

        for commit in repo.iter_commits(paths=[path_spec], max_count=limit):
            # Parse commit message to extract sender and recipients
            # Format: "mail: Sender -> Recipient1, Recipient2 | Subject"
            message_str = _ensure_str(commit.message)
            subject = message_str.split("\n")[0]

            if not subject.startswith("mail: "):
                continue

            # Extract sender and recipients
            try:
                rest = subject[len("mail: "):]
                sender_part, _ = rest.split(" | ", 1) if " | " in rest else (rest, "")

                if " -> " not in sender_part:
                    continue

                sender, recipients_str = sender_part.split(" -> ", 1)
                sender = str(sender).strip()
                recipients = [r.strip() for r in recipients_str.split(",")]

                # Update sender stats
                if sender not in agent_stats:
                    agent_stats[sender] = {"sent": 0, "received": 0}
                agent_stats[sender]["sent"] = agent_stats[sender].get("sent", 0) + 1

                # Update recipient stats and connections
                for recipient in recipients:
                    if not recipient:
                        continue

                    recipient = str(recipient)
                    if recipient not in agent_stats:
                        agent_stats[recipient] = {"sent": 0, "received": 0}
                    agent_stats[recipient]["received"] = agent_stats[recipient].get("received", 0) + 1

                    # Track connection
                    conn_key: tuple[str, str] = (sender, recipient)
                    connections[conn_key] = int(connections.get(conn_key, 0)) + 1

            except Exception:
                # Skip malformed commit messages
                continue

        # Build nodes list
        nodes = []
        for agent_name, stats in agent_stats.items():
            total = stats["sent"] + stats["received"]
            nodes.append({
                "id": agent_name,
                "label": agent_name,
                "sent": stats["sent"],
                "received": stats["received"],
                "total": total,
            })

        # Build edges list
        edges = []
        for (sender, recipient), count in connections.items():
            edges.append({
                "from": sender,
                "to": recipient,
                "count": count,
            })

        return {
            "nodes": nodes,
            "edges": edges,
        }

    result: dict[str, Any] = await _to_thread(_analyze_graph)
    return result


async def get_timeline_commits(
    repo: Repo,
    project_slug: str,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """
    Get commits formatted for timeline visualization with Mermaid.js.

    Args:
        repo: GitPython Repo object
        project_slug: Project slug to analyze
        limit: Maximum number of commits

    Returns:
        List of commit dicts with timeline-specific metadata
    """
    def _get_timeline() -> list[dict[str, Any]]:
        path_spec = f"projects/{project_slug}"

        timeline = []
        for commit in repo.iter_commits(paths=[path_spec], max_count=limit):
            message_str = _ensure_str(commit.message)
            subject = message_str.split("\n")[0]
            commit_time = datetime.fromtimestamp(commit.authored_date, tz=timezone.utc)

            # Classify commit type
            commit_type = "other"
            sender = None
            recipients = []

            if subject.startswith("mail: "):
                commit_type = "message"
                # Parse sender and recipients
                try:
                    rest = subject[len("mail: "):]
                    sender_part, _ = rest.split(" | ", 1) if " | " in rest else (rest, "")
                    if " -> " in sender_part:
                        sender, recipients_str = sender_part.split(" -> ", 1)
                        sender = sender.strip()
                        recipients = [r.strip() for r in recipients_str.split(",")]
                except Exception:
                    pass
            elif subject.startswith("file_reservation: "):
                commit_type = "file_reservation"
            elif subject.startswith("chore: "):
                commit_type = "chore"

            timeline.append({
                "sha": commit.hexsha,
                "short_sha": commit.hexsha[:8],
                "date": commit_time.isoformat(),
                "timestamp": commit.authored_date,
                "subject": subject,
                "type": commit_type,
                "sender": sender,
                "recipients": recipients,
                "author": commit.author.name,
            })

        # Sort by timestamp (oldest first for timeline)
        def _get_timestamp(x: dict[str, Any]) -> int:
            ts = x.get("timestamp", 0)
            return int(ts) if isinstance(ts, (int, float)) else 0
        timeline.sort(key=_get_timestamp)

        return timeline

    result: list[dict[str, Any]] = await _to_thread(_get_timeline)
    return result


async def get_historical_inbox_snapshot(
    archive: ProjectArchive,
    agent_name: str,
    timestamp: str,
    limit: int = 100,
) -> dict[str, Any]:
    """
    Get historical snapshot of agent inbox at specific timestamp.

    Traverses Git history to find the commit closest to (but not after)
    the specified timestamp, then lists all message files in the agent's
    inbox directory at that point in history.

    Args:
        archive: ProjectArchive instance with Git repo
        agent_name: Agent name to get inbox for
        timestamp: ISO 8601 timestamp (e.g., "2024-01-15T10:30:00")
        limit: Maximum messages to return (capped at 500)

    Returns:
        Dict with keys:
            - messages: List of message dicts with id, subject, date, from, importance
            - snapshot_time: ISO timestamp of the actual commit used
            - commit_sha: Git commit hash
            - requested_time: The original requested timestamp
    """
    # Cap limit for safety
    limit = max(1, min(limit, 500))

    def _get_snapshot() -> dict[str, Any]:
        try:
            # Parse timestamp - handle both with and without timezone
            timestamp_clean = timestamp.replace('Z', '+00:00')
            target_time = datetime.fromisoformat(timestamp_clean)

            # If naive datetime (no timezone), assume UTC
            # This handles datetime-local input which doesn't include timezone
            if target_time.tzinfo is None:
                target_time = target_time.replace(tzinfo=timezone.utc)

            target_timestamp = target_time.timestamp()
        except (ValueError, AttributeError) as e:
            return {
                "messages": [],
                "snapshot_time": None,
                "commit_sha": None,
                "requested_time": timestamp,
                "error": f"Invalid timestamp format: {e}"
            }

        # Find commit closest to (but not after) target timestamp
        closest_commit = None
        for commit in archive.repo.iter_commits(max_count=10000):
            if commit.authored_date <= target_timestamp:
                closest_commit = commit
                break

        if not closest_commit:
            # No commits before this time
            return {
                "messages": [],
                "snapshot_time": None,
                "commit_sha": None,
                "requested_time": timestamp,
                "note": "No commits found before this timestamp"
            }

        # Get agent inbox directory at that commit
        inbox_path = f"projects/{archive.slug}/agents/{agent_name}/inbox"

        messages = []
        try:
            # Navigate to the inbox folder in the commit tree
            tree = closest_commit.tree
            for part in inbox_path.split("/"):
                tree = tree / part

            # Recursively traverse inbox subdirectories (YYYY/MM/) to find message files
            def traverse_tree(subtree: Any, depth: int = 0) -> None:
                """Recursively traverse git tree looking for .md files"""
                if depth > 3:  # Safety limit: inbox/YYYY/MM is 2 levels, add buffer
                    return

                for item in subtree:
                    if item.type == "blob" and item.name.endswith(".md"):
                        # Parse filename: YYYY-MM-DDTHH-MM-SSZ__subject-slug__id.md
                        parts = item.name.rsplit("__", 2)

                        if len(parts) >= 2:
                            date_str = parts[0]
                            # Handle both 2-part and 3-part filenames
                            if len(parts) == 3:
                                subject_slug = parts[1]
                                msg_id = parts[2].replace(".md", "")
                            else:
                                # 2-part filename: date__subject.md
                                subject_slug = parts[1].replace(".md", "")
                                msg_id = "unknown"

                            # Convert slug back to readable subject
                            subject = subject_slug.replace("-", " ").replace("_", " ").title()

                            # Read file content to get From field and other metadata
                            from_agent = "unknown"
                            importance = "normal"

                            try:
                                blob_content = item.data_stream.read().decode('utf-8', errors='ignore')

                                # Parse JSON frontmatter (format: ---json\n{...}\n---)
                                if blob_content.startswith('---json\n') or blob_content.startswith('---json\r\n'):
                                    # Find the closing --- delimiter
                                    end_marker = blob_content.find('\n---\n', 8)
                                    if end_marker == -1:
                                        end_marker = blob_content.find('\r\n---\r\n', 8)

                                    if end_marker > 0:
                                        # Extract JSON between markers
                                        # '---json\n' is 8 chars, '---json\r\n' is 9 chars
                                        json_start = 8 if blob_content.startswith('---json\n') else 9
                                        json_str = blob_content[json_start:end_marker]

                                        try:
                                            metadata = json.loads(json_str)
                                            # Extract sender from 'from' field
                                            if 'from' in metadata:
                                                from_agent = str(metadata['from'])
                                            # Extract importance
                                            if 'importance' in metadata:
                                                importance = str(metadata['importance'])
                                            # Extract actual subject
                                            if 'subject' in metadata:
                                                actual_subject = str(metadata['subject']).strip()
                                                if actual_subject:
                                                    subject = actual_subject
                                        except (json.JSONDecodeError, KeyError, TypeError):
                                            pass  # Use defaults if JSON parsing fails

                            except Exception:
                                pass  # Use defaults if parsing fails

                            messages.append({
                                "id": msg_id,
                                "subject": subject,
                                "date": date_str,
                                "from": from_agent,
                                "importance": importance,
                            })

                            if len(messages) >= limit:
                                return  # Stop when we hit the limit

                    elif item.type == "tree":
                        # Recursively traverse subdirectory
                        traverse_tree(item, depth + 1)
                        if len(messages) >= limit:
                            return  # Stop when we hit the limit

            # Start recursive traversal
            traverse_tree(tree)

        except (KeyError, AttributeError):
            # Inbox directory didn't exist at that time
            pass

        # Sort messages by date (newest first)
        messages.sort(key=lambda m: m["date"], reverse=True)

        return {
            "messages": messages,
            "snapshot_time": closest_commit.authored_datetime.isoformat(),
            "commit_sha": closest_commit.hexsha,
            "requested_time": timestamp,
        }

    result: dict[str, Any] = await _to_thread(_get_snapshot)
    return result
