"""Bounded generation jobs whose lifetime is independent of an HTTP reader.

Only the newest progress snapshot and the terminal event are retained. Progress
sequence numbers may therefore skip for a slow or reconnecting listener. These
events contain status and the authoritative saved-WAV URL, not PCM payloads;
coalescing progress never removes generated audio from the synthesis iterator.
"""

import copy
import json
import math
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from uuid import uuid4


MAX_EVENT_BYTES = 64 * 1024
MAX_SUBSCRIBERS = 64


class JobBusy(RuntimeError):
    """The shared inference slot or listener capacity is occupied."""


class JobConflict(ValueError):
    """A retained request identifier was reused for different input."""


class JobNotFound(LookupError):
    """The requested job does not exist or its retention period expired."""


def _finite_seconds(value, name, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f"{name} must be {'nonnegative' if allow_zero else 'positive'} and finite.")
    return float(value)


def _identifier(value, name, maximum):
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters.")
    return value


class GenerationJob:
    """One background iterator and its independently reconnectable listeners.

    ``subscribe()`` returns an opaque token. Pass that token to
    ``unsubscribe(token)`` exactly when that response closes; repeated cleanup
    of an old token is harmless. ``read(cursor)`` returns the latest event newer
    than the caller's sequence, or None after the bounded wait. Once ``done`` is
    set, the source is closed and the shared inference lock has been released.
    """

    def __init__(self, manager, job_id, payload_digest, source_factory, serializer):
        self.id = job_id
        self.payload_digest = payload_digest
        self.cancel_event = threading.Event()
        self.done = threading.Event()
        self.completed_at = None
        self._manager = manager
        self._source_factory = source_factory
        self._serializer = serializer
        self._condition = threading.Condition()
        self._latest = None
        self._sequence = 0
        self._subscribers = set()
        self._disconnect_epoch = 0
        self._disconnect_timer = None
        self._lease_released = False
        self._thread = threading.Thread(target=self._run, name=f"generation-{job_id[:32]}", daemon=True)

    def _release_lease(self):
        with self._condition:
            if self._lease_released:
                return
            self._lease_released = True
        self._manager.shared_lock.release()

    def _prepare_event(self, value):
        if not isinstance(value, Mapping):
            raise TypeError("The generation serializer must return an event object.")
        event = copy.deepcopy(dict(value))
        if "done" in event and not isinstance(event["done"], bool):
            raise ValueError("A generation event's done field must be a boolean.")
        event.setdefault("done", False)
        # Limit the retained event as well as the number of retained jobs.
        if len(json.dumps(event, allow_nan=False).encode("utf-8")) > MAX_EVENT_BYTES:
            raise ValueError("The generation update exceeds the event size limit.")
        return event

    def _publish_locked(self, event):
        self._sequence += 1
        self._latest = {**event, "generation_id": self.id, "sequence": self._sequence}
        self._condition.notify_all()

    @staticmethod
    def _error_event(exc, *, terminal=None):
        detail = f"{type(exc).__name__}: {exc}"[:2000]
        event = dict(terminal or {})
        event.update(status=f"Generation failed: {detail}", error=detail, done=True)
        return event

    def _complete(self, terminal):
        # No callbacks or manager locks while holding the job condition.
        with self._condition:
            timer, self._disconnect_timer = self._disconnect_timer, None
            self._disconnect_epoch += 1
            self._subscribers.clear()
            self.completed_at = self._manager.clock()
            self._publish_locked(terminal)
            self.done.set()
            self._condition.notify_all()
        if timer is not None:
            timer.cancel()
        self._manager._finished(self)

    def _run(self):
        source = None
        terminal = None
        try:
            source = iter(self._source_factory(self.cancel_event))
            for update in source:
                event = self._prepare_event(self._serializer(update))
                if event["done"]:
                    terminal = event
                    break
                with self._condition:
                    self._publish_locked(event)
            if terminal is None:
                raise RuntimeError("The generation stream ended before a completion event.")
        except BaseException as exc:
            terminal = self._error_event(exc)
        finally:
            try:
                if source is not None:
                    close = getattr(source, "close", None)
                    if close is not None:
                        close()
            except BaseException as exc:
                terminal = self._error_event(exc, terminal=terminal)
            finally:
                self._release_lease()
                self._complete(terminal or self._error_event(RuntimeError("Generation did not complete.")))

    def read(self, cursor=0, timeout=0.5):
        """Return a defensive copy of the latest event newer than cursor."""
        if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
            raise ValueError("The event cursor must be a nonnegative integer.")
        timeout = _finite_seconds(timeout, "timeout", allow_zero=True)
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._latest is None or self._latest["sequence"] <= cursor:
                if self.done.is_set():
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return copy.deepcopy(self._latest)

    def subscribe(self):
        """Attach a response and revoke any earlier disconnect deadline."""
        with self._condition:
            if len(self._subscribers) >= MAX_SUBSCRIBERS:
                raise JobBusy("Too many listeners are attached to this generation.")
            token = uuid4().hex
            self._subscribers.add(token)
            self._disconnect_epoch += 1
            timer, self._disconnect_timer = self._disconnect_timer, None
        if timer is not None:
            timer.cancel()
        return token

    def unsubscribe(self, token):
        """Detach once; cancel only this job after a reconnect grace period."""
        with self._condition:
            if token not in self._subscribers:
                return
            self._subscribers.remove(token)
            if self._subscribers or self.done.is_set():
                return
            self._disconnect_epoch += 1
            epoch = self._disconnect_epoch
            timer = threading.Timer(self._manager.disconnect_grace, self._disconnected, args=(epoch,))
            timer.daemon = True
            self._disconnect_timer = timer
        try:
            timer.start()
        except RuntimeError:
            # If a timer cannot start, avoid leaving inference orphaned forever.
            self._disconnected(epoch)

    def _disconnected(self, epoch):
        with self._condition:
            if self.done.is_set() or self._subscribers or epoch != self._disconnect_epoch:
                return
            self._disconnect_timer = None
            self.cancel_event.set()
            self._condition.notify_all()

    def stop(self):
        """A late stop of a completed job never affects a later generation."""
        with self._condition:
            if not self.done.is_set():
                self.cancel_event.set()
                self._condition.notify_all()


class GenerationJobs:
    """Serialize jobs with the legacy API and retain bounded completed results.

    TTL and maximum-count cleanup are opportunistic on submit/get/prune and at
    completion. Active jobs are never evicted. Idempotency is guaranteed only
    while a job remains retained; an expired identifier may be submitted again.
    No source factory, serializer or source close runs under the registry lock.
    """

    def __init__(self, *, shared_lock, max_completed=16, ttl_seconds=900,
                 disconnect_grace=30, clock=None):
        if isinstance(max_completed, bool) or not isinstance(max_completed, int) or max_completed < 1:
            raise ValueError("max_completed must be a positive integer.")
        self.shared_lock = shared_lock
        self.max_completed = max_completed
        self.ttl_seconds = _finite_seconds(ttl_seconds, "ttl_seconds")
        self.disconnect_grace = _finite_seconds(disconnect_grace, "disconnect_grace", allow_zero=True)
        self.clock = time.monotonic if clock is None else clock
        if not callable(self.clock):
            raise ValueError("clock must be callable.")
        self._jobs = OrderedDict()
        self._lock = threading.Lock()

    def _prune_locked(self):
        now = self.clock()
        completed = [(key, job) for key, job in self._jobs.items() if job.done.is_set()]
        for key, job in completed:
            if now - job.completed_at >= self.ttl_seconds:
                self._jobs.pop(key, None)
        retained = [(key, job) for key, job in completed if key in self._jobs]
        retained.sort(key=lambda item: item[1].completed_at)
        for key, _job in retained[:max(0, len(retained) - self.max_completed)]:
            self._jobs.pop(key, None)

    def prune(self):
        with self._lock:
            before = len(self._jobs)
            self._prune_locked()
            return before - len(self._jobs)

    def _finished(self, _job):
        with self._lock:
            self._prune_locked()

    def submit(self, job_id, payload_digest, source_factory, serializer):
        job_id = _identifier(job_id, "job_id", 128)
        payload_digest = _identifier(payload_digest, "payload_digest", 256)
        if not callable(source_factory) or not callable(serializer):
            raise ValueError("A source factory and serializer are required.")
        with self._lock:
            self._prune_locked()
            existing = self._jobs.get(job_id)
            if existing is not None:
                if existing.payload_digest != payload_digest:
                    raise JobConflict("This generation identifier already belongs to different input.")
                return existing
            if not self.shared_lock.acquire(blocking=False):
                raise JobBusy("Speech generation is already running. Wait or stop that generation.")
            try:
                job = GenerationJob(self, job_id, payload_digest, source_factory, serializer)
            except BaseException:
                self.shared_lock.release()
                raise
            self._jobs[job_id] = job
        try:
            job._thread.start()
        except BaseException as exc:
            job._release_lease()
            job._complete(job._error_event(exc))
            with self._lock:
                if self._jobs.get(job_id) is job:
                    self._jobs.pop(job_id)
            raise
        return job

    def get(self, job_id):
        job_id = _identifier(job_id, "job_id", 128)
        with self._lock:
            self._prune_locked()
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound("This generation is unavailable or its retention period expired.")
        return job

    def stop(self, job_id):
        job = self.get(job_id)
        job.stop()
        return job

    def __len__(self):
        with self._lock:
            self._prune_locked()
            return len(self._jobs)
