import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from t1000.jobs import GenerationJobs, JobBusy, JobConflict, JobNotFound, MAX_EVENT_BYTES


def identity(update):
    return update


class GenerationJobTests(unittest.TestCase):
    def setUp(self):
        self.lock = threading.Lock()
        self.manager = GenerationJobs(shared_lock=self.lock, disconnect_grace=0.08)
        self.to_stop = []
        self.addCleanup(self.cleanup_jobs)

    def cleanup_jobs(self):
        for job in self.to_stop:
            job.stop()
        for job in self.to_stop:
            self.assertTrue(job.done.wait(2), f"Worker did not stop: {job.id}")

    def submit(self, name, source, *, manager=None, digest="input-a", serializer=identity):
        selected = self.manager if manager is None else manager
        job = selected.submit(name, digest, source, serializer)
        self.to_stop.append(job)
        return job

    @staticmethod
    def blocking_source(cancel):
        yield {"status": "Working", "done": False}
        cancel.wait(2)
        yield {"status": "Stopped", "done": True, "audio_url": "/api/audio/partial"}

    def assert_released(self):
        self.assertTrue(self.lock.acquire(blocking=False))
        self.lock.release()

    def test_completed_request_replays_without_synthesizing_again(self):
        calls = []

        def source(cancel):
            calls.append(cancel)
            yield {"status": "First", "done": False}
            yield {"status": "Second", "done": False}
            yield {"status": "Saved", "done": True, "audio_url": "/api/audio/final"}

        job = self.submit("same-job", source)
        self.assertTrue(job.done.wait(2))
        duplicate = self.manager.submit("same-job", "input-a", Mock(), Mock())
        self.assertIs(duplicate, job)
        self.assertEqual(len(calls), 1)
        token = duplicate.subscribe()
        event = duplicate.read(0)
        duplicate.unsubscribe(token)
        self.assertEqual(event["generation_id"], "same-job")
        self.assertEqual(event["sequence"], 3)
        self.assertEqual(event["audio_url"], "/api/audio/final")
        self.assertIsNone(duplicate.read(event["sequence"]))
        self.assert_released()

    def test_concurrent_retries_share_one_worker(self):
        release = threading.Event()
        calls = []
        barrier = threading.Barrier(8)

        def source(cancel):
            calls.append(cancel)
            yield {"done": False}
            while not release.wait(0.005) and not cancel.is_set():
                pass
            yield {"done": True}

        def submit_once(_index):
            barrier.wait(timeout=2)
            return self.manager.submit("concurrent", "input-a", source, identity)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(submit_once, range(8)))
        job = results[0]
        self.to_stop.append(job)
        self.assertTrue(all(candidate is job for candidate in results))
        job.read(0)
        self.assertEqual(len(calls), 1)
        release.set()
        self.assertTrue(job.done.wait(2))
        self.assert_released()

    def test_disconnect_reconnect_and_duplicate_cleanup_do_not_restart_or_cancel(self):
        release = threading.Event()
        calls = []

        def source(cancel):
            calls.append(cancel)
            yield {"status": "Working", "done": False}
            while not release.wait(0.005) and not cancel.is_set():
                pass
            yield {"status": "Finished", "done": True}

        job = self.submit("reconnect", source)
        old = job.subscribe()
        progress = job.read(0)
        self.assertFalse(progress["done"])
        job.unsubscribe(old)
        current = job.subscribe()
        # A second cleanup of an old response must not detach the new response.
        job.unsubscribe(old)
        self.assertFalse(job.cancel_event.wait(0.13))
        self.assertEqual(len(calls), 1)
        self.assertIs(self.manager.get("reconnect"), job)
        release.set()
        self.assertTrue(job.done.wait(2))
        terminal = job.read(progress["sequence"])
        self.assertTrue(terminal["done"])
        job.unsubscribe(current)
        self.assert_released()

    def test_disconnected_job_cancels_only_after_grace_and_saves_terminal(self):
        job = self.submit("orphan", self.blocking_source)
        token = job.subscribe()
        job.read(0)
        job.unsubscribe(token)
        self.assertTrue(job.done.wait(2))
        self.assertTrue(job.cancel_event.is_set())
        self.assertEqual(job.read(0)["audio_url"], "/api/audio/partial")
        self.assert_released()

    def test_shared_slot_conflict_and_late_stop_are_request_scoped(self):
        first = self.submit("first", lambda cancel: iter([{"done": True}]))
        self.assertTrue(first.done.wait(2))
        second = self.submit("second", self.blocking_source)
        token = second.subscribe()
        second.read(0)
        with self.assertRaises(JobBusy):
            self.manager.submit("third", "third-input", self.blocking_source, identity)
        with self.assertRaises(JobConflict):
            self.manager.submit("second", "changed-input", self.blocking_source, identity)
        self.assertIs(self.manager.submit("second", "input-a", Mock(), Mock()), second)
        self.manager.stop("first")
        self.assertFalse(second.cancel_event.is_set())
        self.manager.stop("second")
        self.assertTrue(second.done.wait(2))
        second.unsubscribe(token)
        self.assert_released()

    def test_legacy_inference_lock_prevents_new_job(self):
        self.lock.acquire()
        try:
            with self.assertRaises(JobBusy):
                self.manager.submit("occupied", "input", Mock(), identity)
            self.assertEqual(len(self.manager), 0)
        finally:
            self.lock.release()

    def test_completed_count_and_ttl_never_evict_active_generation(self):
        now = [0.0]
        manager = GenerationJobs(shared_lock=self.lock, max_completed=2,
                                 ttl_seconds=10, clock=lambda: now[0])
        for index in range(3):
            now[0] = index
            job = self.submit(f"completed-{index}", lambda cancel: iter([{"done": True}]), manager=manager)
            self.assertTrue(job.done.wait(2))
            # Completion pruning may run immediately after the Event is set.
            manager.prune()
        self.assertEqual(len(manager), 2)
        with self.assertRaises(JobNotFound):
            manager.get("completed-0")
        active = self.submit("active", self.blocking_source, manager=manager)
        token = active.subscribe()
        active.read(0)
        now[0] = 100
        manager.prune()
        self.assertIs(manager.get("active"), active)
        self.assertEqual(len(manager), 1)
        self.assertFalse(self.lock.acquire(blocking=False))
        manager.stop("active")
        self.assertTrue(active.done.wait(2))
        active.unsubscribe(token)
        now[0] = 111
        with self.assertRaises(JobNotFound):
            manager.get("active")
        self.assert_released()

    def test_serializer_failure_closes_source_and_releases_slot(self):
        closed = threading.Event()

        def source(cancel):
            try:
                yield {"done": False}
                yield {"done": True}
            finally:
                closed.set()

        job = self.submit("bad-serialization", source,
                          serializer=Mock(side_effect=ValueError("bad update")))
        self.assertTrue(job.done.wait(2))
        self.assertTrue(closed.is_set())
        result = job.read(0)
        self.assertTrue(result["done"])
        self.assertIn("bad update", result["error"])
        self.assert_released()

    def test_missing_terminal_and_factory_failure_are_visible(self):
        missing = self.submit("missing-terminal", lambda cancel: iter([{"done": False}]))
        self.assertTrue(missing.done.wait(2))
        self.assertIn("before a completion", missing.read(0)["error"])
        failure = self.submit("factory-failure", Mock(side_effect=RuntimeError("factory failed")))
        self.assertTrue(failure.done.wait(2))
        self.assertIn("factory failed", failure.read(0)["error"])
        self.assert_released()

    def test_close_failure_is_terminal_before_releasing_a_success(self):
        class BadClose:
            def __init__(self):
                self.used = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.used:
                    raise StopIteration
                self.used = True
                return {"done": True, "audio_url": "/api/audio/saved"}

            def close(self):
                raise RuntimeError("cleanup failed")

        job = self.submit("bad-close", lambda cancel: BadClose())
        self.assertTrue(job.done.wait(2))
        event = job.read(0)
        self.assertIn("cleanup failed", event["error"])
        self.assertEqual(event["audio_url"], "/api/audio/saved")
        self.assert_released()

    def test_thread_start_failure_does_not_keep_slot_or_cached_job(self):
        source = Mock()
        with patch("t1000.jobs.threading.Thread.start", side_effect=RuntimeError("no thread")):
            with self.assertRaisesRegex(RuntimeError, "no thread"):
                self.manager.submit("cannot-start", "input-a", source, identity)
        source.assert_not_called()
        self.assertEqual(len(self.manager), 0)
        self.assert_released()

    def test_thread_construction_failure_does_not_leak_shared_slot(self):
        source = Mock()
        with patch("t1000.jobs.GenerationJob", side_effect=RuntimeError("cannot create thread")):
            with self.assertRaisesRegex(RuntimeError, "cannot create thread"):
                self.manager.submit("cannot-construct", "input-a", source, identity)
        source.assert_not_called()
        self.assertEqual(len(self.manager), 0)
        self.assert_released()

    def test_events_are_bounded_defensive_copies_with_server_identity(self):
        job = self.submit("metadata", lambda cancel: iter([
            {"done": True, "generation_id": "forged", "sequence": 999, "timings": {"x": 1}},
        ]))
        self.assertTrue(job.done.wait(2))
        first = job.read(0)
        self.assertEqual((first["generation_id"], first["sequence"]), ("metadata", 1))
        first["timings"]["x"] = 2
        self.assertEqual(job.read(0)["timings"]["x"], 1)
        oversized = self.submit("oversized", lambda cancel: iter([
            {"done": False, "status": "x" * MAX_EVENT_BYTES},
        ]))
        self.assertTrue(oversized.done.wait(2))
        self.assertIn("size limit", oversized.read(0)["error"])
        self.assert_released()

    def test_progress_snapshots_can_skip_but_terminal_is_retained(self):
        ready, finish = threading.Event(), threading.Event()

        def source(cancel):
            yield {"done": False, "status": "First"}
            yield {"done": False, "status": "Second"}
            ready.set()
            while not finish.wait(0.005) and not cancel.is_set():
                pass
            yield {"done": True, "status": "Saved"}

        job = self.submit("snapshots", source)
        token = job.subscribe()
        self.assertTrue(ready.wait(2))
        snapshot = job.read(0)
        self.assertEqual((snapshot["sequence"], snapshot["status"]), (2, "Second"))
        finish.set()
        self.assertTrue(job.done.wait(2))
        terminal = job.read(snapshot["sequence"])
        self.assertEqual((terminal["sequence"], terminal["status"]), (3, "Saved"))
        self.assertIsNone(job.read(terminal["sequence"], timeout=0))
        job.unsubscribe(token)


if __name__ == "__main__":
    unittest.main()
