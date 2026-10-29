import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from t1000 import api


class RecordingFinishConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "reference.wav"
        self.path.write_bytes(b"published recording")
        self.token = "recording-retry-identity-0001"
        self.payload = api.RecordingPCM(token=self.token, pcm="AQI=")
        api.finished_references.clear()
        api.finish_locks.clear()
        with api.audio_lock:
            api.audio_files.clear()
        self.validation = patch.object(api, "_validate_reference", return_value=None)
        self.validation.start()
        self.addCleanup(self.validation.stop)
        self.addCleanup(api.finished_references.clear)
        self.addCleanup(api.finish_locks.clear)

    def consume(self, *_args, **_kwargs):
        return str(self.path), "Exact reference words", "Ready"

    async def test_failed_finish_handoff_retains_one_lock_and_publishes_once(self):
        started, release_first, release_third, release_finish, entered_finish = [asyncio.Event() for _ in range(5)]
        calls = []

        async def finish(_data):
            calls.append(1)
            if len(calls) == 1:
                started.set()
                await release_first.wait()
                raise ValueError("Transient first attempt")
            entered_finish.set()
            await release_finish.wait()
            return {"token": self.token}

        consume = Mock(side_effect=self.consume)
        recorder = SimpleNamespace(finish=finish, consume=consume)
        with patch.object(api, "recorder", recorder):
            first = asyncio.create_task(api.recording_finish(self.payload))
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(api.recording_finish(self.payload))

            async def third_request():
                await release_third.wait()
                return await api.recording_finish(self.payload)

            third = asyncio.create_task(third_request())
            await asyncio.sleep(0)
            # The third request is ready before release wakes the existing
            # waiter. Dropping an unlocked-but-used lock creates two holders.
            release_first.set()
            release_third.set()
            try:
                await asyncio.wait_for(entered_finish.wait(), 1)
                await asyncio.sleep(0)
                self.assertEqual(len(calls), 2)
                self.assertEqual(api.finish_locks[self.token].users, 2)
            finally:
                release_finish.set()
                results = await asyncio.gather(first, second, third, return_exceptions=True)
        self.assertIsInstance(results[0], HTTPException)
        self.assertEqual(results[1], results[2])
        self.assertEqual(consume.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(api.finish_locks[self.token].users, 0)

    async def test_cancel_handoff_does_not_remove_a_waiting_retry_lock(self):
        cancel_started, release_cancel, release_third, release_finish, entered_finish = [asyncio.Event() for _ in range(5)]
        calls = []

        async def cancel(_data):
            cancel_started.set()
            await release_cancel.wait()
            return {"cancelled": True}

        async def finish(_data):
            calls.append(1)
            entered_finish.set()
            await release_finish.wait()
            return {"token": self.token}

        # The fake keeps a recoverable result to isolate the API lease handoff;
        # the real recorder can instead report that this token was cancelled.
        consume = Mock(side_effect=self.consume)
        recorder = SimpleNamespace(finish=finish, cancel=cancel, consume=consume)
        with patch.object(api, "recorder", recorder):
            cancelling = asyncio.create_task(api.recording_cancel(api.RecordingToken(token=self.token)))
            await asyncio.wait_for(cancel_started.wait(), 1)
            second = asyncio.create_task(api.recording_finish(self.payload))

            async def third_request():
                await release_third.wait()
                return await api.recording_finish(self.payload)

            third = asyncio.create_task(third_request())
            await asyncio.sleep(0)
            release_cancel.set()
            release_third.set()
            try:
                await asyncio.wait_for(entered_finish.wait(), 1)
                await asyncio.sleep(0)
                self.assertEqual(len(calls), 1)
                self.assertEqual(api.finish_locks[self.token].users, 2)
            finally:
                release_finish.set()
                results = await asyncio.gather(cancelling, second, third, return_exceptions=True)
        self.assertEqual(results[0], {"cancelled": True})
        self.assertEqual(results[1], results[2])
        self.assertEqual(consume.call_count, 1)

    async def test_cancelled_waiter_decrements_lease_without_removing_holder(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def holder():
            async with api._recording_lease(self.token):
                entered.set()
                await release.wait()

        async def waiter():
            async with api._recording_lease(self.token):
                self.fail("A cancelled waiter must not acquire the lease")

        owner = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 1)
        waiting = asyncio.create_task(waiter())
        try:
            await asyncio.sleep(0)
            lease = api.finish_locks[self.token]
            self.assertEqual(lease.users, 2)
            waiting.cancel()
            result = await asyncio.gather(waiting, return_exceptions=True)
            self.assertIsInstance(result[0], asyncio.CancelledError)
            self.assertIs(api.finish_locks[self.token], lease)
            self.assertEqual(lease.users, 1)
        finally:
            release.set()
            await owner
        self.assertNotIn(self.token, api.finish_locks)

    async def test_expiry_and_capacity_eviction_keep_an_in_use_lease(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def holder():
            async with api._recording_lease(self.token):
                entered.set()
                await release.wait()

        async def waiter():
            async with api._recording_lease(self.token):
                pass

        owner = asyncio.create_task(holder())
        await asyncio.wait_for(entered.wait(), 1)
        waiting = asyncio.create_task(waiter())
        try:
            await asyncio.sleep(0)
            lease = api.finish_locks[self.token]
            api.finished_references[self.token] = (time.monotonic() - api.FINISH_CACHE_TTL - 1, "digest", {}, str(self.path))
            api._prune_finished_references()
            self.assertNotIn(self.token, api.finished_references)
            self.assertIs(api.finish_locks[self.token], lease)
            self.assertEqual(lease.users, 2)
            api.finished_references[self.token] = (time.monotonic(), "digest", {}, str(self.path))
            api.finished_references["newer-recording-token"] = (time.monotonic(), "digest", {}, str(self.path))
            with patch.object(api, "FINISH_CACHE_LIMIT", 1):
                api._prune_finished_references()
            self.assertNotIn(self.token, api.finished_references)
            self.assertIs(api.finish_locks[self.token], lease)
        finally:
            release.set()
            await asyncio.gather(owner, waiting)
        self.assertNotIn(self.token, api.finish_locks)


if __name__ == "__main__":
    unittest.main()
