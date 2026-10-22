import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import unittest

from simulation_core.input_events import (
    InputEvent, InputEventConflict, InputEventType, MockInputWebSocketServer,
    RecordingInputEventSink,
)


def base(kind, **fields):
    values = dict(event_id="event-1", run_id="run-1", task_id="task-1",
                  type=kind, occurred_at=datetime.now(timezone.utc))
    values.update(fields); return values


class InputEventTests(unittest.TestCase):
    def test_keyboard_pressed_released_and_validation(self):
        for action in ("pressed", "released"):
            self.assertEqual(InputEvent(**base("KEYBOARD", key="W", action=action)).action, action)
        with self.assertRaises(ValueError): InputEvent(**base("KEYBOARD", key="Escape;rm", action="pressed"))
        with self.assertRaises(ValueError): InputEvent(**base("KEYBOARD", key="W", action="repeat"))

    def test_mouse_move_button_and_coordinates(self):
        InputEvent(**base("MOUSE_MOVE", x=0.0, y=1.0))
        for action in ("pressed", "released"):
            InputEvent(**base("MOUSE_BUTTON", button="left", action=action, x=.4, y=.6))
        for coordinate in (-.1, 1.1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError): InputEvent(**base("MOUSE_MOVE", x=coordinate, y=.5))

    def test_malformed_unknown_and_invalid_ids(self):
        for value in ("{", json.dumps({"type": "UNKNOWN"})):
            with self.assertRaises(ValueError): InputEvent.from_json(value)
        with self.assertRaises(ValueError): InputEvent(**base("KEYBOARD", run_id="../run", key="W", action="pressed"))

    def test_json_contract_rejects_unknown_fields_and_non_utc_times(self):
        valid = base("KEYBOARD", key="W", action="pressed")
        valid["occurred_at"] = valid["occurred_at"].isoformat()
        self.assertEqual("event-1", InputEvent.from_json(json.dumps(valid)).event_id)
        for occurred_at in ("2026-08-28T01:00:00+01:00", "2026-08-28T00:00:00", "bad"):
            invalid = dict(valid, occurred_at=occurred_at)
            with self.subTest(occurred_at=occurred_at), self.assertRaises(ValueError):
                InputEvent.from_json(json.dumps(invalid))
        with self.assertRaises(ValueError):
            InputEvent.from_json(json.dumps(dict(valid, extra=True)))
        with self.assertRaises(ValueError):
            InputEvent(**base("KEYBOARD", occurred_at=datetime.now(timezone(timedelta(hours=1))), key="W", action="pressed"))

    def test_recording_sink_preserves_order(self):
        sink = RecordingInputEventSink()
        items = [InputEvent(**base("KEYBOARD", event_id="event-%d" % i, key="W", action="pressed")) for i in range(3)]
        [sink.record(item) for item in items]
        self.assertEqual([item.event_id for item in sink.events], ["event-0", "event-1", "event-2"])

    def test_recording_sink_deduplicates_conflicts_and_isolates_tasks(self):
        sink = RecordingInputEventSink()
        first = InputEvent(**base("KEYBOARD", key="W", action="pressed"))
        self.assertFalse(sink.record(first).duplicate)
        self.assertTrue(sink.record(first).duplicate)
        with self.assertRaises(InputEventConflict):
            sink.record(InputEvent(**base("KEYBOARD", key="A", action="pressed")))
        other = InputEvent(**base("KEYBOARD", task_id="task-2", key="A", action="pressed"))
        self.assertFalse(sink.record(other).duplicate)
        self.assertEqual(1, len(sink.events_for_task("task-1")))
        self.assertEqual(1, len(sink.events_for_task("task-2")))

    def test_recording_sink_capacity_evicts_oldest_and_resets_dedup_window(self):
        sink = RecordingInputEventSink(max_events_per_task=2)
        items = [
            InputEvent(**base("KEYBOARD", event_id="event-%d" % index, key="W", action="pressed"))
            for index in range(3)
        ]
        for item in items:
            sink.record(item)
        self.assertEqual(["event-1", "event-2"], [item.event_id for item in sink.events_for_task("task-1")])
        self.assertFalse(sink.record(items[0]).duplicate)
        self.assertEqual(["event-2", "event-0"], [item.event_id for item in sink.events_for_task("task-1")])

    def test_recording_sink_is_concurrent_duplicate_safe(self):
        sink = RecordingInputEventSink()
        event = InputEvent(**base("KEYBOARD", key="W", action="pressed"))
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: sink.record(event), range(64)))
        self.assertEqual(1, sum(not result.duplicate for result in results))
        self.assertEqual(1, len(sink.events_for_task("task-1")))
        external = sink.events_for_task("task-1")
        external.clear()
        self.assertEqual(1, len(sink.events_for_task("task-1")))

    def test_websocket_handler_error_isolation_without_socket(self):
        class FakeSocket:
            def __init__(self, messages): self.messages = iter(messages); self.sent = []
            def __aiter__(self): return self
            async def __anext__(self):
                try: return next(self.messages)
                except StopIteration: raise StopAsyncIteration
            async def send(self, value): self.sent.append(json.loads(value))
        valid = base("KEYBOARD", key="W", action="pressed")
        valid["occurred_at"] = valid["occurred_at"].isoformat()
        socket = FakeSocket(["{", json.dumps(valid)])
        sink = RecordingInputEventSink(); server = MockInputWebSocketServer(sink)
        asyncio.run(server.handle(socket))
        self.assertFalse(socket.sent[0]["ok"]); self.assertTrue(socket.sent[1]["ok"])
        self.assertEqual(len(sink.events), 1)


if __name__ == "__main__": unittest.main()
