#!/usr/bin/env python3
"""Local HTTP Event/Callback/Outbox integration smoke test."""

from datetime import datetime, timedelta, timezone
import json

from simulation_core import (
    BackendMessageFactory, CallbackDispatcher, DeliveryStatus,
    HttpBackendClient, InMemoryCallbackOutbox, MockBackendServer,
    MockBackendState,
)


def main():
    state = MockBackendState(callback_failures=1)
    with MockBackendServer(state) as server:
        client = HttpBackendClient(server.base_url, request_timeout=1)
        factory = BackendMessageFactory()
        run_id, task_id = "http-smoke", "task-http-smoke"
        event_specs = (
            ("state", {"previous_status": "initializing", "status": "running"}),
            ("metric", {"step": 16, "metrics": {
                "average_reward": 0.2, "policy_loss": 0.1,
                "value_loss": 0.05, "entropy": 7.4,
            }}),
            ("heartbeat", {"status": "running", "training_steps": 16}),
            ("error", {"code": "TRAINING_FAILED", "message": "smoke", "retryable": False}),
        )
        events = []
        for event_type, payload in event_specs:
            event = factory.event(event_type, run_id, task_id, payload)
            client.publish_event(event)
            events.append(event)
        callback = factory.terminal_callback(
            run_id, task_id, status="completed",
            started_at="2026-08-23T10:00:00+00:00",
            ended_at="2026-08-23T10:01:00+00:00",
            training_steps=16, update_steps=1, stop_reason=None,
            metrics={"average_reward": 0.2},
            model={"type": "checkpoint_best", "path": "checkpoints/model_best.pt"},
            artifacts=[{"type": "manifest", "path": "manifest.json"}], error=None,
        )
        now = datetime.now(timezone.utc)
        outbox = InMemoryCallbackOutbox()
        record = outbox.put_if_absent(callback, now)
        dispatcher = CallbackDispatcher(outbox, client, max_attempts=3, base_delay_seconds=0)
        first = dispatcher.dispatch_event(record.event_id, now=now)
        second = dispatcher.dispatch_event(record.event_id, now=now + timedelta(microseconds=1))
        assert first.delivery_status == DeliveryStatus.PENDING
        assert second.delivery_status == DeliveryStatus.DELIVERED
        assert state.events == [event.to_dict() for event in events]
        assert state.callback_attempts == [callback.to_dict(), callback.to_dict()]
        assert state.callbacks == [callback.to_dict()]
        assert state.callback_attempts[0]["event_id"] == state.callback_attempts[1]["event_id"]
        assert state.callback_attempts[0]["sequence"] == state.callback_attempts[1]["sequence"]
        print(json.dumps({
            "base_url": server.base_url, "events_received": len(state.events),
            "callback_attempts": len(state.callback_attempts),
            "callback_status": second.delivery_status.value,
            "callback_event_id": callback.event_id,
            "callback_sequence": callback.sequence,
        }, indent=2))
    print("HTTP_BACKEND_SMOKE_RESULT=success")


if __name__ == "__main__":
    main()
