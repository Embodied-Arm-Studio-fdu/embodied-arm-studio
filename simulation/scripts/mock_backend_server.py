#!/usr/bin/env python3
"""Run the tiny Phase 5.4 local mock backend; no database or platform APIs."""

import argparse
import time

from simulation_core import MockBackendServer, MockBackendState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--event-failures", type=int, default=0)
    parser.add_argument("--callback-failures", type=int, default=0)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    args = parser.parse_args()
    state = MockBackendState(
        args.event_failures, args.callback_failures, args.delay_seconds
    )
    server = MockBackendServer(state, args.host, args.port).start()
    print("MOCK_BACKEND_URL=%s" % server.base_url, flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("mock backend stopping", flush=True)
    finally:
        server.close()


if __name__ == "__main__":
    main()
