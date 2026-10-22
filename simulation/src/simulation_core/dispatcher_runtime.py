"""Minimal polling runtime for a CallbackDispatcher."""

import time


def run_polling_loop(dispatcher, polling_interval=1.0, stop_requested=None, sleep=None):
    if polling_interval <= 0:
        raise ValueError("polling_interval must be positive")
    should_stop = stop_requested or (lambda: False)
    wait = sleep or time.sleep
    while not should_stop():
        dispatcher.dispatch_once()
        wait(polling_interval)
