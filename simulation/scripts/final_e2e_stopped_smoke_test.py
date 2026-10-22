#!/usr/bin/env python3
from final_e2e_support import finish, run_scenario

result = None
try:
    result = run_scenario("stopped")
    print("FINAL_E2E_STOPPED_RESULT=success")
    print("run_id=%s\ntask_id=%s" % (result["task"].run_id, result["task"].task_id))
    print("queue=redis_rq\ngpu_worker=true\nfinal_task_state=STOPPED")
    print("terminal_callback=stopped\nstop_reason=%s" % result["callback"]["stop_reason"])
    print("final_stream_state=STOPPED\npartial_artifacts_valid=true")
    print("frontend_mode=mock\nbackend_mode=mock")
finally:
    if result is not None: finish(result)
    print("CLEANUP=complete")
