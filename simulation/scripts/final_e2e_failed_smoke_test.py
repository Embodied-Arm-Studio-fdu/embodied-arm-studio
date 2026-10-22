#!/usr/bin/env python3
from final_e2e_support import finish, run_scenario

result = None
try:
    result = run_scenario("failed")
    print("FINAL_E2E_FAILED_RESULT=success")
    print("run_id=%s\ntask_id=%s" % (result["task"].run_id, result["task"].task_id))
    print("failure_injection=training_after_running")
    print("error_event=%s" % str(any(item["type"] == "error" for item in result["events"])).lower())
    print("final_task_state=FAILED\nterminal_callback=failed")
    print("final_stream_state=STOPPED\npartial_artifacts_valid=true")
    print("frontend_mode=mock\nbackend_mode=mock")
finally:
    if result is not None: finish(result)
    print("CLEANUP=complete")
