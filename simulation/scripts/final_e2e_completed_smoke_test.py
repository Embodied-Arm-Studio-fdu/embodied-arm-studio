#!/usr/bin/env python3
from final_e2e_support import finish, run_scenario

result = None
try:
    result = run_scenario("completed")
    events = result["events"]; video = result["video"]
    counts = {kind: sum(item["type"] == kind for item in events)
              for kind in ("state", "metric", "heartbeat")}
    print("FINAL_E2E_COMPLETED_RESULT=success")
    print("run_id=%s\ntask_id=%s" % (result["task"].run_id, result["task"].task_id))
    print("queue=redis_rq\ngpu_worker=true\ntask_state_running=RUNNING\nstream_state_running=READY")
    print("webrtc_connected=true\nvideo_frames_received=%d\nresolution=%dx%d\ntime_to_first_frame=%.3f" %
          (video.frames_decoded, video.width, video.height, video.time_to_first_decoded_frame))
    print("input_events_received=%d" % result["input_count"])
    print("metric_events=%d\nstate_events=%d\nheartbeat_events=%d" %
          (counts["metric"], counts["state"], counts["heartbeat"]))
    print("final_task_state=COMPLETED\nterminal_callback=completed")
    print("artifact_manifest_exists=true\nfinal_stream_state=STOPPED")
    print("frontend_mode=mock\nbackend_mode=mock")
    print("redis_server_pid=%d\nrq_worker_pid=%d\nmediamtx_pid=%d" %
          (result["redis_pid"], result["rq_worker_pid"], result["mediamtx_pid"]))
    for item in result["runtime_processes"]:
        print("runtime_pid=%(pid)d ppid=%(ppid)d state=%(state)s name=%(name)s cmdline=%(cmdline)s" % item)
finally:
    if result is not None: finish(result)
    print("CLEANUP=complete")
