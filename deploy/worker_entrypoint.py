"""Finite GPU preflight; isolate native probes before starting exactly one RQ worker."""
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

PROBES=('cuda','physx','nvenc','vulkan','camera')

def mark(probe,step,status):
    print('A2_NATIVE '+json.dumps({'probe':probe,'step':step,'status':status,'pid':os.getpid()},sort_keys=True),flush=True)

def native(probe,step,function,*args):
    mark(probe,step,'BEGIN')
    value=function(*args)
    mark(probe,step,'PASS')
    return value

def validate_reference_models(root=Path('/app/simulation/artifacts'),manifest=Path('/app/deploy/reference-models.json')):
    for item in json.loads(Path(manifest).read_text())['files']:
        p=Path(root)/item['path']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('Missing or mismatched reference model: '+item['path'])

def check_cuda():
    mark('cuda','import_isaacgym','BEGIN')
    from isaacgym import gymapi
    mark('cuda','import_isaacgym','PASS')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    native('cuda','cuda_init',torch.cuda.init)
    device=torch.cuda.current_device()
    mark('cuda','compute_and_synchronize','BEGIN')
    value=torch.ones(8,device='cuda:%d'%device).square().sum()
    torch.cuda.synchronize(device)
    if value.item()!=8.0:raise RuntimeError('CUDA computation mismatch')
    mark('cuda','compute_and_synchronize','PASS')

def check_scene(camera):
    probe='camera' if camera else 'physx'
    mark(probe,'import_isaacgym','BEGIN')
    from isaacgym import gymapi
    mark(probe,'import_isaacgym','PASS')
    # No PyTorch CUDA context is created in this process before Gym/Vulkan.
    gym=native(probe,'acquire_gym',gymapi.acquire_gym)
    params=gymapi.SimParams()
    params.dt=1.0/60.0;params.substeps=2
    params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81)
    params.physx.use_gpu=True;params.physx.solver_type=1
    params.physx.num_position_iterations=4;params.physx.num_velocity_iterations=1
    # Camera uses CPU RGBA readback, while PhysX still executes on GPU.
    # CPU readback does not disable GPU PhysX (Preview 4 SDK tensor documentation).
    params.use_gpu_pipeline=False
    sim=native(probe,'create_sim',gym.create_sim,0,0 if camera else -1,gymapi.SIM_PHYSX,params)
    if sim is None:raise RuntimeError('Failed to create '+probe+' simulator')
    env=None;handle=None
    try:
        plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1)
        native(probe,'add_ground',gym.add_ground,sim,plane)
        options=gymapi.AssetOptions();options.fix_base_link=False
        box=native(probe,'create_box',gym.create_box,sim,0.1,0.1,0.1,options)
        if box is None:raise RuntimeError('Failed to create finite test rigid body')
        env=native(probe,'create_env',gym.create_env,sim,gymapi.Vec3(-1,-1,0),gymapi.Vec3(1,1,1),1)
        if env is None:raise RuntimeError('Failed to create test environment')
        pose=gymapi.Transform();pose.p=gymapi.Vec3(0,0,0.5)
        actor=native(probe,'create_actor',gym.create_actor,env,box,pose,'a2_preflight_box',0,0)
        if actor is None or actor<0:raise RuntimeError('Failed to create test actor')
        if camera:
            props=gymapi.CameraProperties();props.width=props.height=64;props.enable_tensors=False
            handle=native(probe,'create_camera_sensor',gym.create_camera_sensor,env,props)
            if handle is None or handle<0:raise RuntimeError('Camera sensor creation failed')
            native(probe,'set_camera_location',gym.set_camera_location,handle,env,gymapi.Vec3(1.5,1.5,1),gymapi.Vec3(0,0,0.3))
        native(probe,'prepare_sim',gym.prepare_sim,sim)
        native(probe,'simulate',gym.simulate,sim)
        native(probe,'fetch_results',gym.fetch_results,sim,True)
        if camera:
            native(probe,'step_graphics',gym.step_graphics,sim)
            native(probe,'render_camera',gym.render_all_camera_sensors,sim)
            frame=native(probe,'read_rgba',gym.get_camera_image,sim,env,handle,gymapi.IMAGE_COLOR)
            if frame is None or frame.size!=64*64*4 or str(frame.dtype)!='uint8':raise RuntimeError('Camera did not return 64x64 RGBA8 data')
        else:
            states=native(probe,'read_actor_state',gym.get_actor_rigid_body_states,env,actor,gymapi.STATE_POS)
            import math
            if states is None or len(states)<1 or not all(math.isfinite(float(states['pose']['p'][axis][0])) for axis in ('x','y','z')):raise RuntimeError('Non-finite GPU PhysX actor state')
    finally:
        try:
            if handle is not None and handle>=0:native(probe,'destroy_camera_sensor',gym.destroy_camera_sensor,sim,env,handle)
        finally:native(probe,'destroy_sim',gym.destroy_sim,sim)

def check_vulkan():
    import ctypes as c
    mark('vulkan','load_and_enumerate','BEGIN')
    class Application(c.Structure):
        _fields_=[('sType',c.c_uint32),('pNext',c.c_void_p),('pApplicationName',c.c_char_p),('applicationVersion',c.c_uint32),('pEngineName',c.c_char_p),('engineVersion',c.c_uint32),('apiVersion',c.c_uint32)]
    class Create(c.Structure):
        _fields_=[('sType',c.c_uint32),('pNext',c.c_void_p),('flags',c.c_uint32),('pApplicationInfo',c.POINTER(Application)),('enabledLayerCount',c.c_uint32),('ppEnabledLayerNames',c.POINTER(c.c_char_p)),('enabledExtensionCount',c.c_uint32),('ppEnabledExtensionNames',c.POINTER(c.c_char_p))]
    vk=c.CDLL('libvulkan.so.1')
    create=vk.vkCreateInstance;create.argtypes=[c.POINTER(Create),c.c_void_p,c.POINTER(c.c_void_p)];create.restype=c.c_int32
    destroy=vk.vkDestroyInstance;destroy.argtypes=[c.c_void_p,c.c_void_p];destroy.restype=None
    enum=vk.vkEnumeratePhysicalDevices;enum.argtypes=[c.c_void_p,c.POINTER(c.c_uint32),c.POINTER(c.c_void_p)];enum.restype=c.c_int32
    props=vk.vkGetPhysicalDeviceProperties;props.argtypes=[c.c_void_p,c.c_void_p];props.restype=None
    app=Application(0,None,b'A2 GPU preflight',1,b'A2',1,1<<22)
    config=Create(1,None,0,c.pointer(app),0,None,0,None)
    instance=c.c_void_p()
    rc=create(c.byref(config),None,c.byref(instance))
    if rc!=0 or not instance.value:raise RuntimeError('Vulkan vkCreateInstance failed: '+str(rc)+'; Camera will not run')
    try:
        count=c.c_uint32()
        rc=enum(instance,c.byref(count),None)
        if rc!=0 or not 1<=count.value<=64:raise RuntimeError('Vulkan physical-device count invalid: '+str((rc,count.value)))
        devices=(c.c_void_p*count.value)();rc=enum(instance,c.byref(count),devices)
        if rc!=0:raise RuntimeError('Vulkan physical-device enumeration failed: '+str(rc))
        facts=[]
        for device in devices[:count.value]:
            # VkPhysicalDeviceProperties is smaller than this aligned buffer. First fields are fixed by Vulkan 1.0.
            data=(c.c_uint64*512)();props(device,c.byref(data));raw=c.string_at(c.addressof(data),4096)
            import struct
            api,driver,vendor,deviceid,kind=struct.unpack_from('=5I',raw)
            name=raw[20:276].split(b'\0',1)[0].decode('utf-8',errors='replace')
            facts.append({'api_version':api,'vendor_id':vendor,'device_id':deviceid,'type':kind,'name':name})
        if facts[0]['vendor_id']!=0x10de or facts[0]['type']!=2:raise RuntimeError('Vulkan graphics device 0 is not an NVIDIA discrete GPU: '+json.dumps(facts))
        print('A2_VULKAN_DEVICES '+json.dumps(facts,sort_keys=True),flush=True)
    finally:destroy(instance,None)
    mark('vulkan','load_and_enumerate','PASS')

def check_nvenc():
    ffmpeg=os.environ['SIMULATION_FFMPEG_PATH']
    if not Path(ffmpeg).is_file():raise RuntimeError('Configured FFmpeg is missing')
    with tempfile.TemporaryDirectory(prefix='a2-nvenc-') as directory:
        target=Path(directory)/'probe.h264'
        mark('nvenc','encode_two_frames','BEGIN')
        result=subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-f','lavfi','-i','color=c=black:s=1280x720:r=5','-frames:v','2','-an','-c:v','h264_nvenc','-gpu','0','-pix_fmt','yuv420p','-f','h264',str(target)],capture_output=True,text=True,timeout=30)
        if result.returncode:raise RuntimeError('NVENC short encoding failed: '+result.stderr[-3000:])
        if not target.is_file() or target.stat().st_size==0:raise RuntimeError('NVENC produced no data')
        mark('nvenc','encode_two_frames','PASS')

def run_probe(probe):
    if probe=='cuda':check_cuda()
    elif probe=='physx':check_scene(False)
    elif probe=='vulkan':check_vulkan()
    elif probe=='camera':check_scene(True)
    elif probe=='nvenc':check_nvenc()
    else:raise RuntimeError('Unknown probe')
    mark(probe,'complete','PASS')

def main():
    faulthandler.enable(all_threads=True)
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if sys.version_info[:2]!=(3,8):raise RuntimeError('Isaac Gym Preview 4 requires Python 3.8')
    os.environ.pop('DISPLAY',None)
    if '--probe' in sys.argv:
        if len(sys.argv)!=3 or sys.argv[1]!='--probe' or sys.argv[2] not in PROBES:raise RuntimeError('Invalid native probe invocation')
        run_probe(sys.argv[2]);return
    if any(arg not in ('--check-only',) for arg in sys.argv[1:]):raise RuntimeError('Unknown entrypoint option')
    validate_reference_models()
    from simulation_core.s1_isaac_gym_adapter import S1IsaacGymConfig
    S1IsaacGymConfig().validate_asset_path()
    for probe in PROBES:
        mark(probe,'isolated_process','BEGIN')
        started=time.monotonic()
        # Child inherits only this worker's configured environment; no shell or RQ registration.
        result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--probe',probe],check=False,timeout=120)
        if result.returncode:
            detail=('signal '+str(-result.returncode)) if result.returncode<0 else ('exit '+str(result.returncode))
            raise RuntimeError('Native '+probe+' probe failed with '+detail+'; see last A2_NATIVE BEGIN and faulthandler stack')
        mark(probe,'isolated_process','PASS')
        print('A2_PROBE_SECONDS '+probe+' '+str(round(time.monotonic()-started,3)),flush=True)
    if '--check-only' in sys.argv:
        print('CUDA computation, GPU PhysX finite scene, Vulkan NVIDIA enumeration and Isaac Camera frame, NVENC short encoding, SDK and asset checks passed; no training started.',flush=True);return
    print('A2_NATIVE all probes passed; registering the configured RQ worker',flush=True)
    os.execv(sys.executable,[sys.executable,'simulation/scripts/start_rq_worker.py'])

if __name__=='__main__':main()
