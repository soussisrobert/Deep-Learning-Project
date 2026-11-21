"""Read-only activity telemetry, NEVER compute permission or idle/context proof.

Windows x64 ABI checked against installed nvml.h with MSVC. SM blends 3D/compute.
Missing/zero/stale samples do not establish context absence. Native call counts
are bounded; production wall-time containment is the launcher's parent watchdog.
"""
import ctypes as C
import hashlib
import os
from pathlib import Path
import re
import time

U,Q,P=C.c_uint32,C.c_uint64,C.c_void_p
MAX_RECORDS=512
SEMANTICS='activity_only_SM_blends_3D_compute_empty_is_not_idle_or_context_absence'

class ActivityError(RuntimeError):
    def __init__(self,message,calls=()):
        super().__init__(message);self.calls=tuple(dict(row) for row in calls)

class UtilV1(C.Structure):
    _fields_=[('timeStamp',Q),('pid',U),('smUtil',U),('memUtil',U),('encUtil',U),('decUtil',U),('jpgUtil',U),('ofaUtil',U)]
class InfoV1(C.Structure):
    _fields_=[('version',U),('processSamplesCount',U),('lastSeenTimeStamp',Q),('procUtilArray',C.POINTER(UtilV1))]
class Legacy(C.Structure):
    _fields_=[('pid',U),('timeStamp',Q),('smUtil',U),('memUtil',U),('encUtil',U),('decUtil',U)]

# Actual MSVC/header measurements, not guessed packing.
if (C.sizeof(P),C.sizeof(UtilV1),C.sizeof(InfoV1),C.sizeof(Legacy),InfoV1.procUtilArray.offset,Legacy.timeStamp.offset)!=(8,40,24,32,16,8):
    raise ActivityError('Unsupported ABI: Windows/x64 layout required')

def _uint(value,name,maximum=2**64-1):
    if type(value) is not int or not 0<=value<=maximum:raise ActivityError('Invalid integer '+name)
    return value

def parse_samples(raw,kind,count,capacity,*,last_seen_us,now_us,max_age_us):
    """Validate complete caller-owned buffers; never classify applications or absence."""
    for key,value in [('last_seen_us',last_seen_us),('now_us',now_us),('max_age_us',max_age_us)]:_uint(value,key)
    if last_seen_us>now_us:raise ActivityError('Future cursor or wall-clock rollback')
    _uint(capacity,'capacity',MAX_RECORDS);_uint(count,'count',capacity)
    if kind not in ('v1','legacy') or not isinstance(raw,(bytes,bytearray,memoryview)):raise ActivityError('Malformed sample buffer/type')
    typ=UtilV1 if kind=='v1' else Legacy
    if len(raw)!=capacity*C.sizeof(typ):raise ActivityError('Malformed sample buffer length')
    unique,accepted={},[];stats={'overlap':0,'stale':0,'duplicates':0};cursor=last_seen_us
    for index in range(count):
        native=typ.from_buffer_copy(raw,index*C.sizeof(typ));pid,stamp=int(native.pid),int(native.timeStamp)
        if not pid or not stamp or stamp>now_us:raise ActivityError('Unattributable PID/zero/future timestamp')
        sample={'pid':pid,'timestamp_us':stamp}
        for field in ('smUtil','memUtil','encUtil','decUtil','jpgUtil','ofaUtil'):
            value=int(getattr(native,field)) if hasattr(native,field) else None
            if value==0xFFFFFFFF:value=None
            if value is not None and not 0<=value<=100:raise ActivityError('Malformed utilization percentage')
            sample[field]=value
        key=(pid,stamp)
        if key in unique:
            if unique[key]!=sample:raise ActivityError('Conflicting overlapping sample')
            stats['duplicates']+=1;continue
        unique[key]=sample;cursor=max(cursor,stamp)
        if stamp<=last_seen_us:stats['overlap']+=1
        elif now_us-stamp>max_age_us:stats['stale']+=1
        else:accepted.append(sample)
    return {'samples':sorted(accepted,key=lambda row:(row['timestamp_us'],row['pid'])),'cursor_us':cursor,'discarded':stats}

def _hash(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def native_library_path():
    if os.name!='nt':raise ActivityError('Native Windows required')
    kernel=C.WinDLL('kernel32',use_last_error=True);kernel.GetSystemDirectoryW.argtypes=[C.c_wchar_p,U];kernel.GetSystemDirectoryW.restype=U
    buffer=C.create_unicode_buffer(32768);size=kernel.GetSystemDirectoryW(buffer,len(buffer))
    if not size or size>=len(buffer) or not Path(buffer.value).is_absolute():raise ActivityError('Uncertain native system directory')
    return (Path(buffer.value)/'nvml.dll').resolve(strict=True)

def load_native_library(path):return C.CDLL(str(path),winmode=0x900) # DLL_LOAD_DIR|SYSTEM32, not PATH/CWD.

def _bind(library,name,arguments):
    function=getattr(library,name,None)
    if function is not None:function.argtypes,function.restype=arguments,C.c_int32
    return function

def _call(calls,name,phase,function,*args):
    if len(calls)>=8:raise ActivityError('Native call budget exceeded')
    try:code=int(function(*args))
    except Exception as exc:
        calls.append({'api':name,'phase':phase,'return_code':None,'status':'native_call_raised','exception':type(exc).__name__})
        raise ActivityError('Native activity call raised',calls) from exc
    row={'api':name,'phase':phase,'return_code':code};calls.append(row);return row

def collect_samples(library,device,calls,*,last_seen_us,max_age_us,max_records,now_us=None):
    """At most four activity calls; empty/unsupported never indicates permission."""
    _uint(last_seen_us,'last_seen_us');_uint(max_age_us,'max_age_us')
    if not max_age_us or type(max_records) is not int or not 1<=max_records<=MAX_RECORDS:raise ActivityError('Invalid age/capacity budget')
    if now_us is not None:_uint(now_us,'now_us')
    if last_seen_us>(time.time_ns()//1000 if now_us is None else now_us):raise ActivityError('Future cursor or wall-clock rollback')
    for kind,name in [('v1','nvmlDeviceGetProcessesUtilizationInfo'),('legacy','nvmlDeviceGetProcessUtilization')]:
        typ=UtilV1 if kind=='v1' else Legacy
        args=[P,C.POINTER(InfoV1)] if kind=='v1' else [P,C.POINTER(Legacy),C.POINTER(U),Q]
        function=_bind(library,name,args)
        if function is None:
            calls.append({'api':name,'phase':'resolve','return_code':None,'status':'export_absent'});continue
        info,count=InfoV1(),U();info.version,info.lastSeenTimeStamp=C.sizeof(InfoV1)|(1<<24),last_seen_us
        def invoke(storage=None):
            if kind=='v1':
                if storage is not None:info.processSamplesCount,info.procUtilArray=len(storage),storage
                return _call(calls,name,'size' if storage is None else 'data',function,device,C.byref(info))
            count.value=0 if storage is None else len(storage)
            return _call(calls,name,'size' if storage is None else 'data',function,device,storage,C.byref(count),Q(last_seen_us))
        row=invoke();required=int(info.processSamplesCount if kind=='v1' else count.value);row['count']=required
        if row['return_code'] in (3,13,25):continue # Explicit unsupported/function/version errors only.
        if row['return_code']==6:return {'status':'no_recent_samples','samples':[],'cursor_us':last_seen_us,'api':name}
        if row['return_code'] not in (0,7):return {'status':'native_error','samples':[],'cursor_us':last_seen_us,'api':name}
        if row['return_code']==7 and not required:raise ActivityError('Inconsistent insufficient-size zero count')
        if required>max_records:return {'status':'capacity_exceeded','samples':[],'cursor_us':last_seen_us,'api':name}
        if not required:return {'status':'no_recent_samples','samples':[],'cursor_us':last_seen_us,'api':name}
        capacity=min(max_records,required+8);storage=(typ*capacity)();row=invoke(storage)
        written=int(info.processSamplesCount if kind=='v1' else count.value);row['count'],row['capacity']=written,capacity
        if row['return_code'] in (3,13,25):continue
        if row['return_code']!=0:return {'status':'native_error','samples':[],'cursor_us':last_seen_us,'api':name}
        if kind=='v1' and C.cast(info.procUtilArray,P).value!=C.addressof(storage):raise ActivityError('Driver changed caller-owned array pointer')
        parsed=parse_samples(bytes(storage),kind,written,capacity,last_seen_us=last_seen_us,
            now_us=time.time_ns()//1000 if now_us is None else now_us,max_age_us=max_age_us)
        return {'status':'ok' if parsed['samples'] else 'no_recent_samples','api':name,**parsed}
    return {'status':'unsupported','samples':[],'cursor_us':last_seen_us,'api':None}

def sample_activity(gpu_uuid,*,last_seen_us=0,max_age_us=5_000_000,max_records=MAX_RECORDS):
    """One observation with exact UUID readback, hash-stable DLL and finally cleanup."""
    if not isinstance(gpu_uuid,str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',gpu_uuid):raise ActivityError('Exact physical GPU UUID required')
    _uint(last_seen_us,'last_seen_us');_uint(max_age_us,'max_age_us')
    if not max_age_us or type(max_records) is not int or not 1<=max_records<=MAX_RECORDS:raise ActivityError('Invalid age/capacity budget')
    path=native_library_path();sha=_hash(path);library,calls,initialized,result=load_native_library(path),[],False,None;failure=None
    try:
        init=_bind(library,'nvmlInit_v2',[])
        if init is None or _call(calls,'nvmlInit_v2','init',init)['return_code']!=0:raise ActivityError('NVML initialization unavailable')
        initialized=True;get_handle=_bind(library,'nvmlDeviceGetHandleByUUID',[C.c_char_p,C.POINTER(P)]);get_uuid=_bind(library,'nvmlDeviceGetUUID',[P,C.c_char_p,U])
        if get_handle is None or get_uuid is None:raise ActivityError('Mandatory UUID identity API absent')
        device,readback=P(),C.create_string_buffer(96)
        if _call(calls,'nvmlDeviceGetHandleByUUID','identity',get_handle,gpu_uuid.encode(),C.byref(device))['return_code']!=0:raise ActivityError('Exact UUID handle unavailable')
        if _call(calls,'nvmlDeviceGetUUID','readback',get_uuid,device,readback,U(len(readback)))['return_code']!=0:raise ActivityError('UUID readback unavailable')
        if readback.value.decode('ascii').casefold()!=gpu_uuid.casefold():raise ActivityError('UUID readback mismatch')
        result=collect_samples(library,device,calls,last_seen_us=last_seen_us,max_age_us=max_age_us,max_records=max_records)
    except BaseException as exc:failure=exc;raise
    finally:
        cleanup_error=None
        try:
            if initialized:
                shutdown=_bind(library,'nvmlShutdown',[])
                if shutdown is None or _call(calls,'nvmlShutdown','cleanup',shutdown)['return_code']!=0:cleanup_error='NVML shutdown failed/absent'
        except Exception as exc:cleanup_error='NVML shutdown raised: '+type(exc).__name__
        finally:
            try:
                if _hash(path)!=sha:cleanup_error='Native DLL bytes changed during observation'
            except OSError:cleanup_error='Native DLL identity cannot be reverified'
        if isinstance(failure,ActivityError):failure.calls=tuple(dict(row) for row in calls)
        if cleanup_error:raise ActivityError(cleanup_error,calls)
    return {**result,'calls':calls,'uuid':gpu_uuid,'native_DLL':str(path),'native_DLL_sha256':sha,'native_DLL_hash_stable':True,
        'semantics':SEMANTICS,'can_authorize_compute':False,'classifies_compute_vs_graphics':False,'empty_proves_idle':False}
