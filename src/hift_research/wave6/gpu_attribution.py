"""Read-only Windows GPU attribution capabilities; never an availability waiver.

WDDM <2.9 has RESERVED bytes where newer SDKs expose ClientHint. Neither
unsupported metadata, idle counters, nor a new layout proves context absence.
Native calls run inside the launcher's existing external parent deadline.
"""
import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass,field
from enum import Enum
from pathlib import Path
import re
import uuid as uuidlib

U32,I32,P=C.c_uint32,C.c_int32,C.c_void_p

class Status(str,Enum):
    UNSUPPORTED_CLIENT_HINT='UNSUPPORTED_CLIENT_HINT'
    SUPPORTED_UNCALIBRATED='SUPPORTED_UNCALIBRATED'
    ADAPTER_BINDING_UNKNOWN='ADAPTER_BINDING_UNKNOWN'
    NATIVE_API_ERROR='NATIVE_API_ERROR'

@dataclass
class CapabilityObservation:
    status:Status
    gpu_uuid:str
    reason:str
    wddm_version:int|None=None
    luid:str|None=None
    pci:dict|None=None
    calls:list=field(default_factory=list)
    cleanup_errors:list=field(default_factory=list)
    dll_paths:list=field(default_factory=list)
    competing_compute:str='UNKNOWN'
    idle_context_absence_proven:bool=False
    virtual_alias_coverage_proven:bool=False

class NativeError(RuntimeError):pass

class LUID(C.Structure):
    _fields_=[('low',U32),('high',I32)]
class GUID(C.Structure):
    _fields_=[('raw',C.c_ubyte*16)]
class Desc1(C.Structure):
    _fields_=[('description',W.WCHAR*128),('vendor',U32),('device',U32),('subsystem',U32),('revision',U32),
        ('video_bytes',C.c_size_t),('system_bytes',C.c_size_t),('shared_bytes',C.c_size_t),('luid',LUID),('flags',U32)]
class PCI(C.Structure):
    _fields_=[('legacy',C.c_char*16),('domain',U32),('bus',U32),('device',U32),('device_id',U32),('subsystem_id',U32),('bus_id',C.c_char*32)]
class OpenAdapter(C.Structure):
    _fields_=[('luid',LUID),('handle',U32)]
class AdapterInfo(C.Structure):
    _fields_=[('handle',U32),('type',U32),('data',P),('size',U32)]

def assert_abi():
    actual=[C.sizeof(P),C.sizeof(LUID),C.sizeof(Desc1),C.sizeof(PCI),C.sizeof(OpenAdapter),C.sizeof(AdapterInfo),AdapterInfo.data.offset]
    if actual!=[8,8,312,68,12,24,8]:raise NativeError(f'Unsupported native ABI {actual}')
    return actual

def function(dll,name,restype,*types):
    f=getattr(dll,name);f.argtypes,f.restype=list(types),restype
    return f

def method(obj,index,restype,*types):
    table=C.cast(obj,C.POINTER(C.POINTER(P))).contents
    return C.WINFUNCTYPE(restype,P,*types)(table[index])

def version_status(version):
    # SDK22621 d3dkmthk.h gates ClientHint at WDDM2.9; older bytes are RESERVED.
    # A new layout is still not a calibrated or complete context-classification API.
    return Status.UNSUPPORTED_CLIENT_HINT if version<2900 else Status.SUPPORTED_UNCALIBRATED

def matching_adapters(pci,adapters):
    # KMT_ADAPTERADDRESS omits PCI domain: do not guess nonzero-domain mappings.
    if pci.get('domain')!=0:return []
    matches=[]
    for row in adapters:
        address=row.get('pci_address')
        if (row.get('physical_count')==1 and address
                and row['vendor']==(pci['device_id']&65535) and row['device']==(pci['device_id']>>16)
                and row['subsystem']==pci['subsystem_id']
                and address==[pci['bus'],pci['device'],int(pci['bus_id'].split('.')[-1],16)]):matches.append(row)
    return matches

def probe(gpu_uuid):
    observation=CapabilityObservation(Status.NATIVE_API_ERROR,gpu_uuid,'not completed')
    calls,cleanup=observation.calls,observation.cleanup_errors
    def checked(name,value):
        calls.append({'api':name,'return':value,'hex':f'0x{value&0xffffffff:08x}'})
        if value!=0:raise NativeError(f'{name} returned 0x{value&0xffffffff:08x}')
    try:
        if not isinstance(gpu_uuid,str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',gpu_uuid):
            raise NativeError('Exact physical NVIDIA GPU UUID required')
        assert_abi()
        kernel=C.WinDLL('kernel32.dll') # Windows KnownDLL, not PATH/CWD lookup.
        buf=C.create_unicode_buffer(32768)
        length=function(kernel,'GetSystemDirectoryW',U32,W.LPWSTR,U32)(buf,len(buf))
        if not 0<length<len(buf):raise NativeError('GetSystemDirectoryW failed')
        system32=Path(buf.value)
        if not system32.is_absolute():raise NativeError('native system directory is not absolute')
        def load(name):
            path=system32/name;observation.dll_paths.append(str(path))
            return C.WinDLL(str(path))
        nv=load('nvml.dll')
        checked('nvmlInit_v2',function(nv,'nvmlInit_v2',I32)())
        try:
            handle=P()
            checked('nvmlDeviceGetHandleByUUID',function(nv,'nvmlDeviceGetHandleByUUID',I32,C.c_char_p,C.POINTER(P))(gpu_uuid.encode('ascii'),C.byref(handle)))
            returned_uuid=C.create_string_buffer(96)
            checked('nvmlDeviceGetUUID',function(nv,'nvmlDeviceGetUUID',I32,P,P,U32)(handle,returned_uuid,len(returned_uuid)))
            if returned_uuid.value.decode('ascii').lower()!=gpu_uuid.lower():raise NativeError('NVML UUID readback mismatch')
            pci=PCI()
            checked('nvmlDeviceGetPciInfo_v3',function(nv,'nvmlDeviceGetPciInfo_v3',I32,P,C.POINTER(PCI))(handle,C.byref(pci)))
            observation.pci={'bus_id':pci.bus_id.decode('ascii'),'domain':pci.domain,'bus':pci.bus,'device':pci.device,
                'device_id':pci.device_id,'subsystem_id':pci.subsystem_id}
        finally:
            rc=function(nv,'nvmlShutdown',I32)();calls.append({'api':'nvmlShutdown','return':rc})
            if rc:cleanup.append({'api':'nvmlShutdown','return':rc})
        dxgi,gdi=load('dxgi.dll'),load('gdi32.dll');factory=P()
        iid=GUID.from_buffer_copy(uuidlib.UUID('770aae78-f26f-4dba-a829-253c83d1b387').bytes_le)
        checked('CreateDXGIFactory1',function(dxgi,'CreateDXGIFactory1',I32,C.POINTER(GUID),C.POINTER(P))(C.byref(iid),C.byref(factory)))
        rows=[]
        try:
            enum=method(factory,12,I32,U32,C.POINTER(P))
            for index in range(32):
                adapter=P();rc=enum(factory,index,C.byref(adapter))
                if rc&0xffffffff==0x887a0002:break
                checked('IDXGIFactory1.EnumAdapters1',rc)
                try:
                    desc=Desc1();checked('IDXGIAdapter1.GetDesc1',method(adapter,10,I32,C.POINTER(Desc1))(adapter,C.byref(desc)))
                    row={'luid':f'0x{desc.luid.high&0xffffffff:08x}_0x{desc.luid.low:08x}',
                        'vendor':desc.vendor,'device':desc.device,'subsystem':desc.subsystem};rows.append(row)
                    opened=OpenAdapter(desc.luid,0)
                    checked('D3DKMTOpenAdapterFromLuid',function(gdi,'D3DKMTOpenAdapterFromLuid',I32,C.POINTER(OpenAdapter))(C.byref(opened)))
                    try:
                        def query(kind,value):
                            info=AdapterInfo(opened.handle,kind,C.cast(C.byref(value),P),C.sizeof(value))
                            checked(f'D3DKMTQueryAdapterInfo({kind})',function(gdi,'D3DKMTQueryAdapterInfo',I32,C.POINTER(AdapterInfo))(C.byref(info)))
                        address,count,version=(U32*3)(),U32(),U32()
                        query(6,address);query(30,count);query(13,version)
                        row.update(pci_address=list(address),physical_count=count.value,wddm_version=version.value)
                    finally:
                        rc=function(gdi,'D3DKMTCloseAdapter',I32,C.POINTER(U32))(C.byref(U32(opened.handle)))
                        calls.append({'api':'D3DKMTCloseAdapter','return':rc})
                        if rc:cleanup.append({'api':'D3DKMTCloseAdapter','return':rc})
                finally:
                    method(adapter,2,U32)(adapter);calls.append({'api':'IDXGIAdapter.Release','called':True})
            else:raise NativeError('DXGI enumeration exceeded32adapter bound')
        finally:
            method(factory,2,U32)(factory);calls.append({'api':'IDXGIFactory.Release','called':True})
        matches=matching_adapters(observation.pci,rows);calls.append({'adapter_inventory':rows,'unique_pci_match_count':len(matches)})
        if len(matches)!=1:
            observation.status=Status.ADAPTER_BINDING_UNKNOWN
            observation.reason='No unique domain0 PCI/vendor/device/subsystem match; aliases or linked adapters are not guessed'
        else:
            match=matches[0];observation.luid,observation.wddm_version=match['luid'],match['wddm_version']
            observation.status=version_status(match['wddm_version'])
            observation.reason=('WDDM<2.9: process-adapter ClientHint field is RESERVED, not a client/API observation'
                if observation.status==Status.UNSUPPORTED_CLIENT_HINT else 'ClientHint layout available; mixed/idle context completeness uncalibrated')
        if cleanup:
            observation.status=Status.NATIVE_API_ERROR;observation.reason='Native cleanup failed; observation is not complete'
    except (OSError,AttributeError,ValueError,NativeError) as exc:
        observation.status=Status.NATIVE_API_ERROR;observation.reason=f'{type(exc).__name__}: {exc}'
    return observation
