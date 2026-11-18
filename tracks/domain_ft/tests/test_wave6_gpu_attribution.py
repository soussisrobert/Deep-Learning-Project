"""Native capability fixtures: no GPU APIs/workloads or user grants."""
import ctypes as C
from unittest.mock import patch
import pytest
from hift_research.wave6 import gpu_attribution as m
UUID='GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'

class FakeNative:
    def __init__(self,version=2700,fail=None,duplicate=False):
        self.version,self.fail,self.duplicate=version,fail,duplicate;self.seen,self.loaded=[],[]
    def load(self,path):self.loaded.append(path);return path
    def function(self,dll,name,restype,*types):
        def invoke(*a):
            self.seen.append(name)
            if name==self.fail:return -1
            if name=='GetSystemDirectoryW':a[0].value='C:\\Windows\\System32';return len(a[0].value)
            if name=='nvmlDeviceGetHandleByUUID':a[1]._obj.value=333
            elif name=='nvmlDeviceGetUUID':a[1].value=UUID.encode()
            elif name=='nvmlDeviceGetPciInfo_v3':
                pci=a[1]._obj;pci.domain,pci.bus,pci.device=0,1,0
                pci.device_id,pci.subsystem_id=(0x2c05<<16)|0x10de,123;pci.bus_id=b'00000000:01:00.0'
            elif name=='CreateDXGIFactory1':a[1]._obj.value=111
            elif name=='D3DKMTOpenAdapterFromLuid':a[0]._obj.handle=444
            elif name=='D3DKMTQueryAdapterInfo':
                info=a[0]._obj;data=C.cast(info.data,C.POINTER(m.U32))
                if info.type==6:data[0],data[1],data[2]=1,0,0
                elif info.type==13:data[0]=self.version
                elif info.type==30:data[0]=1
            return 0
        return invoke
    def method(self,obj,index,restype,*types):
        def invoke(*a):
            if index==12:
                if a[1]>=(2 if self.duplicate else 1):return 0x887a0002
                a[2]._obj.value=222+a[1]
            elif index==10:
                desc=a[1]._obj;desc.vendor,desc.device,desc.subsystem=0x10de,0x2c05,123;desc.luid.low=obj.value
            elif index==2:self.seen.append('release_factory' if obj.value==111 else 'release_adapter')
            return 0
        return invoke
    def run(self):
        with patch.object(m.C,'WinDLL',side_effect=self.load),patch.object(m,'function',side_effect=self.function),patch.object(m,'method',side_effect=self.method):
            return m.probe(UUID)

def test_abi():assert m.assert_abi()==[8,8,312,68,12,24,8]

def test_wddm27_unsupported_is_not_busy_and_native_handles_are_closed():
    fake=FakeNative();result=fake.run()
    assert result.status==m.Status.UNSUPPORTED_CLIENT_HINT and result.competing_compute=='UNKNOWN'
    assert not result.idle_context_absence_proven and result.wddm_version==2700
    for action in ['nvmlShutdown','D3DKMTCloseAdapter','release_adapter','release_factory']:assert fake.seen.count(action)==1
    for name in fake.loaded[1:]:assert 'Windows' in name and 'System32' in name

@pytest.mark.parametrize('version',[2900,3000,3100,9999])
def test_new_layout_never_becomes_an_absence_or_availability_claim(version):
    result=FakeNative(version=version).run()
    assert result.status==m.Status.SUPPORTED_UNCALIBRATED and result.competing_compute=='UNKNOWN'
    assert not result.idle_context_absence_proven and not result.virtual_alias_coverage_proven

def test_duplicate_pci_is_ambiguous_not_first_match():
    result=FakeNative(duplicate=True).run()
    assert result.status==m.Status.ADAPTER_BINDING_UNKNOWN and result.luid is None

def test_gdi_failure_preserves_cleanup_and_unknown():
    fake=FakeNative(fail='D3DKMTQueryAdapterInfo');result=fake.run()
    assert result.status==m.Status.NATIVE_API_ERROR and result.competing_compute=='UNKNOWN'
    for action in ['nvmlShutdown','D3DKMTCloseAdapter','release_adapter','release_factory']:assert fake.seen.count(action)==1

def test_nvml_failure_shuts_down_before_dxgi():
    fake=FakeNative(fail='nvmlDeviceGetPciInfo_v3')
    assert fake.run().status==m.Status.NATIVE_API_ERROR and fake.seen.count('nvmlShutdown')==1
    assert 'CreateDXGIFactory1' not in fake.seen

def test_init_failure_does_not_shutdown_uninitialized_session():
    fake=FakeNative(fail='nvmlInit_v2');assert fake.run().status==m.Status.NATIVE_API_ERROR
    assert 'nvmlShutdown' not in fake.seen

def test_cleanup_failure_cannot_report_capability_success():
    result=FakeNative(fail='D3DKMTCloseAdapter').run()
    assert result.status==m.Status.NATIVE_API_ERROR and result.cleanup_errors

@pytest.mark.parametrize('value',['0','GPU-../escape',None,123])
def test_bad_uuid_does_not_load_native_code(value):
    with patch.object(m.C,'WinDLL',side_effect=AssertionError('must not load')):assert m.probe(value).status==m.Status.NATIVE_API_ERROR

def test_nonzero_pci_domain_is_not_guessed():assert m.matching_adapters({'domain':2},[])==[]

@pytest.mark.parametrize('status',[m.Status.UNSUPPORTED_CLIENT_HINT,m.Status.SUPPORTED_UNCALIBRATED,m.Status.NATIVE_API_ERROR])
def test_availability_integration_preserves_opaque_context_failure_for_every_capability_status(monkeypatch,status):
    from types import SimpleNamespace
    from hift_research.wave6 import lease,gpu_activity
    monkeypatch.setattr(gpu_activity,'sample_activity',lambda gpu:{'status':'no_recent_samples','samples':[], 'can_authorize_compute':False,'classifies_compute_vs_graphics':False,'empty_proves_idle':False})
    fixture_xml=f'<nvidia_smi_log><gpu><uuid>{UUID}</uuid><driver_model><current_dm>WDDM</current_dm></driver_model><fb_memory_usage><total>16384 MiB</total><used>128 MiB</used><free>16256 MiB</free></fb_memory_usage><processes><process_info><pid>123</pid><type>C+G</type><used_memory>N/A</used_memory></process_info></processes></gpu></nvidia_smi_log>'
    monkeypatch.setattr(lease,'native_smi_identity',lambda:{'path':'fixture-only','sha256':'a'*64,'argv':['fixture-only','-q','-x']})
    monkeypatch.setattr(lease,'_file_sha256',lambda path:'a'*64)
    monkeypatch.setattr(lease.subprocess,'run',lambda *a,**k:SimpleNamespace(stdout=fixture_xml))
    monkeypatch.setattr(lease,'host_memory',lambda:{'available_host_bytes':16*1024**3,'process_rss_bytes':64*1024**2,'process_peak_rss_bytes':128*1024**2})
    monkeypatch.setattr(m,'probe',lambda gpu:m.CapabilityObservation(status,gpu,'metadata fixture',wddm_version=2700))
    report=lease.probe_physical_availability(UUID)
    assert report['eligible'] is False and report['ownership_assessment']=='UNPROVEN_CONTEXT_CLASSIFICATION'
    assert not report['active_competing_workload_proven']
    assert report['windows_attribution_capability']['status']==status
    assert report['windows_attribution_capability']['idle_context_absence_proven'] is False


def test_physical_alias_or_linked_adapter_cannot_match():
    pci={'domain':0,'bus':1,'device':0,'device_id':(0x2c05<<16)|0x10de,'subsystem_id':123,'bus_id':'00000000:01:00.0'}
    row={'physical_count':2,'pci_address':[1,0,0],'vendor':0x10de,'device':0x2c05,'subsystem':123}
    assert m.matching_adapters(pci,[row])==[]
