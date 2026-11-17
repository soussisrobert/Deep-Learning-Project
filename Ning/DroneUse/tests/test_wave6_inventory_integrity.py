"""Pure XML/CPU regressions: unavailable native telemetry must not look empty."""
import copy,math
import pytest
from hift_research.wave6.lease import DEFAULT_RESOURCES,LeaseError,parse_physical_inventory
GPU='GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'
HOST={'available_host_bytes':32*1024**3,'process_rss_bytes':64*1024**2,'process_peak_rss_bytes':128*1024**2}
def xml(processes='',used='256 MiB',free='16128 MiB'):
    return f'<nvidia_smi_log><gpu><uuid>{GPU}</uuid><product_name>fixture-not-hardware</product_name><driver_model><current_dm>WDDM</current_dm></driver_model><fb_memory_usage><total>16384 MiB</total><used>{used}</used><free>{free}</free></fb_memory_usage><processes>{processes}</processes></gpu></nvidia_smi_log>'
def row(pid=123,kind='G'):
    return f'<process_info><pid>{pid}</pid><type>{kind}</type><process_name>not-an-allowlist</process_name><used_memory>N/A</used_memory></process_info>'

@pytest.mark.parametrize('body',['N/A','Not Supported','unknown','<unsupported/>','<unexpected>0</unexpected>'])
def test_unavailable_or_unknown_inventory_is_not_an_empty_available_gpu(body):
    with pytest.raises(LeaseError):parse_physical_inventory(xml(body),GPU,DEFAULT_RESOURCES,HOST)

@pytest.mark.parametrize('pid',[0,-1,-123])
def test_graphics_tag_does_not_waive_invalid_process_identity(pid):
    with pytest.raises(LeaseError):parse_physical_inventory(xml(row(pid)),GPU,DEFAULT_RESOURCES,HOST)

def test_duplicate_process_identity_is_not_silently_accepted():
    with pytest.raises(LeaseError):parse_physical_inventory(xml(row()+row()),GPU,DEFAULT_RESOURCES,HOST)

@pytest.mark.parametrize('key',['available_host_bytes','process_rss_bytes','process_peak_rss_bytes'])
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,'100000000000',-1])
def test_native_host_resource_fields_are_finite_nonnegative_integer_observations(key,bad):
    host={**HOST,key:bad}
    with pytest.raises(LeaseError):parse_physical_inventory(xml(),GPU,DEFAULT_RESOURCES,host)

@pytest.mark.parametrize('fault',['processes','uuid','memory_group','total','used','free','pid','type','uuid_child','pid_child','row_text','group_tail','root_text','gpu_text','memory_text'])
def test_duplicate_or_nested_critical_fields_do_not_hide_conflicting_native_output(fault):
    value=xml(row())
    replacements={
        'processes':('</processes>','</processes><processes>'+row(kind='C+G')+'</processes>'),
        'uuid':(f'<uuid>{GPU}</uuid>',f'<uuid>{GPU}</uuid><uuid>GPU-00000000-0000-0000-0000-000000000000</uuid>'),
        'memory_group':('</fb_memory_usage>','</fb_memory_usage><fb_memory_usage><total>16384 MiB</total><used>16384 MiB</used><free>0 MiB</free></fb_memory_usage>'),
        'total':('<total>16384 MiB</total>','<total>16384 MiB</total><total>1 MiB</total>'),
        'used':('<used>256 MiB</used>','<used>256 MiB</used><used>16384 MiB</used>'),
        'free':('<free>16128 MiB</free>','<free>16128 MiB</free><free>0 MiB</free>'),
        'pid':('<pid>123</pid>','<pid>123</pid><pid>456</pid>'),
        'type':('<type>G</type>','<type>G</type><type>C+G</type>'),
        'uuid_child':(f'<uuid>{GPU}</uuid>',f'<uuid>{GPU}<unexpected/></uuid>'),
        'pid_child':('<pid>123</pid>','<pid>123<unexpected/></pid>'),
        'row_text':('<process_info>','<process_info>N/A'),
        'group_tail':('</processes>','</processes>N/A'),
        'root_text':('<nvidia_smi_log>','<nvidia_smi_log>N/A'),
        'gpu_text':('<gpu>','<gpu>N/A'),
        'memory_text':('<fb_memory_usage>','<fb_memory_usage>N/A')}
    old,new=replacements[fault]
    with pytest.raises(LeaseError):parse_physical_inventory(value.replace(old,new),GPU,DEFAULT_RESOURCES,HOST)


def test_available_host_memory_cannot_exceed_reported_total():
    with pytest.raises(LeaseError):parse_physical_inventory(xml(),GPU,DEFAULT_RESOURCES,{**HOST,'total_host_bytes':8*1024**3})

def test_gpu_free_plus_used_cannot_exceed_total():
    with pytest.raises(LeaseError):parse_physical_inventory(xml(used='2000 MiB',free='16000 MiB'),GPU,DEFAULT_RESOURCES,HOST)

def test_actually_empty_or_graphics_inventory_preserves_old_availability_semantics():
    for body in ['', '\n  ', row()]:assert parse_physical_inventory(xml(body),GPU,DEFAULT_RESOURCES,HOST)['eligible'] is True

def test_opaque_contexts_remain_unknown_not_proven_training_workloads():
    result=parse_physical_inventory(xml(row(kind='C+G')),GPU,DEFAULT_RESOURCES,HOST)
    assert result['eligible'] is False
