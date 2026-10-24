"""Portable CPU buffer/lifecycle tests; no Windows/NVML/GPU calls.

Adapted from the native-activity candidate tests. The final replay is explicitly
synthetic inline data, not a live GPU observation or permission/idle proof.
"""
import ctypes as C
from types import SimpleNamespace
import pytest
from hift_research.wave6 import gpu_activity as a

NOW = 10_000_000
UUID = 'GPU-4d0142b1-6cb1-48e2-1936-ea596332cd7f'


@pytest.fixture(autouse=True)
def forbid_unmocked_native_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU test attempted an unmocked native library/path operation')
    monkeypatch.setattr(a, 'native_library_path', forbidden)
    monkeypatch.setattr(a, 'load_native_library', forbidden)


def native(typ, row):
    value = typ()
    for key, item in row.items():
        setattr(value, key, item)
    return value


def buffer(rows, kind='v1'):
    typ = a.UtilV1 if kind == 'v1' else a.Legacy
    array = (typ * len(rows))(*[native(typ, row) for row in rows])
    return bytes(array)


def parse(rows, kind='v1', last_seen_us=0):
    return a.parse_samples(buffer(rows, kind), kind, len(rows), len(rows),
                           last_seen_us=last_seen_us, now_us=NOW, max_age_us=2_000_000)


def test_overlapping_stale_samples_sorted_and_deduplicated_without_idle_inference():
    valid = {'pid': 1, 'timeStamp': 9_000_000, 'smUtil': 5}
    result = parse([{'pid': 2, 'timeStamp': 9_500_000}, valid, valid,
                    {'pid': 3, 'timeStamp': 6_000_000}, {'pid': 4, 'timeStamp': 1_000_000}], last_seen_us=1_000_000)
    assert [row['pid'] for row in result['samples']] == [1, 2]
    assert result['discarded'] == {'overlap': 1, 'stale': 1, 'duplicates': 1}
    assert result['cursor_us'] == 9_500_000


def test_conflicting_duplicate_is_malformed_not_silently_averaged():
    with pytest.raises(a.ActivityError, match='Conflicting'):
        parse([{'pid': 1, 'timeStamp': 9_000_000, 'smUtil': 2},
               {'pid': 1, 'timeStamp': 9_000_000, 'smUtil': 3}])


def test_legacy_unknown_fields_and_native_unavailable_remain_none_not_zero():
    sample = parse([{'pid': 1, 'timeStamp': 9_000_000, 'smUtil': 0xFFFFFFFF}], kind='legacy')['samples'][0]
    assert sample['smUtil'] is None and sample['jpgUtil'] is None and sample['ofaUtil'] is None
    assert sample['memUtil'] == 0


@pytest.mark.parametrize('fault', ['truncated', 'overflow', 'negative_count', 'bool_count', 'future', 'zero_pid',
                                  'zero_timestamp', 'bad_percentage', 'bad_kind', 'bad_buffer', 'capacity_limit'])
def test_malformed_buffers_fail_before_metrics_are_consumed(fault):
    row, kind, count, capacity = {'pid': 1, 'timeStamp': 9_000_000}, 'v1', 1, 1
    if fault == 'future': row['timeStamp'] = NOW + 1
    if fault == 'zero_pid': row['pid'] = 0
    if fault == 'zero_timestamp': row['timeStamp'] = 0
    if fault == 'bad_percentage': row['smUtil'] = 101
    raw = buffer([row])
    if fault == 'truncated': raw = raw[:-1]
    if fault == 'overflow': count = 2
    if fault == 'negative_count': count = -1
    if fault == 'bool_count': count = True
    if fault == 'bad_kind': kind = 'guessed_v2'
    if fault == 'bad_buffer': raw = 'not native bytes'
    if fault == 'capacity_limit': capacity = 513
    with pytest.raises(a.ActivityError):
        a.parse_samples(raw, kind, count, capacity, last_seen_us=0, now_us=NOW, max_age_us=2_000_000)


class Fn:
    def __init__(self, callback): self.callback = callback
    def __call__(self, *args): return self.callback(*args)


def v1(rows=(), size_code=None, data_code=0, required=None, written=None, rewrite_pointer=False):
    def call(device, pointer):
        info = C.cast(pointer, C.POINTER(a.InfoV1)).contents
        assert info.version == 16777240
        if not info.procUtilArray:
            info.processSamplesCount = len(rows) if required is None else required
            return (7 if rows else 0) if size_code is None else size_code
        for index, row in enumerate(rows): info.procUtilArray[index] = native(a.UtilV1, row)
        info.processSamplesCount = len(rows) if written is None else written
        if rewrite_pointer: info.procUtilArray = C.POINTER(a.UtilV1)()
        return data_code
    return Fn(call)


def legacy(rows=(), code=None):
    def call(device, storage, pointer, last_seen):
        count = C.cast(pointer, C.POINTER(a.U))
        count[0] = len(rows)
        if storage is None: return (7 if rows else 0) if code is None else code
        for index, row in enumerate(rows): storage[index] = native(a.Legacy, row)
        return 0
    return Fn(call)


def collect(lib, **kwargs):
    calls = []
    result = a.collect_samples(lib, a.P(1), calls, last_seen_us=0,
                               max_age_us=2_000_000, max_records=512, now_us=NOW, **kwargs)
    return result, calls


ROWS = [{'pid': 1, 'timeStamp': 9_000_000, 'smUtil': 5}]


def test_preferred_v1_supported_and_no_fallback():
    lib = SimpleNamespace(nvmlDeviceGetProcessesUtilizationInfo=v1(ROWS),
                          nvmlDeviceGetProcessUtilization=Fn(lambda *args: pytest.fail('unexpected fallback')))
    result, calls = collect(lib)
    assert result['status'] == 'ok' and result['samples'][0]['smUtil'] == 5
    assert [row['return_code'] for row in calls] == [7, 0]


@pytest.mark.parametrize('unsupported', [3, 13, 25, None])
def test_explicit_unsupported_or_absent_preferred_api_allows_typed_legacy_fallback(unsupported):
    lib = SimpleNamespace(nvmlDeviceGetProcessUtilization=legacy(ROWS))
    if unsupported is not None: lib.nvmlDeviceGetProcessesUtilizationInfo = v1(size_code=unsupported)
    result, calls = collect(lib)
    assert result['status'] == 'ok' and result['api'] == 'nvmlDeviceGetProcessUtilization'
    assert result['samples'][0]['jpgUtil'] is None and len(calls) == 3


def test_permission_error_does_not_fallback_or_masquerade_as_no_activity():
    result, calls = collect(SimpleNamespace(nvmlDeviceGetProcessesUtilizationInfo=v1(size_code=4),
                                           nvmlDeviceGetProcessUtilization=Fn(lambda *args: pytest.fail('permission bypass'))))
    assert result['status'] == 'native_error' and result['samples'] == []
    assert len(calls) == 1 and calls[0]['return_code'] == 4


@pytest.mark.parametrize('code', [0, 6])
def test_no_samples_is_observation_only(code):
    result, calls = collect(SimpleNamespace(nvmlDeviceGetProcessesUtilizationInfo=v1(size_code=code)))
    assert result['status'] == 'no_recent_samples' and len(calls) == 1


def test_all_unsupported_keeps_raw_codes_and_unknown_not_permission():
    result, calls = collect(SimpleNamespace(nvmlDeviceGetProcessesUtilizationInfo=v1(size_code=3),
                                           nvmlDeviceGetProcessUtilization=legacy(code=3)))
    assert result['status'] == 'unsupported' and result['samples'] == []
    assert [row['return_code'] for row in calls] == [3, 3]


def test_oversized_count_refuses_unbounded_allocation():
    result, calls = collect(SimpleNamespace(nvmlDeviceGetProcessesUtilizationInfo=v1(required=513, size_code=7)))
    assert result['status'] == 'capacity_exceeded' and len(calls) == 1


def test_changing_count_insufficient_size_stops_after_one_data_call():
    result, calls = collect(SimpleNamespace(nvmlDeviceGetProcessesUtilizationInfo=v1(ROWS, data_code=7, written=513)))
    assert result['status'] == 'native_error' and result['samples'] == [] and len(calls) == 2


@pytest.mark.parametrize('fault', ['zero_size', 'count_overflow', 'pointer_rewrite'])
def test_native_malformed_descriptor_rejected(fault):
    function = {'zero_size': v1(size_code=7), 'count_overflow': v1(ROWS, written=513),
                'pointer_rewrite': v1(ROWS, rewrite_pointer=True)}[fault]
    with pytest.raises(a.ActivityError): collect(SimpleNamespace(nvmlDeviceGetProcessesUtilizationInfo=function))


def lifecycle(tmp_path, monkeypatch, fault=None):
    path = tmp_path / 'inert-native-DLL-marker.dll'
    path.write_bytes(b'inert unit-test bytes; never loaded or executed')
    events = []
    def get_handle(uuid, pointer):
        C.cast(pointer, C.POINTER(a.P))[0] = a.P(1)
        return 0
    def get_uuid(device, target, capacity):
        target.value = (UUID if fault != 'mismatch' else 'GPU-00000000-0000-0000-0000-000000000000').encode()
        return 4 if fault == 'uuid_error' else 0
    def shutdown():
        events.append('shutdown')
        if fault == 'dll_drift': path.write_bytes(b'changed inert marker')
        if fault == 'shutdown_exception': raise OSError('inert mocked native call exception')
        return 3 if fault == 'shutdown_error' else 0
    lib = SimpleNamespace(nvmlInit_v2=Fn(lambda: 3 if fault == 'init_error' else 0),
                          nvmlDeviceGetHandleByUUID=Fn(get_handle), nvmlDeviceGetUUID=Fn(get_uuid),
                          nvmlDeviceGetProcessesUtilizationInfo=v1(size_code=3), nvmlShutdown=Fn(shutdown))
    monkeypatch.setattr(a, 'native_library_path', lambda: path)
    monkeypatch.setattr(a, 'load_native_library', lambda path: lib)
    return events


def test_identity_readback_shutdown_and_immutable_no_permission_semantics(tmp_path, monkeypatch):
    events = lifecycle(tmp_path, monkeypatch)
    result = a.sample_activity(UUID)
    assert result['status'] == 'unsupported' and events == ['shutdown']
    assert result['uuid'] == UUID and result['native_DLL_hash_stable'] is True
    assert result['can_authorize_compute'] is False
    assert result['classifies_compute_vs_graphics'] is False and result['empty_proves_idle'] is False


@pytest.mark.parametrize('fault', ['mismatch', 'uuid_error', 'shutdown_error', 'shutdown_exception', 'dll_drift', 'init_error'])
def test_identity_or_lifecycle_uncertainty_never_yields_a_success_result(tmp_path, monkeypatch, fault):
    events = lifecycle(tmp_path, monkeypatch, fault)
    with pytest.raises(a.ActivityError) as failure: a.sample_activity(UUID)
    assert failure.value.calls
    assert events == ([] if fault == 'init_error' else ['shutdown'])


def test_future_cursor_is_clock_uncertainty_not_idle():
    with pytest.raises(a.ActivityError, match='Future cursor'):
        a.parse_samples(b'', 'v1', 0, 0, last_seen_us=NOW + 1, now_us=NOW, max_age_us=1)


def test_representative_synthetic_v1_capture_replayed_offline():
    # SYNTHETIC portable fixture, not the ignored LIVE-001 measurement. PID/time/
    # utilization values are invented solely to exercise typed parser replay.
    rows = [
        {'pid': 101, 'timeStamp': 8_500_000, 'smUtil': 35, 'memUtil': 4,
         'encUtil': 0, 'decUtil': 0, 'jpgUtil': 0, 'ofaUtil': 0},
        {'pid': 202, 'timeStamp': 9_000_000, 'smUtil': 0, 'memUtil': 0,
         'encUtil': 0, 'decUtil': 0, 'jpgUtil': 0, 'ofaUtil': 0},
        {'pid': 303, 'timeStamp': 9_500_000, 'smUtil': 0xFFFFFFFF, 'memUtil': 2,
         'encUtil': 1, 'decUtil': 0, 'jpgUtil': 0xFFFFFFFF, 'ofaUtil': 0},
    ]
    parsed = a.parse_samples(buffer(rows), 'v1', len(rows), len(rows), last_seen_us=0,
                             now_us=max(row['timeStamp'] for row in rows) + 1_000_000, max_age_us=10_000_000)
    assert len(parsed['samples']) == 3
    assert {row['pid'] for row in parsed['samples']} == {row['pid'] for row in rows}
    assert [row['smUtil'] for row in parsed['samples']] == [35, 0, None]
    assert parsed['samples'][-1]['jpgUtil'] is None
