import pytest
from hift_research.wave4_feasibility import StageObserver,latency,project_budget


def test_external_stage_observer():
    observer=StageObserver()
    observer.record('decode',.002)
    observer.record('decode',0,cache_hit=True)
    data=observer.summary()['decode']
    assert data['total_ms']==2 and data['cache_hits']==1 and data['cache_misses']==1
    assert data['median_ms']==1


@pytest.mark.parametrize('values',[[],[-1],[float('nan')],[float('inf')],[[1,2]]])
def test_invalid_timing_observations(values):
    with pytest.raises(ValueError):latency(values)


def test_budget_includes_real_input_and_development_not_only_synthetic():
    result=project_budget(candidate_ms=[10,10],collate_transfer_ms=[1],forward_ms=[2],
        backward_clip_step_ms=[3],initialization_seconds=1,checkpoint_seconds=.25)
    assert result['mean_training_seconds']==pytest.approx(18.4)
    assert result['mean_development_seconds']==pytest.approx(5.504)
    assert result['central_seconds']==pytest.approx(25.904)
    assert result['engineering_range_seconds'][1]>result['central_seconds']
    assert result['suggested_hard_wall_seconds']>=300


def test_budget_uses_actual_complete_development_cost_when_measured():
    result=project_budget(candidate_ms=[10],collate_transfer_ms=[1],forward_ms=[2],
        backward_clip_step_ms=[3],initialization_seconds=1,checkpoint_seconds=.25,
        development_full_check_seconds=1.25)
    assert result['mean_development_seconds']==5
    assert result['measured_complete_development_check_seconds']==1.25
