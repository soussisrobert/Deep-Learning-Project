import numpy as np
import pytest
from hift_research.wave3_diagnostics.analysis import (
    containment_class,candidate_boxes_in_image,candidate_ious,response_features,
    image_quality_proxies,enrich_rows,describe_sequence,select_oracle_frames,cluster_interval)
from hift_research.wave3_diagnostics.oracle import crop_conditions,emitted_probe_box


@pytest.mark.parametrize('fraction,label',[(0,'outside'),(.2,'partial'),(1,'full'),(1-1e-10,'full')])
def test_containment(fraction,label):
    assert containment_class(fraction)==label


@pytest.mark.parametrize('bad',[float('nan'),-.1,1.1])
def test_containment_invalid(bad):
    with pytest.raises(ValueError): containment_class(bad)


def test_mapping_and_response_separate_candidate_and_selection():
    crop = np.array([[0.,50.],[0.,50.],[10.,10.],[10.,10.]])
    boxes = candidate_boxes_in_image(crop,[10,10],1)
    np.testing.assert_array_equal(boxes,[[5,5,10,10],[55,55,10,10]])
    np.testing.assert_array_equal(candidate_ious(boxes,[5,5,10,10]),[1,0])
    r = response_features(boxes,[5,5,10,10],[.3,.9],[0,4],[.1,.9],1)
    assert r['best_decoded_candidate_iou']==1 and r['selected_candidate_iou']==0
    assert r['B0_best_GT_candidate_minus_best_background_score']<0
    assert 0 <= r['class_mass_entropy_proxy'] <= 1


def test_invalid_candidate_not_baseline_prediction_fallback():
    assert candidate_ious([[0,0,-1,1],[float('nan'),0,1,1]],[0,0,1,1]).tolist()==[0,0]
    with pytest.raises(ValueError): candidate_ious([[0,0,1,1]],[0,0,-1,1])
    with pytest.raises(ValueError): candidate_boxes_in_image(np.zeros((4,2)),[0,0],0)


def test_native_image_quality_does_not_resize_or_invent_tiny_texture():
    image = np.zeros((20,20),np.uint8); image[5:10,5:10]=255
    q = image_quality_proxies(image,[5,5,5,5])
    assert q['native_roi_sampled_pixels']==25
    assert q['native_intensity_variance']==q['native_sobel_energy']==q['native_laplacian_variance']==0
    assert q['target_surround_histogram_distance']==1
    assert image_quality_proxies(image,[5,5,1,1])['native_sobel_energy'] is None
    assert image_quality_proxies(image,[100,100,1,1])['native_roi_sampled_pixels']==0
    assert 'not_validated' in q['quality_interpretation']


def fixture():
    gt = np.array([[5,5,8,8],[5,5,8,8],[100,100,8,8],[np.nan]*4])
    pred = np.array([[5,5,8,8],[5,5,8,8],[5,5,8,8],[5,5,8,8]])
    logs=[]
    for i,p in enumerate(pred):
        logs.append({'frame':i,'bbox':p.tolist(),'image_shape':[200,200,3],
            'crop_geometry':None if i==0 else {'origin_x':0,'origin_y':0,'source_size':40,'model_size':287,'ideal_decoder_scale_z':1.},
            'prior_state':{'center':[9,9],'size':[8,8]},'decision':{'B0':{'selected_index':0},'B1-score':{'selected_index':0}}})
    grids={'predicted_crop_boxes':np.tile(np.array([[0,90],[0,90],[8,8],[8,8]],float),(3,1,1)),
           'class_probability':np.ones((3,2))*.5,'bce_logits':np.zeros((3,2)),
           'B0_selection':np.tile([1.,0.],(3,1)),'B1_selection':np.tile([1.,0.],(3,1))}
    return gt,pred,logs,grids


def test_join_preserves_absence_and_actual_delivered_support():
    gt,pred,logs,grids=fixture()
    rows=enrich_rows(gt,pred,logs,grids,baseline='B0',annotation_origin=0)
    assert rows[1]['crop_containment_class']=='full'
    assert rows[2]['crop_containment_class']=='outside'
    assert rows[2]['resized_min_side_px']>=32
    assert rows[2]['resized_contained_target_area_px2']==0
    assert rows[3]['ground_truth_unavailable'] and rows[3]['source_area_px2'] is None
    summary=describe_sequence(rows)
    assert summary['unavailable_GT']==1 and summary['zero_overlap_frames']==1
    assert summary['small_source_large_footprint']['fully_available_frames']==1
    assert summary['small_source_large_footprint']['fully_available_zero_overlap']==0
    assert summary['good_prediction_before_crop_loss_events']==1
    assert select_oracle_frames(rows)==[{'frame':1,'observed_containment_group':'full'},{'frame':2,'observed_containment_group':'outside'}]


def test_unavailable_GT_gap_is_not_fabricated_crop_loss_transition():
    gt,pred,logs,grids=fixture()
    gt=np.vstack((gt,[100,100,8,8]));pred=np.vstack((pred,pred[-1]))
    logs.append({**logs[-1],'frame':4})
    grids={key:np.concatenate((value,value[-1:]),axis=0) for key,value in grids.items()}
    rows=enrich_rows(gt,pred,logs,grids,baseline='B0',annotation_origin=0)
    assert rows[4]['outside_after_unavailable_GT']
    assert not rows[4]['new_crop_loss_event'] and not rows[4]['drift_preceded_crop_loss']
    assert describe_sequence(rows)['outside_after_unavailable_GT_not_counted_as_observed_transition']==1


def test_select_spread_not_failure_picking_and_cluster_unit():
    rows=[{'initialization':False,'ground_truth_unavailable':False,'crop_containment_class':'full','frame':i} for i in range(20)]
    assert [x['frame'] for x in select_oracle_frames(rows)]==[0,6,13,19]
    a=cluster_interval([.1,.2,.9]); b=cluster_interval([.1,.2,.9])
    assert a==b and a['unit']=='sequence_cluster_not_frame'
    assert cluster_interval([.1,.2])['interval_95'] is None


def test_oracle_translation_vs_scale_factors_and_budget():
    conditions=crop_conditions([100,100,8,8],[10,10],40,7.4,[200,200,3])
    a,b,c,d=conditions
    assert a['joint_containment_fraction']==0 and b['fully_available']
    assert c['fully_available'] and d['fully_available']
    assert a['source_side']==b['source_side']==40 and c['source_side']==d['source_side']>40
    assert a['decoder_scale_z']==b['decoder_scale_z']==7.4
    assert c['nominal_resized_min_side_px']<b['nominal_resized_min_side_px']
    assert not a['uses_current_GT'] and all(x['uses_current_GT'] for x in conditions[1:])
    assert all('NOT_tracking' in x['kind'] for x in conditions)
    assert crop_conditions([10000,10000,8,8],[10,10],40,7.4,[200,200,3])[2]['status']=='source_crop_budget_exceeded'


def test_oracle_cannot_restore_source_pixels_and_large_gt_same_side():
    c=crop_conditions([-10,-10,40,40],[10,10],20,1.,[100,100,3])
    assert not c[1]['fully_available'] and not c[3]['fully_available']
    assert c[3]['joint_containment_fraction']<1
    identical=crop_conditions([5,5,8,8],[10,10],40,7.4,[100,100,3])
    assert identical[0]['source_side']==identical[2]['source_side']
    assert identical[0]['decoder_scale_z']==identical[2]['decoder_scale_z']


def test_observer_compatibility_is_narrow_and_hash_pinned():
    import hashlib
    from pathlib import Path
    from hift_research.wave3_diagnostics.observer_compat import reverse_known_observer_edits,RECORDED_WAVE2_OBSERVER_SHA256
    root=Path(__file__).resolve().parents[1]
    current=(root/'scripts/run_sequence.py').read_bytes()
    previous=reverse_known_observer_edits(current)
    assert hashlib.sha256(previous).hexdigest()==RECORDED_WAVE2_OBSERVER_SHA256
    with pytest.raises(ValueError): reverse_known_observer_edits(current+b'\n# arbitrary new change\n')


def test_emit_literal_smoothing_min_size_and_clipping():
    boxes=np.array([[1.],[2.],[4.],[8.]])
    decision={'selected_index':0,'interpolation_lr':.5}
    np.testing.assert_array_equal(emitted_probe_box(boxes,decision,[12,16],[10,10],1,[100,100]),[6,6,10,12])
    # Zero/negative LR remains literal; no hidden score correction.
    decision['interpolation_lr']=-.5
    out=emitted_probe_box(boxes,decision,[12,16],[10,10],1,[100,100])
    np.testing.assert_array_equal(out,[3,2,16,20])
