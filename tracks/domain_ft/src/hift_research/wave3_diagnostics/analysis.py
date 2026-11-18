"""Retrospective containment/response/source proxies; never imported by tracking.

All GT support is geometric. Image statistics are exploratory, not optical detail,
validated blur measures, semantic distractor labels, or causal resolution evidence.
"""
import math
import numpy as np
from hift_research.diagnostics.source_support import retrospective_rows


def containment_class(fraction, *, tolerance=1e-9):
    if not np.isfinite(fraction) or fraction < -tolerance or fraction > 1+tolerance:
        raise ValueError('containment fraction outside [0,1]')
    return 'full' if fraction >= 1-tolerance else ('outside' if fraction <= tolerance else 'partial')


def candidate_boxes_in_image(crop_boxes, center, scale):
    boxes = np.asarray(crop_boxes, dtype=float)
    center = np.asarray(center, dtype=float)
    if boxes.ndim != 2 or boxes.shape[0] != 4 or center.shape != (2,) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('expected [4,N] center/size candidates and positive decoder scale')
    size = boxes[2:].T / scale
    centers = boxes[:2].T / scale + center
    return np.column_stack((centers-size/2, size))


def candidate_ious(boxes, gt):
    boxes, gt = np.asarray(boxes, float), np.asarray(gt, float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or gt.shape != (4,) or not np.isfinite(gt).all() or min(gt[2:]) <= 0:
        raise ValueError('Nx4 candidates and finite positive GT required')
    valid = np.isfinite(boxes).all(axis=1) & (boxes[:,2:] > 0).all(axis=1)
    result = np.zeros(len(boxes))
    a = boxes[valid]
    intersection = np.maximum(0., np.minimum(a[:,:2]+a[:,2:], gt[:2]+gt[2:])-np.maximum(a[:,:2], gt[:2])).prod(axis=1)
    union = a[:,2:].prod(axis=1)+gt[2:].prod()-intersection
    result[valid] = intersection/union
    return result


def response_features(boxes, gt, probabilities, logits, selection, selected_index):
    ious = candidate_ious(boxes, gt)
    p, z, selection = [np.asarray(value, float) for value in (probabilities, logits, selection)]
    if any(value.shape != ious.shape for value in (p,z,selection)) or not 0 <= selected_index < len(ious):
        raise ValueError('response arrays/index mismatch')
    finite = np.isfinite(p) & np.isfinite(z) & np.isfinite(selection)
    bounded_p = np.clip(p[finite], 0, 1)
    mass = bounded_p/bounded_p.sum() if bounded_p.sum() > 0 else np.array([])
    entropy = float(-(mass[mass>0]*np.log(mass[mass>0])).sum()/np.log(len(mass))) if len(mass)>1 else None
    score = (p+z)/2  # B0 raw mixture, not a probability
    positive, background = (ious >= .5) & finite, (ious <= .1) & finite
    contrast = float(score[positive].max()-score[background].max()) if positive.any() and background.any() else None
    ordered = np.sort(p[finite])
    return {'best_decoded_candidate_iou': float(ious.max()), 'selected_candidate_iou': float(ious[selected_index]),
            'candidate_count_iou_ge_0_5': int((ious >= .5).sum()),
            'B0_best_GT_candidate_minus_best_background_score': contrast,
            'class_mass_entropy_proxy': entropy,
            'class_top_two_margin': float(ordered[-1]-ordered[-2]) if len(ordered)>1 else None,
            'nonfinite_response_candidates': int((~finite).sum())}


def image_quality_proxies(image, bbox):
    """Native-image bbox/ring statistics; no resizing or information hallucination."""
    import cv2
    x,y,w,h = np.asarray(bbox,float)
    if not np.isfinite([x,y,w,h]).all() or min(w,h) <= 0:
        raise ValueError('finite positive bbox required')
    ih, iw = image.shape[:2]
    x0,x1 = int(np.ceil(np.clip(x,0,iw))), int(np.ceil(np.clip(x+w,0,iw)))
    y0,y1 = int(np.ceil(np.clip(y,0,ih))), int(np.ceil(np.clip(y+h,0,ih)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    patch = gray[y0:y1,x0:x1]
    result = {'native_roi_sampled_pixels': int(patch.size), 'native_intensity_variance': None,
              'native_sobel_energy': None, 'native_laplacian_variance': None,
              'target_surround_histogram_distance': None,
              'quality_interpretation': 'exploratory_bbox_statistics_not_validated_optical_or_blur_measure'}
    if not patch.size:
        return result
    result['native_intensity_variance'] = float(np.var(patch.astype(float)))
    if min(patch.shape) >= 3:
        sx = cv2.Sobel(patch,cv2.CV_64F,1,0,ksize=3)
        sy = cv2.Sobel(patch,cv2.CV_64F,0,1,ksize=3)
        result['native_sobel_energy'] = float(np.mean(sx*sx+sy*sy))
        result['native_laplacian_variance'] = float(cv2.Laplacian(patch,cv2.CV_64F).var())
    pad = max(2., .5*min(w,h))
    a,b = int(np.floor(max(0,x-pad))), int(np.floor(max(0,y-pad)))
    c,d = int(np.ceil(min(iw,x+w+pad))), int(np.ceil(min(ih,y+h+pad)))
    context = gray[b:d,a:c]
    ring = np.ones(context.shape,dtype=bool)
    ring[max(0,y0-b):max(0,y1-b),max(0,x0-a):max(0,x1-a)] = False
    background = context[ring]
    if background.size >= 8:
        if image.ndim == 3:
            target_color = image[y0:y1,x0:x1].reshape(-1,3)
            surround_color = image[b:d,a:c][ring]
            hist1 = np.histogramdd(target_color,bins=(4,4,4),range=((0,256),)*3)[0].reshape(-1)
            hist2 = np.histogramdd(surround_color,bins=(4,4,4),range=((0,256),)*3)[0].reshape(-1)
        else:
            hist1 = np.histogram(patch,bins=16,range=(0,256))[0].astype(float)
            hist2 = np.histogram(background,bins=16,range=(0,256))[0].astype(float)
        hist1 /= hist1.sum(); hist2 /= hist2.sum()
        result['target_surround_histogram_distance'] = float(np.sqrt(max(0.,1-np.sqrt(hist1*hist2).sum())))
    return result


def enrich_rows(ground_truth, predictions, logs, grids, *, baseline, annotation_origin, quality=None):
    rows = retrospective_rows(ground_truth,predictions,logs,annotation_origin=annotation_origin,invalid_policy='count_as_failure')
    gt = np.asarray(ground_truth,float).copy()
    gt[:,:2] -= annotation_origin
    for i,row in enumerate(rows):
        row['annotation_origin_assumption'] = annotation_origin
        if not row['initialization']:
            row['selected_score'] = logs[i]['decision'][baseline].get('selected_score')
            row['interpolation_lr'] = logs[i]['decision'][baseline].get('interpolation_lr')
            row['same_response_counterfactual_rank_differs'] = logs[i]['decision'].get('same_state_counterfactual_ranking_differs')
        if row['initialization'] or row['ground_truth_unavailable']:
            continue
        row['crop_containment_class'] = containment_class(row['crop_containment_fraction'])
        row['joint_input_availability_class'] = containment_class(row['joint_crop_image_containment_fraction'])
        row['source_min_side_px'] = min(row['source_width_px'],row['source_height_px'])
        row['min10_ideal_iou_upper_bound'] = row['source_area_px2']/(max(10.,row['source_width_px'])*max(10.,row['source_height_px']))
        row['resized_min_side_px'] = min(row['resized_width_px'],row['resized_height_px'])
        row['motion_in_target_min_sides'] = row['motion_px']/row['source_min_side_px'] if row['motion_px'] is not None else None
        row['source_size_bin'] = '<16' if row['source_min_side_px'] < 16 else ('16-32' if row['source_min_side_px'] < 32 else '>=32')
        row['footprint_bin'] = '<32' if row['resized_min_side_px'] < 32 else ('32-64' if row['resized_min_side_px'] < 64 else '>=64')
        motion = row['motion_in_target_min_sides']
        row['motion_bin'] = 'unknown' if motion is None else ('<0.5' if motion < .5 else ('0.5-1' if motion < 1 else '>=1'))
        row['elongated'] = max(row['aspect_ratio'],1/row['aspect_ratio']) >= 3
        previous = rows[i-1]
        known_previous = not previous['ground_truth_unavailable']
        row['previous_frame_iou'] = previous['iou'] if known_previous else None
        newly_outside = row['crop_containment_class'] == 'outside' and known_previous and (i==1 or previous.get('crop_containment_class') != 'outside')
        row['outside_after_unavailable_GT'] = row['crop_containment_class'] == 'outside' and not known_previous
        row['new_crop_loss_event'] = newly_outside
        row['drift_preceded_crop_loss'] = bool(newly_outside and known_previous and previous['iou'] < .1)
        row['good_prediction_before_crop_loss'] = bool(newly_outside and known_previous and previous['iou'] >= .5)
        mapping = candidate_boxes_in_image(grids['predicted_crop_boxes'][i-1],logs[i]['prior_state']['center'],logs[i]['crop_geometry']['ideal_decoder_scale_z'])
        features = response_features(mapping,gt[i],grids['class_probability'][i-1],grids['bce_logits'][i-1],
                                     grids['B0_selection' if baseline == 'B0' else 'B1_selection'][i-1],
                                     logs[i]['decision'][baseline]['selected_index'])
        row.update(features)
        if row['iou'] < .1:
            row['failure_response_context'] = ('input_absent' if row['joint_input_availability_class']=='outside' else
                ('update_policy_limited' if row['selected_candidate_iou'] >= .5 else
                 ('selection_or_scoring_limited' if row['best_decoded_candidate_iou'] >= .5 else 'candidate_localization_limited')))
        else:
            row['failure_response_context'] = 'not_low_iou_failure'
        if quality is not None:
            row.update(quality.get(i,{}))
    return rows


def cluster_interval(sequence_rates, *, repeats=2000, seed=20261023):
    rates = np.asarray(sequence_rates,float)
    if rates.ndim != 1 or not len(rates) or not np.isfinite(rates).all():
        return {'sequences':len(rates),'mean':None,'interval_95':None}
    mean = float(rates.mean())
    if len(rates) < 3:
        return {'sequences':len(rates),'mean':mean,'interval_95':None,'limitation':'fewer_than_3_sequence_clusters'}
    rng = np.random.default_rng(seed)
    samples = rates[rng.integers(0,len(rates),(repeats,len(rates)))].mean(axis=1)
    return {'sequences':len(rates),'mean':mean,'interval_95':np.quantile(samples,[.025,.975]).tolist(),
            'unit':'sequence_cluster_not_frame','limitation':'selected_panel_bootstrap_not_population_or_causal_inference'}


def describe_sequence(rows):
    tracked = [r for r in rows if not r['initialization']]
    valid = [r for r in tracked if not r['ground_truth_unavailable']]
    failures = [r for r in valid if r['zero_overlap_failure']]
    categories = {}
    for label in ('full','partial','outside'):
        subset = [r for r in valid if r['crop_containment_class']==label]
        categories[label] = {'frames':len(subset),'zero_overlap_frames':sum(r['zero_overlap_failure'] for r in subset),
                             'low_iou_frames':sum(r['iou_below_0_1'] for r in subset),
                             'mean_iou':float(np.mean([r['iou'] for r in subset])) if subset else None}
    contexts = {label:sum(r['failure_response_context']==label for r in valid) for label in
                ('input_absent','update_policy_limited','selection_or_scoring_limited','candidate_localization_limited')}
    critical = [r for r in valid if r['source_min_side_px']<16 and r['resized_min_side_px']>=32]
    contained_critical = [r for r in critical if r['joint_input_availability_class']=='full']
    # Match only within sequence, footprint/motion/aspect strata; no cross-video pooling.
    strata = {}
    for row in valid:
        if row['joint_input_availability_class'] != 'full' or row['motion_bin']=='unknown' or row['min10_ideal_iou_upper_bound'] < .5:
            continue
        key = (row['footprint_bin'],row['motion_bin'],row['elongated'])
        group = strata.setdefault(key,{'small':[],'larger':[]})
        group['small' if row['source_min_side_px']<16 else 'larger'].append(row)
    comparisons = []
    for key,group in strata.items():
        if min(len(group['small']),len(group['larger'])) < 5:
            continue
        comparisons.append({'stratum':list(key),'small_n':len(group['small']),'larger_n':len(group['larger']),
            'small_minus_larger_zero_overlap_rate':float(np.mean([r['zero_overlap_failure'] for r in group['small']])-np.mean([r['zero_overlap_failure'] for r in group['larger']])),
            'interpretation':'descriptive_within_sequence_time_appearance_and_history_confounding_remain'})
    return {'tracked_frames':len(tracked),'unavailable_GT':len(tracked)-len(valid),'valid_GT':len(valid),
            'zero_overlap_frames':len(failures),'containment':categories,'low_iou_response_contexts':contexts,
            'fraction_zero_overlap_failures_not_fully_contained':sum(r['crop_containment_class']!='full' for r in failures)/len(failures) if failures else None,
            'outside_after_unavailable_GT_not_counted_as_observed_transition':sum(r['outside_after_unavailable_GT'] for r in valid),
            'source_bbox_not_fully_in_image_frames':sum(r['source_image_containment_fraction'] < 1-1e-9 for r in valid),
            'source_bbox_completely_outside_image_frames':sum(r['source_image_containment_fraction'] <= 1e-9 for r in valid),
            'full_crop_but_source_bbox_clipped_frames':sum(r['source_image_containment_fraction'] < 1-1e-9 and r['crop_containment_class']=='full' for r in valid),
            'policy':{'selected_score_outside_0_1_frames':sum(r.get('selected_score') is not None and not 0 <= r['selected_score'] <= 1 for r in tracked),
                      'interpolation_lr_outside_0_1_frames':sum(r.get('interpolation_lr') is not None and not 0 <= r['interpolation_lr'] <= 1 for r in tracked),
                      'counterfactual_same_response_rank_flip_frames':sum(bool(r.get('same_response_counterfactual_rank_differs')) for r in tracked)},
            'new_crop_loss_events':sum(r['new_crop_loss_event'] for r in valid),
            'drift_preceded_crop_loss_events':sum(r['drift_preceded_crop_loss'] for r in valid),
            'good_prediction_before_crop_loss_events':sum(r['good_prediction_before_crop_loss'] for r in valid),
            'small_source_large_footprint':{'frames':len(critical),'zero_overlap':sum(r['zero_overlap_failure'] for r in critical),
                'fully_available_frames':len(contained_critical),'fully_available_zero_overlap':sum(r['zero_overlap_failure'] for r in contained_critical)},
            'min10_excludes_iou_0_5_frames':sum(r['min10_ideal_iou_upper_bound'] < .5 for r in valid),
            'fully_available_min10_excludes_iou_0_5_frames':sum(r['min10_ideal_iou_upper_bound'] < .5 and r['joint_input_availability_class']=='full' for r in valid),
            'source_information_matched_strata':comparisons}


def select_oracle_frames(rows, *, per_group=4):
    """Deterministic spread, not worst-case selection; reuse same cases across origins."""
    selected = []
    for label in ('full','partial','outside'):
        candidates = [r['frame'] for r in rows if not r['initialization'] and not r['ground_truth_unavailable'] and r['crop_containment_class']==label]
        if candidates:
            positions = np.unique(np.rint(np.linspace(0,len(candidates)-1,min(per_group,len(candidates)))).astype(int))
            selected.extend({'frame':candidates[i],'observed_containment_group':label} for i in positions)
    return sorted(selected,key=lambda r:r['frame'])
