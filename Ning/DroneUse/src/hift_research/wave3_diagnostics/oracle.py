"""Offline current-GT crop probes. NEVER a deployable tracker or tracking score."""
import math
import numpy as np
from hift_research.geometry import CropGeometry


def crop_conditions(gt, observed_center, observed_side, observed_decoder_scale, image_shape, *, model_size=287, max_side=4096):
    gt, center = np.asarray(gt,float), np.asarray(observed_center,float)
    if gt.shape != (4,) or center.shape != (2,) or not np.isfinite(np.r_[gt,center,observed_decoder_scale]).all() or min(gt[2:]) <= 0 or observed_decoder_scale <= 0:
        raise ValueError('valid GT, observed center and decoder scale required')
    if isinstance(observed_side,bool) or int(observed_side)!=observed_side or observed_side < 1:
        raise ValueError('positive integer observed source side required')
    target_center = gt[:2]+gt[2:]/2
    distance = np.max(np.abs(np.r_[gt[:2]-center,gt[:2]+gt[2:]-center]))
    enlarged = max(observed_side,math.ceil(2*distance)+3)
    result = []
    for name, current_center, side in (
        ('observed_center_observed_side',center,observed_side),
        ('GT_center_observed_side',target_center,observed_side),
        ('observed_center_enlarged_side',center,enlarged),
        ('GT_center_enlarged_side',target_center,enlarged)):
        item = {'condition':name,'center':current_center.tolist(),'source_side':int(side),
                'uses_current_GT':name!='observed_center_observed_side',
                'kind':'offline_oracle_frame_probe_NOT_tracking',
                'status':'ready' if side <= max_side else 'source_crop_budget_exceeded'}
        if item['status']=='ready':
            geometry = CropGeometry.from_tracker(current_center,side,model_size)
            support = geometry.support(gt,image_shape)
            item.update({'decoder_scale_z':float(observed_decoder_scale if side==observed_side else model_size/side),
                         'origin_x':geometry.origin_x,'origin_y':geometry.origin_y,
                         'delivered_geometric_source_area_px2':support['genuine_visible_support_px2'],
                         'joint_containment_fraction':support['containment_fraction'],
                         'delivered_resized_bbox_area_px2':support['genuine_visible_support_px2']*geometry.scale**2,
                         'nominal_resized_min_side_px':float(min(gt[2:])*geometry.scale),
                         'fully_available':bool(support['containment_fraction']>=1-1e-9)})
        result.append(item)
    return result


def emitted_probe_box(crop_boxes, decision, prior_size, center, decoder_scale, image_shape):
    """Literal original single-frame decode/smooth/clip, without modifying tracker state."""
    boxes = np.asarray(crop_boxes)
    bbox = boxes[:,decision['selected_index']]/decoder_scale
    lr = decision['interpolation_lr']
    cx,cy = bbox[:2]+np.asarray(center)
    width = prior_size[0]*(1-lr)+bbox[2]*lr
    height = prior_size[1]*(1-lr)+bbox[3]*lr
    ih,iw = image_shape[:2]
    cx,cy = max(0,min(cx,iw)),max(0,min(cy,ih))
    width,height = max(10,min(width,iw)),max(10,min(height,ih))
    result = np.array([cx-width/2,cy-height/2,width,height])
    if not np.isfinite(result).all():
        raise ValueError('nonfinite offline probe; no repair')
    return result
