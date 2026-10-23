"""Narrow, hash-pinned Wave2 observer compatibility, never a blanket source bypass."""
import hashlib
from pathlib import Path
from hift_research.evaluation.assets import read_json,sha256_file

RELEASE_OBSERVER_SHA256='af7246e098ceef4504ce0b8af7e74f7efde440c4623e8cb3c71438d688e21ce2'
RECORDED_WAVE2_OBSERVER_SHA256='c9f0dc897e43fbf22265c2d3f75ec7231f5911280a47ed3a8aeeb3524ff3d7d2'
STRICT_GUARD=("    if ('annotation_to_tracker_transform' in data and data['annotation_to_tracker_transform'] !=\n"
              "            'upstream_numeric_passthrough_plus_half_pixel_no_origin_shift'):\n"
              "        raise ValueError('Only the declared literal upstream initialization transform is supported')\n").encode()


def reverse_known_observer_edits(current):
    """Reconstruct recorded script EXACTLY from the two disclosed Wave2 handoff edits."""
    if hashlib.sha256(current).hexdigest()!=RELEASE_OBSERVER_SHA256:
        raise ValueError('current observer is not the immutable Wave2 delivery version')
    grouped=b'sx = sz * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)'
    ungrouped=b'sx = sz * cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE'
    if current.count(grouped)!=1 or current.count(STRICT_GUARD)!=1:
        raise ValueError('known observer edit context differs')
    previous=current.replace(grouped,ungrouped).replace(STRICT_GUARD,b'')
    if hashlib.sha256(previous).hexdigest()!=RECORDED_WAVE2_OBSERVER_SHA256:
        raise ValueError('two disclosed observer edits do not reconstruct recorded hash')
    return previous


def validate_recorded_observer(root, manifest_path, recorded_hash):
    root,manifest_path=Path(root),Path(manifest_path)
    current=root/'scripts/run_sequence.py'
    current_hash=sha256_file(current)
    if recorded_hash==current_hash:
        return {'compatibility':'exact_current_source','sha256':current_hash}
    if recorded_hash!=RECORDED_WAVE2_OBSERVER_SHA256:
        raise ValueError('unrecognized observer source change')
    reverse_known_observer_edits(current.read_bytes())
    audit_path=root/'artifacts/audits/WAVE2-GEOMETRY-001.json'
    audit=read_json(audit_path)
    if not audit['passed'] or audit['failures'] or audit['tracked_rows']!=524:
        raise ValueError('known Wave2 observer geometry evidence not passed')
    matching=[r for r in audit['runs'] if (root/r['run_manifest']).resolve()==manifest_path.resolve()]
    if len(matching)!=1 or sha256_file(manifest_path)!=matching[0]['sha256']:
        raise ValueError('ordinary run is not hash-bound to known Wave2 geometry audit')
    return {'compatibility':'exactly_two_reviewed_Wave2_observer_only_edits',
            'recorded_sha256':recorded_hash,'release_sha256':current_hash,
            'edits':['added_input_transform_validation_only','grouped_observer_ratio_matches_literal_upstream'],
            'all524_saved_crops_verified':True,'geometry_audit_sha256':sha256_file(audit_path),
            'model_and_tracker_source_unchanged':True}
