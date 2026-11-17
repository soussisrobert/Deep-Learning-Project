"""Read-only preservation and independent pair algebra; no model/training execution."""
from pathlib import Path
import subprocess
from .evaluation.assets import read_json,sha256_file
from . import UPSTREAM_COMMIT,OFFICIAL_CHECKPOINT_SHA256

WAVE3_HEAD='28e9e197c62a21450587465e4c32abdf97fb88d3'
BOOKKEEPING_MUTABLE={'README.md','experiments/registry.json'}


def verify_prior(root):
    root=Path(root).resolve()
    def git(*args,raw=False):
        result=subprocess.check_output(['git',*args],cwd=root)
        return result if raw else result.decode('utf-8').strip()
    if git('-C','third_party/HiFT','rev-parse','HEAD')!=UPSTREAM_COMMIT:
        raise ValueError('immutable upstream commit changed')
    if git('-C','third_party/HiFT','status','--porcelain'):
        raise ValueError('immutable upstream has working-tree changes')
    subprocess.run(['git','merge-base','--is-ancestor',WAVE3_HEAD,'HEAD'],cwd=root,check=True)
    protected={}
    for name in git('ls-tree','-r','--name-only',WAVE3_HEAD).splitlines():
        if name in BOOKKEEPING_MUTABLE or name=='third_party/HiFT':
            continue
        path=root/name
        # Git's historical text filters normalize existing CRLF working files;
        # compare canonical blobs, while retaining actual SHA256 separately.
        if not path.is_file() or git('hash-object','--path='+name,name)!=git('rev-parse',f'{WAVE3_HEAD}:{name}'):
            raise ValueError('prior file changed or missing:'+name)
        protected[name]=sha256_file(path)
    start_path=root/'artifacts/wave4-integration/START-001.json'
    start_bytes_equal=None
    if start_path.is_file():
        previous=read_json(start_path)
        if protected!=previous['protected_prior_files_sha256']:
            raise ValueError('actual prior working-file bytes changed since Wave4 starting receipt')
        start_bytes_equal=True
    references={}
    for receipt in ('experiments/wave2_validation.json','experiments/wave3_validation.json'):
        original=read_json(root/receipt)
        references[receipt]={}
        for name,ref in original['references'].items():
            target=Path(ref['path'])
            target=target if target.is_absolute() else root/target
            if sha256_file(target)!=ref['sha256']:
                raise ValueError('prior experiment reference changed:'+receipt+':'+name)
            references[receipt][name]=ref['sha256']
    checkpoint=root/'artifacts/checkpoints/general_model.pth'
    if sha256_file(checkpoint)!=OFFICIAL_CHECKPOINT_SHA256:
        raise ValueError('official checkpoint changed')
    return {'verified_starting_delivery':WAVE3_HEAD,'current_HEAD':git('rev-parse','HEAD'),
            'upstream_commit':UPSTREAM_COMMIT,'upstream_clean':True,
            'official_checkpoint_sha256':OFFICIAL_CHECKPOINT_SHA256,
            'protected_prior_file_count':len(protected),'protected_prior_files_sha256':protected,
            'file_preservation_comparison':'Git canonical blobs (existing text EOL filters honored); actual working-byte SHA256 recorded separately',
            'actual_working_bytes_equal_wave4_start':start_bytes_equal,
            'prior_receipt_references':references,
            'current_git_status':git('status','--short'),
            'no_prior_experiments_executed':True}


def independent_pair_algebra(manifest_path,ledger_path,*,split='train'):
    """Re-derive source endpoints, physical footprints and grid offsets without the sampler.

    This supplements, not replaces, the separate fresh-production-pipeline/all-array
    replay. Reads small target members of every accepted NPZ with pickle disabled.
    """
    import csv,json,math
    from collections import Counter
    import numpy as np
    manifest=read_json(manifest_path)
    source={}
    for video in manifest['videos']:
        gt={}
        with Path(video['source_annotation_path']).open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.reader(stream):
                if not row:continue
                values=list(map(float,row))
                key=(int(values[0]),str(int(values[1])))
                if key in gt:raise ValueError('duplicate original GT endpoint')
                gt[key]=values
        source[video['video_lineage']]={'gt':gt,'frames':{f['frame_id']:f for f in video['frames']},'split':video['split']}
    with Path(ledger_path).open(encoding='utf-8') as stream:records=[json.loads(line) for line in stream if line.strip()]
    expected=1024 if split=='train' else 128
    if [r['candidate_index'] for r in records]!=list(range(expected)):raise ValueError('non-exhaustive candidate ledger')
    totals=Counter();max_offset_error=0.;max_affine_error=0.;source_tracks=set();source_frames=set()
    def check(condition,message):
        if not condition:raise ValueError(message)
    def overlap(a,b):return max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    for r in records:
        negative=r['candidate_negative'];index=r['candidate_index']
        requested=(np.random.RandomState(20261023+index).random_sample()<.05) if split=='train' else index%2==1
        check(bool(requested)==negative,'class request does not match frozen indexed law')
        check(r['replacement'] is None and r['replacement_count']==0,'hidden replacement')
        video=source[r['video_lineage']];check(video['split']==split,'source split contradiction')
        endpoint_boxes=[]
        for role in ('template','search'):
            identity=r[role+'_identity'];frame=r[role+'_frame_id'];track=identity[2]
            check(identity[:2]==['MOT17',r['video_lineage']],'canonical release/video identity namespace mismatch')
            check(int(track)>0,'invalid source track')
            original=video['gt'][(frame,track)]
            check(original[6]==1 and original[7]==1 and original[8]>=.5,'unusable author endpoint')
            check(np.array_equal(original[2:6],r[role+'_source_bbox_raw']),'source bbox/raw GT contradiction')
            check(video['frames'][frame]['sha256']==r[role+'_source_sha256'],'source image identity contradiction')
            x,y,w,h=original[2:6];box=np.asarray([x-1,y-1,x-1+w,y-1+h]);endpoint_boxes.append(box)
            geometry=r[role+'_geometry'];roi=geometry['augmentation']['crop_xyxy'];size=127 if role=='template' else 287
            side=math.sqrt((w+.5*(w+h))*(h+.5*(w+h)))*511/127
            a=510/side;cx=x-1+w/2;cy=y-1+h/2
            prep=np.array([[a,0,-a*(cx-side/2)],[0,a,-a*(cy-side/2)],[0,0,1.]])
            ax=(size-1)/(roi[2]-roi[0]);ay=(size-1)/(roi[3]-roi[1])
            aug=np.array([[ax,0,-ax*roi[0]],[0,ay,-ay*roi[1]],[0,0,1.]])
            error=float(np.abs((aug@prep)[:2]-np.asarray(geometry['source_to_output_affine_2x3'])).max())
            max_affine_error=max(max_affine_error,error);check(error<1e-8,'independent composed affine mismatch')
            if negative and role=='search':
                guarded=np.asarray([max(0,roi[0]-18),max(0,roi[1]-18),min(510,roi[2]+18),min(510,roi[3]+18)])
                guarded=(guarded-np.array([prep[0,2],prep[1,2],prep[0,2],prep[1,2]]))/a+np.array([-2,-2,2,2])
                check(np.allclose(guarded,r['negative_guarded_source_footprint_xyxy'],rtol=0,atol=1e-8),'negative JPEG footprint mismatch')
            source_tracks.add((r['video_lineage'],track));source_frames.add((r['video_lineage'],frame))
        if negative:
            check(r['template_frame_id']==r['search_frame_id'] and r['template_identity']!=r['search_identity'],'negative lacks simultaneous distinct witness')
            check(overlap(*endpoint_boxes)==0,'negative boxes overlap')
            visible=np.asarray([max(0,endpoint_boxes[0][0]),max(0,endpoint_boxes[0][1]),
                min(r['source_width'],endpoint_boxes[0][2]),min(r['source_height'],endpoint_boxes[0][3])])
            area=overlap(visible,guarded)
            if r['status']=='accepted':check(area==0 and r['negative_exclusion_proved'] is True,'contradictory negative target in search input')
            else:check(area>0 and r['reason']=='template_target_visible_in_negative_search','unexpected negative rejection')
        else:
            check(r['template_identity']==r['search_identity'] and 1<=abs(r['template_frame_id']-r['search_frame_id'])<=19,'invalid temporal positive')
        if r['status']!='accepted':totals['rejected_negative' if negative else 'rejected_positive']+=1;continue
        with np.load(r['array_archive_path'],allow_pickle=False) as archive:
            cls=archive['label_cls1'];soft=archive['labelcls2'];weight=archive['weightxff'];offset=archive['labelxff'];bbox=archive['bbox']
            check(all(np.isfinite(v).all() for v in (cls,soft,weight,offset,bbox)),'nonfinite targets')
            check(set(np.unique(cls)).issubset({-2,-1,0,1}) and ((soft>=0)&(soft<=1)).all() and (weight>=0).all(),'mask domain mismatch')
            if negative:
                check(all(not np.any(v) for v in (cls,soft,weight,offset)),'negative has positive/localization supervision')
            else:
                check(np.any(cls==1) and np.any(soft>0) and np.any(weight>0),'empty positive support')
                xx,yy=np.meshgrid(np.arange(11)*16+63,np.arange(11)*16+63)
                independently_encoded=np.tanh(np.stack((xx-bbox[0],bbox[2]-xx,yy-bbox[1],bbox[3]-yy))/143)
                error=float(np.max(np.abs(offset-independently_encoded)));max_offset_error=max(max_offset_error,error)
                check(error<2e-6,'grid stride/origin/inverse-tanh offset semantics mismatch')
            totals['weight64_cells']+=int(np.count_nonzero(weight==64))
            totals['regression_active_cells']+=int(np.count_nonzero(weight))
        totals['accepted_negative' if negative else 'accepted_positive']+=1
    check(totals['accepted_positive']>0 and totals['accepted_negative']>0,'both supervision classes required')
    return {'kind':'independent_original_GT_footprint_and_grid_algebra_no_model','split':split,'result':'passed',
            'candidate_count':len(records),'counts':dict(totals),'unique_videos':len({v for v,t in source_tracks}),
            'unique_tracks':len(source_tracks),'unique_frames':len(source_frames),
            'maximum_composed_affine_error':max_affine_error,'maximum_offset_vs_independent_tanh_error':max_offset_error,
            'all_accepted_target_archives_checked':totals['accepted_positive']+totals['accepted_negative'],
            'manifest_sha256':sha256_file(manifest_path),'ledger_sha256':sha256_file(ledger_path)}
