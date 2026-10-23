"""CPU/random-synthetic-only safety guards and journal invariance."""
import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from hift_research.training.lifecycle import apply_epoch_modes, lifecycle_optimizer, EpochSchedule, capture_rng, window_denominators, window_loss
from hift_research.training.graph import synthetic_batch
from hift_research.training.wave4_loader import SyntheticCandidateDataset
from hift_research.training.wave4_pilot import attempt_batch
from hift_research.wave6.runtime import RuntimeGuard, GuardViolation, FatalInputError, AllowedGeometryRejection, gradient_norms, loss_components
from hift_research.wave6.telemetry import DurableJournal, read_last_good, JournalCorruptionError


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4))
        self.grader=nn.Sequential(nn.Linear(4,4),nn.Dropout(.2),nn.Linear(4,1))
    def forward(self,x):
        return self.grader(self.backbone(x))


def build(**options):
    model=apply_epoch_modes(TinyModel(),0)
    optimizer=lifecycle_optimizer(model)
    EpochSchedule(optimizer).set_epoch(0)
    guard=RuntimeGuard(model,optimizer,synthetic=True,**options)
    return model,optimizer,guard


def same(a,b):
    if isinstance(a,torch.Tensor):
        return torch.equal(a,b)
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def test_guards_and_durable_logging_preserve_rng_parameters_and_gradients(tmp_path):
    model,opt,guard=build()
    model(torch.arange(16,dtype=torch.float32).reshape(4,4)).sum().backward()
    before=copy.deepcopy(model.state_dict())
    grads={n:p.grad.clone() if p.grad is not None else None for n,p in model.named_parameters()}
    rng=capture_rng()
    journal=DurableJournal(tmp_path/'events.jsonl')
    journal.emit('backward',**guard.check('after_backward'))
    guard.check('pre_clip')
    assert same(rng,capture_rng()) and same(before,model.state_dict())
    assert same(grads,{n:p.grad for n,p in model.named_parameters()})
    torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
    guard.check('pre_step');opt.step();result=guard.check('post_step')
    assert result['digests']['backbone']==guard.initial_snapshot['backbone']
    assert result['digests']['grader']!=guard.initial_snapshot['grader']
    assert gradient_norms(model)['finite']


@pytest.mark.parametrize('mutation',['backbone','bn','trainability','modes','group_order','lr','nan_grader','nan_grad','buffer_shape'])
def test_fatal_runtime_mutations_before_publish(mutation):
    model,opt,guard=build()
    with torch.no_grad():
        if mutation=='backbone':model.backbone[0].weight.add_(1)
        elif mutation=='bn':model.backbone[1].running_mean.add_(1)
        elif mutation=='trainability':model.backbone[0].weight.requires_grad_(True)
        elif mutation=='modes':model.backbone[1].train()
        elif mutation=='group_order':opt.param_groups.reverse()
        elif mutation=='lr':opt.param_groups[1]['lr']=.9
        elif mutation=='nan_grader':model.grader[0].weight.fill_(float('nan'))
        elif mutation=='buffer_shape':model.backbone[1].running_mean=torch.zeros(8)
        else:model.grader[0].weight.grad=torch.full_like(model.grader[0].weight,float('nan'))
    with pytest.raises(GuardViolation):guard.check('post_step')


def test_evaluation_modes_allowed_only_explicitly_and_restored():
    model,opt,guard=build()
    model.eval()
    with pytest.raises(GuardViolation):guard.check('dev')
    guard.check('dev',allow_evaluation=True)
    apply_epoch_modes(model,0);guard.check('after_dev')


def test_clipped_norm_scope_source_and_momentum_fail_closed():
    model,opt,guard=build()
    model.grader[0].weight.grad=torch.ones_like(model.grader[0].weight)*100
    guard.check('pre_clip')
    with pytest.raises(GuardViolation,match='clipped norm'):guard.check('pre_step')
    torch.nn.utils.clip_grad_norm_(model.parameters(),10);guard.check('pre_step')
    opt.state[next(model.backbone.parameters())]['momentum_buffer']=torch.zeros_like(next(model.backbone.parameters()))
    with pytest.raises(GuardViolation,match='optimizer state'):guard.check('post_step')
    with pytest.raises(GuardViolation):build(source_check=lambda:False)


@pytest.mark.parametrize('field,value',[('reserved_bytes',81),('allocated_bytes',81),('host_rss_bytes',9*1024**3),
                                       ('available_host_bytes',3*1024**3),('total_bytes',float('nan')),('wall_exhausted',True)])
def test_resource_limits_every_callback(field,value):
    probe={'reserved_bytes':80,'total_bytes':100,'allocated_bytes':70,'host_rss_bytes':1024**3,'available_host_bytes':8*1024**3}
    model,opt,guard=build(resource_probe=lambda:dict(probe),max_host_rss_bytes=8*1024**3,min_available_host_bytes=4*1024**3)
    probe[field]=value
    with pytest.raises(GuardViolation):guard.check('after_backward')


def test_partial_batch_and_allowed_geometry_rejections_do_not_refill():
    dataset=SyntheticCandidateDataset(length=4,rejected=(1,))
    batch=attempt_batch(dataset,[0,1,2,3],kind='synthetic')
    model,opt,guard=build(expected_identity={'manifest_sha256':dataset.manifest_sha256},seed=dataset.seed)
    summary=guard.validate_batch(batch)
    assert summary['attempts']==4 and summary['accepted']==3 and summary['rejected']==1
    assert summary['allowed_rejections']==[{'candidate_index':1,'reason':'synthetic_geometry_reject'}]
    empty=attempt_batch(SyntheticCandidateDataset(length=4,rejected=(0,1,2,3)),[0,1,2,3],kind='synthetic')
    assert build()[2].validate_batch(empty)['accepted']==0


@pytest.mark.parametrize('fault',['dtype','shape','nonfinite','mask','negative','identity','seed','reason'])
def test_malformed_inputs_are_fatal_not_rejected_or_refilled(fault):
    dataset=SyntheticCandidateDataset(length=4,rejected=(1,))
    batch=attempt_batch(dataset,[0,1,2,3],kind='synthetic')
    guard=build(expected_identity={'manifest_sha256':dataset.manifest_sha256},seed=dataset.seed)[2]
    if fault=='dtype':batch['template']=batch['template'].double()
    elif fault=='shape':batch['template']=batch['template'][:,:,:126,:]
    elif fault=='nonfinite':batch['search'].fill_(float('nan'))
    elif fault=='mask':batch['targets']['label_cls1'].fill_(4)
    elif fault=='negative':batch['targets']['weightxff'][-1].fill_(1)
    elif fault=='identity':batch['attempt_records'][0]['manifest_sha256']='wrong'
    elif fault=='seed':batch['attempt_records'][0]['seed']+=1
    else:batch['rejected_records'][0]['reasons']=['decode failure']
    with pytest.raises(FatalInputError):guard.validate_batch(batch)


def test_real_geometry_whitelist_requires_evidence_and_corruption_is_fatal():
    guard=build()[2]
    allowed={'candidate_index':3,'candidate_negative':True,'reason':'template_target_visible_in_negative_search',
             'negative_template_A_overlap_px2':2.,'negative_exclusion_proved':False}
    assert isinstance(guard.classify_rejection(allowed),AllowedGeometryRejection)
    for reason in ('invalid_input_target_shape_dtype_or_finite','invalid_mask_domains','negative_dense_zero_mask_violation',
                   'empty_positive_support','OpenCV decode failed','source hash mismatch'):
        with pytest.raises(FatalInputError):guard.classify_rejection({**allowed,'reason':reason})
    with pytest.raises(FatalInputError):guard.classify_rejection({**allowed,'negative_template_A_overlap_px2':0})


def test_detached_components_match_actual_frozen_loss_and_preserve_grad_rng():
    _,_,targets=synthetic_batch(2,negative=[False,True])
    heads=(torch.rand(2,4,11,11,requires_grad=True)*.2,torch.randn(2,2,11,11,requires_grad=True),torch.randn(2,1,11,11,requires_grad=True))
    denominators=window_denominators([{'targets':targets}])
    reference=window_loss(heads,targets,denominators)
    rng=capture_rng()
    components=loss_components(heads,targets,denominators)
    assert components['total']==float(reference.detach()) and same(rng,capture_rng())
    before=torch.autograd.grad(reference,heads,retain_graph=True)
    loss_components(heads,targets,denominators)
    after=torch.autograd.grad(reference,heads)
    assert same(before,after)


def test_journal_durable_top_level_fields_and_partial_tail_recovery(tmp_path):
    path=tmp_path/'journal.jsonl';journal=DurableJournal(path)
    journal.emit('update_commit',update=1,cursor=16)
    journal.emit('checkpoint',path='synthetic.pt',sha256='checkpoint-hash')
    good=path.read_bytes()
    with path.open('ab') as stream:stream.write(b'{"seq":2,"event":"failure"')
    state=read_last_good(path)
    assert state['incomplete_tail'] and state['last_event']['sha256']=='checkpoint-hash'
    assert state['events'][0]['update']==1 and state['events'][0]['cursor']==16
    resumed=DurableJournal(path,resume=True)
    assert path.read_bytes()==good
    resumed.emit('failure',reason='injected',last_committed_update=1)
    assert len(read_last_good(path)['events'])==3


def test_optional_authentic_first_four_saved_archives_validate_input_only_without_model_call():
    from itertools import islice
    from hift_research.training.recipe import ROOT
    from hift_research.provenance import sha256
    from hift_research.training.wave4_loader import ordered_collate
    from hift_research.training.wave4_layout import canonicalize_model_inputs
    ledger=ROOT/'artifacts/wave4-preflight/PREFLIGHT-002/train/candidate_ledger.jsonl'
    policy=ROOT/'artifacts/wave4-preflight/FROZEN-002/train_policy.json'
    if not ledger.exists() or not policy.exists():
        pytest.skip('optional immutable authentic preflight archives absent; never acquire data from tests')
    with ledger.open(encoding='utf-8') as stream:
        records=[json.loads(line) for line in islice(stream,4)]
    assert [r['candidate_index'] for r in records]==[0,1,2,3]
    assert [r['candidate_negative'] for r in records]==[False,False,False,True]
    pairs=[]
    for record in records:
        assert sha256(record['array_archive_path'])==record['array_archive_sha256']
        with np.load(record['array_archive_path'],allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        for key in ('template','search'):
            arrays[key]=np.ascontiguousarray(arrays[key].transpose(1,2,0)).transpose(2,0,1)
        pairs.append((record,arrays))
    collated=ordered_collate(pairs)
    values=canonicalize_model_inputs(collated['model_inputs'])
    batch={'data_kind':'authorized_future_real','sample_ids':[0,1,2,3],
           'template':values['template'],'search':values['search'],
           'targets':{k:values[k] for k in ('label_cls1','labelcls2','labelxff','weightxff')},
           **{k:collated[k] for k in ('attempt_records','accepted_records','rejected_records','replacement_count')}}
    model,opt,_=build()
    identity={k:records[0][k] for k in ('manifest_sha256','reservations_sha256')}
    identity['policy_sha256']=sha256(policy)
    guard=RuntimeGuard(model,opt,synthetic=False,expected_identity=identity,seed=20261023)
    before=guard.snapshot();rng=capture_rng()
    counts=guard.validate_batch(batch)  # absolutely no forward/backward/optimizer.step with these real arrays
    assert counts=={'attempts':4,'accepted':4,'rejected':0,'positive':3,'negative':1,'allowed_rejections':[]}
    assert guard.snapshot()==before and same(rng,capture_rng()) and not opt.state
    assert all(p.grad is None for p in model.parameters())
    corrupt=copy.deepcopy(batch)
    corrupt['template'][0,0,0,0]+=1
    with pytest.raises(FatalInputError,match='hash changed'):guard.validate_batch(corrupt)
    corrupt_mask=copy.deepcopy(batch)
    first_positive=torch.nonzero(corrupt_mask['targets']['label_cls1'][0]==1)[0]
    corrupt_mask['targets']['label_cls1'][(0,*first_positive.tolist())]=0
    assert bool(corrupt_mask['targets']['label_cls1'][0].eq(1).any())  # still domain-valid/nonempty
    with pytest.raises(FatalInputError,match='hash changed:label_cls1'):guard.validate_batch(corrupt_mask)


def test_journal_complete_corruption_is_not_silently_repaired(tmp_path):
    path=tmp_path/'journal.jsonl';journal=DurableJournal(path);journal.emit('update_commit',update=1)
    raw=path.read_bytes().replace(b'"update":1',b'"update":2');path.write_bytes(raw)
    with pytest.raises(JournalCorruptionError):read_last_good(path,repair=True)
    assert path.read_bytes()==raw
    with pytest.raises(ValueError):journal.emit('bad',norm=float('nan'))
    with pytest.raises(ValueError):journal.emit('bad',model_tensor=torch.ones(1))
