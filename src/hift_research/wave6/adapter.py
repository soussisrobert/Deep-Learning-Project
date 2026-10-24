"""Narrow observations around the BYTE-UNCHANGED Wave4 serial controller.

No new loss/optimizer/sampler implementation. Only the original controller may
perform backward, clipping and SGD. Failed poststep checks taint live state and
never publish a successful update or permit a checkpoint/continuation.
"""
import contextlib
import math
import time
from pathlib import Path
from unittest.mock import patch

import torch

from hift_research.training import wave4_pilot as pilot
from hift_research.training import wave4_pilot_engine as engine
from hift_research.training.lifecycle import apply_epoch_modes, actual_heads, window_loss
from hift_research.training.wave4_lifecycle import device_tree
from .runtime import RuntimeGuard, loss_components, gradient_norms


class GuardedController(engine.SerialPilotController):
    def __init__(self,*args,journal,source_check=None,resource_probe=None,fault=None,development_identity=None,
                 min_device_free_bytes=None,expected_gpu_uuid=None,**kwargs):
        self._cooperative=min_device_free_bytes is not None
        self._training_identity=dict(kwargs.get('data_identity',{}));self._development_identity=development_identity
        super().__init__(*args,**kwargs)
        self.journal=journal;self.fault=fault;self.tainted=False
        self.guard=RuntimeGuard(self.model,self.optimizer,source_check=source_check,resource_probe=resource_probe,
                                expected_device=str(self.device),synthetic=self.synthetic,
                                max_host_rss_bytes=8*1024**3,min_available_host_bytes=4*1024**3,
                                expected_identity=self._training_identity,seed=None if self.synthetic else 20261023,
                                min_device_free_bytes=min_device_free_bytes,expected_gpu_uuid=expected_gpu_uuid)
        self.last_published_update=0;self.last_published_cursor=0;self.last_checkpoint=None
        self._micro=0;self._phase='train';self._optimizer_started=False
        self.journal.emit('controller_initialized',synthetic=self.synthetic,update=0,cursor=0,initial_snapshot=self.guard.initial_snapshot,
                          optimizer_groups=[{'name':g.get('name'),'lr':g['lr']} for g in self.optimizer.param_groups])

    def inject(self,stage):
        if self.fault is not None:self.fault(stage,self)

    def timed(self,function):
        if self._cooperative:self.guard.check('before_operation')
        name=getattr(function,'__name__','')
        is_step=getattr(function,'__self__',None) is self.optimizer and name=='step'
        if name=='backward':
            result=super().timed(function);self._micro+=1;self.inject('after_backward')
            check=self.guard.check('after_backward')
            self.journal.emit('micro_backward',micro=self._micro,checks=check,gradient_norms=gradient_norms(self.model))
            return result
        if name=='validate_and_clip':
            self.guard.check('pre_clip');self.inject('pre_clip')
            before=gradient_norms(self.model);result=super().timed(function)
            self.journal.emit('gradient_clip',before=before,after=gradient_norms(self.model),max_norm=10.0)
            return result
        if is_step:
            self.inject('pre_step');check=self.guard.check('pre_step')
            self.journal.emit('optimizer_step_begin',next_update=self.update+1,cursor=self.cursor,checks=check)
            self._optimizer_started=True
            result=super().timed(function)
            self.inject('post_step');check=self.guard.check('post_step')
            self.journal.emit('optimizer_step_validated',next_update=self.update+1,checks=check)
            return result
        result=super().timed(function)
        if self._cooperative:self.guard.check('after_operation')
        return result

    @contextlib.contextmanager
    def bindings(self,phase='train',dataset=None):
        previous_identity,previous_seed=self.guard.expected_identity,self.guard.seed
        if not self.synthetic:
            if phase=='development' and self._development_identity is None:raise ValueError('frozen DEV identity required')
            self.guard.expected_identity=self._development_identity if phase=='development' else self._training_identity
            self.guard.seed=20261024 if phase=='development' else 20261023
        original_buffer=pilot.buffered_attempt_window;original_loss=engine.window_loss
        def observed_buffer(dataset,ids,physical=4,**kwargs):
            batches,denominators=original_buffer(dataset,ids,physical,**kwargs)
            for batch in batches:self.guard.validate_batch(batch)
            records=[r for b in batches for r in b['attempt_records']]
            accepted=[r for b in batches for r in b['accepted_records']]
            rejected=[r for b in batches for r in b['rejected_records']]
            self.journal.emit('attempt_window',phase=phase,attempt_ids=ids,denominators=denominators,
                requested_positive=sum(not bool(r['candidate_negative']) for r in records),
                requested_negative=sum(bool(r['candidate_negative']) for r in records),
                accepted_positive=sum(not bool(r['candidate_negative']) for r in accepted),
                accepted_negative=sum(bool(r['candidate_negative']) for r in accepted),
                rejection_reasons=[{'index':r['candidate_index'],'reason':r.get('reason',';'.join(r.get('reasons',[])))} for r in rejected],replacements=0)
            self.inject('after_input')
            return batches,denominators
        def observed_loss(heads,targets,denominators):
            value=original_loss(heads,targets,denominators)
            if not bool(torch.isfinite(value)):raise ValueError('nonfinite loss before backward')
            components=loss_components(heads,targets,denominators)
            if not math.isclose(float(value.detach()),components['total'],rel_tol=1e-5,abs_tol=1e-6):
                raise ValueError('detached telemetry components differ from executing frozen loss')
            self.journal.emit('loss',phase=phase,micro=self._micro+1,components=components,denominators=denominators)
            self.inject('after_loss')
            return value
        try:
            with patch.object(pilot,'buffered_attempt_window',observed_buffer),patch.object(engine,'window_loss',observed_loss):
                yield
        finally:self.guard.expected_identity,self.guard.seed=previous_identity,previous_seed

    def step(self,**kwargs):
        if self.tainted:raise RuntimeError('tainted guarded controller cannot continue')
        if self.cursor>=self.max_attempts or self.update>=self.max_updates:return None
        self._micro=0;self._optimizer_started=False
        ids=list(range(self.cursor,min(self.cursor+self.attempts_per_window,self.max_attempts)))
        self.journal.emit('window_begin',next_window=self.window+1,last_update=self.update,last_cursor=self.cursor,attempt_ids=ids)
        try:
            self.guard.check('pre_window')
            with self.bindings():result=super().step(**kwargs)
            self.guard.check('commit')
            self.inject('before_publish_commit')
            self.journal.emit('update_commit',update=self.update,cursor=self.cursor,semantic=result['semantic'],measurement=result['measurement'])
            self.last_published_update=self.update;self.last_published_cursor=self.cursor
            return result
        except BaseException as exc:
            self.tainted=True;self.in_window=True
            try:
                self.journal.emit('window_failure',error=type(exc).__name__+':'+str(exc),
                    last_committed_update=self.last_published_update,last_committed_cursor=self.last_published_cursor,
                    attempt_ids=ids,micro_backward_count=self._micro,optimizer_may_have_mutated=self._optimizer_started,
                    live_state_tainted=True,automatic_retry=False)
            finally:raise

    def save_boundary(self,path):
        if self.tainted:raise RuntimeError('tainted guarded controller cannot publish checkpoint')
        try:
            self.guard.check('checkpoint')
            self.journal.emit('checkpoint_begin',update=self.update,cursor=self.cursor,path=str(Path(path).resolve()))
            self.inject('checkpoint_begin')
            # Policy B keeps bytes unpublished until post-save safety validation.
            # A killed/failed save retains only explicitly untrusted evidence.
            target=Path(path).resolve()
            staging=target.with_name(target.name+'.unvalidated') if self._cooperative else target
            receipt=super().save_boundary(staging)
            self.inject('checkpoint_saved')
            self.guard.check('checkpoint_published')
            if self._cooperative:
                import os
                os.link(staging,target)  # no overwrite; publication follows guard
                staging.unlink()  # exact resolved staging selected immediately above
                receipt={**receipt,'path':str(target)}
            self.journal.emit('checkpoint',update=self.update,cursor=self.cursor,**receipt)
            self.last_checkpoint=receipt
            return receipt
        except BaseException as exc:
            self.tainted=True
            self.journal.emit('checkpoint_failure',error=type(exc).__name__+':'+str(exc),last_checkpoint=self.last_checkpoint,
                last_committed_update=self.last_published_update,last_committed_cursor=self.last_published_cursor,live_state_tainted=True)
            raise

    def restore_boundary(self,path,*,expected_sha256):
        if not self.synthetic:raise PermissionError('first authentic guarded pilot has no resume authority')
        super().restore_boundary(path,expected_sha256=expected_sha256)
        self.guard.check('synthetic_restored')
        self.last_published_update,self.last_published_cursor=self.update,self.cursor
        self.journal.emit('synthetic_restore',update=self.update,cursor=self.cursor,path=str(Path(path).resolve()),sha256=expected_sha256)
        return self


def guarded_development(controller,dataset,*,expected_attempts=128):
    """Same frozen eval ordering/objective; guards add no model forward or RNG draws."""
    model=controller.model
    with controller.bindings('development'):
        batches,denominators=pilot.buffered_attempt_window(dataset,list(range(expected_attempts)),kind='real_development_forward_only')
    records=[r for batch in batches for r in batch['accepted_records']]
    positive=sum(not r['candidate_negative'] for r in records);negative=sum(bool(r['candidate_negative']) for r in records)
    if (positive,negative)!=(64,64) or any(b['rejected_records'] for b in batches):
        raise ValueError('fixed64positive64negativeDEV cohort incomplete; no replacement')
    total=0.0;components=[];model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                controller.guard.check('dev_before_micro',allow_evaluation=True)
                batch=device_tree(batch,controller.device)
                heads=actual_heads(model,batch);value=window_loss(heads,batch['targets'],denominators)
                if not bool(torch.isfinite(value)):raise ValueError('nonfiniteDEV loss')
                total+=float(value);components.append(loss_components(heads,batch['targets'],denominators))
                controller.guard.check('dev_after_micro',allow_evaluation=True)
    finally:apply_epoch_modes(model,0)
    result={'update':controller.update,'total_loss':total,'positive_pairs':positive,'negative_pairs':negative,
        'attempts':expected_attempts,'tracking_auc':None,'selection':'fixed pair supervised loss only','micro_components':components}
    controller.journal.emit('development',**result)
    return result
