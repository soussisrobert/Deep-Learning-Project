"""Additive read-only guards around the frozen W4 epoch0 grader-only controller."""
import hashlib
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from hift_research.training.graph import masked_classification_log_probs, masked_decode_regression
from hift_research.corrections.masked_losses import masked_bce_with_logits, masked_iou_loss


class GuardViolation(RuntimeError):
    """Fatal runtime condition: no further SGD, publication or automatic retry."""


class FatalInputError(GuardViolation):
    """Corruption/identity/decode/contract failure, never a geometry rejection."""


@dataclass(frozen=True)
class AllowedGeometryRejection:
    candidate_index: int
    reason: str


GEOMETRY_REJECTIONS = frozenset({"no_available_target_source_support", "template_target_visible_in_negative_search"})


def state_digest(state):
    """Byte digest without RNG, state mutation, or floating-point arithmetic."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def gradient_norms(model):
    """Detached native-dtype L2 diagnostics; never clip or otherwise modify grads."""
    values = {"backbone": [], "grader": []}
    all_values = []
    present = 0
    finite = True
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        present += 1
        grad = param.grad.detach()
        if grad.layout != torch.strided or not bool(torch.isfinite(grad).all()):
            finite = False
            continue
        norm = torch.linalg.vector_norm(grad)
        if not bool(torch.isfinite(norm)):
            finite = False
            continue
        all_values.append(norm)
        for prefix in values:
            if name.startswith(prefix+"."):
                values[prefix].append(norm)
    def total(norms):
        return float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0
    norm = total(all_values) if finite else None
    finite = finite and norm is not None and math.isfinite(norm)
    return {"finite": finite, "norm": norm if finite else None, "total": norm if finite else None,
            "backbone": total(values["backbone"]) if finite else None,
            "grader": total(values["grader"]) if finite else None, "present": present}


def loss_components(heads, targets, denominators):
    """Detached recomputation of frozen window_loss; no forward/backward/RNG call."""
    with torch.no_grad():
        loc, cls1, cls2 = [x.detach() for x in heads]
        labels = targets["label_cls1"].detach()
        logp = masked_classification_log_probs(cls1, labels).reshape(-1, 2)
        label = labels.reshape(-1).long()
        nll = logp.reshape(-1)[:0].sum()
        for class_id, key in ((1,"positive"),(0,"negative")):
            selected = label.eq(class_id)
            count = denominators[key]
            if count:
                nll = nll + 0.5*F.nll_loss(logp[selected],label[selected],reduction="sum")/count
            elif bool(selected.any()):
                raise FatalInputError("global class denominator excludes observations")
        bce_count = denominators["bce"]
        if bce_count <= 0 or bce_count < cls2.numel():
            raise FatalInputError("invalid global BCE denominator")
        bce = masked_bce_with_logits(cls2,targets["labelcls2"])*cls2.numel()/bce_count
        weight = targets["weightxff"]
        local_weight, global_weight = float(weight.double().sum()), denominators["localization"]
        if global_weight < 0 or global_weight < local_weight:
            raise FatalInputError("invalid global localization denominator")
        iou = masked_iou_loss(masked_decode_regression(loc,weight),masked_decode_regression(targets["labelxff"],weight),weight)
        iou = iou*((local_weight+1e-6)/(global_weight+1e-6)) if global_weight else iou
        weighted_iou = 2.4*iou
        total = weighted_iou+nll+bce
        result = {key:float(value) for key,value in (("nll",nll),("bce",bce),("iou",iou),("weighted_iou",weighted_iou),("total",total))}
        if not all(math.isfinite(v) for v in result.values()):
            raise FatalInputError("nonfinite loss components")
        return result


class RuntimeGuard:
    def __init__(self,model,optimizer,*,source_check=None,resource_probe=None,max_reserved_fraction=.8,
                 max_grad_norm=10.,expected_device=None,max_host_rss_bytes=None,min_available_host_bytes=None,deadline_monotonic=None,
                 synthetic=False,expected_identity=None,seed=None,min_device_free_bytes=None,expected_gpu_uuid=None):
        self.model,self.optimizer = model,optimizer
        self.source_check,self.resource_probe = source_check,resource_probe
        self.max_reserved_fraction,self.max_grad_norm = max_reserved_fraction,max_grad_norm
        self.max_host_rss_bytes,self.deadline_monotonic = max_host_rss_bytes,deadline_monotonic
        self.min_available_host_bytes = min_available_host_bytes
        self.min_device_free_bytes,self.expected_gpu_uuid = min_device_free_bytes,expected_gpu_uuid
        self._resource_uuid = None
        self._trainability = {n:p.requires_grad for n,p in model.named_parameters()}
        if any(p.requires_grad for p in model.backbone.parameters()) or not any(p.requires_grad for p in model.grader.parameters()):
            raise GuardViolation("initial trainability must be frozen BB and active grader")
        self.synthetic,self.expected_identity,self.seed = synthetic,dict(expected_identity or {}),seed
        if not 0 < max_reserved_fraction <= 1 or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("invalid guard limits")
        self.expected_device = torch.device(expected_device) if expected_device is not None else next(model.parameters()).device
        self._parameter_ids = {n:id(p) for n,p in model.named_parameters()}
        self._modes = {n:m.training for n,m in model.named_modules()}
        self._buffer_specs = {n:(tuple(b.shape),b.dtype,b.device) for n,b in model.named_buffers()}
        if not model.training or not model.grader.training:
            raise GuardViolation("construct guard in source epoch0 train mode; DEV uses allow_evaluation")
        self._group_options = [{k:v for k,v in g.items() if k!="params"} for g in optimizer.param_groups]
        self.initial_snapshot = self.snapshot()
        self.check("initial")

    def snapshot(self):
        return {"model":state_digest(self.model.state_dict()),"backbone":state_digest(self.model.backbone.state_dict()),
                "grader":state_digest(self.model.grader.state_dict())}

    def _optimizer_scope(self):
        if not isinstance(self.optimizer,torch.optim.SGD) or len(self.optimizer.param_groups)!=2:
            raise GuardViolation("exact two-group source SGD required")
        for group, name, lr, initial in zip(self.optimizer.param_groups,("backbone","grader"),(.00025,.0025),(.0005,.005)):
            if (group.get("name")!=name or [id(p) for p in group["params"]]!=[id(p) for p in getattr(self.model,name).parameters()]
                    or group.get("lr")!=lr or group.get("initial_lr")!=initial or group.get("momentum")!=.9
                    or group.get("weight_decay")!=.0001 or group.get("dampening")!=0
                    or group.get("nesterov") or group.get("maximize") or group.get("differentiable")):
                raise GuardViolation("optimizer ordered scope/source epoch0 options changed")
        if [{k:v for k,v in g.items() if k!="params"} for g in self.optimizer.param_groups]!=self._group_options:
            raise GuardViolation("optimizer group options changed")
        allowed = set(self.model.grader.parameters())
        for parameter,state in self.optimizer.state.items():
            if parameter not in allowed or set(state)-{"momentum_buffer"}:
                raise GuardViolation("unauthorized optimizer state")
            for value in state.values():
                if (not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=parameter.dtype
                        or value.device!=parameter.device or not bool(torch.isfinite(value).all())):
                    raise GuardViolation("invalid momentum state")

    def check(self,stage,allow_evaluation=False):
        try:
            if self.source_check is not None and self.source_check() is False:
                raise GuardViolation("source identity check failed")
            if {n:id(p) for n,p in self.model.named_parameters()}!=self._parameter_ids:
                raise GuardViolation("model parameter identity changed")
            for name,param in self.model.named_parameters():
                if (param.device!=self.expected_device or param.dtype!=torch.float32 or not bool(torch.isfinite(param).all())
                        or param.requires_grad != self._trainability[name] or (param.requires_grad and not name.startswith("grader."))):
                    raise GuardViolation("parameter scope/device/dtype/finite violation:"+name)
                if name.startswith("backbone.") and param.grad is not None:
                    raise GuardViolation("frozen backbone acquired gradient:"+name)
            if {n:(tuple(b.shape),b.dtype,b.device) for n,b in self.model.named_buffers()}!=self._buffer_specs:
                raise GuardViolation("model buffer schema/device/dtype changed")
            for name,value in self.model.named_buffers():
                if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                    raise GuardViolation("nonfinite model buffer:"+name)
            expected = {n:False for n in self._modes} if allow_evaluation else self._modes
            if {n:m.training for n,m in self.model.named_modules()}!=expected:
                raise GuardViolation("model train/eval modes changed")
            if any(m.training for m in self.model.backbone.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)):
                raise GuardViolation("backbone BN must remain eval")
            if state_digest(self.model.backbone.state_dict())!=self.initial_snapshot["backbone"]:
                raise GuardViolation("frozen backbone/BN bytes changed")
            self._optimizer_scope()
            gradients = gradient_norms(self.model)
            if not gradients["finite"]:
                raise GuardViolation("nonfinite gradient/norm")
            if stage in ("pre_step","pre_sgd","post_step") and gradients["norm"] > self.max_grad_norm*(1+1e-5):
                raise GuardViolation("gradient exceeds clipped norm before/after SGD")
            resources = self.resource_probe() if self.resource_probe is not None else {}
            if not isinstance(resources,dict):
                raise GuardViolation("resource probe must return mapping")
            for key in ("reserved_bytes","total_bytes","allocated_bytes","host_rss_bytes","available_host_bytes"):
                if key in resources and (isinstance(resources[key],bool) or not isinstance(resources[key],(int,float))
                        or not math.isfinite(resources[key]) or resources[key]<0):
                    raise GuardViolation("malformed resource probe:"+key)
            if "reserved_bytes" in resources:
                total = resources.get("total_bytes",0)
                if total<=0 or resources["reserved_bytes"] > total*self.max_reserved_fraction:
                    raise GuardViolation("CUDA reserved-memory limit")
            if "allocated_bytes" in resources and ("reserved_bytes" not in resources or resources["allocated_bytes"]>resources["reserved_bytes"]):
                raise GuardViolation("allocated CUDA memory exceeds reserved")
            if self.max_host_rss_bytes is not None and ("host_rss_bytes" not in resources or resources["host_rss_bytes"]>self.max_host_rss_bytes):
                raise GuardViolation("host memory limit/missing RSS")
            if self.min_available_host_bytes is not None and ("available_host_bytes" not in resources or resources["available_host_bytes"]<self.min_available_host_bytes):
                raise GuardViolation("available host memory limit/missing probe")
            if self.min_device_free_bytes is not None or self.expected_gpu_uuid is not None:
                from .monitor import validate_resources
                validate_resources(resources,min_available_host_bytes=self.min_available_host_bytes,
                    max_host_rss_bytes=self.max_host_rss_bytes,max_reserved_fraction=self.max_reserved_fraction,
                    min_device_free_bytes=self.min_device_free_bytes,expected_gpu_uuid=self.expected_gpu_uuid,require_cuda=True)
            if "device" in resources and torch.device(resources["device"])!=self.expected_device:
                raise GuardViolation("resource device identity mismatch")
            if "device_uuid" in resources:
                if not isinstance(resources["device_uuid"],str) or not resources["device_uuid"]:
                    raise GuardViolation("invalid resource device UUID")
                if self._resource_uuid is None:
                    self._resource_uuid=resources["device_uuid"]
                elif self._resource_uuid!=resources["device_uuid"]:
                    raise GuardViolation("resource device UUID changed")
            elif self._resource_uuid is not None:
                raise GuardViolation("resource device UUID disappeared")
            if resources.get("wall_exhausted") or (self.deadline_monotonic is not None and time.monotonic()>=self.deadline_monotonic):
                raise GuardViolation("wall deadline exhausted")
            return {"stage":stage,"gradients":gradients,"resources":dict(resources),"backbone_digest":self.initial_snapshot["backbone"],
                    "lr":[g["lr"] for g in self.optimizer.param_groups],
                    "digests":self.snapshot() if stage in ("post_step","checkpoint","post_sgd") else None}
        except GuardViolation:
            raise
        except Exception as exc:
            raise GuardViolation(f"{stage} guard failed: {type(exc).__name__}: {exc}") from exc

    def classify_rejection(self,record):
        reason = record.get("reason")
        if self.synthetic and record.get("data_kind")=="synthetic" and record.get("reasons")==["synthetic_geometry_reject"]:
            reason = "synthetic_geometry_reject"
        if reason not in GEOMETRY_REJECTIONS and not (self.synthetic and reason=="synthetic_geometry_reject"):
            raise FatalInputError("fatal input, not allowed geometry rejection:"+str(reason))
        if reason=="no_available_target_source_support":
            values=[record.get(k) for k in ("template_available_source_support_px2","search_available_source_support_px2")]
            if any(type(v) not in (float,int) or not math.isfinite(v) for v in values) or min(values)>0:
                raise FatalInputError("geometry rejection lacks matching source-support evidence")
        if reason=="template_target_visible_in_negative_search":
            overlap=record.get("negative_template_A_overlap_px2")
            if (record.get("candidate_negative") is not True or type(overlap) not in (float,int)
                    or not math.isfinite(overlap) or overlap<=0 or record.get("negative_exclusion_proved") is not False):
                raise FatalInputError("geometry rejection lacks matching negative-overlap evidence")
        return AllowedGeometryRejection(record["candidate_index"],reason)

    def validate_batch(self,batch):
        try:
            ids,records = batch["sample_ids"],batch["attempt_records"]
            allowed_kinds=("synthetic",) if self.synthetic else ("authorized_future_real","real_development_forward_only")
            if batch.get("data_kind") not in allowed_kinds:
                raise FatalInputError("batch data-kind scope mismatch")
            if (not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids)
                    or ids!=list(range(ids[0],ids[0]+len(ids))) or [r["candidate_index"] for r in records]!=ids
                    or batch["replacement_count"]!=0):
                raise FatalInputError("invalid attempt order/ledger/replacement")
            accepted=[r for r in records if r.get("status")=="accepted"]
            rejected=[r for r in records if r.get("status")=="rejected"]
            if len(accepted)+len(rejected)!=len(records) or accepted!=batch["accepted_records"] or rejected!=batch["rejected_records"]:
                raise FatalInputError("attempt status/receipt disagreement")
            for record in records:
                if type(record.get("candidate_negative")) is not bool:
                    raise FatalInputError("missing/bad candidate class")
                if self.seed is not None and record.get("seed")!=self.seed+record["candidate_index"]:
                    raise FatalInputError("candidate seed identity changed")
                for key,value in self.expected_identity.items():
                    if key.endswith("sha256") and (key in record or not self.synthetic) and record.get(key)!=value:
                        raise FatalInputError("candidate identity/hash changed:"+key)
            allowed=[self.classify_rejection(r) for r in rejected]
            n=len(accepted)
            shapes={"template":(n,3,127,127),"search":(n,3,287,287)}
            targets=batch["targets"]
            shapes.update({"label_cls1":(n,1,11,11),"labelcls2":(n,1,11,11),"labelxff":(n,4,11,11),"weightxff":(n,1,11,11)})
            for name,shape in shapes.items():
                value=batch[name] if name in ("template","search") else targets[name]
                if (not isinstance(value,torch.Tensor) or value.layout!=torch.strided or value.device.type!="cpu"
                        or value.dtype!=torch.float32 or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all())):
                    raise FatalInputError("malformed shape/dtype/device/finite:"+name)
                if name in ("template","search") and not value.is_contiguous():
                    raise FatalInputError("model images must pass CPU contiguous bridge")
            a,c,w=targets["label_cls1"],targets["labelcls2"],targets["weightxff"]
            if not bool(((a==-2)|(a==-1)|(a==0)|(a==1)).all()) or bool(((c<0)|(c>1)).any()) or bool((w<0).any()):
                raise FatalInputError("invalid mask domains")
            for i,record in enumerate(accepted):
                if record["candidate_negative"]:
                    if any(bool(targets[k][i].ne(0).any()) for k in targets):
                        raise FatalInputError("negative dense-zero-mask violation")
                elif not (bool(a[i].eq(1).any()) and bool(c[i].gt(0).any()) and bool(w[i].gt(0).any())):
                    raise FatalInputError("empty positive support")
                if not self.synthetic:
                    ti,si=record.get("template_identity"),record.get("search_identity")
                    if (not isinstance(ti,list) or not isinstance(si,list) or len(ti)!=3 or len(si)!=3
                            or ti[:2]!=si[:2] or ti[0]!="MOT17" or ti[1]!=record.get("video_lineage")):
                        raise FatalInputError("source fragment identity mismatch")
                    gap=record.get("native_gap_abs")
                    if record["candidate_negative"]:
                        if (ti==si or record.get("template_frame_id")!=record.get("search_frame_id") or gap!=0
                                or record.get("negative_exclusion_proved") is not True or record.get("negative_template_A_overlap_px2")!=0):
                            raise FatalInputError("unsafe negative identity/temporal/exclusion proof")
                    elif ti!=si or type(gap) is not int or not 1<=gap<=19:
                        raise FatalInputError("positive identity/temporal gap violation")
                    from hift_research.data_contract.pipeline import array_digest
                    for name in ("template","search","labelxff","label_cls1"):
                        value=batch[name][i] if name in ("template","search") else targets[name][i]
                        raw=value.numpy()
                        # CLS1's validated integer domain is exactly representable both ways.
                        # Do NOT roundtrip fractional FP64 CLS2/weight targets: that is lossy.
                        if name=="label_cls1":
                            raw=raw.astype("float64")
                        if array_digest(raw)!=record.get("array_hashes",{}).get(name):
                            raise FatalInputError("array pixel/target hash changed:"+name)
            return {"attempts":len(ids),"accepted":n,"rejected":len(rejected),
                    "positive":sum(not r["candidate_negative"] for r in accepted),"negative":sum(r["candidate_negative"] for r in accepted),
                    "allowed_rejections":[{"candidate_index":r.candidate_index,"reason":r.reason} for r in allowed]}
        except FatalInputError:
            raise
        except Exception as exc:
            raise FatalInputError(f"malformed batch: {type(exc).__name__}: {exc}") from exc
