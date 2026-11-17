"""Nonsemantic timing collection and conservative full-input pilot projection."""
from collections import defaultdict
import math
import numpy as np


def latency(values):
    data=np.asarray(values,dtype=np.float64)
    if data.ndim!=1 or not len(data) or not np.isfinite(data).all() or (data<0).any():
        raise ValueError('nonempty finite nonnegative timing observations required')
    return {'count':len(data),'total_ms':float(data.sum()),'mean_ms':float(data.mean()),
            'median_ms':float(np.median(data)),'p95_ms':float(np.quantile(data,.95)),
            'min_ms':float(data.min()),'max_ms':float(data.max())}


class StageObserver:
    """Accept externally measured seconds; never draw RNG or modify candidate records."""
    def __init__(self):self.rows=[]

    def record(self,stage_name,elapsed_seconds,*,cache_hit=False):
        if not isinstance(stage_name,str) or not stage_name or not math.isfinite(elapsed_seconds) or elapsed_seconds<0:
            raise ValueError('invalid timing stage')
        self.rows.append({'stage':stage_name,'ms':float(elapsed_seconds)*1000,'cache_hit':bool(cache_hit)})

    def summary(self):
        groups=defaultdict(list)
        for row in self.rows:groups[row['stage']].append(row)
        return {name:{**latency([v['ms'] for v in rows]),
                      'cache_hits':sum(v['cache_hit'] for v in rows),
                      'cache_misses':sum(not v['cache_hit'] for v in rows)}
                for name,rows in sorted(groups.items())}


def project_budget(*,candidate_ms,collate_transfer_ms,forward_ms,backward_clip_step_ms,
                   initialization_seconds,checkpoint_seconds,attempts=1600,micro_slots=4,
                   development_attempts=128,development_checks=4,development_full_check_seconds=None):
    """Measured component sum, deliberately not resident-only synthetic throughput.

    The upper bound uses p95 observations, twice the component sum and a30second
    fixed I/O/validation contingency. This is a bounded engineering estimate,
    not a statistical wall-time confidence interval or convergence prediction.
    """
    timings=(candidate_ms,collate_transfer_ms,forward_ms,backward_clip_step_ms)
    stats=[latency(values) for values in timings]
    if min(attempts,micro_slots,development_attempts,development_checks)<=0:
        raise ValueError('positive declared work counts required')
    if (not math.isfinite(initialization_seconds) or initialization_seconds<0 or
            not math.isfinite(checkpoint_seconds) or checkpoint_seconds<0):
        raise ValueError('invalid measured initialization/checkpoint cost')
    train_micro=math.ceil(attempts/micro_slots)
    dev_micro=math.ceil(development_attempts/micro_slots)*development_checks
    def cost(field):
        data,transfer,forward,backward=[row[field] for row in stats]
        train=(attempts*data+train_micro*(transfer+forward+backward))/1000
        # Conservative: reuse representative TRAIN candidate cost for unaugmented DEV.
        dev=(development_attempts*development_checks*data+dev_micro*(transfer+forward))/1000
        return train,dev
    central_train,central_dev=cost('mean_ms')
    high_train,high_dev=cost('p95_ms')
    if development_full_check_seconds is not None:
        if not math.isfinite(development_full_check_seconds) or development_full_check_seconds<0:
            raise ValueError('invalid measured complete development cost')
        central_dev=development_full_check_seconds*development_checks
        high_dev=max(high_dev,central_dev*1.5)
    fixed=initialization_seconds+checkpoint_seconds*development_checks
    central=central_train+central_dev+fixed
    lower=max(1.,central*.75)
    upper=max(lower,2*(high_train+high_dev+fixed)+30.)
    return {'kind':'measured_component_engineering_projection_NOT_training_or_confidence_interval',
            'train_attempts':attempts,'train_microbatches_max':train_micro,'development_microbatches_max':dev_micro,
            'component_observations':dict(zip(('input_per_attempt','collate_transfer_per_micro',
                'real_forward_per_micro','synthetic_backward_clip_step_per_micro'),stats)),
            'mean_training_seconds':central_train,'mean_development_seconds':central_dev,
            'measured_complete_development_check_seconds':development_full_check_seconds,
            'measured_fixed_overhead_seconds':fixed,'central_seconds':central,
            'engineering_range_seconds':[lower,upper],
            'suggested_hard_wall_seconds':int(math.ceil(max(300.,upper*1.5)/60)*60),
            'assumptions':['serial/no-prefetch; no overlap speedup assumed','up to4accepted pairs per4attempt slots',
                'no hidden replacements; all-reject window skips rather than completing100updates',
                'development uses conservative representative training input costs',
                'short samples/cache locality and CPU/storage contention limit extrapolation',
                'backward estimate uses random synthetic data; no real-image gradient or SGD measured']}
