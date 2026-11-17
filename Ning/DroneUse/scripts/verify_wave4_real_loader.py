"""CPU-only actual temporal Dataset/Windows-spawn verification; never model/SGD/CUDA."""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from hift_research.evaluation.assets import write_json_exclusive
from hift_research.provenance import sha256,source_identity
from hift_research.training.recipe import ROOT,canonical_hash
from hift_research.training.wave4_loader import indexed_loader,ConsumedCursor,dataset_identity
from hift_research.temporal_contract.torch_adapter import TemporalCandidateDataset
from hift_research.temporal_contract.pipeline import TemporalPairPipeline


def batch_digest(batch):
    tensors=batch['model_inputs']
    return {'indices':batch['attempt_indices'],'records_sha256':canonical_hash(batch['attempt_records']),
            'accepted':len(batch['accepted_records']),'rejected':len(batch['rejected_records']),
            'tensor_sha256':None if tensors is None else {key:hashlib.sha256(value.numpy().tobytes()).hexdigest() for key,value in tensors.items()}}


def consume(dataset,workers,*,seed=1,start=0,count=16):
    loader=indexed_loader(dataset,num_workers=workers,seed=seed,start=start)
    iterator=iter(loader)
    try:return [batch_digest(row) for row in itertools.islice(iterator,count//4)]
    finally:
        if workers:iterator._shutdown_workers()
        del iterator,loader;gc.collect()


def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    def dataset():return TemporalCandidateDataset(args.manifest,args.policy)
    start={'schema':'hift-wave4-authentic-production-loader-v1','started_utc':datetime.now(timezone.utc).isoformat(),
        'source':source_identity(ROOT),'manifest_sha256':sha256(args.manifest),'policy_sha256':sha256(args.policy),
        'device':'cpu_only_no_model_or_SGD','tested_indices':list(range(16)),
        'worker_modes':[0,2],'async_exact_resume_claimed':False}
    write_json_exclusive(output/'manifest.start.json',start)
    original_rng=torch.get_rng_state().clone()
    serial=consume(dataset(),0,seed=17)
    spawn1=consume(dataset(),2,seed=17)
    spawn2=consume(dataset(),2,seed=932)
    flags={'serial_equals_spawn2':serial==spawn1,'different_worker_base_seeds_same_indexed_output':spawn1==spawn2,
           'dedicated_loader_generator_preserves_main_torch_rng':torch.equal(original_rng,torch.get_rng_state()),
           'ordered_complete_0_to15':[i for row in serial for i in row['indices']]==list(range(16))}
    first=dataset();identity=dataset_identity(first)
    cursor=ConsumedCursor(identity)
    batches=iter(indexed_loader(first,num_workers=0,seed=17))
    cursor.commit(next(batches));cursor.commit(next(batches))
    del batches
    saved=cursor.state_dict();restored=ConsumedCursor.from_state(saved,identity)
    flags['serial_consumed_cursor_resume_equals_full']=consume(dataset(),0,seed=932,start=restored.next_index,count=8)==serial[2:]
    faults=[]
    # Inject failures below the actual Dataset wrapper, WITHOUT altering frozen files.
    for kind in ('missing','decode','exception'):
        value=dataset();value._pipeline=TemporalPairPipeline(args.manifest,args.policy);value._pid=os.getpid()
        if kind=='missing':
            original=Path.read_bytes
            def missing(path):
                if path.suffix.lower()=='.jpg':raise FileNotFoundError('injected actual media read failure')
                return original(path)
            injection=patch('hift_research.temporal_contract.pipeline.Path.read_bytes',missing)
        elif kind=='decode':injection=patch('hift_research.temporal_contract.pipeline.cv2.imdecode',return_value=None)
        else:injection=patch('hift_research.temporal_contract.pipeline.cv2.imdecode',side_effect=RuntimeError('injected actual decoder exception'))
        error=None
        with injection:
            try:next(iter(indexed_loader(value,num_workers=0,errors='raise')))
            except RuntimeError as exc:error={'type':type(exc).__name__,'message':str(exc),'cause':str(exc.__cause__)}
        recovered=consume(dataset(),0,seed=17,count=4)==serial[:1]
        faults.append({'kind':kind,'exception_propagated':error is not None,'error':error,'clean_reconstruction_same_first_batch':recovered})
    flags['actual_media_faults_stop_and_recovery_replays_same_index']=all(row['exception_propagated'] and row['clean_reconstruction_same_first_batch'] for row in faults)
    receipt={**start,'completed_utc':datetime.now(timezone.utc).isoformat(),'comparisons':flags,'serial':serial,
        'spawn_seed17':spawn1,'spawn_seed932':spawn2,'saved_consumed_cursor':saved,'faults':faults,
        'result':'passed' if all(flags.values()) else 'failed','recommended_mode':'num_workers0/no-prefetch; atomic effective-update boundary checkpoints',
        'scope':'Actual authenticated image decoding/crop/augmentation/targets/collation; only16indices for transport checks; no optimization or full asynchronous replay claim'}
    write_json_exclusive(output/'manifest.json',receipt)
    print(json.dumps({'result':receipt['result'],'comparisons':flags,'faults':faults},indent=2))
    if not all(flags.values()):raise ValueError('actual temporal loader verification failed; receipt retained')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
