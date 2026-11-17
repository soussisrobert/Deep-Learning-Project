"""Compile measured real-input and synthetic-backward costs; never optimize a model."""
import argparse
from pathlib import Path
from hift_research.evaluation.assets import read_json,sha256_file,write_json_exclusive
from hift_research.wave4_feasibility import project_budget


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real',type=Path,required=True);parser.add_argument('--synthetic',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    real=read_json(args.real);synthetic=read_json(args.synthetic)
    if real['result']!='passed_forward_only_no_SGD' or not all(synthetic['comparisons'].values()):raise ValueError('failed prerequisite')
    if real['real_data_optimizer_steps'] or real['real_data_backward_calls']:raise ValueError('no real optimization permitted')
    warm=synthetic['uninterrupted_measurements'][1]['timings']
    cold=synthetic['uninterrupted_measurements'][0]['timings']
    syn_fwd_ms=warm['forward_loss_s']*1000/4
    backward_ms=(warm['backward_s']+warm['gradient_audit_clip_s']+warm['optimizer_step_s'])*1000/4
    transfers=[row['collation_ms']+row.get('transfer_ms',0) for row in real['rows']]
    initialization=real['pipeline_initialization_seconds']+real['model_initialization_seconds']+max(0,cold['window_total_s']-warm['window_total_s'])
    projection=project_budget(candidate_ms=real['candidate_ms'],collate_transfer_ms=transfers,
        forward_ms=[max(value,syn_fwd_ms) for value in real['forward_loss_ms']],
        backward_clip_step_ms=[backward_ms],initialization_seconds=initialization,
        checkpoint_seconds=synthetic['checkpoint_serialization_seconds'],
        development_full_check_seconds=real['development_initial_forward_only']['full_input_forward_seconds'])
    receipt={**projection,'prerequisite_references':{
        'real_pipeline':{'path':str(args.real.resolve()),'sha256':sha256_file(args.real)},
        'synthetic_production_layout_replay':{'path':str(args.synthetic.resolve()),'sha256':sha256_file(args.synthetic)}},
        'backward_cost_selection':'Second uninterrupted effective update only: first separately counts cold-start overhead. n=1warmwindow is a limited engineering sample, not a throughput confidence interval.',
        'forward_cost_policy':'max(each actual real no-grad forward/loss batch, warm random-synthetic train/autograd forward/loss per micro); no speedup assumed',
        'synthetic_forward_ms_per_micro':syn_fwd_ms,'synthetic_backward_audit_clip_step_ms_per_micro':backward_ms,
        'real_profile_counts':real['counts'],'real_profile_attempts':32,'real_development_pairs':128,
        'real_input_classes':'first32 happened28positive/4negative; not selected to force populationratio',
        'training_plan':{'physical_attempt_slots':4,'effective_attempt_slots':16,'accumulation_max':4,
            'max_effective_updates':100,'max_attempts':1600,'num_workers':0,'prefetch':False,
            'source_epoch':0,'head_only':True,'grader_lr':.0025,'momentum':.9,'weight_decay':.0001,'clip_norm':10},
        'GPU_memory':{'real_forward_peak_allocated_bytes':real['peak_cuda_allocated_bytes'],
            'real_forward_peak_reserved_bytes':real['peak_cuda_reserved_bytes'],
            'synthetic_controller_measured_max_allocated_bytes':max(row['memory']['peak_allocated'] for row in synthetic['uninterrupted_measurements']),
            'synthetic_controller_measured_max_reserved_bytes':max(row['memory']['peak_reserved'] for row in synthetic['uninterrupted_measurements']),
            'limitation':'Torch allocator only, not full driver/context VRAM; comparison process may contain additional models; two updates not long-run stability proof'},
        'host_memory_real_profile':real['host_process_memory'],
        'no_training_launched':True,'GO_does_not_authorize_real_SGD':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json_exclusive(args.output,receipt)
    print({key:receipt[key] for key in ('central_seconds','engineering_range_seconds','suggested_hard_wall_seconds','mean_training_seconds','mean_development_seconds')})
