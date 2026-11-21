"""Authoritative Wave6 entry. Legacy run_b2_pilot.py is NOT the production pilot path.

This small stdlib bootstrap is the reviewed trust anchor, not an administrator-
resistant sandbox. Declared source bytes are checked before mutable project code
is imported. No boolean/CLI flag substitutes for an exact human acknowledgment.
"""
import sys
import time
BOOT_START=time.perf_counter()
# sys/time are built-ins. Remove script/cwd/PYTHONPATH shadowing BEFORE importing
# any stdlib source module. Operator-installed Python/site startup hooks remain
# an explicitly trusted pre-script boundary, not an administrator-proof claim.
if __name__=='__main__':
    if sys.flags.optimize!=0 or sys.orig_argv[1:]!=sys.argv:
        raise PermissionError('exact direct interpreter invocation required; flags/optimization are not approved')
    _base=sys.base_prefix.replace('\\','/')
    _venv=sys.prefix.replace('\\','/')
    sys.path=[_base+'/python'+str(sys.version_info.major)+str(sys.version_info.minor)+'.zip',
        _base+'/DLLs',_base+'/Lib',_venv+'/Lib/site-packages']
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from datetime import datetime,timezone

# This pre-ACK watchdog grants no authority. It bounds even bootstrap hashing and
# is only tightened, never reset, when an untrusted proposal declares a smaller cap.
_BOOT_TIMERS=[]
def arm_boot_deadline(deadline):
    if __name__!='__main__':return  # importing this bootstrap never arms a process-exit timer
    timer=threading.Timer(max(0,deadline-time.perf_counter()),lambda:os._exit(124))
    timer.daemon=True;timer.start();_BOOT_TIMERS.append(timer)
arm_boot_deadline(BOOT_START+900)
ROOT=Path(__file__).resolve().parents[1]


def install_verified_project_loader(sources,*,force=False):
    """Compile the SAME approved bytes read once; never trust project .pyc.

    Dependency/interpreter startup remains the explicitly trusted environment.
    Its normal bytecode cache is not needlessly disabled for Torch startup.
    """
    if __name__!='__main__' and not force:return
    import importlib.machinery
    loader_class=importlib.machinery.SourceFileLoader;original=loader_class.get_code
    roots=tuple((ROOT/p).resolve() for p in ('src','scripts','third_party/HiFT'))
    expected={(Path(p) if Path(p).is_absolute() else ROOT/p).resolve():sha for p,sha in sources.items()}
    fingerprint=hashlib.sha256(json.dumps(sources,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    def approved_code(loader,fullname):
        path=Path(loader.path).resolve()
        if any(root in path.parents for root in roots):
            raw=loader.get_data(loader.path)
            if path.suffix.lower()!='.py' or hashlib.sha256(raw).hexdigest()!=expected.get(path):
                raise PermissionError('loaded project source bytes differ from exact approved inventory:'+str(path))
            return loader.source_to_code(raw,loader.path)
        return original(loader,fullname)
    approved_code._hift_wave6_source_map_sha256=fingerprint
    loader_class.get_code=approved_code;sys.dont_write_bytecode=True
    return original


def bootstrap_source_closure(sources):
    roots=[ROOT/'src',ROOT/'scripts',ROOT/'third_party/HiFT']
    expected={Path(name).resolve() if Path(name).is_absolute() else (ROOT/name).resolve() for name in sources}
    expected={p for p in expected if any(root.resolve() in p.parents for root in roots) and p.suffix.lower()=='.py'}
    actual=set()
    for root in roots:
        if root.is_symlink() or root.is_junction():raise PermissionError('project source root aliases are forbidden')
        for path in root.rglob('*'):
            if '__pycache__' in path.parts:continue  # never read under the fresh cache policy
            if path.is_symlink() or path.is_junction():raise PermissionError('project source aliases are forbidden')
            if path.is_file():
                if path.suffix.lower() in ('.pyc','.pyo','.pyd','.dll','.so'):
                    raise PermissionError('unbound project native/sourceless executable:'+str(path))
                if path.suffix.lower()=='.py':actual.add(path.resolve())
    if actual!=expected:raise PermissionError('project executable source set differs from exact inventory')


def bootstrap_sources(proposal_path):
    proposal=json.loads(Path(proposal_path).read_text(encoding='utf-8'))
    cap=proposal.get('limits',{}).get('max_wall_seconds')
    if type(cap) is not int or not 0<cap<=900:raise PermissionError('bounded bootstrap duration required')
    expiry=datetime.fromisoformat(proposal['expires_utc'].replace('Z','+00:00'))
    if expiry.tzinfo is None:raise PermissionError('explicit bootstrap expiry timezone required')
    arm_boot_deadline(min(BOOT_START+cap,time.perf_counter()+max(0,(expiry-datetime.now(timezone.utc)).total_seconds())))
    sources=proposal.get('source_files_sha256')
    if not isinstance(sources,dict) or not sources:raise PermissionError('exact source inventory absent')
    bootstrap_source_closure(sources)
    for name,expected in sources.items():
        path=Path(name)
        if not path.is_absolute():path=ROOT/path
        value=hashlib.sha256(path.read_bytes()).hexdigest()
        if value!=expected:raise PermissionError('source hash mismatch before project imports:'+str(path))
    return proposal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal');parser.add_argument('--authorization');parser.add_argument('--output')
    parser.add_argument('--synthetic-test',action='store_true',help='temporary explicitly synthetic envelopes ONLY, never authentic images')
    parser.add_argument('--_worker',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args._worker:
        if not os.environ.get('HIFT_W6_CAPABILITY'):raise PermissionError('private worker requires supervised parent capability')
        dispatch=json.loads(Path(args._worker).read_text(encoding='utf-8'));proposal=bootstrap_sources(dispatch['proposal_path'])
        install_verified_project_loader(proposal['source_files_sha256'])
        sys.path.insert(0,str(ROOT/'src'))
        from hift_research.wave6.worker import execute
        return execute(args._worker)
    if not all((args.proposal,args.authorization,args.output)):raise PermissionError('exact proposal, human acknowledgment file and bound output required')
    if not Path(args.authorization).is_file():raise PermissionError('human acknowledgment missing; no worker/optimizer constructed')
    proposal=bootstrap_sources(args.proposal)
    install_verified_project_loader(proposal['source_files_sha256'])
    sys.path.insert(0,str(ROOT/'src'))
    from hift_research.wave6.entry import run_guarded
    result=run_guarded(args.proposal,args.authorization,args.output,
        actual_argv=[str(Path(sys.executable).resolve()),*sys.argv],
        starting_monotonic=BOOT_START,allow_synthetic=args.synthetic_test)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    return 0 if (result.get('status')=='COMPLETED' and result.get('child_exit_code')==0
        and result.get('process_tree_empty_verified') and result.get('job_closed')
        and not result.get('ledger_pending_entry_finalization',True)
        and not result.get('ledger_finalization',{}).get('timing',{}).get('budget_breached',True)
        and not result.get('final_timing',{}).get('budget_breached',True)) else 1

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({'guarded_launch':'rejected_or_failed','error':type(exc).__name__+':'+str(exc),'no_automatic_retry':True}),file=sys.stderr)
        raise SystemExit(1)
