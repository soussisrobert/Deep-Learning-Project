"""Actual Windows CPU-only process trees; no data, CUDA, model or SGD."""
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from hift_research.wave6.ledger import reserve_attempt,read_ledger,AttemptConsumedError,atomic_json
from hift_research.wave6.supervisor import supervise
from hift_research.wave6.windows_job import SuspendedJobProcess,is_process_running

ROOT=Path(__file__).resolve().parents[1]
WINDOWS=pytest.mark.skipif(os.name!="nt",reason="real Windows Job Objects required")


def expiry(seconds=30):return (datetime.now(timezone.utc)+timedelta(seconds=seconds)).isoformat()


def reserve(tmp_path,seconds=2.,margin=.6,start=None,name="attempt",expires=None,watchdog=True):
    return reserve_attempt(tmp_path/(name+".time.jsonl"),experiment_id=name,authorization_id="synthetic-ack-digest",
                           max_seconds=seconds,cleanup_margin_seconds=margin,starting_monotonic=start,expiry_utc=expires or expiry(),reservation_watchdog=watchdog)


def wait_for(predicate,seconds=2):
    end=time.perf_counter()+seconds
    while time.perf_counter()<end:
        if predicate():return True
        time.sleep(.01)
    return bool(predicate())


def run(tmp_path,script,ledger=None,**kwargs):
    ledger=ledger or reserve(tmp_path)
    kwargs.setdefault("physical_executable",sys._base_executable)
    environment=dict(kwargs.pop("env",{}) or {});environment.setdefault("__PYVENV_LAUNCHER__",sys.executable)
    kwargs["env"]=environment
    return supervise([sys.executable,"-c",script],cwd=ROOT,output=tmp_path/"output",experiment_id=ledger.experiment_id,
                     authorization_id=ledger.authorization_id,ledger=ledger,**kwargs)


def test_reservation_before_preflight_and_all_failure_work_is_consumed(tmp_path):
    start=time.perf_counter()-.35
    ledger=reserve(tmp_path,start=start)
    assert ledger.elapsed()>=.35 and ledger.remaining()<1.7
    ledger.record("preflight_failed",reason="synthetic bound-file failure")
    finish=ledger.finish("FAIL_PREFLIGHT")
    assert finish["timing"]["elapsed_seconds"]>=.35
    assert read_ledger(ledger.path)[-1]["event"]=="entry_finished"
    with pytest.raises(AttemptConsumedError):reserve(tmp_path)
    with pytest.raises(AttemptConsumedError):ledger.claim_supervision()


def test_partial_crashed_reservation_never_restarts(tmp_path):
    path=tmp_path/"broken.time.jsonl";path.write_bytes(b'{"partial":')
    with pytest.raises(AttemptConsumedError):reserve_attempt(path,experiment_id="x",authorization_id="ack",max_seconds=2,cleanup_margin_seconds=.5,expiry_utc=expiry())
    with pytest.raises(ValueError):read_ledger(path)


def test_append_hash_chain_detects_mutation_and_no_receipt_overwrite(tmp_path):
    ledger=reserve(tmp_path);ledger.record("preflight_start")
    rows=read_ledger(ledger.path);assert len(rows)==2
    payload=ledger.path.read_bytes();ledger.path.write_bytes(payload.replace(b'preflight_start',b'preflight_other'))
    with pytest.raises(ValueError):read_ledger(ledger.path)
    with pytest.raises(ValueError):ledger.record("cannot_append")
    output=tmp_path/"receipt.json";atomic_json(output,{"first":1})
    with pytest.raises(AttemptConsumedError):atomic_json(output,{"first":2})
    assert json.loads(output.read_text())=={"first":1}


def test_expiry_and_original_start_are_not_extendable(tmp_path):
    ledger=reserve(tmp_path,expires=expiry(-1),watchdog=False)  # arithmetic-only expired synthetic probe
    assert ledger.remaining()==0
    assert ledger.deadline<=time.perf_counter()
    with pytest.raises(ValueError):reserve_attempt(tmp_path/"bad",experiment_id="x",authorization_id="ack",max_seconds=901,cleanup_margin_seconds=5,expiry_utc=expiry())
    with pytest.raises(ValueError):reserve_attempt(tmp_path/"future",experiment_id="x",authorization_id="ack",max_seconds=2,cleanup_margin_seconds=.5,starting_monotonic=time.perf_counter()+10,expiry_utc=expiry())
    with pytest.raises(ValueError):reserve_attempt(tmp_path/"naive",experiment_id="x",authorization_id="ack",max_seconds=2,cleanup_margin_seconds=.5,expiry_utc="2030-01-01T00:00:00")


@WINDOWS
def test_native_child_is_suspended_until_assigned_and_explicitly_resumed(tmp_path):
    marker=tmp_path/"started"
    job=SuspendedJobProcess([sys.executable,"-c",f"from pathlib import Path; Path({str(marker)!r}).write_text('started')"],ROOT,
                            stdout_path=tmp_path/"out.log",stderr_path=tmp_path/"err.log")
    try:
        assert job.assigned and not job.resumed
        assert job.trace[0]["atomic_job_assignment"] and job.trace[1]["atomic_at_process_creation"]
        time.sleep(.08)
        assert not marker.exists() and job.poll() is None
        assert [row["event"] for row in job.trace]==["created_suspended","assigned_to_job"]
        job.resume();assert wait_for(marker.exists)
        assert job.wait_empty(1.)
        assert job.poll()==0 and job.trace[2]["previous_suspend_count"]==1
    finally:job.close()


@WINDOWS
def test_deadline_kills_real_child_grandchild_tree_and_reads_last_commit(tmp_path):
    child_pid=tmp_path/"child.pid";grand_pid=tmp_path/"grand.pid";journal=tmp_path/"output"/"telemetry.jsonl"
    grand=f"import os,time; from pathlib import Path; Path({str(grand_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    child=f"import os,subprocess,sys,time; from pathlib import Path; Path({str(child_pid)!r}).write_text(str(os.getpid())); subprocess.Popen([sys._base_executable,'-c',{grand!r}]); time.sleep(60)"
    script=("import subprocess,sys,time; from hift_research.wave6.telemetry import DurableJournal; "
            f"j=DurableJournal({str(journal)!r}); "
            "j.emit('update_commit',update=5,cursor=80); j.emit('checkpoint',path='synthetic-only.pt',sha256='not-revalidated',update=5,cursor=80); "
            "j.emit('window_begin',attempt_ids=list(range(80,96)),lastupdate=5); "
            "j.emit('attempt_window',phase='training',attempt_ids=[80,81,82,83],counts={'accepted':4}); "
            "j.emit('optimizer_step_begin',update=6,cursor=96); "
            f"subprocess.Popen([sys._base_executable,'-c',{child!r}]); time.sleep(60)")
    ledger=reserve(tmp_path,seconds=2.5,margin=.8)
    receipt=run(tmp_path,script,ledger)
    assert receipt["status"]=="STOP_DEADLINE" and not receipt["timing"]["budget_breached"]
    assert receipt["process_tree_empty_verified"] and receipt["forced_tree_termination"]
    assert receipt["peak_active_processes"]>=3
    assert receipt["last_committed"]["update"]==5 and receipt["last_committed"]["cursor"]==80
    assert receipt["failed_work"]["optimizer_mutation_unpublished_possible"]
    assert receipt["failed_work"]["last_optimizer_step_begin"]["update"]==6
    assert receipt["consumed_attempts"]["observed_by_phase"]["training"]["count"]==4
    assert receipt["last_checkpoint"]["update"]==5 and not receipt["checkpoint_bytes_revalidated"]
    for path in (child_pid,grand_pid):assert not is_process_running(int(path.read_text()))
    finished=ledger.finish(receipt["status"],outer_lease_cleanup_complete=True)
    assert not finished["timing"]["budget_breached"]
    with pytest.raises(AttemptConsumedError):run(tmp_path/"retry",script,ledger)


@WINDOWS
@pytest.mark.parametrize("phase",["initialization","development","checkpoint","cleanup"])
def test_hung_phase_never_receives_a_fresh_budget(tmp_path,phase):
    journal=tmp_path/"output"/"telemetry.jsonl"
    script=f"import time; from hift_research.wave6.telemetry import DurableJournal; j=DurableJournal({str(journal)!r}); j.emit('phase_begin',phase={phase!r}); time.sleep(60)"
    ledger=reserve(tmp_path,seconds=1.3,margin=.5)
    receipt=run(tmp_path,script,ledger)
    assert receipt["status"]=="STOP_DEADLINE"
    assert receipt["telemetry"]["last_events"]["phase_begin"]["phase"]==phase
    assert receipt["process_tree_empty_verified"] and receipt["timing"]["elapsed_seconds"]<1.3
    ledger.finish(receipt["status"])


@WINDOWS
def test_preflight_spent_budget_does_not_schedule_child(tmp_path):
    ledger=reserve(tmp_path,seconds=2.,margin=.6,start=time.perf_counter()-1.5)
    marker=tmp_path/"must_not_execute"
    receipt=run(tmp_path,f"from pathlib import Path; Path({str(marker)!r}).write_text('bad')",ledger)
    assert receipt["status"]=="STOP_DEADLINE_PREFLIGHT" and not receipt["child_resumed"]
    assert receipt["child_pid"] is None and not marker.exists()
    assert receipt["timing"]["elapsed_seconds"]>=1.5
    ledger.finish(receipt["status"])


@WINDOWS
def test_child_exit_with_live_descendant_is_not_success(tmp_path):
    marker=tmp_path/"desc.pid"
    desc=f"import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    script=f"import subprocess,sys,time\nfrom pathlib import Path\nsubprocess.Popen([sys._base_executable,'-c',{desc!r}])\nwhile not Path({str(marker)!r}).exists():time.sleep(.01)\n"
    receipt=run(tmp_path,script)
    assert receipt["status"]=="FAIL_CHILD_EXIT" and receipt["process_tree_empty_verified"]
    assert receipt["reason"]=="child_exited_with_live_descendants"
    assert not is_process_running(int(marker.read_text()))


@WINDOWS
def test_cooperative_stop_is_inside_same_hard_budget(tmp_path):
    # Avoid shell/heredoc: this is an actual Python child whose stop loop polls
    # only the supervisor's fixed stop file.
    script="import os,time\nfrom pathlib import Path\np=Path(os.environ['HIFT_W6_STOP_FILE'])\nwhile not p.exists():time.sleep(.01)\n"
    ledger=reserve(tmp_path,seconds=1.5,margin=.6)
    receipt=run(tmp_path,script,ledger)
    assert receipt["status"]=="STOP_DEADLINE" and receipt["cooperative_stop_requested"]
    assert not receipt["forced_tree_termination"] and receipt["process_tree_empty_verified"]
    ledger.finish(receipt["status"])


@WINDOWS
def test_nonzero_exit_incomplete_tail_never_invents_optimizer_commit(tmp_path):
    journal=tmp_path/"output"/"telemetry.jsonl"
    script=("from hift_research.wave6.telemetry import DurableJournal; "
            f"j=DurableJournal({str(journal)!r}); j.emit('update_commit',update=2,cursor=32); j.emit('optimizer_step_begin',update=3); "
            f"f=open({str(journal)!r},'ab'); f.write(b'{{partial'); f.flush(); f.close(); raise SystemExit(7)")
    receipt=run(tmp_path,script)
    assert receipt["status"]=="FAIL_CHILD_EXIT" and receipt["child_exit_code"]==7
    assert receipt["last_committed"]["update"]==2
    assert receipt["telemetry"]["incomplete_tail"] and receipt["failed_work"]["optimizer_mutation_unpublished_possible"]


@WINDOWS
def test_parent_receipt_overhead_overshoot_is_reported_fail(tmp_path,monkeypatch):
    import hift_research.wave6.supervisor as module
    original=module._telemetry_summary
    def delayed(path):time.sleep(1.);return original(path)
    monkeypatch.setattr(module,"_telemetry_summary",delayed)
    ledger=reserve(tmp_path,seconds=.9,margin=.3)
    receipt=run(tmp_path,"pass",ledger)
    assert receipt["status"]=="FAIL_WALL_BUDGET_OVERSHOOT" and receipt["timing"]["budget_breached"]
    assert Path(receipt["receipt_path"]).name=="supervisor_budget_breach.json"
    assert ledger.finish("COMPLETED")["status"]=="FAIL_WALL_BUDGET_OVERSHOOT"


@WINDOWS
def test_environment_capability_delivered_but_not_logged(tmp_path):
    marker=tmp_path/"capability.txt";secret="synthetic-not-a-real-secret"
    script=f"import os; from pathlib import Path; Path({str(marker)!r}).write_text(os.environ['HIFT_W6_DISPATCH_CAPABILITY'])"
    receipt=run(tmp_path,script,env={"HIFT_W6_DISPATCH_CAPABILITY":secret})
    assert receipt["status"]=="COMPLETED" and marker.read_text()==secret
    assert secret not in json.dumps(receipt)
    assert secret not in Path(receipt["ledger_path"]).read_text()


@WINDOWS
def test_killing_supervisor_process_closes_noninherited_job_and_descendants(tmp_path):
    root_pid=tmp_path/"root.pid";grand_pid=tmp_path/"grand.pid";ready=tmp_path/"ready"
    grand=f"import os,time; from pathlib import Path; Path({str(grand_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    root=f"import os,subprocess,sys,time; from pathlib import Path; Path({str(root_pid)!r}).write_text(str(os.getpid())); subprocess.Popen([sys._base_executable,'-c',{grand!r}]); time.sleep(60)"
    outer=("import sys,time; from pathlib import Path; from hift_research.wave6.windows_job import SuspendedJobProcess; "
           f"j=SuspendedJobProcess([sys.executable,'-c',{root!r}],{str(ROOT)!r},stdout_path={str(tmp_path/'out.log')!r},stderr_path={str(tmp_path/'err.log')!r},physical_executable=sys._base_executable,env={{'__PYVENV_LAUNCHER__':sys.executable}}); j.resume(); Path({str(ready)!r}).write_text('ready'); time.sleep(60)")
    process=subprocess.Popen([sys._base_executable,"-c",outer],cwd=ROOT,env={**os.environ,"__PYVENV_LAUNCHER__":sys.executable})
    try:
        assert wait_for(lambda:ready.exists() and grand_pid.exists())
        process.kill();process.wait(timeout=2)
        assert wait_for(lambda:not is_process_running(int(root_pid.read_text())) and not is_process_running(int(grand_pid.read_text())),seconds=2)
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=2)


@WINDOWS
def test_reservation_atomic_race_allows_one_process_only(tmp_path):
    path=tmp_path/"race.jsonl"
    code=("from hift_research.wave6.ledger import reserve_attempt,AttemptConsumedError; from datetime import datetime,timedelta,timezone; "
          f"p={str(path)!r}\ntry:\n reserve_attempt(p,experiment_id='race',authorization_id='ack',max_seconds=2,cleanup_margin_seconds=.5,expiry_utc=(datetime.now(timezone.utc)+timedelta(seconds=20)).isoformat())\nexcept AttemptConsumedError:\n raise SystemExit(23)\n")
    children=[subprocess.Popen([sys.executable,"-c",code],cwd=ROOT) for _ in range(2)]
    assert sorted(child.wait(timeout=3) for child in children)==[0,23]
    assert len(read_ledger(path))==1


@WINDOWS
def test_direct_base_physical_launch_preserves_venv_and_exact_parent(tmp_path):
    marker=tmp_path/"prefix.json"
    script=f"import os,sys,json; from pathlib import Path; Path({str(marker)!r}).write_text(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),executable=sys.executable,prefix=sys.prefix)))"
    receipt=run(tmp_path,script,physical_executable=sys._base_executable,expected_prefix=sys.prefix,
                env={"__PYVENV_LAUNCHER__":sys.executable})
    actual=json.loads(marker.read_text())
    assert receipt["status"]=="COMPLETED"
    assert actual["ppid"]==os.getpid() and actual["pid"]==receipt["child_pid"]
    assert Path(actual["prefix"]).resolve()==Path(sys.prefix).resolve()
    assert Path(actual["executable"]).resolve()==Path(sys.executable).resolve()
    assert Path(receipt["physical_executable"]).resolve()==Path(sys._base_executable).resolve()


def guard_process(tmp_path,body,seconds=1.3,margin=.5):
    path=tmp_path/"guard.time.jsonl"
    prefix=("import time,sys,os; from datetime import datetime,timedelta,timezone; "
            "from hift_research.wave6.ledger import reserve_attempt,atomic_json; "
            "from hift_research.wave6.supervisor import parent_budget_guard,supervise; "
            f"l=reserve_attempt({str(path)!r},experiment_id='guard',authorization_id='synthetic-ack-digest',max_seconds={seconds},cleanup_margin_seconds={margin},expiry_utc=(datetime.now(timezone.utc)+timedelta(seconds=20)).isoformat()); "
            "g=parent_budget_guard(l); ")
    return subprocess.Popen([sys._base_executable,"-c",prefix+body],cwd=ROOT,
                            env={**os.environ,"__PYVENV_LAUNCHER__":sys.executable}),path


@WINDOWS
@pytest.mark.parametrize("phase",["preflight","cleanup"])
def test_root_watchdog_bounds_parent_preflight_and_outer_cleanup(tmp_path,phase):
    marker=tmp_path/"child_complete.json";out=tmp_path/"output";journal=out/"telemetry.jsonl"
    script=f"from hift_research.wave6.telemetry import DurableJournal; j=DurableJournal({str(journal)!r}); j.emit('update_commit',update=7,cursor=112)"
    body="g.__enter__(); "
    if phase=="cleanup":
        body+=(f"r=supervise([sys.executable,'-c',{script!r}],cwd={str(ROOT)!r},output={str(out)!r},experiment_id='guard',authorization_id='synthetic-ack-digest',ledger=l,physical_executable=sys._base_executable,env={{'__PYVENV_LAUNCHER__':sys.executable}}); "
               f"atomic_json({str(marker)!r},r); ")
    body+="time.sleep(60)"  # no finish: final timer cannot be silently disarmed
    began=time.perf_counter();process,path=guard_process(tmp_path,body,seconds=1.6,margin=.6)
    try:assert process.wait(timeout=3)==124
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=2)
    assert time.perf_counter()-began<2.6
    rows=read_ledger(path);assert not any(row["event"]=="entry_finished" for row in rows)
    emergency=json.loads(Path(str(path)+".deadline.json").read_text())
    assert emergency["recovery_requires_new_approval"] and emergency["hard_exit_code"]==124
    if phase=="cleanup":
        assert json.loads(marker.read_text())["process_tree_empty_verified"]
        assert emergency["jobs"][0]["last_committed"]["update"]==7
        assert emergency["jobs"][0]["process_tree_empty_verified"]
    else:assert not marker.exists() and emergency["jobs"]==[]
    with pytest.raises(AttemptConsumedError):reserve(tmp_path,name="guard")


@WINDOWS
@pytest.mark.parametrize("block",["ledger_lock","emergency_io"])
def test_root_hard_exit_never_waits_for_ledger_lock_or_emergency_io(tmp_path,block):
    body=""
    if block=="emergency_io":body+="g._emergency=lambda:time.sleep(60); "
    body+="g.__enter__(); "
    if block=="ledger_lock":body+="l._lock.acquire(); "
    body+="time.sleep(60)"
    process,path=guard_process(tmp_path,body,seconds=1.1,margin=.4)
    try:assert process.wait(timeout=2.5)==124
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=2)
    assert read_ledger(path)[0]["event"]=="attempt_reserved"
    with pytest.raises(AttemptConsumedError):reserve(tmp_path,name="guard")


@WINDOWS
def test_root_watchdog_terminates_registered_actual_descendant_tree(tmp_path):
    child_pid=tmp_path/"child.pid";grand_pid=tmp_path/"grand.pid"
    grand=f"import os,time; from pathlib import Path; Path({str(grand_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    child=f"import os,subprocess,sys,time; from pathlib import Path; Path({str(child_pid)!r}).write_text(str(os.getpid())); subprocess.Popen([sys._base_executable,'-c',{grand!r}]); time.sleep(60)"
    body=("g.__enter__(); from hift_research.wave6.windows_job import SuspendedJobProcess; "
          f"j=SuspendedJobProcess([sys.executable,'-c',{child!r}],{str(ROOT)!r},stdout_path={str(tmp_path/'out.log')!r},stderr_path={str(tmp_path/'err.log')!r},physical_executable=sys._base_executable,env={{'__PYVENV_LAUNCHER__':sys.executable}}); "
          "g.register_job(j); j.resume(); time.sleep(60)")
    process,path=guard_process(tmp_path,body,seconds=1.7,margin=.6)
    try:assert process.wait(timeout=3)==124
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=2)
    assert child_pid.exists() and grand_pid.exists()
    assert wait_for(lambda:not is_process_running(int(child_pid.read_text())) and not is_process_running(int(grand_pid.read_text())),seconds=1)
    emergency=json.loads(Path(str(path)+".deadline.json").read_text())
    assert emergency["jobs"][0]["process_tree_empty_verified"]


@WINDOWS
def test_parent_guard_disarms_only_after_durable_finish(tmp_path):
    body="g.__enter__(); l.finish('FAIL_PREFLIGHT',outer_cleanup_complete=True); g.__exit__(None,None,None); time.sleep(1.3)"
    process,path=guard_process(tmp_path,body,seconds=.8,margin=.3)
    try:assert process.wait(timeout=2.5)==0
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=2)
    assert read_ledger(path)[-1]["event"]=="entry_finished"
    assert Path(str(path)+".deadline.json").read_bytes()==b''


@WINDOWS
def test_reservation_own_initial_durable_io_cannot_hang_unbounded(tmp_path):
    path=tmp_path/"reserve.time.jsonl"
    code=("import time\nfrom datetime import datetime,timedelta,timezone\nimport hift_research.wave6.ledger as m\n"
          "original=m._exclusive_bytes\ndef blocked(path,payload):\n original(path,payload)\n time.sleep(60)\n"
          "m._exclusive_bytes=blocked\n"
          f"m.reserve_attempt({str(path)!r},experiment_id='reserve',authorization_id='synthetic-ack-digest',max_seconds=1.1,cleanup_margin_seconds=.4,expiry_utc=(datetime.now(timezone.utc)+timedelta(seconds=20)).isoformat())\n")
    process=subprocess.Popen([sys._base_executable,"-c",code],cwd=ROOT,env={**os.environ,"__PYVENV_LAUNCHER__":sys.executable})
    try:assert process.wait(timeout=2.5)==124
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=2)
    assert read_ledger(path)[0]["event"]=="attempt_reserved"
    with pytest.raises(AttemptConsumedError):reserve(tmp_path,name="reserve")
