"""External inclusive deadline + Windows descendant containment; no SGD authority.

A valid-ACK entry must reserve its ledger before preflight, then keep its GPU
lease held through this function and job-empty confirmation. With a caller-owned
ledger, entry calls ledger.finish AFTER outer lease cleanup. No time reset.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import threading
import time
from .ledger import AttemptLedger,reserve_attempt,expiry_epoch,atomic_json
from .windows_job import SuspendedJobProcess,JobLaunchError


def _telemetry_summary(path):
    empty=dict(path=str(path) if path else None,available=False,integrity="missing",incomplete_tail=False,
               last_committed=None,last_checkpoint=None,checkpoint_bytes_revalidated=False,
               failed_work={"optimizer_mutation_unpublished_possible":None},consumed_attempts={},last_events={})
    if path is None or not Path(path).exists():return empty
    try:
        from .telemetry import read_last_good
        state=read_last_good(path,repair=False)
    except Exception as exc:
        return dict(empty,integrity="corrupt",error=type(exc).__name__+": "+str(exc))
    events=state["events"];last={}
    # The native journal uses top-level fields. data/details are accepted only as
    # a compatibility input; top-level names win and no unverified file is read.
    normalized=[]
    for event in events:
        row=dict(event)
        nested=row.get("data",row.get("details",{}))
        if isinstance(nested,dict):row={**nested,**row}
        normalized.append(row);last[row["event"]]=row
    commit=last.get("update_commit");commit_seq=commit.get("seq",-1) if commit else -1
    pending=[row for row in normalized if row.get("seq",-1)>commit_seq]
    step=next((row for row in reversed(pending) if row["event"]=="optimizer_step_begin"),None)
    validated=next((row for row in reversed(pending) if row["event"]=="optimizer_step_validated"),None)
    failure=last.get("window_failure",{})
    planned=[];observed=[];by_phase={}
    def extend_unique(target,values):
        if isinstance(values,list):
            for value in values:
                if value not in target:target.append(value)
    for row in normalized:
        if row["event"]=="attempt_window":
            phase=str(row.get("phase","unknown"));extend_unique(by_phase.setdefault(phase,[]),row.get("attempt_ids",[]))
        if row.get("seq",-1)>commit_seq:
            if row["event"]=="window_begin":extend_unique(planned,row.get("attempt_ids",[]))
            if row["event"]=="attempt_window":extend_unique(observed,row.get("attempt_ids",[]))
    tainted=bool(step or validated or state["incomplete_tail"] or (failure.get("optimizer_may_have_mutated") and failure.get("seq",-1)>commit_seq))
    return dict(empty,available=True,integrity="valid_prefix",incomplete_tail=state["incomplete_tail"],
                verified_event_count=len(events),last_committed=commit,last_checkpoint=last.get("checkpoint"),
                checkpoint_bytes_revalidated=False,
                failed_work=dict(last_window_begin=last.get("window_begin"),last_optimizer_step_begin=step,
                                 last_optimizer_step_validated=validated,last_window_failure=last.get("window_failure"),
                                 optimizer_mutation_unpublished_possible=tainted,
                                 observed_uncommitted_attempt_ids=observed,planned_uncommitted_attempt_ids=planned),
                consumed_attempts=dict(last_committed_cursor=commit.get("cursor") if commit else None,
                                       observed_by_phase={phase:{"attempt_ids":ids,"count":len(ids)} for phase,ids in by_phase.items()},
                                       pending_planned_are_not_claimed_consumed=True),last_events=last)


_ACTIVE_PARENT_GUARD=None


class ParentBudgetGuard:
    """Root-process watchdog spanning preflight, child work, outer cleanup, finish.

    Final timer performs ONLY os._exit(124): no ledger lock, fsync or callbacks.
    Emergency telemetry is independent/best effort. Watchdog threads must remain
    OS/GIL-schedulable; this is not a defense against hostile native GIL capture.
    """
    def __init__(self,ledger,emergency_path=None):
        if not isinstance(ledger,AttemptLedger):raise TypeError("reserved ledger required")
        self.ledger=ledger;self.emergency_path=Path(emergency_path or (str(ledger.path)+".deadline.json")).resolve()
        self._jobs=[];self._fd=None;self._started=False;self._emergency_lock=threading.Lock()
        self._cancelled=threading.Event();self.emergency_triggered=False;self.cancelled=False

    @staticmethod
    def _hard_exit():os._exit(124)

    def _emergency(self):
        if not self._emergency_lock.acquire(False):return
        try:
            if self.emergency_triggered or self.cancelled:return
            self.emergency_triggered=True;snapshots=[]
            for item in tuple(self._jobs):
                job=item["job"];empty=bool(item.get("tree_empty_verified",False)) if job.closed else False
                try:
                    if not job.closed:
                        job.terminate_tree();empty=job.wait_empty(min(.05,self.ledger.remaining()*.1))
                except Exception:empty=False
                telemetry=_telemetry_summary(item.get("telemetry_path"))
                snapshots.append(dict(child_pid=job.pid,process_tree_empty_verified=empty,
                                      last_committed=telemetry["last_committed"],last_checkpoint=telemetry["last_checkpoint"],
                                      failed_work=telemetry["failed_work"],consumed_attempts=telemetry["consumed_attempts"],
                                      journal_integrity=telemetry["integrity"],incomplete_tail=telemetry["incomplete_tail"]))
            value=dict(schema="hift-wave6-parent-deadline-emergency-v1",status="STOP_PARENT_SHUTDOWN_MARGIN",
                       experiment_id=self.ledger.experiment_id,authorization_id=self.ledger.authorization_id,
                       attempt_id=self.ledger.attempt_id,timing=self.ledger.timing(),jobs=snapshots,
                       recovery_requires_new_approval=True,automatic_retry=False,
                       hard_exit_code=124,checkpoint_bytes_revalidated=False,
                       note="best effort before independent no-I/O hard-exit timer; empty preopened file means no published emergency")
            if self._fd is not None:
                payload=json.dumps(value,sort_keys=True,allow_nan=False,separators=(",",":")).encode()+b"\n"
                os.write(self._fd,payload);os.fsync(self._fd)
        except Exception:
            # Never delay/cancel the independent hard timer because emergency
            # reporting, native job calls or disk I/O failed or hung.
            pass
        finally:self._emergency_lock.release()

    def _watch_expiry(self):
        while not self._cancelled.is_set():
            remaining=self.ledger.remaining()
            if remaining<=0:self._hard_exit();return
            if remaining<=self.ledger.cleanup_margin_seconds/2 and not self.emergency_triggered and not getattr(self,"_extra_emergency_started",False):
                self._extra_emergency_started=True
                threading.Thread(target=self._emergency,daemon=True).start()
            self._cancelled.wait(min(.05,max(.001,remaining)))

    def __enter__(self):
        global _ACTIVE_PARENT_GUARD
        if self._started or _ACTIVE_PARENT_GUARD is not None:raise RuntimeError("only one root parent guard/invocation allowed")
        self.ledger._assert_owner();self._started=True;_ACTIVE_PARENT_GUARD=self
        remaining=self.ledger.remaining()
        self._hard=threading.Timer(remaining,self._hard_exit);self._hard.daemon=True
        self._early=threading.Timer(max(0.,remaining-self.ledger.cleanup_margin_seconds/2),self._emergency);self._early.daemon=True
        # Arm final no-I/O escape BEFORE any directory/open/fsync or ledger lock.
        self._hard.start();self._early.start()
        self._expiry=threading.Thread(target=self._watch_expiry,daemon=True);self._expiry.start()
        if self.emergency_path.parent!=self.ledger.path.parent:raise ValueError("emergency file must remain inside time-ledger parent")
        self._fd=os.open(self.emergency_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_BINARY",0),0o600)
        # Preopened, noninheritable file: no path discovery/open at hard deadline.
        # It stays empty after a healthy run; only emergency emits JSON once.
        self.ledger.record("parent_budget_guard_armed",emergency_path=str(self.emergency_path),hard_exit_code=124)
        return self

    def register_job(self,job,telemetry_path=None):
        if self.cancelled:raise RuntimeError("parent deadline guard already closed")
        self._jobs.append(dict(job=job,telemetry_path=str(telemetry_path) if telemetry_path is not None else None,active=True))
        if self.emergency_triggered and not job.closed:job.terminate_tree()

    def unregister_job(self,job,tree_empty_verified=False):
        # Retain completed job journal paths for a later hung outer cleanup.
        for item in self._jobs:
            if item["job"] is job:
                item["active"]=False;item["tree_empty_verified"]=bool(tree_empty_verified)

    def __exit__(self,*exc):
        global _ACTIVE_PARENT_GUARD
        if not self.ledger._finished:
            # Failed finalization MUST NOT silently disarm the only parent bound.
            return False
        if self._fd is not None:
            os.close(self._fd);self._fd=None
        if self.ledger.remaining()<=0:self._hard_exit()
        self._cancelled.set();self._hard.cancel();self._early.cancel();self.cancelled=True
        _ACTIVE_PARENT_GUARD=None
        return False


def parent_budget_guard(ledger,emergency_path=None):
    return ParentBudgetGuard(ledger,emergency_path)


def supervise(argv,*,cwd,output,experiment_id,authorization_id=None,authorization_digest=None,max_seconds=None,
              expiry_utc=None,starting_monotonic=None,ledger=None,ledger_path=None,cleanup_margin_seconds=None,
              telemetry_path=None,env=None,stop_file=None,poll_interval=.02,physical_executable=None,expected_prefix=None,
              resource_monitor_factory=None):
    """Contain one fresh child tree. Returned normal receipt includes last journal commit.

    Child logs are files, never pipes. Environment override VALUES (including
    dispatch capability) are never recorded. Supplying a later start/expiry or
    different identity cannot extend a reserved attempt. Deadline overruns FAIL.
    """
    entered=time.perf_counter();output=Path(output).resolve();cwd=Path(cwd).resolve()
    auth=authorization_id if authorization_id is not None else authorization_digest
    own_ledger=ledger is None
    if ledger is None:
        duration=900 if max_seconds is None else max_seconds
        margin=5 if cleanup_margin_seconds is None else cleanup_margin_seconds
        ledger=reserve_attempt(ledger_path or output.with_name(output.name+".time.jsonl"),experiment_id=experiment_id,
                              authorization_id=auth,max_seconds=duration,expiry_utc=expiry_utc,
                              starting_monotonic=entered if starting_monotonic is None else starting_monotonic,
                              cleanup_margin_seconds=margin)
    if not isinstance(ledger,AttemptLedger):raise TypeError("live reserved AttemptLedger required; no restart/reopen")
    ledger.claim_supervision()  # every invocation, including invalid binding, is one-shot
    if ledger.experiment_id!=experiment_id or ledger.authorization_id!=auth:raise PermissionError("supervisor differs from reserved experiment/authorization")
    if max_seconds is not None and float(max_seconds)!=ledger.max_seconds:raise ValueError("cannot reset/change reserved time budget")
    if starting_monotonic is not None and float(starting_monotonic)!=ledger.starting_monotonic:raise ValueError("cannot replace original entry start")
    if expiry_utc is not None and expiry_epoch(expiry_utc)!=ledger.expiry_epoch:raise ValueError("cannot replace reserved expiry")
    if cleanup_margin_seconds is not None and float(cleanup_margin_seconds)!=ledger.cleanup_margin_seconds:raise ValueError("cannot replace reserved cleanup margin")
    if not 0<float(poll_interval)<=.1:raise ValueError("poll interval must be (0,.1] seconds")
    output.mkdir(parents=True,exist_ok=True)
    stop_path=Path(stop_file).resolve() if stop_file is not None else output/"STOP_REQUEST.json"
    if not stop_path.is_relative_to(output):raise ValueError("stop request must be inside bound output")
    if stop_path.exists():raise FileExistsError("preexisting stop request: never launch/retry")
    child_env=dict(env or {});child_env["HIFT_W6_STOP_FILE"]=str(stop_path)
    child_env["HIFT_W6_HARD_DEADLINE_MONOTONIC"]=str(ledger.deadline)
    telemetry_path=Path(telemetry_path).resolve() if telemetry_path is not None else output/"telemetry.jsonl"
    job=None;exit_code=None;peak_active=0;tree_empty=True;accounting=None;errors=[];status="FAIL_SUPERVISOR"
    reason=None;stop_requested=False;forced=False;launched=False;pid=None;trace=[]
    resource_monitor=None;resource_receipt=None
    try:
        if ledger.remaining()<=ledger.cleanup_margin_seconds:
            status="STOP_DEADLINE_PREFLIGHT";reason="preflight_consumed_inclusive_time_budget"
        else:
            ledger.record("child_launch_begin",argv=list(map(str,argv)),cwd=str(cwd),output=str(output),
                          environment_override_names=sorted(child_env),time_scope="original entry start; no preflight reset")
            job=SuspendedJobProcess(argv,cwd,stdout_path=output/"child.stdout.log",stderr_path=output/"child.stderr.log",env=child_env,physical_executable=physical_executable)
            pid=job.pid;tree_empty=False
            if _ACTIVE_PARENT_GUARD is not None:_ACTIVE_PARENT_GUARD.register_job(job,telemetry_path)
            ledger.record("child_suspended_assigned",pid=pid,creation_trace=list(job.trace))
            if resource_monitor_factory is not None:
                resource_monitor=resource_monitor_factory(job)
                resource_monitor.start();resource_monitor.check()
            if ledger.remaining()<=ledger.cleanup_margin_seconds:
                status="STOP_DEADLINE_PREFLIGHT";reason="launch_overhead_consumed_budget_before_resume"
            else:
                job.resume();launched=True;ledger.record("child_resumed",pid=pid)
                while True:
                    current=job.accounting();peak_active=max(peak_active,current["active_processes"])
                    exit_code=job.poll()
                    if exit_code is not None:
                        # Main process can exit between accounting and poll;
                        # recheck rather than mislabel it as a live descendant.
                        job.wait_empty(min(.03,ledger.remaining()*.1))
                        current=job.accounting()
                        if current["active_processes"]:
                            status="FAIL_CHILD_EXIT";reason="child_exited_with_live_descendants"
                        elif exit_code:
                            status="FAIL_CHILD_EXIT";reason="child_nonzero_exit"
                        elif stop_requested:
                            status="STOP_DEADLINE";reason="cooperative_stop_before_hard_deadline"
                        else:status="COMPLETED";reason="child_exit_zero_and_job_empty"
                        break
                    if resource_monitor is not None:resource_monitor.check()
                    now=time.perf_counter();hard=ledger.effective_deadline();margin=ledger.cleanup_margin_seconds
                    if now>=hard-margin and not stop_requested:
                        atomic_json(stop_path,dict(schema="hift-wave6-external-stop-v1",reason="inclusive_deadline_shutdown_margin",
                                                   experiment_id=experiment_id,authorization_id=auth,attempt_id=ledger.attempt_id,
                                                   hard_deadline_monotonic=hard,remaining_seconds=ledger.remaining()))
                        stop_requested=True;ledger.record("cooperative_stop_requested",pid=pid,stop_file=str(stop_path))
                    if now>=hard-margin/2:
                        status="STOP_DEADLINE";reason="forced_tree_stop_before_hard_deadline";forced=True;break
                    next_boundary=(hard-margin) if not stop_requested else (hard-margin/2)
                    time.sleep(min(float(poll_interval),max(.001,next_boundary-time.perf_counter())))
    except Exception as exc:
        status="FAIL_SUPERVISOR";reason=type(exc).__name__+": "+str(exc);errors.append(reason)
        if isinstance(exc,JobLaunchError):
            pid=exc.pid;trace=list(exc.trace);tree_empty=exc.tree_empty_verified
    finally:
        if resource_monitor is not None:
            try:
                resource_receipt=resource_monitor.close()
                if resource_receipt.get("failure"):
                    status="STOP_RESOURCE_POLICY";reason=resource_receipt["failure"]
            except Exception as exc:
                status="FAIL_RESOURCE_MONITOR";reason=type(exc).__name__+":"+str(exc);errors.append(reason)
        if job is not None:
            try:
                accounting=job.accounting()
                if accounting["active_processes"]:
                    job.terminate_tree();forced=True
                # All parent cleanup shares the original bound. Never grant a
                # fresh full timeout to an unresponsive descendant or to cleanup.
                tree_empty=job.wait_empty(max(0.,ledger.remaining()*.55))
                accounting=job.accounting();peak_active=max(peak_active,accounting["active_processes"])
                exit_code=job.poll();trace=list(job.trace)
            except Exception as exc:
                tree_empty=False;errors.append("job_cleanup:"+type(exc).__name__+": "+str(exc))
            finally:
                try:job.close()
                except Exception as exc:errors.append("job_close:"+type(exc).__name__+": "+str(exc));tree_empty=False
                if _ACTIVE_PARENT_GUARD is not None:_ACTIVE_PARENT_GUARD.unregister_job(job,tree_empty)
        if not tree_empty:
            status="FAIL_CONTAINMENT";reason="process_tree_empty_not_verified_before_job_close"
    telemetry=_telemetry_summary(telemetry_path)
    if telemetry["integrity"]=="corrupt":status="FAIL_TELEMETRY";reason="complete_child_journal_record_corrupt"
    timing=ledger.timing()
    if timing["budget_breached"]:status="FAIL_WALL_BUDGET_OVERSHOOT"
    receipt=dict(schema="hift-wave6-supervisor-receipt-v1",status=status,reason=reason,experiment_id=experiment_id,
                 authorization_id=auth,attempt_id=ledger.attempt_id,argv=list(map(str,argv)),cwd=str(cwd),output=str(output),
                 child_pid=pid,child_resumed=launched,child_exit_code=exit_code,process_tree_empty_verified=tree_empty,
                 logical_interpreter=str(argv[0]) if argv else None,
                 physical_executable=job.physical_executable if job is not None else str(physical_executable) if physical_executable else None,
                 expected_venv_launcher=child_env.get("__PYVENV_LAUNCHER__"),expected_prefix=str(expected_prefix) if expected_prefix else None,
                 job_closed=bool(job is None or job.closed),peak_active_processes=peak_active,final_job_accounting=accounting,
                 creation_trace=trace,cooperative_stop_requested=stop_requested,forced_tree_termination=forced,errors=errors,
                 resource_monitor=resource_receipt,
                 timing=timing,remaining_seconds=timing["remaining_seconds"],telemetry=telemetry,
                 last_committed=telemetry["last_committed"],
                 last_committed_update=telemetry["last_committed"].get("update",0) if telemetry["last_committed"] else 0,
                 last_committed_cursor=telemetry["last_committed"].get("cursor") if telemetry["last_committed"] else None,
                 last_good_checkpoint=telemetry["last_checkpoint"],failed_work=telemetry["failed_work"],
                 consumed_attempts=telemetry["consumed_attempts"],last_checkpoint=telemetry["last_checkpoint"],
                 checkpoint_bytes_revalidated=False,recovery_requires_new_approval=True,automatic_retry=False,
                 ledger_path=str(ledger.path),ledger_pending_entry_finalization=not own_ledger,
                 timing_scope="entry start through job cleanup; caller-owned ledger finalizes AFTER outer lease cleanup",
                 receipt_elapsed_snapshot_before_publication=True)
    ledger.record("supervisor_complete",status=status,process_tree_empty_verified=tree_empty,child_exit_code=exit_code,
                  last_committed=telemetry["last_committed"],
                 last_committed_update=telemetry["last_committed"].get("update",0) if telemetry["last_committed"] else 0,
                 last_committed_cursor=telemetry["last_committed"].get("cursor") if telemetry["last_committed"] else None,
                 last_good_checkpoint=telemetry["last_checkpoint"],failed_work=telemetry["failed_work"],timing=timing)
    normal_path=output/"supervisor_receipt.json";atomic_json(normal_path,receipt)
    receipt["receipt_path"]=str(normal_path);receipt["timing"]=ledger.timing();receipt["remaining_seconds"]=ledger.remaining()
    if own_ledger:
        finished=ledger.finish(status,process_tree_empty_verified=tree_empty,supervisor_receipt_path=str(normal_path))
        receipt["ledger_finalization"]=finished;receipt["ledger_pending_entry_finalization"]=False
        if finished["status"]=="FAIL_WALL_BUDGET_OVERSHOOT":receipt["status"]=finished["status"]
    actual=ledger.timing();receipt["timing"]=actual;receipt["remaining_seconds"]=actual["remaining_seconds"]
    if actual["budget_breached"]:
        receipt["status"]="FAIL_WALL_BUDGET_OVERSHOOT"
        breach_path=output/"supervisor_budget_breach.json"
        receipt["normal_receipt_path"]=str(normal_path);receipt["receipt_path"]=str(breach_path)
        atomic_json(breach_path,receipt)
        receipt["timing"]=ledger.timing();receipt["remaining_seconds"]=ledger.remaining()
    return receipt
