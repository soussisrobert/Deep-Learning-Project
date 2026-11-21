"""Durable one-shot inclusive time accounting; this module grants no authority.

The entry validates the cheap human ACK first, reserves before expensive work,
then finally calls finish AFTER outer lease/resource cleanup. No reopen/resume
API exists. A partial/crashed reservation permanently consumes its path.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid


class AttemptConsumedError(FileExistsError):
    pass


def _json_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")


def expiry_epoch(value):
    if isinstance(value,bool) or value is None:raise ValueError("explicit aware UTC expiry required")
    if isinstance(value,(int,float)):
        epoch=float(value)
    else:
        date=value if isinstance(value,datetime) else datetime.fromisoformat(str(value).replace("Z","+00:00"))
        if date.tzinfo is None or date.utcoffset() is None:raise ValueError("expiry must include timezone")
        epoch=date.timestamp()
    if not math.isfinite(epoch):raise ValueError("expiry must be finite")
    return epoch


def _exclusive_bytes(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY|getattr(os,"O_BINARY",0),0o600)
    except FileExistsError as exc:raise AttemptConsumedError("attempt already consumed: "+str(path)) from exc
    try:
        with os.fdopen(fd,"wb",buffering=0) as stream:
            view=memoryview(payload)
            while view:
                written=stream.write(view)
                if not written:raise OSError("short durable reservation write")
                view=view[written:]
            os.fsync(stream.fileno())
    except BaseException:
        # Never delete/retry a partial reservation: it consumed this attempt.
        raise


def atomic_json(path,value):
    """Atomic no-replace publication; staging remnants never authorize retry."""
    path=Path(path).resolve();path.parent.mkdir(parents=True,exist_ok=True)
    stage=path.with_name("."+path.name+"."+uuid.uuid4().hex+".pending")
    _exclusive_bytes(stage,_json_bytes(value)+b"\n")
    try:os.link(stage,path)
    except FileExistsError as exc:raise AttemptConsumedError("receipt already exists: "+str(path)) from exc
    finally:
        # This exact private staging file was created above; never remove an
        # arbitrary/computed external path or an existing published artifact.
        if stage.resolve().parent!=path.parent:raise RuntimeError("staging path escaped receipt parent")
        stage.unlink()


def read_ledger(path):
    """Verify complete append-only hash chain; partial journals fail closed."""
    payload=Path(path).read_bytes()
    if not payload or not payload.endswith(b"\n"):raise ValueError("incomplete consumed attempt ledger")
    rows=[];previous="0"*64
    for number,line in enumerate(payload.splitlines()):
        row=json.loads(line)
        digest=row.pop("event_sha256")
        if row.get("sequence")!=number or row.get("previous_sha256")!=previous or hashlib.sha256(_json_bytes(row)).hexdigest()!=digest:
            raise ValueError("ledger sequence/hash integrity failure")
        row["event_sha256"]=digest;rows.append(row);previous=digest
    if rows[0].get("event")!="attempt_reserved":raise ValueError("missing initial one-shot reservation")
    if any(row.get("attempt_id")!=rows[0]["attempt_id"] for row in rows):raise ValueError("mixed attempt ledger")
    return rows


class AttemptLedger:
    """Only the process which reserved may append or finalize this live object."""
    def __init__(self,path,initial):
        self.path=Path(path).resolve();self.initial=initial
        self.attempt_id=initial["attempt_id"];self.experiment_id=initial["experiment_id"]
        self.authorization_id=initial["authorization_id"];self.owner_pid=initial["owner_pid"]
        self.starting_monotonic=initial["starting_monotonic"];self.max_seconds=initial["max_seconds"]
        self.expiry_epoch=initial["expiry_epoch"];self.deadline=initial["hard_deadline_monotonic"]
        self.cleanup_margin_seconds=initial["cleanup_margin_seconds"]
        self._sequence=1;self._previous=initial["event_sha256"]
        self._file_sha=hashlib.sha256(self.path.read_bytes()).hexdigest();self._finished=False
        self._lock=threading.Lock();self._supervision_claimed=False

    @property
    def authorization_digest(self):return self.authorization_id

    def elapsed(self):return max(0.,time.perf_counter()-self.starting_monotonic)

    def remaining(self):
        # UTC forward jumps shorten the run; rollback never extends the anchored
        # monotonic deadline. No restart can reset it (O_EXCL reservation).
        return max(0.,min(self.deadline-time.perf_counter(),self.expiry_epoch-time.time()))

    def effective_deadline(self):
        now=time.perf_counter()
        return min(self.deadline,now+max(0.,self.expiry_epoch-time.time()))

    def timing(self):
        elapsed=self.elapsed();remaining=self.remaining()
        breached=bool(time.perf_counter()>self.deadline or time.time()>self.expiry_epoch)
        return dict(starting_monotonic=self.starting_monotonic,elapsed_seconds=elapsed,remaining_seconds=remaining,
                    max_seconds=self.max_seconds,hard_deadline_monotonic=self.deadline,
                    expiry_epoch=self.expiry_epoch,budget_breached=breached)

    def _assert_owner(self):
        if os.getpid()!=self.owner_pid:raise PermissionError("time ledger belongs to a different process")
        if self._finished:raise AttemptConsumedError("attempt already finalized")

    def record(self,event,**details):
        with self._lock:
            self._assert_owner()
            raw=self.path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=self._file_sha:raise ValueError("time ledger changed outside owner")
            row=dict(event=str(event),sequence=self._sequence,previous_sha256=self._previous,attempt_id=self.attempt_id,
                     experiment_id=self.experiment_id,authorization_id=self.authorization_id,
                     monotonic=time.perf_counter(),utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=self.elapsed(),remaining_seconds=self.remaining(),details=details)
            digest=hashlib.sha256(_json_bytes(row)).hexdigest();row["event_sha256"]=digest
            line=_json_bytes(row)+b"\n"
            fd=os.open(self.path,os.O_WRONLY|os.O_APPEND|getattr(os,"O_BINARY",0))
            try:
                written=os.write(fd,line)
                if written!=len(line):raise OSError("partial append consumes ledger; no repair/retry")
                os.fsync(fd)
            finally:os.close(fd)
            self._sequence+=1;self._previous=digest;self._file_sha=hashlib.sha256(raw+line).hexdigest()
            return row

    def claim_supervision(self):
        self._assert_owner()
        claim=self.path.with_name(self.path.name+".supervision-claimed")
        _exclusive_bytes(claim,_json_bytes(dict(attempt_id=self.attempt_id,owner_pid=self.owner_pid))+b"\n")
        self._supervision_claimed=True
        self.record("supervision_claimed",claim_path=str(claim))

    def finish(self,status,**details):
        """Entry invokes once AFTER lease release and all outer cleanup."""
        timing=self.timing()
        final_status="FAIL_WALL_BUDGET_OVERSHOOT" if timing["budget_breached"] else str(status)
        row=self.record("entry_finished",status=final_status,timing=timing,recovery_requires_new_approval=True,**details)
        # This timestamp additionally accounts for the durable final append.
        actual=self.timing()
        if actual["budget_breached"] and final_status!="FAIL_WALL_BUDGET_OVERSHOOT":
            row=self.record("finalization_overshoot",status="FAIL_WALL_BUDGET_OVERSHOOT",timing=actual,recovery_requires_new_approval=True)
            actual=self.timing()
        self._finished=True
        return dict(status="FAIL_WALL_BUDGET_OVERSHOOT" if actual["budget_breached"] else final_status,
                    timing=actual,ledger_path=str(self.path),attempt_id=self.attempt_id,
                    final_event_sha256=row["event_sha256"],recovery_requires_new_approval=True)


def reserve_attempt(ledger_path,*,experiment_id,authorization_id=None,authorization_digest=None,max_seconds=900,
                    starting_monotonic=None,expiry_utc=None,cleanup_margin_seconds=5,reservation_watchdog=True):
    """Consume valid-ACK invocation BEFORE preflight/lease/model initialization.

    This is a time reservation, NOT an execution/GPU claim or authorization.
    Supplied earlier start is never replaced by a later preflight completion.
    """
    now=time.perf_counter();wall=time.time()
    duration=float(max_seconds);margin=float(cleanup_margin_seconds)
    start=now if starting_monotonic is None else float(starting_monotonic)
    if not math.isfinite(duration) or not 0<duration<=900:raise ValueError("inclusive budget must be (0,900] seconds")
    if not math.isfinite(margin) or not 0<margin<duration:raise ValueError("cleanup margin must be positive and below total budget")
    if not math.isfinite(start) or start<0 or start>now:raise ValueError("start must be an earlier perf_counter reading")
    expiry=expiry_epoch(expiry_utc)
    auth=authorization_id if authorization_id is not None else authorization_digest
    if not isinstance(experiment_id,str) or not experiment_id or not isinstance(auth,str) or not auth:
        raise ValueError("experiment and authorization digest identities required")
    if authorization_id is not None and authorization_digest is not None and authorization_id!=authorization_digest:
        raise ValueError("authorization identity aliases disagree")
    initial=dict(schema="hift-wave6-inclusive-time-ledger-v1",event="attempt_reserved",sequence=0,previous_sha256="0"*64,
                 attempt_id=uuid.uuid4().hex,experiment_id=experiment_id,authorization_id=auth,owner_pid=os.getpid(),
                 starting_monotonic=start,reservation_monotonic=now,reservation_utc=datetime.now(timezone.utc).isoformat(),
                 max_seconds=duration,cleanup_margin_seconds=margin,expiry_epoch=expiry,
                 hard_deadline_monotonic=min(start+duration,now+max(0.,expiry-wall)),
                 max_invocations=1,resume_allowed=False,recovery_requires_new_approval=True,
                 clock=time.get_clock_info("perf_counter").implementation,authority_granted=False)
    initial["event_sha256"]=hashlib.sha256(_json_bytes(initial)).hexdigest()
    # Bound the reservation's OWN mkdir/open/fsync/read as well: parent_guard
    # can only be armed after this function returns a live ledger. Production
    # defaults on; only explicit synthetic expired-time arithmetic probes omit it.
    watchdog=None
    if reservation_watchdog:
        watchdog=threading.Timer(max(0.,initial["hard_deadline_monotonic"]-time.perf_counter()),lambda:os._exit(124))
        watchdog.daemon=True;watchdog.start()
    try:
        _exclusive_bytes(ledger_path,_json_bytes(initial)+b"\n")
        ledger=AttemptLedger(ledger_path,initial)
        return ledger
    finally:
        if watchdog is not None:watchdog.cancel()
