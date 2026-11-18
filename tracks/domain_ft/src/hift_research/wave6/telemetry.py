"""Single-writer, fsynced, hash-chained JSONL. Only an incomplete final tail is recoverable.

No RNG draws, tensor arithmetic, model evaluation, loss backward, or optimizer calls.
Checkpoint payload `sha256` is separate from `journal_sha256` integrity metadata.
"""
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path


class JournalCorruptionError(RuntimeError):
    pass


def _encode(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode("utf-8")


def _validate(value):
    if value is None or type(value) in (str,bool,int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value,(list,tuple)):
        for item in value:
            _validate(item)
        return
    if isinstance(value,dict) and all(isinstance(k,str) for k in value):
        for item in value.values():
            _validate(item)
        return
    raise ValueError("journal fields must be finite detached JSON primitives, not tensors/RNG objects")


def read_last_good(path,repair=False):
    path=Path(path)
    events=[];offset=0;previous=None
    with path.open("rb") as stream:
        data=stream.read()
    incomplete=False
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            incomplete=True
            break
        try:
            entry=json.loads(line)
            _validate(entry)
            checksum=entry.pop("journal_sha256")
            if (entry.get("seq")!=len(events) or type(entry.get("seq")) is not int
                    or entry.get("previous_journal_sha256")!=previous
                    or not isinstance(entry.get("event"),str) or not entry["event"]
                    or checksum!=hashlib.sha256(_encode(entry)).hexdigest()):
                raise ValueError("journal chain/sequence/checksum mismatch")
            entry["journal_sha256"]=checksum
        except (ValueError,KeyError,TypeError,UnicodeError) as exc:
            raise JournalCorruptionError(f"complete journal record corrupt at byte {offset}") from exc
        events.append(entry);offset+=len(line);previous=checksum
    if repair and incomplete:
        # Verified absolute target is the exact caller-selected journal, never a computed deletion.
        resolved=path.resolve()
        with resolved.open("r+b") as stream:
            stream.truncate(offset);stream.flush();os.fsync(stream.fileno())
    return {"events":events,"last_event":events[-1] if events else None,"last_good_offset":offset,"incomplete_tail":incomplete}


class DurableJournal:
    def __init__(self,path,resume=False):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._lock=threading.Lock()
        if resume:
            state=read_last_good(self.path,repair=True)
            self._seq=len(state["events"])
            self._previous=state["last_event"]["journal_sha256"] if state["last_event"] else None
        else:
            with self.path.open("xb") as stream:
                stream.flush();os.fsync(stream.fileno())
            self._seq=0;self._previous=None

    def emit(self,event,**fields):
        if not isinstance(event,str) or not event:
            raise ValueError("nonempty event required")
        reserved={"event","seq","time","previous_journal_sha256","journal_sha256"}
        if reserved.intersection(fields):
            raise ValueError("journal metadata field collision")
        _validate(fields)
        with self._lock:
            entry={**fields,"event":event,"seq":self._seq,"time":time.time(),"previous_journal_sha256":self._previous}
            entry["journal_sha256"]=hashlib.sha256(_encode(entry)).hexdigest()
            payload=_encode(entry)+b"\n"
            with self.path.open("ab") as stream:
                stream.write(payload);stream.flush();os.fsync(stream.fileno())
            self._seq+=1;self._previous=entry["journal_sha256"]
            return entry

    def read_last_good(self,repair=False):
        return read_last_good(self.path,repair=repair)
