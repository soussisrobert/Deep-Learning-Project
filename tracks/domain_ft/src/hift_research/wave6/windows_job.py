"""Real Windows Job Object containment, assigned BEFORE the first child instruction.

No shell, stdout pipe, breakaway flag, GPU query or model import. The job handle
is non-inheritable; KILL_ON_JOB_CLOSE therefore also covers supervisor death.
"""
from __future__ import annotations
import ctypes
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import time

CREATE_SUSPENDED=0x00000004
CREATE_UNICODE_ENVIRONMENT=0x00000400
CREATE_NO_WINDOW=0x08000000
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE=0x00002000
WAIT_OBJECT_0=0
WAIT_TIMEOUT=258
INFINITE=0xFFFFFFFF
INVALID_HANDLE_VALUE=ctypes.c_void_p(-1).value


class IO_COUNTERS(ctypes.Structure):
    _fields_=[(name,ctypes.c_ulonglong) for name in ("ReadOperationCount","WriteOperationCount","OtherOperationCount","ReadTransferCount","WriteTransferCount","OtherTransferCount")]


class BASIC_LIMIT(ctypes.Structure):
    _fields_=[("PerProcessUserTimeLimit",ctypes.c_longlong),("PerJobUserTimeLimit",ctypes.c_longlong),
              ("LimitFlags",W.DWORD),("MinimumWorkingSetSize",ctypes.c_size_t),("MaximumWorkingSetSize",ctypes.c_size_t),
              ("ActiveProcessLimit",W.DWORD),("Affinity",ctypes.c_size_t),("PriorityClass",W.DWORD),("SchedulingClass",W.DWORD)]


class EXTENDED_LIMIT(ctypes.Structure):
    _fields_=[("BasicLimitInformation",BASIC_LIMIT),("IoInfo",IO_COUNTERS),("ProcessMemoryLimit",ctypes.c_size_t),
              ("JobMemoryLimit",ctypes.c_size_t),("PeakProcessMemoryUsed",ctypes.c_size_t),("PeakJobMemoryUsed",ctypes.c_size_t)]


class ACCOUNTING(ctypes.Structure):
    _fields_=[("TotalUserTime",ctypes.c_longlong),("TotalKernelTime",ctypes.c_longlong),
              ("ThisPeriodTotalUserTime",ctypes.c_longlong),("ThisPeriodTotalKernelTime",ctypes.c_longlong),
              ("TotalPageFaultCount",W.DWORD),("TotalProcesses",W.DWORD),("ActiveProcesses",W.DWORD),("TotalTerminatedProcesses",W.DWORD)]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_=[("nLength",W.DWORD),("lpSecurityDescriptor",W.LPVOID),("bInheritHandle",W.BOOL)]


class STARTUPINFO(ctypes.Structure):
    _fields_=[("cb",W.DWORD),("lpReserved",W.LPWSTR),("lpDesktop",W.LPWSTR),("lpTitle",W.LPWSTR),
              ("dwX",W.DWORD),("dwY",W.DWORD),("dwXSize",W.DWORD),("dwYSize",W.DWORD),
              ("dwXCountChars",W.DWORD),("dwYCountChars",W.DWORD),("dwFillAttribute",W.DWORD),("dwFlags",W.DWORD),
              ("wShowWindow",W.WORD),("cbReserved2",W.WORD),("lpReserved2",ctypes.POINTER(ctypes.c_byte)),
              ("hStdInput",W.HANDLE),("hStdOutput",W.HANDLE),("hStdError",W.HANDLE)]


class STARTUPINFOEX(ctypes.Structure):
    _fields_=[("StartupInfo",STARTUPINFO),("lpAttributeList",W.LPVOID)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_=[("hProcess",W.HANDLE),("hThread",W.HANDLE),("dwProcessId",W.DWORD),("dwThreadId",W.DWORD)]


def _kernel():
    if os.name!="nt":raise OSError("Windows Job Object supervision requires Windows")
    k=ctypes.WinDLL("kernel32",use_last_error=True)
    signatures={
        "CreateJobObjectW":([W.LPVOID,W.LPCWSTR],W.HANDLE),
        "SetInformationJobObject":([W.HANDLE,ctypes.c_int,W.LPVOID,W.DWORD],W.BOOL),
        "QueryInformationJobObject":([W.HANDLE,ctypes.c_int,W.LPVOID,W.DWORD,ctypes.POINTER(W.DWORD)],W.BOOL),
        "AssignProcessToJobObject":([W.HANDLE,W.HANDLE],W.BOOL),
        "InitializeProcThreadAttributeList":([W.LPVOID,W.DWORD,W.DWORD,ctypes.POINTER(ctypes.c_size_t)],W.BOOL),
        "UpdateProcThreadAttribute":([W.LPVOID,W.DWORD,ctypes.c_size_t,W.LPVOID,ctypes.c_size_t,W.LPVOID,W.LPVOID],W.BOOL),
        "DeleteProcThreadAttributeList":([W.LPVOID],None),
        "IsProcessInJob":([W.HANDLE,W.HANDLE,ctypes.POINTER(W.BOOL)],W.BOOL),
        "TerminateJobObject":([W.HANDLE,W.UINT],W.BOOL),
        "TerminateProcess":([W.HANDLE,W.UINT],W.BOOL),
        "CreateProcessW":([W.LPCWSTR,W.LPWSTR,W.LPVOID,W.LPVOID,W.BOOL,W.DWORD,W.LPVOID,W.LPCWSTR,ctypes.POINTER(STARTUPINFO),ctypes.POINTER(PROCESS_INFORMATION)],W.BOOL),
        "ResumeThread":([W.HANDLE],W.DWORD),
        "WaitForSingleObject":([W.HANDLE,W.DWORD],W.DWORD),
        "GetExitCodeProcess":([W.HANDLE,ctypes.POINTER(W.DWORD)],W.BOOL),
        "CloseHandle":([W.HANDLE],W.BOOL),
        "CreateFileW":([W.LPCWSTR,W.DWORD,W.DWORD,ctypes.POINTER(SECURITY_ATTRIBUTES),W.DWORD,W.DWORD,W.HANDLE],W.HANDLE),
        "OpenProcess":([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),
    }
    for name,(args,result) in signatures.items():
        function=getattr(k,name);function.argtypes=args;function.restype=result
    return k


def _check(ok,label):
    if not ok:raise ctypes.WinError(ctypes.get_last_error(),label)


class JobLaunchError(RuntimeError):
    def __init__(self,message,pid,trace,tree_empty_verified=False):
        super().__init__(message);self.pid=pid;self.trace=trace;self.tree_empty_verified=tree_empty_verified


class SuspendedJobProcess:
    def __init__(self,argv,cwd,*,stdout_path,stderr_path,env=None,physical_executable=None):
        self.k=_kernel();self.job=None;self.process=None;self.thread=None;self.pid=None
        self.assigned=False;self.resumed=False;self.closed=False;self.trace=[];self._streams=[]
        argv=[str(item) for item in argv]
        if not argv or any("\0" in value for value in argv):raise ValueError("nonempty NUL-free argv required")
        cwd=Path(cwd).resolve()
        application=Path(physical_executable if physical_executable is not None else argv[0]);application=application if application.is_absolute() else cwd/application
        application=application.resolve();self.physical_executable=str(application);self.logical_argv=list(argv)
        if not application.is_file() or not cwd.is_dir():raise FileNotFoundError("explicit executable and cwd must exist; no PATH/shell fallback")
        try:
            self.job=self.k.CreateJobObjectW(None,None);_check(self.job,"CreateJobObjectW")
            limit=EXTENDED_LIMIT();limit.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            _check(self.k.SetInformationJobObject(self.job,9,ctypes.byref(limit),ctypes.sizeof(limit)),"SetInformationJobObject(KILL_ON_JOB_CLOSE)")
            attributes=SECURITY_ATTRIBUTES(ctypes.sizeof(SECURITY_ATTRIBUTES),None,True)
            def stream(path,access,creation):
                handle=self.k.CreateFileW(str(path),access,0x1|0x2,ctypes.byref(attributes),creation,0x80,None)
                if handle in (None,INVALID_HANDLE_VALUE):raise ctypes.WinError(ctypes.get_last_error(),"CreateFileW standard stream")
                self._streams.append(handle);return handle
            si=STARTUPINFO();si.cb=ctypes.sizeof(si);si.dwFlags=0x00000100
            si.hStdInput=stream("NUL",0x80000000,3)
            si.hStdOutput=stream(Path(stdout_path).resolve(),0x40000000,1)
            si.hStdError=stream(Path(stderr_path).resolve(),0x40000000,1)
            environment=None
            if env is not None:
                values=dict(os.environ)
                for key,value in env.items():
                    if not isinstance(key,str) or not key or "=" in key or "\0" in key or not isinstance(value,str) or "\0" in value:
                        raise ValueError("invalid child environment override")
                    values[key]=value
                environment=ctypes.create_unicode_buffer("\0".join(key+"="+value for key,value in sorted(values.items(),key=lambda pair:pair[0].upper()))+"\0\0")
            command=ctypes.create_unicode_buffer(subprocess.list2cmdline(argv));pi=PROCESS_INFORMATION()
            # Win10 atomically assigns the job during process creation. This
            # closes even the suspended-but-unassigned orphan window if the
            # parent hard watchdog dies between CreateProcess and a later API.
            # No unsupported-OS fallback silently weakens this guarantee.
            attribute_size=ctypes.c_size_t()
            self.k.InitializeProcThreadAttributeList(None,2,0,ctypes.byref(attribute_size))
            attribute_buffer=ctypes.create_string_buffer(attribute_size.value)
            attribute_pointer=ctypes.cast(attribute_buffer,W.LPVOID)
            _check(self.k.InitializeProcThreadAttributeList(attribute_pointer,2,0,ctypes.byref(attribute_size)),"InitializeProcThreadAttributeList")
            try:
                jobs=(W.HANDLE*1)(self.job);stdio=(W.HANDLE*3)(si.hStdInput,si.hStdOutput,si.hStdError)
                _check(self.k.UpdateProcThreadAttribute(attribute_pointer,0,0x0002000D,ctypes.byref(jobs),ctypes.sizeof(jobs),None,None),"PROC_THREAD_ATTRIBUTE_JOB_LIST")
                _check(self.k.UpdateProcThreadAttribute(attribute_pointer,0,0x00020002,ctypes.byref(stdio),ctypes.sizeof(stdio),None,None),"PROC_THREAD_ATTRIBUTE_HANDLE_LIST")
                extended=STARTUPINFOEX();extended.StartupInfo=si;extended.StartupInfo.cb=ctypes.sizeof(extended);extended.lpAttributeList=attribute_pointer
                _check(self.k.CreateProcessW(str(application),command,None,None,True,
                           CREATE_SUSPENDED|CREATE_UNICODE_ENVIRONMENT|CREATE_NO_WINDOW|0x00080000,
                           ctypes.cast(environment,W.LPVOID) if environment is not None else None,str(cwd),ctypes.byref(extended.StartupInfo),ctypes.byref(pi)),"CreateProcessW(CREATE_SUSPENDED + atomic JOB_LIST)")
            finally:self.k.DeleteProcThreadAttributeList(attribute_pointer)
            self.process=pi.hProcess;self.thread=pi.hThread;self.pid=int(pi.dwProcessId);self.assigned=True
            self.trace.append({"event":"created_suspended","pid":self.pid,"atomic_job_assignment":True,"monotonic":time.perf_counter()})
            in_job=W.BOOL(False);_check(self.k.IsProcessInJob(self.process,self.job,ctypes.byref(in_job)),"IsProcessInJob")
            if not in_job.value:raise RuntimeError("atomic child assignment not confirmed")
            self.trace.append({"event":"assigned_to_job","atomic_at_process_creation":True,"monotonic":time.perf_counter()})
        except BaseException as exc:
            # An unassigned suspended process is NOT killed by job closure.
            # Explicitly terminate it before closing any process handles.
            empty=self.process is None
            try:
                if self.process:
                    if self.assigned:
                        self.k.TerminateJobObject(self.job,0xE001);empty=self.wait_empty(.5)
                    else:
                        self.k.TerminateProcess(self.process,0xE001)
                        empty=self.k.WaitForSingleObject(self.process,500)==WAIT_OBJECT_0
            except Exception:empty=False
            try:self.close()
            except Exception:empty=False
            raise JobLaunchError(str(exc),self.pid,list(self.trace),tree_empty_verified=empty) from exc

    def resume(self):
        if self.closed or not self.assigned or self.resumed:raise RuntimeError("resume requires fresh assigned suspended child")
        previous=self.k.ResumeThread(self.thread)
        if previous==0xFFFFFFFF:raise ctypes.WinError(ctypes.get_last_error(),"ResumeThread")
        if previous!=1:
            self.terminate_tree();raise RuntimeError("unexpected initial suspend count; containment proof failed")
        self.resumed=True;self.trace.append({"event":"resumed_after_assignment","previous_suspend_count":int(previous),"monotonic":time.perf_counter()})

    def poll(self):
        if not self.process:return None
        result=self.k.WaitForSingleObject(self.process,0)
        if result==WAIT_TIMEOUT:return None
        if result!=WAIT_OBJECT_0:raise ctypes.WinError(ctypes.get_last_error(),"WaitForSingleObject process poll")
        code=W.DWORD();_check(self.k.GetExitCodeProcess(self.process,ctypes.byref(code)),"GetExitCodeProcess")
        return int(code.value)

    def accounting(self):
        if self.closed or not self.job:raise RuntimeError("job already closed")
        info=ACCOUNTING()
        _check(self.k.QueryInformationJobObject(self.job,1,ctypes.byref(info),ctypes.sizeof(info),None),"QueryInformationJobObject accounting")
        return dict(total_processes=int(info.TotalProcesses),active_processes=int(info.ActiveProcesses),terminated_processes=int(info.TotalTerminatedProcesses))

    def terminate_tree(self,exit_code=0xE002):
        if not self.closed and self.job:
            _check(self.k.TerminateJobObject(self.job,exit_code),"TerminateJobObject")
            self.trace.append({"event":"terminate_job","monotonic":time.perf_counter()})

    def wait_empty(self,timeout):
        deadline=time.perf_counter()+max(0.,timeout)
        while True:
            if self.accounting()["active_processes"]==0:return True
            left=deadline-time.perf_counter()
            if left<=0:return False
            time.sleep(min(.01,left))

    def close(self):
        if self.closed:return
        errors=[]
        if self.job:
            # KILL_ON_JOB_CLOSE is a second independent cleanup mechanism.
            if not self.k.CloseHandle(self.job):errors.append(ctypes.get_last_error())
            self.job=None
        for handle in (self.thread,self.process,*self._streams):
            if handle and not self.k.CloseHandle(handle):errors.append(ctypes.get_last_error())
        self.thread=None;self.process=None;self._streams=[];self.closed=True
        if errors:raise OSError("Windows handle cleanup failed: "+str(errors))

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def is_process_running(pid):
    k=_kernel();handle=k.OpenProcess(0x00100000|0x1000,False,int(pid))
    if not handle:
        code=ctypes.get_last_error()
        if code==87:return False
        raise ctypes.WinError(code,"OpenProcess liveness")
    try:return k.WaitForSingleObject(handle,0)==WAIT_TIMEOUT
    finally:k.CloseHandle(handle)
