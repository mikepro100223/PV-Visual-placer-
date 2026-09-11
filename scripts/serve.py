"""Run the local map and Python API; Ctrl+C stops only these child processes."""
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser

ROOT = Path(__file__).resolve().parents[1]

def port_open(port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1',port))==0

def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.1',port))
            return True
        except OSError:
            return False

def free_port(preferred,taken=()):
    """Preferred port if free, else the next free one above it."""
    for port in range(preferred,preferred+100):
        if port not in taken and port_free(port) and not port_open(port):
            return port
    raise SystemExit(f'No free port found in {preferred}-{preferred+99}.')

def kill_children_with_us():
    """Windows: put this process in a job that kills all children when we exit,
    even if the console window is closed and no cleanup code runs."""
    if os.name!='nt':
        return
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL('kernel32',use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    class BASIC(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit',ctypes.c_int64),('PerJobUserTimeLimit',ctypes.c_int64),
                    ('LimitFlags',wintypes.DWORD),('MinimumWorkingSetSize',ctypes.c_size_t),
                    ('MaximumWorkingSetSize',ctypes.c_size_t),('ActiveProcessLimit',wintypes.DWORD),
                    ('Affinity',ctypes.c_size_t),('PriorityClass',wintypes.DWORD),('SchedulingClass',wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(n,ctypes.c_uint64) for n in ('R','W','O','RB','WB','OB')]
    class EXTENDED(ctypes.Structure):
        _fields_ = [('Basic',BASIC),('Io',IO),('ProcessMemoryLimit',ctypes.c_size_t),
                    ('JobMemoryLimit',ctypes.c_size_t),('PeakProcessMemoryUsed',ctypes.c_size_t),
                    ('PeakJobMemoryUsed',ctypes.c_size_t)]
    job = k32.CreateJobObjectW(None,None)
    info = EXTENDED()
    info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = job and k32.SetInformationJobObject(wintypes.HANDLE(job),9,ctypes.byref(info),ctypes.sizeof(info)) \
        and k32.AssignProcessToJobObject(wintypes.HANDLE(job),wintypes.HANDLE(k32.GetCurrentProcess()))
    if not ok:
        print('Note: could not bind servers to this window; close them with Ctrl+C.',flush=True)
    globals()['_JOB'] = job  # keep the handle open for our lifetime

def stop_leftovers():
    """Stop API/map servers from this folder left behind by an earlier run."""
    try:
        import psutil
    except ImportError:
        return
    me = {os.getpid()}|{p.pid for p in psutil.Process().parents()}
    root = str(ROOT).lower()
    for proc in psutil.process_iter(['pid','cmdline']):
        cmd = ' '.join(proc.info['cmdline'] or []).lower()
        if proc.info['pid'] in me or root not in cmd:
            continue
        if 'app.main:app' in cmd or ('vinext' in cmd and ' dev' in cmd):
            try:
                proc.kill()
                print(f'Stopped leftover server (PID {proc.pid}) from an earlier run.',flush=True)
            except psutil.Error:
                pass

def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    kill_children_with_us()
    stop_leftovers()
    time.sleep(0.5)  # let killed servers release their ports and lock file
    node = shutil.which('node')
    if node is None or not (ROOT/'web/node_modules').exists():
        raise SystemExit('Frontend dependencies missing: run start.cmd, or cd web; npm install')
    api_port = free_port(8000)
    web_port = free_port(5173,taken={api_port})
    for name,wanted,got in (('API',8000,api_port),('Map',5173,web_port)):
        if got!=wanted:
            print(f'{name} port {wanted} is in use; using {got} instead.',flush=True)
    url = f'http://127.0.0.1:{web_port}'
    env = dict(os.environ,PV_API_PORT=str(api_port),PV_WEB_PORT=str(web_port))
    open_browser = '--open' in sys.argv[1:]
    flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    children = []
    try:
        with (ROOT/'logs/api.log').open('a') as api_log, (ROOT/'logs/web.log').open('a') as web_log:
            api = subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(api_port)],cwd=ROOT,env=env,stdout=api_log,stderr=subprocess.STDOUT,creationflags=flags)
            children.append(api)
            # Launch the JS entry directly, avoiding a lingering npm/cmd child.
            web = subprocess.Popen([node,str(ROOT/'web/node_modules/vinext/dist/cli.js'),'dev'],cwd=ROOT/'web',env=env,stdout=web_log,stderr=subprocess.STDOUT,creationflags=flags)
            children.append(web)
            print(f'PV Visual Placer: {url}',flush=True)
            print('Logs: logs/api.log and logs/web.log. Press Ctrl+C to stop.',flush=True)
            deadline = time.time()+120 if open_browser else 0
            while all(p.poll() is None for p in children):
                if deadline and port_open(web_port):
                    webbrowser.open(url)
                    deadline = 0
                elif deadline and time.time()>deadline:
                    deadline = 0
                time.sleep(1)
            raise SystemExit('A server stopped; check logs/api.log and logs/web.log.')
    except KeyboardInterrupt:
        print('\nStopping local servers.')
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()

if __name__=='__main__':
    main()
