"""Run the local map and Python API; Ctrl+C stops only these child processes."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    (ROOT/'logs').mkdir(exist_ok=True)
    npm = shutil.which('npm.cmd' if os.name=='nt' else 'npm')
    if npm is None or not (ROOT/'web/node_modules').exists():
        raise SystemExit('Install frontend dependencies first: cd web; npm install')
    flags = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    children = []
    try:
        with (ROOT/'logs/api.log').open('a') as api_log, (ROOT/'logs/web.log').open('a') as web_log:
            api = subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000'],cwd=ROOT,stdout=api_log,stderr=subprocess.STDOUT,creationflags=flags)
            children.append(api)
            # Launch the JS entry directly, avoiding a lingering npm/cmd child.
            node = shutil.which('node')
            web = subprocess.Popen([node,str(ROOT/'web/node_modules/vinext/dist/cli.js'),'dev'],cwd=ROOT/'web',stdout=web_log,stderr=subprocess.STDOUT,creationflags=flags)
            children.append(web)
            print('PV Visual Placer: http://127.0.0.1:5173',flush=True)
            print('Logs: logs/api.log and logs/web.log. Press Ctrl+C to stop.',flush=True)
            while all(p.poll() is None for p in children):
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
