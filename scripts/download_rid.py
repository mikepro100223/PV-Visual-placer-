"""Download RID reviewed masks and georeferenced imagery from its public share."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time
import argparse
import zipfile
from urllib.parse import unquote
from xml.etree import ElementTree as ET
import requests

ROOT = Path(__file__).resolve().parents[1] / 'data/raw/rid'
BASE = 'https://dataserv.ub.tum.de'
DAV = '/public.php/dav/files/m1655470/RID_dataset/'
LOCAL = threading.local()

def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.get(BASE + '/index.php/s/m1655470', timeout=60).raise_for_status()
        LOCAL.session.headers['X-Requested-With'] = 'XMLHttpRequest'
    return LOCAL.session

def listing(folder):
    r = session().request('PROPFIND', BASE + DAV + folder, headers={'Depth':'1'}, timeout=90)
    r.raise_for_status()
    return [n.text for n in ET.fromstring(r.content).findall('{DAV:}response/{DAV:}href') if n.text and not n.text.endswith('/') and n.text.rsplit('/',1)[-1].lower().endswith(('.txt','.png','.tif','.tiff','.md'))]

def fetch(href):
    rel = unquote(href).split('RID_dataset/', 1)[1]
    dest = (ROOT / rel).resolve()
    # OneDrive may return an extended-length Windows path for existing files.
    canonical_dest = Path(str(dest).removeprefix('\\\\?\\'))
    canonical_root = Path(str(ROOT.resolve()).removeprefix('\\\\?\\'))
    if not canonical_dest.is_relative_to(canonical_root):
        raise ValueError(f'Invalid remote path: {rel!r} -> {dest}')
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            r = session().get(BASE + href, timeout=(20, 120))
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + '.part')
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)

def archive_download():
    archive = ROOT.parent / 'rid_archive.zip'
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        with session().get(BASE+DAV+'?accept=zip',stream=True,timeout=(20,180)) as r:
            r.raise_for_status()
            tmp = archive.with_suffix('.zip.part')
            with tmp.open('wb') as output:
                for chunk in r.iter_content(1024*1024):
                    output.write(chunk)
            tmp.replace(archive)
    allowed = {'README_data.md','filenames_train_val_test_split','masks_superstructures_reviewed','masks_segments_reviewed','images_roof_centered_geotiff'}
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            parts = Path(member.filename).parts
            if len(parts)<2 or parts[0]!='RID_dataset' or parts[1] not in allowed or member.is_dir():
                continue
            if any(p in {'.','..'} for p in parts):
                raise ValueError('Invalid archive path')
            dest = ROOT.joinpath(*parts[1:])
            canonical_dest = Path(str(dest.resolve()).removeprefix('\\\\?\\'))
            canonical_root = Path(str(ROOT.resolve()).removeprefix('\\\\?\\'))
            if not canonical_dest.is_relative_to(canonical_root):
                raise ValueError('Archive escapes dataset directory')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(z.read(member))  # ZipFile verifies entry CRC.
    print('RID archive extracted and CRC-checked.',flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--individual',action='store_true')
    args=parser.parse_args()
    if not args.individual:
        archive_download()
        raise SystemExit(0)
    fetch(DAV + 'README_data.md')
    print((ROOT / 'README_data.md').read_text(), flush=True)
    folders = ['filenames_train_val_test_split/', 'masks_superstructures_reviewed/', 'masks_segments_reviewed/', 'images_roof_centered_geotiff/']
    with ThreadPoolExecutor(max_workers=8) as pool:
        for folder in folders:
            paths = listing(folder)
            print(f'{folder}: {len(paths)} files', flush=True)
            for i, _ in enumerate(pool.map(fetch, paths), 1):
                if i % 200 == 0 or i == len(paths):
                    print(f'{folder}: {i}/{len(paths)}', flush=True)
