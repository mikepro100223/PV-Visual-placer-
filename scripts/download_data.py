"""Fetch the public Swiss PV training archive; never execute archive contents."""
from pathlib import Path
import argparse
import requests
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = destination.with_suffix(destination.suffix + '.part')
    with requests.get(url, stream=True, timeout=(20, 180)) as response:
        response.raise_for_status()
        with temporary.open('wb') as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    temporary.replace(destination)
    print(f'Downloaded {destination.name}: {destination.stat().st_size:,} bytes', flush=True)
    return destination

def extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('Archive path escapes destination')
            # Dataset may include other authors models: only extract training data.
            if target.suffix.lower() in {'.pt', '.pth', '.pkl', '.h5'}:
                continue
            z.extract(entry, destination)
        print(f'Archive has {len(z.infolist())} entries', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--swiss', action='store_true')
    args = parser.parse_args()
    if args.swiss:
        path = download('https://www.kaggle.com/api/v1/datasets/download/jeanprbt/swiss-solar-panels-segmentation?datasetVersionNumber=1', ROOT / 'data/raw/swiss.zip')
        extract(path, ROOT / 'data/raw/swiss')
