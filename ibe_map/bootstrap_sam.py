"""Fetch official SAM ViT-B weights and a pinned, torch-only source subset."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'third_party' / 'segment_anything'
WEIGHTS = ROOT / 'output/ibe_map/weights/sam_vit_b_01ec64.pth'
URL = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
FILES = ['LICENSE', 'segment_anything/build_sam.py',
         'segment_anything/modeling/__init__.py', 'segment_anything/modeling/sam.py',
         'segment_anything/modeling/common.py', 'segment_anything/modeling/image_encoder.py',
         'segment_anything/modeling/mask_decoder.py', 'segment_anything/modeling/prompt_encoder.py',
         'segment_anything/modeling/transformer.py']


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + '.part')
    for attempt in range(5):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'User-Agent': 'UEVAVD-reproduction'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=45) as src:
                resumed = src.status == 206 and offset > 0
                if resumed and not src.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                    raise ValueError('Unexpected download range response')
                print(f'Download {path.name}: resume={resumed}, offset={offset}', flush=True)
                received = 0
                with part.open('ab' if resumed else 'wb') as dest:
                    while chunk := src.read(1024 * 1024):
                        dest.write(chunk)
                        received += len(chunk)
                expected = src.headers.get('Content-Length')
                if expected and received != int(expected):
                    raise OSError(f'Incomplete download: {received}/{expected} response bytes')
            part.replace(path)
            return
        except (OSError, TimeoutError) as exc:
            if attempt == 4:
                raise
            print(f'Download interrupted; retrying saved bytes: {exc}', flush=True)
            time.sleep(2)


def main():
    manifest_path = VENDOR / 'SOURCE.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        commit = manifest['commit']
    else:
        req = urllib.request.Request('https://api.github.com/repos/facebookresearch/segment-anything/commits/main',
                                     headers={'User-Agent': 'UEVAVD-reproduction'})
        with urllib.request.urlopen(req, timeout=30) as r:
            commit = json.load(r)['sha']
        manifest = {'repository': 'https://github.com/facebookresearch/segment-anything', 'commit': commit}
    print('Official SAM source commit:', commit, flush=True)
    for name in FILES:
        dest = VENDOR / name
        if not dest.exists():
            download(f'https://raw.githubusercontent.com/facebookresearch/segment-anything/{commit}/{name}', dest)
        if name in manifest.get('files_sha256', {}) and sha256(dest) != manifest['files_sha256'][name]:
            raise ValueError(f'Pinned SAM source changed: {name}')
    # Importing the upstream top-level package eagerly imports its automatic
    # generator and torchvision NMS. Our bbox/point decoder uses neither.
    init = VENDOR / 'segment_anything/__init__.py'
    init.write_text('# Local import-only adapter; modeling files are unmodified upstream.\n'
                    'from .build_sam import sam_model_registry, build_sam_vit_b\n', encoding='utf8')
    manifest['local_change'] = 'Top-level __init__ only imports build_sam, avoiding unused torchvision automatic-mask-generator dependency.'
    manifest['files_sha256'] = {name: sha256(VENDOR / name) for name in FILES}
    manifest['adapter_sha256'] = sha256(init)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    if not WEIGHTS.exists():
        print('Downloading official SAM ViT-B weights (~375 MB)', flush=True)
        download(URL, WEIGHTS)
    digest = sha256(WEIGHTS)
    weight_manifest = WEIGHTS.with_suffix('.json')
    if weight_manifest.exists() and json.loads(weight_manifest.read_text())['sha256'] != digest:
        raise ValueError('SAM checkpoint differs from previously recorded download')
    weight_manifest.write_text(json.dumps({'url': URL, 'sha256': digest, 'bytes': WEIGHTS.stat().st_size,
                                           'model': 'SAM ViT-B; official pretrained, no UEVAVD finetuning'}, indent=2))
    print('SAM checkpoint SHA256:', digest, flush=True)


if __name__ == '__main__':
    main()
