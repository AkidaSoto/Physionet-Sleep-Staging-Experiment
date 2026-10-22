"""Download ISRUC-Sleep subgroup I (100 subjects, ~15 GB) from its public MEGA folder into data/raw/isruc/.

The old dataset.isr.uc.pt server is gone; the project page (sleeptight.isr.uc.pt) now links one MEGA folder. MEGA
stores files AES-encrypted, so this script talks to the MEGA API directly: list the folder, decrypt node keys and
names with the folder key, then stream each file through AES-CTR. Resumable: finished files (right size) are
skipped and partial `.part` files continue from where they stopped (HTTP Range, CTR counter offset).

The same folder may also be arriving through the MEGA desktop app (--app-dir). Any file already complete there (right
size, no `.getxfer*` temp file in its folder) is copied instead of downloaded, and a download in progress stops once
the file has appeared in the output (see scripts/sync_isruc_mega_app.py). Subjects go from the highest number down,
because the app works upwards.

Usage: python scripts/download_isruc.py [--out data/raw/isruc] [--subjects 1 2 3] [--jobs 6] [--app-dir DIR]
Requires pycryptodome.
"""
from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from Crypto.Cipher import AES
from Crypto.Util import Counter

REPO = Path(__file__).resolve().parents[1]
FOLDER_HANDLE = "QJgDQDDZ"
FOLDER_KEY = "ZMDj3w82msavACurqP48IA"
API = "https://g.api.mega.co.nz/cs"
CHUNK = 1 << 20
QUOTA_WAIT = 15 * 60
APP_DIR = Path.home() / "OneDrive/Documents/MEGA downloads/ISRUC-Sleep-I"
_app_dir: Path | None = None


def copy_from_app(path: Path, size: int, rel: str, app_dir: Path | None) -> bool:
    """Copy `rel` from the MEGA app's folder if it is complete there; True if `path` now holds the file."""
    if app_dir is None:
        return False
    src = app_dir / rel
    if not src.exists() or src.stat().st_size != size or any(src.parent.glob("*.getxfer*")):
        return False
    import shutil

    tmp = path.with_name(path.name + ".copying")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, tmp)
    if tmp.stat().st_size != size:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    path.with_name(path.name + ".part").unlink(missing_ok=True)
    return True


def b64(s: str) -> bytes:
    s = s.replace("-", "+").replace("_", "/").replace(",", "")
    return base64.b64decode(s + "=" * (-len(s) % 4))


def api(payload: list[dict], retries: int = 6) -> list:
    for k in range(retries):
        try:
            req = Request(f"{API}?id={int(time.time())}&n={FOLDER_HANDLE}", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=120) as r:
                out = json.loads(r.read())
            if isinstance(out, int):
                raise RuntimeError(f"MEGA API error {out}")
            return out
        except Exception as exc:  # noqa: BLE001
            if k == retries - 1:
                raise
            print(f"  api retry {k + 1}: {exc}", flush=True)
            time.sleep(5 * (k + 1))
    raise AssertionError


def node_key(node: dict, folder_key: bytes, root: str) -> bytes:
    """The node key encrypted with the shared folder's key (the 'root:key' entry among several)."""
    enc = b64(dict(seg.split(":", 1) for seg in node["k"].split("/"))[root])
    return AES.new(folder_key, AES.MODE_ECB).decrypt(enc)


def file_key(raw: bytes) -> tuple[bytes, bytes]:
    """32-byte MEGA file key -> (AES key, 8-byte CTR nonce)."""
    a = struct.unpack(">8I", raw)
    key = struct.pack(">4I", a[0] ^ a[4], a[1] ^ a[5], a[2] ^ a[6], a[3] ^ a[7])
    return key, struct.pack(">2I", a[4], a[5])


def name_of(node: dict, key: bytes) -> str:
    plain = AES.new(key, AES.MODE_CBC, b"\0" * 16).decrypt(b64(node["a"]))
    assert plain.startswith(b"MEGA"), "bad attribute decryption"
    return json.loads(plain[4:].rstrip(b"\0").decode())["n"]


def listing() -> list[tuple[str, str, int, bytes, bytes]]:
    """(relative path, handle, size, aes key, nonce) for every file in the folder."""
    fk = b64(FOLDER_KEY)
    nodes = api([{"a": "f", "c": 1, "r": 1, "ca": 1}])[0]["f"]
    handles = {n["h"] for n in nodes}
    root = next(n["h"] for n in nodes if n.get("p") not in handles)
    names, parents, out = {}, {}, []
    for n in nodes:
        raw = node_key(n, fk, root)
        key = file_key(raw) if n["t"] == 0 else (raw, b"")
        names[n["h"]] = name_of(n, key[0])
        parents[n["h"]] = n.get("p")
        if n["t"] == 0:
            out.append((n["h"], n["s"], *key))
    files = []
    for h, size, key, nonce in out:
        parts, p = [names[h]], parents[h]
        while p in names and parents.get(p) in names:  # drop the shared root folder's own name
            parts.append(names[p])
            p = parents[p]
        files.append(("/".join(reversed(parts)), h, size, key, nonce))
    return sorted(files)


def download(path: Path, handle: str, size: int, key: bytes, nonce: bytes) -> None:
    if path.exists() and path.stat().st_size == size:
        return
    rel = path.relative_to(path.parents[1]).as_posix()
    if copy_from_app(path, size, rel, _app_dir):
        return
    part = path.with_name(path.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while attempt < 30:
        done = part.stat().st_size // 16 * 16 if part.exists() else 0
        with open(part, "r+b" if part.exists() else "wb") as fh:
            fh.truncate(done)
        if done < size:
            try:
                url = api([{"a": "g", "g": 1, "n": handle}])[0]["g"]
                ctr = Counter.new(64, prefix=nonce, initial_value=done // 16)
                cipher = AES.new(key, AES.MODE_CTR, counter=ctr)
                req = Request(url, headers={"Range": f"bytes={done}-"})
                with urlopen(req, timeout=120) as r, open(part, "ab") as fh:
                    for i, chunk in enumerate(iter(lambda: r.read(CHUNK), b"")):
                        fh.write(cipher.decrypt(chunk))
                        if i % 16 == 0 and path.exists():  # the app copy arrived first
                            break
                if path.exists():
                    part.unlink(missing_ok=True)
                    return
            except HTTPError as exc:
                if exc.code == 509:  # free transfer quota used up for this IP; it refills over a few hours
                    wait = int(exc.headers.get("X-MEGA-Time-Left") or QUOTA_WAIT) + 30
                    print(f"  {path.name}: MEGA quota exceeded, waiting {wait // 60} min", flush=True)
                    time.sleep(wait)
                    continue
                print(f"  {path.name}: retry {attempt + 1} after {exc}", flush=True)
                time.sleep(min(300, 10 * (attempt + 1)))
                attempt += 1
                continue
            except ConnectionResetError:  # MEGA drops connections while the quota is exhausted
                print(f"  {path.name}: connection reset, waiting {QUOTA_WAIT // 60} min", flush=True)
                time.sleep(QUOTA_WAIT)
                continue
            except Exception as exc:  # noqa: BLE001
                if isinstance(getattr(exc, "reason", None), ConnectionResetError):
                    print(f"  {path.name}: connection reset, waiting {QUOTA_WAIT // 60} min", flush=True)
                    time.sleep(QUOTA_WAIT)
                    continue
                print(f"  {path.name}: retry {attempt + 1} after {exc}", flush=True)
                time.sleep(min(300, 10 * (attempt + 1)))
                attempt += 1
                continue
        if part.stat().st_size == size:
            part.replace(path)
            return
    raise RuntimeError(f"failed to download {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "data/raw/isruc"))
    ap.add_argument("--subjects", nargs="*", default=None, help="subject numbers (folder names); default all")
    ap.add_argument("--list", action="store_true", help="print the listing and exit")
    ap.add_argument("--jobs", type=int, default=6, help="files downloaded in parallel (MEGA throttles each stream)")
    ap.add_argument("--app-dir", default=str(APP_DIR), help="MEGA desktop app copy of the folder ('' for none)")
    args = ap.parse_args()
    out = Path(args.out)
    global _app_dir
    _app_dir = Path(args.app_dir) if args.app_dir and Path(args.app_dir).exists() else None
    files = sorted(listing(), key=lambda f: (-int(f[0].split("/")[0]) if f[0].split("/")[0].isdigit() else 0, f[0]))
    if args.subjects:
        want = set(args.subjects)
        files = [f for f in files if f[0].split("/")[0] in want]
    total = sum(f[2] for f in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB -> {out}", flush=True)
    if args.list:
        for rel, _, size, _, _ in files:
            print(f"{size:>12}  {rel}")
        return
    from concurrent.futures import ThreadPoolExecutor, as_completed

    got = 0
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(download, out / rel, handle, size, key, nonce): (rel, size)
                   for rel, handle, size, key, nonce in files}
        failed = []
        for i, fut in enumerate(as_completed(futures), 1):
            rel, size = futures[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                failed.append(rel)
                print(f"[{i}/{len(files)}] {rel} FAILED: {exc}", flush=True)
                continue
            got += size
            print(f"[{i}/{len(files)}] {rel} ({got / 1e9:.2f}/{total / 1e9:.2f} GB)", flush=True)
    print(f"finished; {len(failed)} failed (rerun to resume): {failed}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
