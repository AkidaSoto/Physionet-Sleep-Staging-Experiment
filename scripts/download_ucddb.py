from __future__ import annotations

import argparse
import os
import posixpath
import sys
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


BASE_URL = "https://physionet.org/files/ucddb/1.0.0/"
USER_AGENT = "PhysioNetUCDDBDownloader/1.0"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def list_links(url: str) -> list[str]:
    parser = LinkParser()
    parser.feed(fetch_text(url))
    return parser.links


def is_same_tree(base_url: str, candidate_url: str) -> bool:
    base = urlparse(base_url)
    candidate = urlparse(candidate_url)
    return (
        base.scheme == candidate.scheme
        and base.netloc == candidate.netloc
        and candidate.path.startswith(base.path)
    )


def relative_path(base_url: str, file_url: str) -> str:
    base_path = urlparse(base_url).path
    full_path = urlparse(file_url).path
    return posixpath.relpath(full_path, start=base_path)


def iter_remote_files(base_url: str):
    queue: deque[str] = deque([base_url])
    seen_dirs: set[str] = set()
    seen_files: set[str] = set()

    while queue:
        current = queue.popleft()
        if current in seen_dirs:
            continue
        seen_dirs.add(current)

        for href in list_links(current):
            if href.startswith("#") or href.startswith("?"):
                continue
            next_url = urljoin(current, href)
            if not is_same_tree(base_url, next_url):
                continue
            parsed = urlparse(next_url)
            if parsed.path.endswith("/"):
                if next_url != current:
                    queue.append(next_url)
                continue
            if next_url not in seen_files:
                seen_files.add(next_url)
                yield next_url


def remote_size(url: str) -> int | None:
    request = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request) as response:
            value = response.headers.get("Content-Length")
            return int(value) if value else None
    except Exception:
        return None


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_size = remote_size(url)
    if destination.exists() and expected_size is not None and destination.stat().st_size == expected_size:
        print(f"skip  {destination}", flush=True)
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request) as response, open(tmp_path, "wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(tmp_path, destination)
    print(f"saved {destination}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the PhysioNet UCDDB dataset.")
    parser.add_argument(
        "--dest",
        default="data/raw/ucddb",
        help="Local destination directory.",
    )
    args = parser.parse_args()

    destination_root = Path(args.dest).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {BASE_URL} into {destination_root}", flush=True)

    try:
        for file_url in iter_remote_files(BASE_URL):
            rel_path = relative_path(BASE_URL, file_url)
            local_path = destination_root / Path(rel_path)
            download_file(file_url, local_path)
    except KeyboardInterrupt:
        print("download interrupted", file=sys.stderr, flush=True)
        return 130

    print("download complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
