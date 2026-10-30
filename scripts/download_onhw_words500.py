"""Download the right-handed writer-independent OnHW-Words500 ZIP from HF mirror.

OnHW is CC BY-NC 4.0; do not commit the archive. This does not unpickle.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.parse
import urllib.request
import zipfile

REPO = "FelixOtt/OnHW_datasets"
BASE_API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
BASE_FILE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"

def remote_archive_name() -> str:
    req = urllib.request.Request(BASE_API+"?recursive=true&limit=1000",
                                 headers={"User-Agent":"IMUpen-OnHW-M4/1.0"})
    with urllib.request.urlopen(req,timeout=45) as response:
        records = json.load(response)
    names = [x.get("path","") for x in records if x.get("type")=="file"]
    matches = [x for x in names if "words500" in x.lower() and
               "indep" in x.lower() and x.lower().endswith(".zip") and
               not x.lower().endswith("_l.zip")]
    if len(matches)!=1:
        raise RuntimeError(f"Expected one right-handed Words500 indep ZIP, got {matches}")
    return matches[0]

def safe_unzip(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True,exist_ok=True)
    root=dest.resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name=PurePosixPath(item.filename)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe ZIP member {item.filename!r}")
            output=(dest/Path(*name.parts)).resolve()
            if not output.is_relative_to(root):
                raise ValueError(f"ZIP escapes extraction path: {item.filename!r}")
            mode=item.external_attr>>16
            if (mode & 0o170000)==0o120000:
                raise ValueError(f"ZIP symlink not allowed: {item.filename!r}")
        z.extractall(dest)

def main():
    p=argparse.ArgumentParser(description="Download OnHW-Words500 indep (noncommercial research)")
    p.add_argument("--dest",type=Path,default=Path("datasets/onhw"))
    p.add_argument("--url",help="Manual verified source URL if HF layout differs")
    args=p.parse_args()
    args.dest.mkdir(parents=True,exist_ok=True)
    name=remote_archive_name() if args.url is None else None
    url=args.url or BASE_FILE+urllib.parse.quote(name,safe="/")
    archive=args.dest/"OnHW-Words500_indep.zip"
    if not archive.exists():
        print(f"Downloading {url}",flush=True)
        with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"IMUpen-OnHW-M4/1.0"}),timeout=120) as response,archive.open("wb") as output:
            shutil.copyfileobj(response,output)
    print(f"ZIP bytes: {archive.stat().st_size}",flush=True)
    safe_unzip(archive,args.dest/"extracted")
    print(f"Extracted to: {args.dest/'extracted'}",flush=True)

if __name__=="__main__":
    main()
