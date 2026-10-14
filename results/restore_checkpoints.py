"""Reassemble Release checkpoint parts and verify their SHA-256 checksums.

Usage: python results/restore_checkpoints.py MANIFEST DOWNLOAD_DIR OUTPUT_DIR
"""

import hashlib
import json
import sys
from pathlib import Path


def verify(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        raise ValueError(f"Missing or wrong-sized file: {path}")
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError(f"SHA-256 mismatch: {path}")


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    download_dir = Path(sys.argv[2])
    output_dir = Path(sys.argv[3])
    for entry in manifest["files"]:
        output = output_dir / entry["path"]
        if output.exists():
            verify(output, entry["size"], entry["sha256"])
            print(f"Already verified: {output}")
            continue
        for part in entry["parts"]:
            verify(download_dir / part["name"], part["size"], part["sha256"])
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_name(output.name + ".tmp")
        digest = hashlib.sha256()
        with temp.open("wb") as target:
            for part in entry["parts"]:
                with (download_dir / part["name"]).open("rb") as source:
                    while block := source.read(1024 * 1024):
                        target.write(block)
                        digest.update(block)
        if temp.stat().st_size != entry["size"] or digest.hexdigest() != entry["sha256"]:
            temp.unlink()
            raise ValueError(f"Reassembled checkpoint checksum mismatch: {output}")
        temp.replace(output)
        print(f"Verified: {output}")


if __name__ == "__main__":
    main()
