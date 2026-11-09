"""Check a local copy of a Hugging Face repo against the Hub's own file list, file by file.

For every file of the repo at a pinned revision, the Hub reports its size and a content hash: SHA-256 for
files stored with Git LFS/Xet, and the git blob SHA-1 for small files kept in git. This script compares
each local file with both, reports missing, extra and mismatched files, and writes a manifest
(``MANIFEST.sha256``, ``sha256sum -c`` format) so later checks need no network.

    python tools/verify_hf_snapshot.py allenai/planteur-benchmark --repo-type dataset \
        --revision <commit sha> --local-dir /lustre/shared/agriscience-benchmark/datasets/planteur

Exit code 0 only when every file is present with the right size and hash.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi

CHUNK = 8 * 1024 * 1024


def hashes(path: Path) -> tuple[str, str]:
    """SHA-256 of the content and git blob SHA-1 (sha1 of b'blob <size>\\0' + content), in one read."""
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(b"blob %d\0" % path.stat().st_size)
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_id")
    parser.add_argument("--repo-type", default="dataset")
    parser.add_argument("--revision", required=True, help="pin a commit sha, not a branch")
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[], help="glob of Hub paths to skip, as for hf download")
    parser.add_argument("--manifest", default="MANIFEST.sha256", help="written inside --local-dir")
    args = parser.parse_args()

    root = args.local_dir
    remote = [
        entry
        for entry in HfApi().list_repo_tree(
            args.repo_id, repo_type=args.repo_type, revision=args.revision, recursive=True
        )
        if hasattr(entry, "size") and hasattr(entry, "blob_id")  # files only, not folders
        and not any(fnmatch.fnmatch(entry.path, pattern) for pattern in args.exclude)
    ]
    print(f"{args.repo_id}@{args.revision}: {len(remote)} files, "
          f"{sum(e.size for e in remote) / 1e9:.1f} GB on the Hub"
          + (f" after excluding {args.exclude}" if args.exclude else ""), flush=True)

    missing, bad, manifest = [], [], []
    for i, entry in enumerate(sorted(remote, key=lambda e: e.path), 1):
        path = root / entry.path
        if not path.is_file():
            missing.append(entry.path)
            continue
        if path.stat().st_size != entry.size:
            bad.append(f"{entry.path}: size {path.stat().st_size} != {entry.size}")
            continue
        sha256, blob_sha1 = hashes(path)
        expected = entry.lfs.sha256 if entry.lfs else None
        if (expected and sha256 != expected) or (not expected and blob_sha1 != entry.blob_id):
            bad.append(f"{entry.path}: hash mismatch")
            continue
        manifest.append(f"{sha256}  {entry.path}")
        if i % 5000 == 0:
            print(f"  checked {i}/{len(remote)}", flush=True)

    wanted = {e.path for e in remote} | {args.manifest}
    extra = [
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and ".cache" not in p.relative_to(root).parts and str(p.relative_to(root)) not in wanted
        and not any(fnmatch.fnmatch(str(p.relative_to(root)), pattern) for pattern in args.exclude)
    ]

    print(f"ok {len(manifest)}, missing {len(missing)}, bad {len(bad)}, extra {len(extra)}")
    for label, items in (("MISSING", missing), ("BAD", bad), ("EXTRA", extra)):
        for item in items[:20]:
            print(f"  {label} {item}")
        if len(items) > 20:
            print(f"  ... {len(items) - 20} more {label.lower()}")

    if missing or bad:
        return 1
    (root / args.manifest).write_text("\n".join(manifest) + "\n")
    print(f"all {len(manifest)} files verified; wrote {root / args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
