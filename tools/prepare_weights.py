"""Resolve and verify the checkpoint files declared in a representation preset."""

import argparse
import hashlib
import os
import re
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "conf"


def preset_path(selection):
    kind, _, name = selection.partition("=")
    if kind not in {"encoder", "representation"} or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]*", name
    ):
        raise ValueError("Use encoder=NAME or representation=NAME")
    return CONFIG / kind / f"{name}.yaml"


def sha256(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def verify(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def fetch(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        subprocess.run(
            ["curl", "--fail", "--location", "--retry", "3", "--silent",
             "--show-error", "--output", str(staged), "--", url],
            check=True,
        )
        verify(staged, expected)
        staged.chmod(0o644)
        try:
            os.link(staged, path)
        except FileExistsError:
            verify(path, expected)
    finally:
        staged.unlink(missing_ok=True)


def prepare(preset, cache, local_dir=None, verify_only=False):
    weights = yaml.safe_load(preset.read_text())["weights"]
    files = weights["files"]
    if not files:
        raise ValueError("Declare at least one checkpoint file")
    for name, expected in files.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
            raise ValueError(f"Invalid checkpoint filename: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise ValueError(f"Expected a SHA-256 digest for {name}")

    if local_dir is None:
        repository = weights["repository"]
        revision = weights["revision"]
        subfolder = weights.get("subfolder", "")
        safe_part = r"[A-Za-z0-9][A-Za-z0-9._-]*"
        if not re.fullmatch(f"{safe_part}/{safe_part}", repository):
            raise ValueError("Expected a Hugging Face repository in owner/name format")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Pin weights to a full Hugging Face commit SHA")
        if subfolder and not all(re.fullmatch(safe_part, p) for p in subfolder.split("/")):
            raise ValueError("Invalid checkpoint subfolder")
        root = cache / repository / revision / subfolder
        base_url = f"https://huggingface.co/{repository}/resolve/{revision}"
        if subfolder:
            base_url += f"/{quote(subfolder)}"
    else:
        root = local_dir

    for name, expected in files.items():
        path = root / name
        expected = str(expected)
        if path.exists() or local_dir is not None or verify_only:
            verify(path, expected)
        else:
            fetch(f"{base_url}/{quote(name)}", path, expected)
    return root.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", metavar="encoder=NAME")
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("AGRIBENCH_WEIGHTS"))
    parser.add_argument("--local-dir", type=Path, help="Verify an existing checkpoint directory")
    parser.add_argument("--verify-only", action="store_true", help="Require cached files")
    args = parser.parse_args()
    if args.cache_dir is None and args.local_dir is None:
        parser.error("Set AGRIBENCH_WEIGHTS or supply --cache-dir or --local-dir")
    try:
        root = prepare(preset_path(args.selection), args.cache_dir, args.local_dir,
                       args.verify_only)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError,
            subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
    print(root)


if __name__ == "__main__":
    main()
