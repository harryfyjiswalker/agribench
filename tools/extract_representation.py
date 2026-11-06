"""Run a registered adapter and write feature files for the PASTIS-R probe."""

import argparse
import importlib
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

from .prepare_weights import CONFIG, prepare, preset_path, sha256


def resolve_config(selection, input_selection):
    preset = preset_path(selection)
    config = yaml.safe_load(preset.read_text())
    if selection.startswith("representation="):
        if input_selection is not None:
            raise ValueError("Use encoder=NAME to select a separate input")
        return preset, preset.stem, config, []

    if input_selection is None or not re.fullmatch(
        r"input=pastis-r/[A-Za-z0-9][A-Za-z0-9_-]*", input_selection
    ):
        raise ValueError("Pass input=pastis-r/NAME")
    input_id = input_selection.removeprefix("input=")
    input_config = yaml.safe_load((CONFIG / "input" / f"{input_id}.yaml").read_text())
    if config["input_format"] != input_config["format"]:
        raise ValueError(f"Encoder expects {config['input_format']}, got {input_config['format']}")
    name = f"{preset.stem}--{input_id.replace('/', '-')}"
    config = {
        **config,
        "encoder": preset.stem,
        "input": input_id,
        "input_profile": input_config["input_profile"],
        "inputs": input_config["inputs"],
        "preprocessing": input_config["preprocessing"],
        "input_config": input_config,
    }
    return preset, name, config, input_config["required_arrays"]


def verify_source(source, source_dir):
    if source is None:
        if source_dir is not None:
            raise ValueError("Declare a source revision in the preset to use --source-dir")
        return
    if source_dir is None:
        raise ValueError("Pass --source-dir pointing to the pinned upstream checkout")
    git = ["git", "-C", str(source_dir)]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    if revision != source["revision"]:
        raise ValueError(f"Upstream revision is {revision}, expected {source['revision']}")
    if subprocess.check_output([*git, "status", "--porcelain", "-uno"], text=True):
        raise ValueError("The upstream checkout has tracked changes")


def check_features(ids, features, expected_ids):
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("The adapter must return a one-dimensional integer ID array")
    order = np.argsort(ids)
    if not np.array_equal(ids[order], expected_ids):
        raise ValueError("Adapter output must contain each PASTIS-R parcel ID exactly once")
    if (features.ndim != 2 or features.shape[0] != len(ids)
            or features.shape[1] == 0 or features.dtype.kind != "f"):
        raise ValueError("Expected a floating-point feature matrix with one row per parcel")
    features = features[order].astype(np.float32, copy=False)
    if np.isinf(features).any() or np.isnan(features).all(axis=1).any():
        raise ValueError("Features contain infinities or parcels without observed features")
    return features


def save(output_dir, name, ids, features, recipe):
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / f"{name}.npz"
    recipe_path = output_dir / f"{name}.json"
    with TemporaryDirectory(dir=output_dir) as temporary:
        staged = Path(temporary)
        np.savez(staged / feature_path.name, ids=ids, features=features)
        (staged / recipe_path.name).write_text(json.dumps(recipe, indent=2) + "\n")
        os.link(staged / recipe_path.name, recipe_path)
        try:
            os.link(staged / feature_path.name, feature_path)
        except OSError:
            recipe_path.unlink()
            raise


def smoke(adapter, options, inputs, arrays, sample_arrays, expected_ids):
    if "ids" not in sample_arrays or not set(sample_arrays).issubset(arrays):
        raise ValueError("For --smoke, declare sample_arrays including ids in the input YAML")
    with np.load(inputs, allow_pickle=False) as table:
        ids = table["ids"]
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
            raise ValueError("Input ids must be a nonempty one-dimensional integer array")
        selected = np.argsort(ids)[:8]
        sample_ids = ids[selected]
        if len(np.unique(sample_ids)) != len(sample_ids) or not np.isin(sample_ids, expected_ids).all():
            raise ValueError("Smoke sample contains duplicate or unknown parcel IDs")
        sample = {}
        for key in arrays:
            values = table[key]
            if key in sample_arrays:
                if values.ndim == 0 or values.shape[0] != len(ids):
                    raise ValueError(f"Sample array {key} must have one row per input ID")
                values = values[selected]
            sample[key] = values
    with TemporaryDirectory() as directory:
        batch_path = Path(directory) / "sample.npz"
        np.savez(batch_path, **sample)
        result_ids, features = adapter(inputs=batch_path, **options)
        return check_features(result_ids, features, sample_ids).shape


def extract(args):
    preset, name, config, required_arrays = resolve_config(args.selection, args.input)
    for field in ("adapter", "version", "input_profile", "inputs", "preprocessing", "pooling"):
        if field not in config:
            raise ValueError(f"Preset is missing {field}")
    if args.check_config:
        print(f"Compatible configuration: {name}")
        return
    sample_arrays = config.get("input_config", {}).get("sample_arrays", [])
    if args.smoke and not sample_arrays:
        raise ValueError("--smoke requires a separate input YAML that declares sample_arrays")
    if not args.smoke and any(
        (args.output_dir / f"{name}.{ext}").exists() for ext in ("npz", "json")
    ):
        raise ValueError(f"Output for {name} already exists in {args.output_dir}")
    if not args.inputs.is_file():
        raise FileNotFoundError(args.inputs)
    if required_arrays:
        with np.load(args.inputs, allow_pickle=False) as table:
            missing = set(required_arrays).difference(table.files)
            if missing:
                raise ValueError(f"Input NPZ is missing arrays: {', '.join(sorted(missing))}")
            expected = config["input_config"].get("expected_provenance", {})
            if expected:
                provenance = json.loads(table["provenance"].item())
                for field, value in expected.items():
                    if provenance.get(field) != value:
                        raise ValueError(f"Input NPZ provenance does not match {field}")

    parcel_ids = pd.read_csv(args.metadata, usecols=["ID_PARCEL"], dtype="Int64").ID_PARCEL
    if len(parcel_ids) != 85_951 or parcel_ids.isna().any() or parcel_ids.duplicated().any():
        raise ValueError("Expected 85,951 unique PASTIS-R parcel IDs")
    expected_ids = np.sort(parcel_ids.to_numpy(dtype=np.int64))

    weights = None
    if "weights" in config:
        if args.weights_dir is None and args.cache_dir is None:
            raise ValueError("Pass --weights-dir or --cache-dir with prepared weights")
        weights = prepare(preset, args.cache_dir, args.weights_dir, verify_only=True)
    elif args.weights_dir is not None:
        raise ValueError("Declare checkpoint hashes in the preset to use --weights-dir")
    verify_source(config.get("source"), args.source_dir)

    module_name, separator, function_name = config["adapter"].partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Use module:function for the adapter")
    module = importlib.import_module(module_name)
    adapter = getattr(module, function_name)
    options = dict(
        weights_dir=weights, source_dir=args.source_dir, device=args.device,
        batch_size=args.batch_size,
    )
    if args.smoke:
        rows, width = smoke(adapter, options, args.inputs, required_arrays, sample_arrays, expected_ids)
        print(f"Smoke passed: {name} returned {rows} vectors of width {width}")
        return
    ids, features = adapter(inputs=args.inputs.resolve(), **options)
    features = check_features(ids, features, expected_ids)
    recipe = {
        **config, "name": name, "device": args.device, "batch_size": args.batch_size,
        "input_sha256": sha256(args.inputs),
        "metadata_sha256": sha256(args.metadata),
        "adapter_sha256": sha256(Path(module.__file__)),
    }
    save(args.output_dir, name, expected_ids, features, recipe)
    print(f"Saved {features.shape[0]:,} parcel vectors to {args.output_dir / (name + '.npz')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", metavar="encoder=NAME")
    parser.add_argument("input", nargs="?", metavar="input=pastis-r/NAME")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-config", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="Check up to eight parcels")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=os.environ.get("AGRIBENCH_WEIGHTS"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.check_config and None in (args.inputs, args.metadata):
        parser.error("Extraction and smoke mode require --inputs and --metadata")
    if not (args.check_config or args.smoke) and args.output_dir is None:
        parser.error("Full extraction requires --output-dir")
    try:
        extract(args)
    except (OSError, ValueError, KeyError, ImportError, AttributeError,
            subprocess.CalledProcessError, yaml.YAMLError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
