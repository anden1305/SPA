#!/usr/bin/env python3

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} did not parse to a dict.")
    return data


def _find_config(start_dir: Path, results_root: Path) -> Path | None:
    current = start_dir
    results_root = results_root.resolve()
    while True:
        candidate = current / "config.json"
        if candidate.exists():
            return candidate
        if current == results_root or current.parent == current:
            return None
        current = current.parent


def _normalize_path(path_str: str, base: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate all decoder-only CVAE checkpoints under a results folder."
    )
    parser.add_argument(
        "--results_root",
        default="results/decoder_only",
        help="Root folder to scan for checkpoints.",
    )
    parser.add_argument(
        "--config_subdir",
        default="validation_gmm",
        help="Subdirectory under each run folder to store temporary validation configs.",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python executable to use for validation runs.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop on the first failed validation instead of continuing.",
    )
    parser.add_argument(
        "--include_shared",
        action="store_true",
        help="Also validate any cvae_decoder_only_model.pth checkpoints.",
    )
    parser.add_argument(
        "--tag_prefix",
        default="",
        help="Optional prefix to prepend to the validation tag name.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    results_root = _normalize_path(args.results_root, repo_root)

    if not results_root.exists():
        print(f"Results root not found: {results_root}")
        return 1

    checkpoints: set[Path] = set(results_root.rglob("cvae_final_model_run*.pth"))
    if args.include_shared:
        checkpoints.update(results_root.rglob("cvae_decoder_only_model.pth"))

    checkpoint_list = sorted(checkpoints)
    if not checkpoint_list:
        print(f"No checkpoints found under {results_root}")
        return 0

    print(f"Found {len(checkpoint_list)} checkpoints under {results_root}")

    failures: list[Path] = []
    for index, checkpoint in enumerate(checkpoint_list, start=1):
        run_dir = checkpoint.parent
        config_path = _find_config(run_dir, results_root)
        if config_path is None:
            print(f"[{index}/{len(checkpoint_list)}] Skipping (no config.json): {checkpoint}")
            continue

        config = _load_config(config_path)
        cvae_cfg = config.get("cvae") or {}
        config["cvae"] = cvae_cfg
        cvae_cfg["model_checkpoint_path"] = str(checkpoint.resolve())

        validation_tag = f"{args.tag_prefix}{checkpoint.stem}"
        config_dir = run_dir / args.config_subdir
        config_dir.mkdir(parents=True, exist_ok=True)
        temp_config = config_dir / f"validation_config_{checkpoint.stem}.json"
        with temp_config.open("w") as handle:
            json.dump(config, handle, indent=2)

        print(f"[{index}/{len(checkpoint_list)}] Validating {checkpoint}")
        try:
            subprocess.run(
                [
                    args.python,
                    "main.py",
                    "--method",
                    "validate_cvae_gmm",
                    "--config_path",
                    str(temp_config),
                    "--validation_tag",
                    validation_tag,
                ],
                check=True,
                cwd=repo_root,
            )
        except subprocess.CalledProcessError as exc:
            print(f"Validation failed for {checkpoint}: {exc}")
            failures.append(checkpoint)
            if args.stop_on_error:
                break

    if failures:
        print(f"{len(failures)} validations failed.")
        for checkpoint in failures:
            print(f"- {checkpoint}")
        return 1

    print("All validations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
