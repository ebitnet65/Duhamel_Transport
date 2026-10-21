#!/usr/bin/env python3
"""Collect portable, machine-readable provenance for numerical runs.

The module deliberately treats JAX as optional.  This lets a fresh macOS or
Linux environment record useful diagnostics even before JAX is installed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for run records."""

    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except (ImportError, OSError):
        return None
    return getattr(module, "__version__", None)


def _jax_metadata() -> dict[str, Any]:
    try:
        jax = importlib.import_module("jax")
    except (ImportError, OSError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    try:
        devices = [
            {
                "id": int(getattr(device, "id", index)),
                "platform": str(getattr(device, "platform", "unknown")),
                "device_kind": str(getattr(device, "device_kind", "unknown")),
            }
            for index, device in enumerate(jax.devices())
        ]
        backend = str(jax.default_backend())
    except Exception as exc:  # Backend discovery can fail on partial installs.
        devices = []
        backend = None
        backend_error = f"{type(exc).__name__}: {exc}"
    else:
        backend_error = None

    try:
        x64_enabled = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64_enabled = None

    result: dict[str, Any] = {
        "available": True,
        "version": getattr(jax, "__version__", None),
        "backend": backend,
        "devices": devices,
        "x64_enabled": x64_enabled,
    }
    if backend_error is not None:
        result["backend_error"] = backend_error
    return result


def _git_metadata(repo_dir: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_dir), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        root = run("rev-parse", "--show-toplevel")
        branch = run("branch", "--show-current") or None
        dirty = bool(run("status", "--porcelain"))
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "root": root,
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def collect_runtime_metadata(
    *,
    command: Sequence[str] | None = None,
    model_metadata: Mapping[str, Any] | None = None,
    seeds: Mapping[str, Any] | None = None,
    repo_dir: str | Path = ".",
    started_at: str | None = None,
    finished_at: str | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    """Collect runtime, dependency, repository, and run-specific metadata."""

    uname = platform.uname()
    elapsed_seconds = None
    if started_at is not None and finished_at is not None:
        try:
            elapsed_seconds = (
                datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)
            ).total_seconds()
        except ValueError:
            pass
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at": utc_timestamp(),
        "host": {
            "hostname": socket.gethostname(),
            "os": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "processor": uname.processor or None,
            "logical_cpu_count": os.cpu_count(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "jax": _jax_metadata(),
        },
        "git": _git_metadata(Path(repo_dir).resolve()),
        "run": {
            "command": list(command) if command is not None else None,
            "working_directory": str(Path.cwd()),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_seconds": elapsed_seconds,
            "exit_code": exit_code,
            "model": dict(model_metadata or {}),
            "seeds": dict(seeds or {}),
        },
    }
    return metadata


def write_runtime_metadata(metadata: Mapping[str, Any], output: str | Path) -> Path:
    """Write a JSON record atomically and return its path."""

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)
    return output_path


def _json_object(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    candidate = Path(value)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model metadata must be a JSON object")
    return parsed


def _seed_map(values: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"seed must have NAME=VALUE form: {value!r}")
        name, raw = value.split("=", 1)
        if not name:
            raise ValueError("seed name cannot be empty")
        try:
            result[name] = int(raw)
        except ValueError:
            result[name] = raw
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--model-metadata-json")
    parser.add_argument("--seed", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--started-at")
    parser.add_argument("--finished-at")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metadata = collect_runtime_metadata(
        command=args.command or None,
        model_metadata=_json_object(args.model_metadata_json),
        seeds=_seed_map(args.seed),
        repo_dir=args.repo_dir,
        started_at=args.started_at,
        finished_at=args.finished_at,
        exit_code=args.exit_code,
    )
    write_runtime_metadata(metadata, args.output)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
