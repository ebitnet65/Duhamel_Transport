import json
from pathlib import Path

import runtime_metadata as runtime


def test_collect_runtime_metadata_without_jax(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_jax_metadata", lambda: {"available": False})
    monkeypatch.setattr(
        runtime,
        "_git_metadata",
        lambda path: {"available": True, "commit": "abc", "dirty": False},
    )

    record = runtime.collect_runtime_metadata(
        command=["python", "benchmark.py"],
        model_metadata={"model": "dimer", "controls": 3},
        seeds={"numpy": 17},
        repo_dir=tmp_path,
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
        exit_code=0,
    )

    assert record["host"]["hostname"]
    assert record["host"]["os"]
    assert record["host"]["architecture"]
    assert record["host"]["logical_cpu_count"]
    assert record["python"]["version"]
    assert record["packages"]["jax"] == {"available": False}
    assert record["git"]["commit"] == "abc"
    assert record["run"]["command"] == ["python", "benchmark.py"]
    assert record["run"]["model"]["controls"] == 3
    assert record["run"]["seeds"] == {"numpy": 17}
    assert record["run"]["exit_code"] == 0
    assert record["run"]["elapsed_seconds"] == 1.0


def test_write_runtime_metadata_is_valid_json(tmp_path):
    output = tmp_path / "nested" / "runtime_metadata.json"
    written = runtime.write_runtime_metadata({"finite": 1.25}, output)

    assert written == output
    assert json.loads(output.read_text(encoding="utf-8")) == {"finite": 1.25}
    assert not output.with_suffix(".json.tmp").exists()


def test_model_metadata_accepts_inline_json_or_file(tmp_path):
    assert runtime._json_object('{"model": "dimer"}') == {"model": "dimer"}

    path = tmp_path / "model.json"
    path.write_text('{"controls": ["J", "k", "gamma_phi"]}', encoding="utf-8")
    assert runtime._json_object(str(path))["controls"][2] == "gamma_phi"


def test_seed_map_parses_integer_and_text_values():
    assert runtime._seed_map(["numpy=7", "dataset=truth-a"]) == {
        "numpy": 7,
        "dataset": "truth-a",
    }


def test_git_metadata_reports_non_repository(tmp_path):
    result = runtime._git_metadata(Path(tmp_path))
    assert result["available"] is False
    assert "error" in result
