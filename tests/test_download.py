import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hf_onnx_pipeline.common import repo_parts, safe_path
from hf_onnx_pipeline.download import choose_files, pinned_revision, check_weights, plan_download, execute_download, MANIFEST


def test_onnx_preferred_without_original_weights():
    selection = choose_files(["config.json", "model.safetensors", "onnx/model.onnx", "onnx/weights.data"], "auto")
    assert selection["kind"] == "onnx"
    assert "model.safetensors" not in selection["files"]
    assert selection["graphs"] == ["onnx/model.onnx"]


def test_original_fallback_includes_every_shard():
    files = ["config.json", "model-1.safetensors", "model-2.safetensors", "model.safetensors.index.json"]
    assert choose_files(files, "auto")["files"] == sorted(files)


def test_multiple_variants_require_selection():
    files = ["model.onnx", "model_q4.onnx", "config.json"]
    with pytest.raises(ValueError, match="Multiple"):
        choose_files(files, "auto")
    assert choose_files(files, "auto", ["model_q4.onnx"])["graphs"] == ["model_q4.onnx"]


def test_multicomponent_selection():
    files = ["decoder.onnx", "vision.onnx", "config.json"]
    assert choose_files(files, "onnx", files[:2])["graphs"] == files[:2]


def test_unsafe_paths_and_ids(tmp_path):
    for p in ("../secret", "/secret", "x\\y"):
        with pytest.raises(ValueError):
            safe_path(tmp_path, p)
    with pytest.raises(ValueError):
        repo_parts("../model")
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_path(tmp_path, "link/secret")


def test_existing_metadata_revision_is_reused(tmp_path):
    (tmp_path / "metadata_manifest.json").write_text(json.dumps({"repo_id": "org/model", "revision": "abc"}))
    assert pinned_revision(tmp_path, "org/model", None) == "abc"
    with pytest.raises(ValueError):
        pinned_revision(tmp_path, "other/model", None)


def test_missing_weight_shard(tmp_path):
    (tmp_path / "model-1.safetensors").write_bytes(b"test")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model-2.safetensors"}}))
    with pytest.raises(ValueError, match="Missing weight shard"):
        check_weights(tmp_path)


def test_revision_conflict_before_transfer(tmp_path):
    root = tmp_path / "org/model"
    root.mkdir(parents=True)
    (root / MANIFEST).write_text(json.dumps({"repo_id": "org/model", "revision": "old"}))
    info = SimpleNamespace(sha="new", siblings=[SimpleNamespace(rfilename="config.json", size=5)])
    with patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.return_value = info
        with pytest.raises(ValueError, match="mix"):
            plan_download("org/model", tmp_path, revision="main")


def test_interrupted_transfer_records_failure_and_resumes(tmp_path):
    plan = {"repo_id": "org/model", "revision": "abc", "destination": str(tmp_path / "model"),
            "kind": "original", "graphs": [], "files": ["model.safetensors"],
            "sizes": {"model.safetensors": 4}, "known_download_bytes": 4, "unknown_size_files": 0}
    with patch("huggingface_hub.snapshot_download", side_effect=OSError("interrupted")):
        with pytest.raises(OSError):
            execute_download(plan)
    assert json.loads((Path(plan["destination"]) / MANIFEST).read_text())["status"] == "failed"
    def transfer(**kwargs):
        (kwargs["local_dir"] / "model.safetensors").write_bytes(b"test")
    with patch("huggingface_hub.snapshot_download", side_effect=transfer):
        assert execute_download(plan)["status"] == "download_complete"


def test_budget_stops_before_transfer(tmp_path):
    plan = {"destination": str(tmp_path), "known_download_bytes": 1000, "unknown_size_files": 0}
    with patch("huggingface_hub.snapshot_download") as download:
        with pytest.raises(ValueError, match="exceeds"):
            execute_download(plan, max_bytes=100)
        download.assert_not_called()
