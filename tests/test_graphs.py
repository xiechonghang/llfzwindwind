from pathlib import Path
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from hf_onnx_pipeline.graphs import external_files, inspect_graphs
from hf_onnx_pipeline.download import execute_download
from hf_onnx_pipeline.validate import validate_model


def make_external_graph(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    weight = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "w")
    graph = helper.make_graph([helper.make_node("Add", ["x", "w"], ["y"])], "add",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.save_model(model, root / "model.onnx", save_as_external_data=True,
                    all_tensors_to_one_file=True, location="arbitrary-weights.payload", size_threshold=0)


def test_external_data_presence(tmp_path):
    make_external_graph(tmp_path)
    assert external_files(tmp_path / "model.onnx", tmp_path) == ["arbitrary-weights.payload"]
    assert inspect_graphs(tmp_path, ["model.onnx"])["graphs"][0]["structure"] == "passed"
    (tmp_path / "arbitrary-weights.payload").unlink()
    with pytest.raises(ValueError, match="Missing external"):
        inspect_graphs(tmp_path, ["model.onnx"])


def test_runtime_load(tmp_path):
    pytest.importorskip("onnxruntime")
    make_external_graph(tmp_path)
    assert inspect_graphs(tmp_path, ["model.onnx"], runtime=True)["graphs"][0]["runtime_load"] == "passed"


def test_validation_uses_selected_snapshot_graphs(tmp_path):
    make_external_graph(tmp_path)
    (tmp_path / "old_variant.onnx").write_bytes(b"stale invalid graph")
    (tmp_path / ".hf-onnx-download.json").write_text(json.dumps({"status": "download_complete", "graphs": ["model.onnx"]}))
    result = validate_model(tmp_path)
    assert [g["file"] for g in result["graphs"]] == ["model.onnx"]


def test_external_path_traversal_rejected(tmp_path):
    make_external_graph(tmp_path)
    model = onnx.load(tmp_path / "model.onnx", load_external_data=False)
    model.graph.initializer[0].external_data[0].value = "../../secret"
    (tmp_path / "model.onnx").write_bytes(model.SerializeToString())
    with pytest.raises(ValueError):
        external_files(tmp_path / "model.onnx", tmp_path)


def test_download_follows_external_references(tmp_path):
    source = tmp_path / "source"
    make_external_graph(source)
    names = ["model.onnx", "arbitrary-weights.payload"]
    plan = {"repo_id": "org/model", "revision": "abc", "kind": "onnx", "graphs": ["model.onnx"],
            "destination": str(tmp_path / "target"), "files": ["model.onnx"],
            "sizes": {"model.onnx": (source / "model.onnx").stat().st_size},
            "known_download_bytes": (source / "model.onnx").stat().st_size, "unknown_size_files": 0}
    def transfer(**kwargs):
        for name in kwargs["allow_patterns"]:
            (kwargs["local_dir"] / name).write_bytes((source / name).read_bytes())
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=(source / n).stat().st_size) for n in names])
    with patch("huggingface_hub.snapshot_download", side_effect=transfer), patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.return_value = info
        state = execute_download(plan)
    assert state["files"] == sorted(names)
    assert state["validation"] == "not_run"


def test_external_weight_budget_is_enforced_before_payload(tmp_path):
    source = tmp_path / "source"
    make_external_graph(source)
    graph_bytes = (source / "model.onnx").stat().st_size
    plan = {"repo_id": "org/model", "revision": "abc", "kind": "onnx", "graphs": ["model.onnx"],
            "destination": str(tmp_path / "target"), "files": ["model.onnx"],
            "sizes": {"model.onnx": graph_bytes}, "known_download_bytes": graph_bytes, "unknown_size_files": 0}
    def transfer(**kwargs):
        for name in kwargs["allow_patterns"]:
            (kwargs["local_dir"] / name).write_bytes((source / name).read_bytes())
    info = SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=(source / n).stat().st_size)
                                    for n in ("model.onnx", "arbitrary-weights.payload")])
    with patch("huggingface_hub.snapshot_download", side_effect=transfer), patch("huggingface_hub.HfApi") as api:
        api.return_value.model_info.return_value = info
        with pytest.raises(ValueError, match="External weights exceed"):
            execute_download(plan, max_bytes=graph_bytes)
    assert not (tmp_path / "target/arbitrary-weights.payload").exists()
