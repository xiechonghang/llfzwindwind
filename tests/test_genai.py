from pathlib import Path
import json

import pytest

from hf_onnx_pipeline.genai import builder_command, check_genai


def test_local_builder_command_keeps_paths_as_arguments():
    command = builder_command(Path("/tmp/model with spaces"), Path("/tmp/output"), "fp32", "cpu")
    assert command[command.index("-i") + 1] == "/tmp/model with spaces"
    assert "-m" in command and "onnxruntime_genai.models.builder" in command
    assert "hf_remote=false" in command


def test_no_silent_architecture_substitution(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    assert not check_genai(tmp_path)["supported"]
    with pytest.raises(ValueError):
        builder_command(tmp_path, tmp_path, "int8", "cpu")
