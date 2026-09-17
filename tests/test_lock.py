import os

import pytest

from hf_onnx_pipeline.common import directory_lock


def test_competing_operation_preserves_owner_lock(tmp_path):
    with directory_lock(tmp_path):
        lock = tmp_path / ".hf-onnx.lock"
        owner = lock.read_bytes()
        with pytest.raises(RuntimeError, match=f"recorded PID: {os.getpid()}"):
            with directory_lock(tmp_path):
                pytest.fail("Competing operation acquired the lock")
        assert lock.read_bytes() == owner
    assert not lock.exists()
