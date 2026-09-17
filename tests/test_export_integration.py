import pytest

from hf_onnx_pipeline.export import check_export, export_model
from hf_onnx_pipeline.validate import validate_model


@pytest.mark.integration
def test_real_export_and_decode_parity(tmp_path):
    """Synthetic tiny checkpoint: tests plumbing, never represents Qwen3.5 validation."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    torch.manual_seed(0)
    source = tmp_path / "tiny-gpt2"
    source.mkdir()
    config = GPT2Config(vocab_size=32, n_positions=64, n_embd=16, n_layer=1, n_head=2,
                        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    GPT2LMHeadModel(config).eval().save_pretrained(source)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", pad_token="t0", eos_token="t2").save_pretrained(source)
    task = "text-generation-with-past"
    assert check_export(source, task)["supported"]
    output = tmp_path / "onnx"
    result = export_model(source, output, task)
    assert result["status"] == "exported_structure_checked"
    validation = validate_model(output, runtime=True, reference=source)
    assert validation["numerical_validation"]["status"] == "passed"
    assert validation["numerical_validation"]["cache_tested"]
