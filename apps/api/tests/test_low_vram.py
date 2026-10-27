import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from t1000.low_vram import _replace_linears, load_low_vram_runtime


class FakeLinear4bit(nn.Linear):
    def __init__(self, in_features, out_features, *, bias, compute_dtype,
                 compress_statistics, quant_type, device):
        super().__init__(in_features, out_features, bias=bias, device=device)
        self.quantization_options = {
            "compute_dtype": compute_dtype,
            "compress_statistics": compress_statistics,
            "quant_type": quant_type,
            "device": device,
        }


def tiny_model():
    model = nn.Module()
    model.core = nn.Module()
    model.core.llm = nn.Module()
    model.core.llm.model = nn.Module()
    model.core.llm.model.embed_tokens = nn.Embedding(8, 4)
    model.core.llm.model.layers = nn.ModuleList([
        nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)),
        nn.Sequential(nn.Sequential(nn.Linear(4, 3, bias=False))),
    ])
    model.core.llm.lm_head = nn.Linear(4, 8, bias=False)
    model.core.llm.lm_head.weight = model.core.llm.model.embed_tokens.weight
    model.core.patch_encoder = nn.Linear(4, 4)
    model.core.velocity_field_predictor = nn.Linear(4, 4)
    model.vocoder = nn.Linear(4, 4)
    model.xvector_extractor = nn.Linear(4, 4)
    return model


class LowVramLoadingTests(unittest.TestCase):
    def modules(self, model, runtime):
        self.load = Mock(return_value=model)
        self.runtime = Mock(return_value=runtime)
        return {
            "bitsandbytes.nn": SimpleNamespace(Linear4bit=FakeLinear4bit),
            "dots_tts.models.dots_tts.model": SimpleNamespace(
                DotsTtsModel=SimpleNamespace(from_pretrained=self.load)
            ),
            "dots_tts.runtime": SimpleNamespace(DotsTtsRuntime=self.runtime),
        }

    def test_quantization_preserves_weights_bias_and_module_boundaries(self):
        model = tiny_model()
        original_weights = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        untouched = {
            name: module for name, module in model.named_modules()
            if not name.startswith("core.llm.model.layers")
        }
        first_norm = model.core.llm.model.layers[0][1]
        runtime = object()
        with patch.dict(sys.modules, self.modules(model, runtime)):
            result = load_low_vram_runtime("/local/model", precision="bfloat16")

        self.assertIs(result, runtime)
        self.load.assert_called_once_with(Path("/local/model"))
        converted = [module for module in model.modules() if isinstance(module, FakeLinear4bit)]
        self.assertEqual(len(converted), 2)
        self.assertIsNotNone(converted[0].bias)
        self.assertIsNone(converted[1].bias)
        for module in converted:
            self.assertEqual(module.quantization_options, {
                "compute_dtype": torch.bfloat16,
                "compress_statistics": True,
                "quant_type": "nf4",
                "device": "cpu",
            })
            self.assertFalse(module.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))
        for name, module in model.named_modules():
            if name in untouched:
                self.assertIs(module, untouched[name])
        self.assertIs(model.core.llm.model.layers[0][1], first_norm)
        self.assertIs(model.core.llm.lm_head.weight, model.core.llm.model.embed_tokens.weight)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original_weights[name], rtol=0, atol=0)

        self.runtime.assert_called_once_with(
            model, Path("/local/model"), precision="bfloat16", optimize=False,
            max_generate_length=128, max_sequence_length=512,
            vocoder_merge_steps=1, warmup_on_optimize=False,
        )

    def test_fp16_uses_fp16_quantized_layer_compute(self):
        model = tiny_model()
        with patch.dict(sys.modules, self.modules(model, object())):
            load_low_vram_runtime(Path("/local/model"), precision="float16")
        for module in model.modules():
            if isinstance(module, FakeLinear4bit):
                self.assertEqual(module.quantization_options["compute_dtype"], torch.float16)

    def test_missing_bitsandbytes_explains_installation(self):
        with patch.dict(sys.modules, {"bitsandbytes.nn": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements/gpu-low-vram.txt"):
                load_low_vram_runtime("/local/model", precision="float16")

    def test_unsupported_precision_is_rejected_before_loading(self):
        with self.assertRaisesRegex(ValueError, "requires bfloat16 or float16"):
            load_low_vram_runtime("/local/model", precision="float32")

    def test_missing_linear_layers_does_not_start_runtime(self):
        model = tiny_model()
        model.core.llm.model.layers = nn.ModuleList([nn.LayerNorm(4)])
        with patch.dict(sys.modules, self.modules(model, object())):
            with self.assertRaisesRegex(RuntimeError, "no supported linear layers"):
                load_low_vram_runtime("/local/model", precision="float16")
        self.runtime.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("bitsandbytes"), "bitsandbytes is an optional dependency")
class RealQuantizationTests(unittest.TestCase):
    def test_qwen_prefill_and_cached_decode_after_runtime_transfer_order(self):
        import bitsandbytes as bnb
        from transformers import Qwen2Config, Qwen2ForCausalLM

        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = Qwen2ForCausalLM(Qwen2Config(
                vocab_size=128, hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=128, tie_word_embeddings=True,
            )).eval()
            _replace_linears(model.model.layers, bnb.nn.Linear4bit, torch.bfloat16)
            quantized = [module for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)]
            self.assertEqual(len(quantized), 14)

            # Match DotsTtsRuntime: cast first, then transfer. An explicit CPU
            # transfer exercises real packing without requiring a CUDA device.
            model.to(dtype=torch.bfloat16)
            self.assertTrue(all(layer.weight.dtype == torch.bfloat16 for layer in quantized))
            self.assertTrue(all(not layer.weight.bnb_quantized for layer in quantized))
            model.to("cpu")
            self.assertTrue(all(layer.weight.dtype == torch.uint8 for layer in quantized))
            self.assertTrue(all(layer.weight.bnb_quantized for layer in quantized))
            self.assertTrue(all(layer.weight.quant_state is not None for layer in quantized))
            self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)

            with torch.inference_mode():
                prefill = model(input_ids=torch.tensor([[1, 3, 7, 4, 9]]), use_cache=True)
                decode = model(input_ids=torch.tensor([[5]]), past_key_values=prefill.past_key_values,
                               use_cache=True)
            self.assertEqual(tuple(prefill.logits.shape), (1, 5, 128))
            self.assertEqual(tuple(decode.logits.shape), (1, 1, 128))
            self.assertTrue(torch.isfinite(prefill.logits).all())
            self.assertTrue(torch.isfinite(decode.logits).all())


if __name__ == "__main__":
    unittest.main()
