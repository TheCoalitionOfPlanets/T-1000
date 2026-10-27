import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from t1000.offload import _OffloadedModel, load_offloaded_runtime


class TrackedModule(nn.Module):
    def __init__(self, name, events, *, replace_weight_on_move=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 2))
        self.bias = None
        self.name = name
        self.events = events
        self.replace_weight_on_move = replace_weight_on_move

    def to(self, *args, **kwargs):
        if args and torch.device(args[0]).type == "cuda":
            self.events.append((self.name, str(args[0]), self.weight.dtype))
            if self.replace_weight_on_move:
                # Simulate Parameter replacement on a real CPU-to-CUDA move.
                self.weight = nn.Parameter(self.weight.detach().clone())
            return self
        return super().to(*args, **kwargs)


class GuardedParent(nn.Module):
    def to(self, *args, **kwargs):
        if args and isinstance(args[0], (str, torch.device)):
            if torch.device(args[0]).type == "cuda":
                raise AssertionError("A complete parent must never move to CUDA.")
        return super().to(*args, **kwargs)


class FakeLLM(GuardedParent):
    def __init__(self, events):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = TrackedModule(
            "embedding", events, replace_weight_on_move=True
        )
        self.model.layers = nn.ModuleList(
            [TrackedModule(f"block-{i}", events) for i in range(4)]
        )
        self.model.norm = TrackedModule("norm", events)
        self.model.rotary_emb = TrackedModule("rotary", events)
        self.lm_head = TrackedModule("head", events)
        self.tie_weights()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class FakeModel(GuardedParent):
    def __init__(self, events):
        super().__init__()
        self.created_in_inference_mode = torch.is_inference_mode_enabled()
        self.core = GuardedParent()
        self.core.llm = FakeLLM(events)
        self.core.acoustic = TrackedModule("acoustic", events)
        self.vocoder = TrackedModule("vocoder", events)
        self.xvector_extractor = TrackedModule("speaker", events)
        self.config = SimpleNamespace(sample_rate=48000)

    def generate_stream(self, text):
        return iter([text])


class OffloadPlacementTests(unittest.TestCase):
    def test_only_selected_blocks_are_resident_and_embeddings_stay_tied(self):
        events = []
        original = FakeModel(events)
        original.core.to(dtype=torch.float16)
        offload = Mock()
        wrapper = _OffloadedModel(original, gpu_layers=2, cpu_offload=offload)

        self.assertIs(wrapper.to(torch.device("cuda")), wrapper)
        self.assertEqual(
            [event[0] for event in events],
            ["acoustic", "vocoder", "speaker", "embedding", "norm", "rotary",
             "block-0", "block-1"],
        )
        self.assertIs(
            original.core.llm.lm_head.weight,
            original.core.llm.model.embed_tokens.weight,
        )
        blocks = original.core.llm.model.layers
        self.assertEqual([call.args[0] for call in offload.call_args_list], list(blocks[2:]))
        for call in offload.call_args_list:
            self.assertEqual(call.kwargs, {
                "execution_device": torch.device("cuda"), "offload_buffers": True,
            })
        # Core precision changes before placement, while the original runtime's
        # vocoder and speaker modules retain their full-precision weights.
        self.assertTrue(all(block.weight.dtype == torch.float16 for block in blocks))
        self.assertEqual(original.vocoder.weight.dtype, torch.float32)
        self.assertEqual(original.xvector_extractor.weight.dtype, torch.float32)
        self.assertEqual(list(wrapper.generate_stream("Hello")), ["Hello"])
        self.assertIs(wrapper.config, original.config)

        wrapper.to(torch.device("cuda"))
        self.assertEqual(offload.call_count, 2)
        with self.assertRaisesRegex(RuntimeError, "change its CUDA device"):
            wrapper.to(torch.device("cuda:1"))

    def test_zero_resident_blocks_uses_offload_for_every_block(self):
        events = []
        offload = Mock()
        model = FakeModel(events)
        _OffloadedModel(model, gpu_layers=0, cpu_offload=offload).to("cuda")
        self.assertEqual(offload.call_count, 4)
        self.assertFalse(any(name.startswith("block-") for name, *_ in events))

    def test_invalid_layer_counts_are_rejected_before_device_moves(self):
        for value in [-1, 5, True, 1.5, "2"]:
            with self.subTest(value=value):
                events = []
                with self.assertRaises(ValueError):
                    _OffloadedModel(FakeModel(events), gpu_layers=value, cpu_offload=Mock())
                self.assertEqual(events, [])

    def test_untied_head_and_cpu_target_are_rejected(self):
        model = FakeModel([])
        model.core.llm.lm_head.weight = nn.Parameter(torch.ones(2, 2))
        model.core.llm.tie_weights = Mock()
        with self.assertRaisesRegex(RuntimeError, "tied input/output"):
            _OffloadedModel(model, gpu_layers=0, cpu_offload=Mock())
        wrapper = _OffloadedModel(FakeModel([]), gpu_layers=0, cpu_offload=Mock())
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA GPU"):
            wrapper.to("cpu")


class OffloadLoadingTests(unittest.TestCase):
    def test_loader_uses_eager_runtime_and_builds_normal_tensors(self):
        events = []
        models = []

        def create_model(path):
            self.assertEqual(path, Path("/fake/model"))
            model = FakeModel(events)
            models.append(model)
            return model

        def create_runtime(model, path, **settings):
            self.assertFalse(torch.is_inference_mode_enabled())
            model.core.to(dtype=torch.bfloat16)
            model.to("cuda").eval()
            return SimpleNamespace(model=model, path=path, settings=settings)

        load_model = Mock(side_effect=create_model)
        runtime_factory = Mock(side_effect=create_runtime)
        offload = Mock()
        modules = {
            "accelerate": SimpleNamespace(cpu_offload=offload),
            "dots_tts.models.dots_tts.model": SimpleNamespace(
                DotsTtsModel=SimpleNamespace(from_pretrained=load_model)
            ),
            "dots_tts.runtime": SimpleNamespace(DotsTtsRuntime=runtime_factory),
        }
        with patch.dict(sys.modules, modules), \
                patch.object(torch.cuda, "is_available", return_value=True), \
                torch.inference_mode():
            runtime = load_offloaded_runtime(
                "/fake/model", precision="bfloat16", gpu_layers=1
            )
        load_model.assert_called_once()
        self.assertFalse(models[0].created_in_inference_mode)
        self.assertFalse(torch.is_inference(models[0].core.llm.model.layers[0].weight))
        self.assertFalse(runtime.model.training)
        self.assertEqual(offload.call_count, 3)
        self.assertEqual(runtime.settings, {
            "precision": "bfloat16", "optimize": False,
            "max_generate_length": 128, "max_sequence_length": 512,
            "vocoder_merge_steps": 1, "warmup_on_optimize": False,
        })

    def test_precision_and_cuda_requirements_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "bfloat16 or float16"):
            load_offloaded_runtime("/fake/model", precision="float32")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires a CUDA GPU"):
                load_offloaded_runtime("/fake/model", precision="float16")

    def test_missing_accelerate_has_installation_guidance(self):
        with patch.dict(sys.modules, {"accelerate": None}), \
                patch.object(torch.cuda, "is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "needs Accelerate"):
                load_offloaded_runtime("/fake/model", precision="float16")


class OffloadHookExecutionTests(unittest.TestCase):
    def test_real_qwen_prefill_and_cached_decode_preserve_logits(self):
        try:
            from accelerate import cpu_offload
        except ImportError:
            self.skipTest("Optional Accelerate dependency is not installed.")
        from transformers import Qwen2Config, Qwen2ForCausalLM

        config = Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=64, attention_dropout=0.0,
            tie_word_embeddings=True, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(19)
            dense = Qwen2ForCausalLM(config).eval()
        offloaded = copy.deepcopy(dense).eval()
        for block in offloaded.model.layers:
            # CPU execution tests the real hooks and cache integration without
            # claiming anything about CUDA placement, peak VRAM or latency.
            cpu_offload(block, execution_device=torch.device("cpu"), offload_buffers=True)

        def assert_weights_released():
            self.assertTrue(all(
                parameter.is_meta
                for block in offloaded.model.layers
                for parameter in block.parameters()
            ))

        assert_weights_released()
        with torch.no_grad():
            prompt = torch.tensor([[1, 4, 7, 3]])
            dense_prefill = dense(input_ids=prompt, use_cache=True)
            offload_prefill = offloaded(input_ids=prompt, use_cache=True)
            torch.testing.assert_close(
                offload_prefill.logits, dense_prefill.logits, rtol=0, atol=0
            )
            assert_weights_released()

            next_token = torch.tensor([[8]])
            dense_decode = dense(
                input_ids=next_token,
                past_key_values=dense_prefill.past_key_values,
                use_cache=True,
            )
            offload_decode = offloaded(
                input_ids=next_token,
                past_key_values=offload_prefill.past_key_values,
                use_cache=True,
            )
            torch.testing.assert_close(
                offload_decode.logits, dense_decode.logits, rtol=0, atol=0
            )
            self.assertEqual(offload_decode.past_key_values.get_seq_length(), 5)
            self.assertEqual(dense_decode.past_key_values.get_seq_length(), 5)
            assert_weights_released()


if __name__ == "__main__":
    unittest.main()
