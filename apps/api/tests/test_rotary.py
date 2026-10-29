import unittest
from types import SimpleNamespace

import torch

from t1000.rotary import configure_rotary_cache


def tiny_dit(dtype=torch.float32):
    from dots_tts.modules.backbone.dit import DiT
    from dots_tts.modules.backbone.dit_inference import FusedAdaLNDiT

    values = dict(hidden_size=8, num_heads=2, num_layers=3,
                  ffn_hidden_size=16, modulation=True, qkv_bias=False,
                  qk_norm=True, rotary_bias=True, attn_dropout=0., dropout=0.)
    config = SimpleNamespace(**values, to_dict=lambda: values)
    torch.manual_seed(918)
    dit = DiT(8, 2, config).eval()
    for parameter in dit.parameters():
        torch.nn.init.uniform_(parameter, -.2, .2)
    return FusedAdaLNDiT(dit).to(dtype=dtype).eval()


def runtime_for(dit):
    return SimpleNamespace(optimize=False, model=SimpleNamespace(
        _optimize_enabled=False,
        core=SimpleNamespace(velocity_field_predictor=dit)))


class RotaryReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def inputs(self, length=6, dtype=torch.float32, positions=None):
        torch.manual_seed(123 + length)
        return dict(x=torch.randn(2, length, 8, dtype=dtype),
                    timesteps=torch.tensor([.2, .2], dtype=dtype),
                    g_cond=torch.randn(2, 8, dtype=dtype), pos_ids=positions)

    def test_real_float32_and_bfloat16_dit_bitexact_and_shared(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                dit = tiny_dit(dtype)
                runtime = runtime_for(dit)
                inputs = self.inputs(dtype=dtype)
                with torch.inference_mode(), torch.autocast(
                        "cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
                    expected = dit(**inputs)
                    configure_rotary_cache(runtime)
                    actual = dit(**inputs)
                self.assertTrue(torch.equal(actual, expected))
                stats = runtime.model._t1000_rotary_reuse.stats
                self.assertEqual(stats["rotary_computations"], 1)
                self.assertEqual(stats["rotary_layer_reuses"], 2)

    def test_positions_mutation_and_lengths_never_reuse_across_forwards(self):
        dit = tiny_dit()
        runtime = runtime_for(dit)
        original = dit.run_blocks
        configure_rotary_cache(runtime)
        for length in (6, 9):
            with torch.inference_mode():
                positions = torch.arange(length, dtype=torch.float32)
                for offset in (0, 5):
                    positions.add_(offset)
                    values = self.inputs(length, positions=positions)
                    patched = dit.run_blocks
                    dit.run_blocks = original
                    expected = dit(**values)
                    dit.run_blocks = patched
                    actual = dit(**values)
                    self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(runtime.model._t1000_rotary_reuse.stats["rotary_computations"], 4)

    def test_distinct_or_mutated_frequencies_are_not_shared(self):
        dit = tiny_dit()
        dit.blocks[1].attn.rotary.inv_freq.add_(.25)
        runtime = runtime_for(dit)
        configure_rotary_cache(runtime)
        original = runtime.model._t1000_rotary_reuse._original
        for mutate in (False, True):
            if mutate:
                dit.blocks[2].attn.rotary.inv_freq.mul_(.75)
            values = self.inputs()
            patched = dit.run_blocks
            dit.run_blocks = original
            with torch.inference_mode():
                expected = dit(**values)
                dit.run_blocks = patched
                actual = dit(**values)
            self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(runtime.model._t1000_rotary_reuse.stats["uncacheable_layers"], 1)

    def test_batched_positions_and_attention_masks_preserve_results(self):
        dit = tiny_dit()
        runtime = runtime_for(dit)
        positions = torch.stack((torch.arange(6), torch.arange(6) + 4))
        for shape in ((2, 6), (2, 6, 6), (2, 1, 6, 6)):
            values = self.inputs(positions=positions)
            values["attn_mask"] = torch.ones(shape, dtype=torch.bool)
            with torch.inference_mode():
                expected = dit(**values)
                configure_rotary_cache(runtime)
                actual = dit(**values)
                configure_rotary_cache(runtime, False)
            self.assertTrue(torch.equal(actual, expected))

    def test_existing_conditioning_proxy_uses_shared_run_blocks(self):
        from t1000.eager import _ConditioningProxy, _RequestCache

        dit = tiny_dit()
        values = self.inputs()
        cache = _RequestCache(1, values["g_cond"].clone())
        proxy = _ConditioningProxy(dit, cache)
        with torch.inference_mode():
            expected = dit(**values)
            configure_rotary_cache(runtime_for(dit))
            actual = proxy(**values)
        self.assertTrue(torch.equal(actual, expected))

    def test_optin_idempotence_restore_and_compiled_passthrough(self):
        dit = tiny_dit()
        runtime = runtime_for(dit)
        original = dit.run_blocks
        self.assertIs(configure_rotary_cache(runtime, False), runtime)
        configure_rotary_cache(runtime)
        installed = dit.run_blocks
        configure_rotary_cache(runtime)
        self.assertIs(dit.run_blocks, installed)
        configure_rotary_cache(runtime, False)
        self.assertEqual(dit.run_blocks, original)
        runtime.optimize = True
        configure_rotary_cache(runtime)
        self.assertEqual(dit.run_blocks, original)
        with self.assertRaises(ValueError):
            configure_rotary_cache(runtime, "1")


if __name__ == "__main__":
    unittest.main()
