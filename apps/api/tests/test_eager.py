import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from t1000.eager import (
    MAX_CACHED_STEPS,
    _CachedEagerSolver,
    configure_eager_cache,
    read_eager_cache_enabled,
)


def tiny_context():
    from dots_tts.modules.backbone.dit import DiT
    from dots_tts.modules.backbone.dit_inference import DiTInferenceContext, FusedAdaLNDiT

    values = dict(hidden_size=8, num_heads=2, num_layers=2,
                  ffn_hidden_size=16, modulation=True, qkv_bias=False,
                  qk_norm=True, rotary_bias=True, attn_dropout=0., dropout=0.)
    config = SimpleNamespace(**values, to_dict=lambda: values)
    torch.manual_seed(918)
    dit = DiT(8, 2, config).eval()
    # Upstream initializes the output and modulation projections to zero;
    # nonzero weights ensure these checks exercise actual conditioned output.
    for parameter in dit.parameters():
        torch.nn.init.uniform_(parameter, -.2, .2)
    return DiTInferenceContext(
        dit=FusedAdaLNDiT(dit), coordinate_proj=torch.nn.Linear(2, 8).eval(),
        hidden_patch_size=1, latent_patch_size=2, latent_dim=2,
        hidden_size=8, fm_sigma=0.,
    )


class EagerConditioningCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        from dots_tts.modules.backbone.dit_inference import DiTSolver, DiTSolverState

        self.State = DiTSolverState
        self.context = tiny_context()
        self.baseline = DiTSolver(self.context, optimize=False, bucket_resolver=lambda n: n)
        self.solver = DiTSolver(self.context, optimize=False, bucket_resolver=lambda n: n)
        self.cached = _CachedEagerSolver(self.solver)
        self.condition = torch.randn(1, 8)
        self.calls = Mock()
        def count_call(*args):
            self.calls(*args)
        handle = self.context.dit.fused_adaln.register_forward_hook(count_call)
        self.addCleanup(handle.remove)

    def kwargs(self, length=4, *, steps=2, condition=None, method="euler"):
        torch.manual_seed(321 + length)
        return dict(sequence=torch.randn(1, length, 8),
                    cfg_sequence=torch.randn(1, length, 8), fm_seq_len=length,
                    null_g_cond=torch.zeros(1, 8),
                    g_cond=self.condition if condition is None else condition,
                    nfe=steps, ode_method=method, guidance_scale=1.2)

    def decode(self, solver, state, values, seed=42):
        torch.manual_seed(seed)
        with torch.inference_mode():
            return solver.decode_next(state, **values)

    def test_real_dit_is_bitexact_across_patches_and_reuses_only_modulation(self):
        state = self.State()
        for index, length in enumerate((4, 7, 10)):
            values = self.kwargs(length)
            expected = self.decode(self.baseline, self.State(), values, seed=42 + index)
            before = self.calls.call_count
            actual = self.decode(self.cached, state, values, seed=42 + index)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(self.calls.call_count - before, 2 if index == 0 else 0)
        cache = state._t1000_eager_conditioning_cache
        expected_bytes = (self.condition.numel() + 2 * 2 * (6 * 2 + 2) * 8 + 2 * 2) * 4
        self.assertEqual(cache.tensor_bytes, expected_bytes)

    def test_changed_voice_steps_and_new_request_invalidate(self):
        state = self.State()
        for condition, steps in ((self.condition, 2),
                                 (self.condition + .7, 2),
                                 (self.condition + .7, 3)):
            values = self.kwargs(steps=steps, condition=condition)
            expected = self.decode(self.baseline, self.State(), values)
            before = self.calls.call_count
            self.assertTrue(torch.equal(self.decode(self.cached, state, values), expected))
            self.assertEqual(self.calls.call_count - before, steps)
        fresh = self.State()
        before = self.calls.call_count
        self.decode(self.cached, fresh, self.kwargs())
        self.assertEqual(self.calls.call_count - before, 2)
        self.assertIsNot(fresh._t1000_eager_conditioning_cache,
                         state._t1000_eager_conditioning_cache)

    def test_inplace_voice_change_invalidates_snapshot(self):
        state = self.State()
        self.decode(self.cached, state, self.kwargs())
        old = state._t1000_eager_conditioning_cache
        self.condition.add_(.2)
        self.decode(self.cached, state, self.kwargs())
        self.assertIsNot(state._t1000_eager_conditioning_cache, old)

    def test_non_euler_falls_back_to_unchanged_solver(self):
        state = self.State()
        for method in ("midpoint", "rk4"):
            values = self.kwargs(method=method)
            expected = self.decode(self.baseline, self.State(), values)
            self.assertTrue(torch.equal(self.decode(self.cached, state, values), expected))
        self.assertFalse(hasattr(state, "_t1000_eager_conditioning_cache"))

    def test_changed_actual_timestep_sequence_fails_before_return_and_clears(self):
        state = self.State()
        values = self.kwargs()
        self.decode(self.cached, state, values)
        runner = self.solver._get_eager_runner(device=torch.device("cpu"), dtype=torch.float32)
        original_dit = runner.dit
        original_velocity = runner._flow_matching_velocity

        def shifted(t, z, **kwargs):
            return original_velocity(t + .125, z, **kwargs)

        with patch.object(runner, "_flow_matching_velocity", shifted):
            with self.assertRaisesRegex(RuntimeError, "changed Euler timestep sequence"):
                self.decode(self.cached, state, values)
        self.assertIsNone(state._t1000_eager_conditioning_cache)
        self.assertIs(runner.dit, original_dit)
        self.assertTrue(torch.equal(self.decode(self.cached, state, values),
                                    self.decode(self.baseline, self.State(), values)))

    def test_interruption_clears_partial_cache_and_restores_runner(self):
        state = self.State()
        runner = self.solver._get_eager_runner(device=torch.device("cpu"), dtype=torch.float32)
        original_dit = runner.dit
        with patch.object(self.solver, "decode_next", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.decode(self.cached, state, self.kwargs())
        self.assertIsNone(state._t1000_eager_conditioning_cache)
        self.assertIs(runner.dit, original_dit)

    def test_step_bound_bypasses_cache(self):
        state = self.State()
        values = self.kwargs(steps=MAX_CACHED_STEPS + 1)
        with patch.object(self.solver, "decode_next", return_value="unchanged") as delegate:
            self.assertEqual(self.cached.decode_next(state, **values), "unchanged")
        delegate.assert_called_once_with(state, **values)
        self.assertFalse(hasattr(state, "_t1000_eager_conditioning_cache"))


class EagerCacheConfigurationTests(unittest.TestCase):
    def test_opt_in_is_strict_and_defaults_off(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(read_eager_cache_enabled())
        for value, expected in (("0", False), ("1", True), (" 1 ", True)):
            with patch.dict(os.environ, {"T1000_TTS_EAGER_CACHE": value}):
                self.assertEqual(read_eager_cache_enabled(), expected)
        with patch.dict(os.environ, {"T1000_TTS_EAGER_CACHE": "auto"}):
            with self.assertRaises(ValueError):
                read_eager_cache_enabled()

    def test_factory_is_per_model_idempotent_and_modes_pass_through(self):
        flow = SimpleNamespace(optimize=False)
        other = object()
        factory = Mock(side_effect=lambda *, solver_mode: flow if solver_mode == "flow_matching" else other)
        model = SimpleNamespace(_optimize_enabled=False, _get_dit_solver=factory)
        runtime = SimpleNamespace(optimize=False, model=model)
        self.assertIs(configure_eager_cache(runtime, True), runtime)
        replacement = model._get_dit_solver
        first = replacement(solver_mode="flow_matching")
        self.assertIsInstance(first, _CachedEagerSolver)
        self.assertIs(replacement(solver_mode="flow_matching"), first)
        self.assertIs(replacement(solver_mode="meanflow"), other)
        configure_eager_cache(runtime, True)
        self.assertIs(model._get_dit_solver, replacement)
        untouched = SimpleNamespace(optimize=False, model=SimpleNamespace(
            _optimize_enabled=False, _get_dit_solver=factory))
        configure_eager_cache(untouched, False)
        self.assertIs(untouched.model._get_dit_solver, factory)

    def test_compiled_mode_passes_through_and_invalid_flag_rejected(self):
        for runtime_opt, model_opt in ((True, False), (False, True)):
            factory = Mock()
            runtime = SimpleNamespace(optimize=runtime_opt, model=SimpleNamespace(
                _optimize_enabled=model_opt, _get_dit_solver=factory))
            configure_eager_cache(runtime, True)
            self.assertIs(runtime.model._get_dit_solver, factory)
        with self.assertRaises(ValueError):
            configure_eager_cache(SimpleNamespace(), "1")


if __name__ == "__main__":
    unittest.main()
