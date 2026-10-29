"""Optional request-scoped reuse of eager DiT timestep conditioning.

The Euler solver still receives the original time grid, noise, arithmetic and
full attention history. Only its repeated, voice-conditioned AdaLN projection
is cached. Actual timestep vectors are checked before returning each patch.
"""

import os
import threading
from dataclasses import dataclass, field

import torch


MAX_CACHED_STEPS = 32


def read_eager_cache_enabled():
    value = os.environ.get("T1000_TTS_EAGER_CACHE", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("T1000_TTS_EAGER_CACHE must be 0 or 1.")
    return value == "1"


@dataclass
class _RequestCache:
    steps: int
    conditioning: torch.Tensor
    mods: list = field(default_factory=list)
    timesteps: torch.Tensor | None = None

    @property
    def tensor_bytes(self):
        values = [self.conditioning, *self.mods]
        if self.timesteps is not None:
            values.append(self.timesteps)
        return sum(value.numel() * value.element_size() for value in values)


class _ConditioningProxy:
    def __init__(self, dit, cache):
        self.dit = dit
        self.cache = cache
        self.reuse = cache.timesteps is not None
        self.actual_times = []

    def __getattr__(self, name):
        return getattr(self.dit, name)

    def __call__(self, x, timesteps, duration=None, attn_mask=None,
                 pos_ids=None, g_cond=None):
        index = len(self.actual_times)
        if index >= self.cache.steps:
            raise RuntimeError("Eager conditioning cache received an unexpected Euler timestep count.")
        self.actual_times.append(timesteps.detach().clone())
        if self.reuse:
            all_mods = self.cache.mods[index]
        else:
            all_mods = self.dit.fused_adaln(
                self.dit.build_condition(timesteps, duration=duration, g_cond=g_cond)
            )
            self.cache.mods.append(all_mods.detach())
        return self.dit.run_blocks(
            x=x, all_mods=all_mods, attn_mask=attn_mask, pos_ids=pos_ids,
        )

    def validate(self):
        if len(self.actual_times) != self.cache.steps:
            raise RuntimeError("Eager conditioning cache received an unexpected Euler timestep count.")
        actual = torch.stack(self.actual_times)
        if self.reuse:
            # One comparison per patch; never fetch a CUDA scalar per ODE step.
            if not torch.equal(actual, self.cache.timesteps):
                raise RuntimeError("Eager conditioning cache detected a changed Euler timestep sequence; retry without T1000_TTS_EAGER_CACHE.")
        else:
            self.cache.timesteps = actual


class _CachedEagerSolver:
    """Delegate the unchanged upstream solve with a runner-local DiT proxy."""

    def __init__(self, solver):
        self.solver = solver
        self._lock = threading.RLock()

    def __getattr__(self, name):
        return getattr(self.solver, name)

    def decode_next(self, state, **kwargs):
        steps = kwargs["nfe"]
        if (self.solver.optimize or self.solver.meanflow
                or kwargs.get("ode_method", "euler") != "euler"
                or isinstance(steps, bool) or not isinstance(steps, int)
                or not 1 <= steps <= MAX_CACHED_STEPS):
            with self._lock:
                return self.solver.decode_next(state, **kwargs)

        sequence = kwargs["sequence"]
        null_condition = kwargs["null_g_cond"]
        condition = kwargs.get("g_cond")
        condition = null_condition if condition is None else condition.to(
            device=null_condition.device, dtype=sequence.dtype,
        )
        with self._lock:
            cache = getattr(state, "_t1000_eager_conditioning_cache", None)
            valid = (isinstance(cache, _RequestCache) and cache.steps == steps
                     and cache.conditioning.shape == condition.shape
                     and cache.conditioning.device == condition.device
                     and cache.conditioning.dtype == condition.dtype
                     and torch.equal(cache.conditioning, condition))
            if not valid:
                cache = _RequestCache(steps, condition.detach().clone())
                state._t1000_eager_conditioning_cache = cache
            runner = self.solver._get_eager_runner(
                device=sequence.device, dtype=sequence.dtype,
            )
            original = runner.dit
            proxy = _ConditioningProxy(original, cache)
            runner.dit = proxy
            try:
                result = self.solver.decode_next(state, **kwargs)
                proxy.validate()
                return result
            except BaseException:
                # Cancellation and failed/incomplete decodes must not retain a
                # partial schedule for a later call using this request state.
                state._t1000_eager_conditioning_cache = None
                raise
            finally:
                runner.dit = original


def configure_eager_cache(runtime, enabled):
    """Install an opt-in per-model factory; compiled runtimes pass through."""
    if not isinstance(enabled, bool):
        raise ValueError("Eager conditioning cache flag must be a boolean.")
    if not enabled or runtime.optimize or runtime.model._optimize_enabled:
        return runtime
    model = runtime.model
    if getattr(model, "_t1000_eager_cache_configured", False):
        return runtime
    original_factory = model._get_dit_solver
    adapters = {}

    def get_solver(*, solver_mode):
        solver = original_factory(solver_mode=solver_mode)
        if solver_mode != "flow_matching" or solver.optimize:
            return solver
        key = id(solver)
        if key not in adapters:
            adapters[key] = _CachedEagerSolver(solver)
        return adapters[key]

    model._get_dit_solver = get_solver
    model._t1000_eager_cache_configured = True
    return runtime
