"""Per-iteration keys, simulated compute times, and gradients. No network calls.

The worker loop calls, for iteration t starting at 1:

    selection = workload.selection(t)
    delay_s = workload.compute_delay_s(t)
    gradients = workload.gradients(t, selection, pulled_flat)

Same seed must reproduce keys, delays, gradients, the straggler set, and
the hidden target. Use a separate random stream per worker, and separate
streams for keys, delays, and noise. Do not share one Random across workers.

Implement in this order: fixed gradients and a simple key selection, then
Zipf sampling, scatter, lognormal delays, stragglers, and target noise.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Selection:
    """Sparse keys, or a dense half-open range. Set one of the two."""

    keys: tuple[int, ...] | None = None
    begin: int | None = None
    end: int | None = None


class Workload:
    def __init__(
        self,
        iterations: int = 1,
        width: int = 1,
        *,
        worker_id: int = 0,
        worker_count: int = 1,
        seed: int = 42,
        num_keys: int = 8,
        access_mode: str = "sparse",
        keys_per_iter: int = 4,
        zipf_s: float = 1.1,
        scatter: bool = True,
        range_size: int | None = None,
        compute_median_ms: float = 20.0,
        compute_sigma: float = 0.4,
        straggler_fraction: float = 0.0,
        straggler_slowdown: float = 1.0,
        gradient_type: str = "fixed",
        noise_std: float = 0.0,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        # Store the fields above. worker.py currently passes only
        # iterations and width; the driver will pass the rest later.
        self.iterations = iterations
        self.width = width
        self.worker_id = worker_id
        self.worker_count = worker_count
        self.seed = seed
        self.num_keys = num_keys
        self.access_mode = access_mode
        self.keys_per_iter = keys_per_iter
        self.zipf_s = zipf_s
        self.scatter = scatter
        self.range_size = range_size
        self.compute_median_ms = compute_median_ms
        self.compute_sigma = compute_sigma
        self.straggler_fraction = straggler_fraction
        self.straggler_slowdown = straggler_slowdown
        self.gradient_type = gradient_type
        self.noise_std = noise_std
        self.target_mean = target_mean
        self.target_std = target_std

        # Filled on first use. Same contents for every worker that shares `seed`.
        self._rank_to_key: list[int] | None = None
        self._zipf_weights: list[float] | None = None
        self._target: list[float] | None = None
        self._stragglers: set[int] | None = None

    def selection(self, t: int) -> Selection:
        """Sorted unique sparse keys, or a dense [begin, end) inside the key space.

        Sparse: draw keys_per_iter Zipf ranks with replacement, drop
        duplicates, then sort. The unique count may be smaller than
        keys_per_iter. scatter=True maps ranks through one permutation
        seeded without worker_id.
        """
        if self.access_mode == "sparse":
            return self._sparse_selection(t)
        elif self.access_mode == "dense":
            return self._dense_selection(t)
        else:
            raise ValueError(f"Invalid access mode: {self.access_mode}")

    def compute_delay_s(self, t: int) -> float:
        """Seconds to sleep after a pull. Multiply by the slowdown if this worker is a straggler."""
        # median of a lognormal is exp(mu), and lognormvariate takes that mu.
        mu = math.log(self.compute_median_ms)
        delay_ms = self._rng("delay", t).lognormvariate(mu, self.compute_sigma)
        if self.worker_id in self._straggler_ids():
            delay_ms *= self.straggler_slowdown
        return delay_ms / 1000.0

    def gradients(self, t: int, selection: Selection, pulled_flat: list[float]) -> list[float]:
        """Length is n_keys * width, same order as pulled_flat.

        fixed: a function of (worker_id, t, key, component) only.
        target_diff: pulled - hidden_target_for_those_keys + noise.
        """
        if self.gradient_type == "fixed":
            return self._fixed_gradients(t, selection, pulled_flat)
        elif self.gradient_type == "target_diff":
            return self._target_diff_gradients(t, selection, pulled_flat)
        else:
            raise ValueError(f"Invalid gradient type: {self.gradient_type}")

    def hidden_target(self) -> list[float]:
        """w* for every key, length num_keys * width. Same vector for every worker."""
        if self._target is None:
            # No worker_id in this stream, so every worker draws the same w*.
            rng = self._rng("target", per_worker=False)
            n = self.num_keys * self.width
            if self.target_std == 0.0:
                self._target = [self.target_mean] * n
            else:
                self._target = [rng.gauss(self.target_mean, self.target_std) for _ in range(n)]
        return self._target

    def _sparse_selection(self, t: int) -> Selection:
        """Sparse keys, draw keys_per_iter Zipf ranks with replacement, drop duplicates, then sort."""
        drawn = self._rng("keys", t).choices(
            self._key_by_rank(),
            weights=self._rank_weights(),
            k=self.keys_per_iter,
        )
        return Selection(keys=tuple(sorted(set(drawn))))

    def _dense_selection(self, t: int) -> Selection:
        """Dense half-open range. If range_size is None, use all keys."""
        if self.range_size is None or self.range_size == self.num_keys:
            return Selection(begin=0, end=self.num_keys)
        if not 1 <= self.range_size <= self.num_keys:
            raise ValueError("range_size must be within 1..num_keys")
        begin = self._rng("keys", t).randint(0, self.num_keys - self.range_size)
        return Selection(begin=begin, end=begin + self.range_size)

    def _fixed_gradients(self, t: int, selection: Selection, pulled_flat: list[float]) -> list[float]:
        keys = self._keys(selection)
        self._check_pulled(keys, pulled_flat)
        out = []
        for key in keys:
            for component in range(self.width):
                # Depends only on worker, iteration, key, and component.
                out.append((self.worker_id + 1) * 1e-2 + t * 1e-4 + key * 1e-6 + component * 1e-8)
        return out

    def _target_diff_gradients(self, t: int, selection: Selection, pulled_flat: list[float]) -> list[float]:
        keys = self._keys(selection)
        self._check_pulled(keys, pulled_flat)
        target = self.hidden_target()
        noise = self._rng("noise", t)
        out = []
        for position, key in enumerate(keys):
            for component in range(self.width):
                pulled = pulled_flat[position * self.width + component]
                star = target[key * self.width + component]
                bump = noise.gauss(0.0, self.noise_std) if self.noise_std else 0.0
                out.append(pulled - star + bump)
        return out

    def _keys(self, selection: Selection) -> list[int]:
        if selection.keys is not None:
            return list(selection.keys)
        if selection.begin is not None and selection.end is not None:
            return list(range(selection.begin, selection.end))
        raise ValueError("selection has neither keys nor a dense range")

    def _check_pulled(self, keys: list[int], pulled_flat: list[float]) -> None:
        expected = len(keys) * self.width
        if len(pulled_flat) != expected:
            raise ValueError(f"expected {expected} pulled values, got {len(pulled_flat)}")

    def _key_by_rank(self) -> list[int]:
        """Key id for rank 1..num_keys. Rank 0 in this list is the hottest."""
        if self._rank_to_key is None:
            keys = list(range(self.num_keys))
            if self.scatter:
                # Shared by every worker: the seed does not include worker_id.
                self._rng("scatter", per_worker=False).shuffle(keys)
            self._rank_to_key = keys
        return self._rank_to_key

    def _rank_weights(self) -> list[float]:
        """Zipf weight of rank r is r ** (-s), with r starting at 1."""
        if self._zipf_weights is None:
            s = self.zipf_s
            self._zipf_weights = [r ** (-s) for r in range(1, self.num_keys + 1)]
        return self._zipf_weights

    def _straggler_ids(self) -> set[int]:
        if self._stragglers is None:
            n = int(self.worker_count * self.straggler_fraction)
            n = max(0, min(self.worker_count, n))
            if n == 0:
                self._stragglers = set()
            else:
                rng = self._rng("stragglers", per_worker=False)
                self._stragglers = set(rng.sample(range(self.worker_count), n))
        return self._stragglers

    def _rng(self, stream: str, t: int = 0, *, per_worker: bool = True) -> random.Random:
        """Fresh generator. Calling the same stream and iteration twice matches.

        A single Random advanced by every method would make keys depend on
        whether a delay or a noise draw ran first, and on which worker thread
        drew next. Python's hash() is also unusable here: it changes per process.
        """
        worker = self.worker_id if per_worker else 0
        digest = hashlib.sha256(f"{self.seed}|{stream}|{worker}|{t}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "little"))
