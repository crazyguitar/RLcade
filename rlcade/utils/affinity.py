from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import pynvml

log = logging.getLogger(__name__)

# nvml returns affinity as a list of 64-bit ints.
_NVML_BIT_AFFINITY = 64
_NVML_AFFINITY_ELEMENTS = (os.cpu_count() + _NVML_BIT_AFFINITY - 1) // _NVML_BIT_AFFINITY


class Affinity:
    """Query and set CPU affinity for NUMA-local GPU binding."""

    def __init__(self, gpu_id: int):
        self._gpu_id = gpu_id
        pynvml.nvmlInit()

    def close(self) -> None:
        pynvml.nvmlShutdown()

    def __enter__(self) -> Affinity:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def gpu_id(self) -> int:
        return self._gpu_id

    def gpu_cpu_affinity(self) -> set[int]:
        """Return the set of CPU cores on the same NUMA node as this GPU."""
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_id)
        mask = pynvml.nvmlDeviceGetCpuAffinityWithinScope(
            handle, _NVML_AFFINITY_ELEMENTS, pynvml.NVML_AFFINITY_SCOPE_NODE
        )
        return self._bitmask_to_cpus(mask)

    def set_cpu_affinity(self, cpus: Optional[Sequence[int]] = None) -> set[int]:
        """Bind the current process to CPUs associated with this GPU."""
        if cpus is not None:
            assigned = set(cpus)
        else:
            assigned = self._available_cpus(self.gpu_cpu_affinity())

        os.sched_setaffinity(0, assigned)
        log.info(
            "GPU %d bound to %d CPUs (cores: %s)",
            self._gpu_id,
            len(assigned),
            self.compact_range(assigned),
        )
        return assigned

    @staticmethod
    def compact_range(cpus: set[int]) -> str:
        """Format CPU indices as a compact range string, e.g. '0-15,32-47'."""
        if not cpus:
            return ""
        sorted_cpus = sorted(cpus)
        ranges: list[str] = []
        start = prev = sorted_cpus[0]
        for c in sorted_cpus[1:]:
            if c == prev + 1:
                prev = c
            else:
                ranges.append(f"{start}-{prev}" if start != prev else str(start))
                start = prev = c
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        return ",".join(ranges)

    @staticmethod
    def _bitmask_to_cpus(mask: list[int]) -> set[int]:
        """Convert a list of 64-bit bitmask integers to a set of CPU indices."""
        bits = "".join(f"{word:064b}" for word in reversed(mask))
        return {i for i, b in enumerate(reversed(bits)) if b == "1"}

    def _available_cpus(self, node_cpus: set[int]) -> set[int]:
        """Intersect NUMA node CPUs with the OS-allowed set (cgroup / taskset)."""
        allowed = os.sched_getaffinity(0)
        available = node_cpus & allowed
        if available:
            return available
        log.warning(
            "No overlap between NUMA CPUs for GPU %d and allowed CPUs; " "falling back to all allowed CPUs.",
            self._gpu_id,
        )
        return allowed


def set_gpu_affinity(local_rank: int, cpus: Optional[Sequence[int]] = None) -> set[int]:
    """Convenience function: set CPU affinity for a GPU rank."""
    with Affinity(local_rank) as affinity:
        return affinity.set_cpu_affinity(cpus=cpus)
