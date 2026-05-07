"""Progress bar and metrics display utilities."""

from tqdm import tqdm


class ProgressBar:
    """Training progress bar with live metrics."""

    def __init__(self, total: int, initial: int = 0, disable: bool = False):
        self.pbar = tqdm(total=total, initial=initial, desc="Training", unit="iter", disable=disable)
        self.metrics: dict[str, float] = {}

    def update(self, metrics: dict[str, float], n: int = 1):
        """Advance ``n`` iterations and update displayed metrics."""
        self.metrics.update(metrics)
        self.pbar.set_postfix({k: f"{v:.4g}" for k, v in self.metrics.items()})
        self.pbar.update(n)

    def close(self):
        self.pbar.close()
