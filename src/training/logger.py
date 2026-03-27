"""Logging utilities for training metrics."""

import csv
import os
import time
from collections import defaultdict
from typing import Any, Dict, Optional


class Logger:
    """Logger that writes key-value metrics to CSV and optionally TensorBoard.

    Usage::

        logger = Logger(log_dir="runs/exp1", use_tensorboard=True)
        logger.log("train/reward", 42.0, step=100)
        logger.log("train/loss", 0.05, step=100)
        logger.dump(step=100)
        logger.close()

    Args:
        log_dir: Directory for log files. Created if it does not exist.
        use_tensorboard: If True, also write to TensorBoard. Requires
            ``torch.utils.tensorboard`` to be available.
    """

    def __init__(self, log_dir: str, use_tensorboard: bool = False) -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir

        # CSV setup
        self._csv_path = os.path.join(log_dir, "progress.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fieldnames: list[str] = []

        # TensorBoard setup
        self._tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
            except ImportError:
                print(
                    "Warning: torch.utils.tensorboard not available. "
                    "TensorBoard logging disabled."
                )

        # Current row buffer
        self._current: Dict[str, Any] = {}

        # Wall clock
        self._start_time = time.time()

    def log(self, key: str, value: Any, step: int) -> None:
        """Record a key-value pair for the current step.

        Values are buffered until :meth:`dump` is called.

        Args:
            key: Metric name (e.g. "train/reward").
            value: Metric value (numeric).
            step: Global step / timestep counter.
        """
        self._current[key] = value
        self._current["step"] = step

        # Write to TensorBoard immediately
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_scalar(key, float(value), global_step=step)
            except (TypeError, ValueError):
                pass  # Skip non-numeric values

    def dump(self, step: int) -> None:
        """Flush buffered metrics to CSV (one row per dump call).

        Also records wall_time automatically.

        Args:
            step: Global step for this row.
        """
        self._current["step"] = step
        self._current["wall_time"] = time.time() - self._start_time

        # Lazily initialize CSV writer once we know the field names
        if self._csv_writer is None:
            self._csv_fieldnames = sorted(self._current.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_fieldnames
            )
            self._csv_writer.writeheader()
        else:
            # Handle new keys that appeared after the header was written
            new_keys = set(self._current.keys()) - set(self._csv_fieldnames)
            if new_keys:
                # Rewrite with expanded fieldnames
                self._csv_fieldnames = sorted(
                    set(self._csv_fieldnames) | new_keys
                )
                self._csv_file.close()
                # Read existing rows
                rows = []
                with open(self._csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        rows.append(row)
                # Rewrite with new header
                self._csv_file = open(self._csv_path, "w", newline="")
                self._csv_writer = csv.DictWriter(
                    self._csv_file, fieldnames=self._csv_fieldnames
                )
                self._csv_writer.writeheader()
                for row in rows:
                    self._csv_writer.writerow(row)

        self._csv_writer.writerow(
            {k: self._current.get(k, float("nan")) for k in self._csv_fieldnames}
        )
        self._csv_file.flush()

        # Clear buffer
        self._current = {}

    def close(self) -> None:
        """Close file handles and flush TensorBoard."""
        self._csv_file.close()
        if self._tb_writer is not None:
            self._tb_writer.close()
