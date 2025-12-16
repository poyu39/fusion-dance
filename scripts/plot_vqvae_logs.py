#!/usr/bin/env python
"""Parse VQ-VAE training logs and plot loss curves."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_EXPERIMENT_LOGS: Dict[str, Path] = {
    "vq_vae_v5.10": Path("outputs/vq_vae_v5.10/vq_vae.log"),
    "vq_vae_pixel_aware_loss": Path("outputs/vq_vae_pixel_aware_loss/vq_vae.log"),
}

EPOCH_PATTERN = re.compile(r"Epoch:\s+(\d+)/(\d+):")
METRIC_PREFIXES = {
    "Train Loss": "train_loss",
    "Train Reconstruction Loss": "train_recon_loss",
    "Train Commitment Loss": "train_commit_loss",
    "Val Loss": "val_loss",
    "Val Reconstruction Loss": "val_recon_loss",
    "Val Commitment Loss": "val_commit_loss",
}
TEST_PATTERN = re.compile(
    r"(Test Metrics - MSE = ([\deE\+\-\.]+), SSIM = ([\deE\+\-\.]+))"
)


def parse_log(
    log_path: Path,
) -> Tuple[List[Dict[str, float]], Dict[str, float], str]:
    """Parse a VQ-VAE training log file."""
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    history: List[Dict[str, float]] = []
    current_epoch: Dict[str, float] | None = None
    test_metrics: Dict[str, float] | None = None
    test_line: str | None = None

    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                if current_epoch:
                    history.append(current_epoch)
                current_epoch = {"epoch": float(epoch_match.group(1))}
                continue

            if current_epoch:
                for prefix, key in METRIC_PREFIXES.items():
                    if line.startswith(prefix):
                        value = float(line.split("=")[1].strip())
                        current_epoch[key] = value
                        break

            if test_metrics is None:
                test_match = TEST_PATTERN.search(line)
                if test_match:
                    test_line = test_match.group(1)
                    test_metrics = {
                        "mse": float(test_match.group(2)),
                        "ssim": float(test_match.group(3)),
                    }

    if current_epoch:
        history.append(current_epoch)

    if not history:
        raise ValueError(f"No epoch metrics found in {log_path}")

    if test_metrics is None or test_line is None:
        raise ValueError(f"No test metrics found in {log_path}")

    return history, test_metrics, test_line


def plot_losses(
    history: List[Dict[str, float]],
    exp_name: str,
    out_path: Path,
    test_line: str,
) -> None:
    """Save loss curves for train/val metrics."""
    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    def _plot_pair(ax, train_key, val_key, title):
        ax.plot(epochs, [entry[train_key] for entry in history], label="Train")
        ax.plot(epochs, [entry[val_key] for entry in history], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

    _plot_pair(axes[0], "train_loss", "val_loss", "Total Loss")
    _plot_pair(
        axes[1],
        "train_recon_loss",
        "val_recon_loss",
        "Reconstruction Loss",
    )
    _plot_pair(
        axes[2],
        "train_commit_loss",
        "val_commit_loss",
        "Commitment Loss",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle(f"{exp_name} Loss Curves")
    fig.text(0.01, 0.01, test_line, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _resolve_logs(log_args: Iterable[str]) -> Dict[str, Path]:
    if log_args:
        resolved: Dict[str, Path] = {}
        for arg in log_args:
            path = Path(arg)
            if path.is_dir():
                path = path / "vq_vae.log"
            if not path.exists():
                raise FileNotFoundError(f"Provided log not found: {path}")
            exp_name = path.parent.name
            resolved[exp_name] = path
        return resolved
    return DEFAULT_EXPERIMENT_LOGS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot VQ-VAE loss curves from training logs."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Optional list of log files or experiment directories. Defaults to built-in experiments.",
    )
    args = parser.parse_args()

    experiments = _resolve_logs(args.logs)
    summary_lines = []
    for exp_name, log_path in experiments.items():
        history, test_metrics, test_line = parse_log(log_path)
        plot_path = log_path.parent / "loss_curves.png"
        plot_losses(history, exp_name, plot_path, test_line)
        (log_path.parent / "test_metrics.txt").write_text(
            test_line + "\n", encoding="utf-8"
        )
        summary_lines.append(
            f"{exp_name}: loss plot -> {plot_path}, "
            f"Test MSE = {test_metrics['mse']:.15g}, "
            f"SSIM = {test_metrics['ssim']:.15g}"
        )

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
