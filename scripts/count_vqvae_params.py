#!/usr/bin/env python
"""Compute parameter counts for VQ-VAE experiments based on their logs."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import vqvae  # noqa: E402

DEFAULT_LOGS = [
    "outputs/vq_vae_v5.10/vq_vae.log",
    "outputs/vq_vae_multi_scale_codebook84_emb8/vq_vae.log",
]


@dataclass
class ExperimentConfig:
    name: str
    log_path: Path
    num_layers: int
    image_size: int
    small_conv: bool
    embedding_dim: int
    num_embeddings: int
    commitment_cost: float
    use_max_filters: bool
    max_filters: int
    use_multi_scale: bool
    structure_embedding_dim: int | None = None
    structure_num_embeddings: int | None = None
    structure_downsample_factor: int | None = None


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def _extract_value(line: str, label: str) -> str | None:
    if label not in line:
        return None
    parts = line.split(f"{label}:")
    if len(parts) < 2:
        return None
    return parts[1].strip()


def parse_log_config(log_path: Path) -> ExperimentConfig:
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    data: Dict[str, str] = {}
    exp_name = log_path.parent.name
    with log_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            for label in [
                "Num Layers",
                "Num Embeddings",
                "Embedding Dim",
                "Commitment Cost",
                "Use Max Filters",
                "Max Filters",
                "Small Conv",
                "Use Multi-Scale Codebook",
                "Structure Downsample Factor",
                "Structure Embedding Dim",
                "Structure Num Embeddings",
                "Image Size",
            ]:
                value = _extract_value(line, label)
                if value is not None:
                    data[label] = value
    required = [
        "Num Layers",
        "Num Embeddings",
        "Embedding Dim",
        "Commitment Cost",
        "Use Max Filters",
        "Max Filters",
        "Small Conv",
        "Image Size",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing config keys {missing} in {log_path}")

    use_multi = _parse_bool(data.get("Use Multi-Scale Codebook", "False"))
    if use_multi:
        extra_required = [
            "Structure Downsample Factor",
            "Structure Embedding Dim",
            "Structure Num Embeddings",
        ]
        missing = [key for key in extra_required if key not in data]
        if missing:
            raise ValueError(
                f"Missing multi-scale config keys {missing} in {log_path}"
            )

    return ExperimentConfig(
        name=exp_name,
        log_path=log_path,
        num_layers=int(float(data["Num Layers"])),
        image_size=int(float(data["Image Size"])),
        small_conv=_parse_bool(data["Small Conv"]),
        embedding_dim=int(float(data["Embedding Dim"])),
        num_embeddings=int(float(data["Num Embeddings"])),
        commitment_cost=float(data["Commitment Cost"]),
        use_max_filters=_parse_bool(data["Use Max Filters"]),
        max_filters=int(float(data["Max Filters"])),
        use_multi_scale=use_multi,
        structure_embedding_dim=int(float(data["Structure Embedding Dim"]))
        if use_multi
        else None,
        structure_num_embeddings=int(float(data["Structure Num Embeddings"]))
        if use_multi
        else None,
        structure_downsample_factor=int(float(data["Structure Downsample Factor"]))
        if use_multi
        else None,
    )


def build_model(cfg: ExperimentConfig) -> torch.nn.Module:
    common_kwargs = dict(
        num_layers=cfg.num_layers,
        input_image_dimensions=cfg.image_size,
        small_conv=cfg.small_conv,
        embedding_dim=cfg.embedding_dim,
        max_filters=cfg.max_filters,
        use_max_filters=cfg.use_max_filters,
        num_embeddings=cfg.num_embeddings,
        commitment_cost=cfg.commitment_cost,
    )
    if cfg.use_multi_scale:
        return vqvae.MultiScaleVQVAE(
            **common_kwargs,
            structure_embedding_dim=cfg.structure_embedding_dim,
            structure_num_embeddings=cfg.structure_num_embeddings,
            structure_downsample_factor=cfg.structure_downsample_factor,
        )
    return vqvae.VQVAE(**common_kwargs)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(param.numel() for param in params)


def resolve_logs(logs: Iterable[str]) -> list[Path]:
    if logs:
        paths = []
        for entry in logs:
            path = Path(entry)
            if path.is_dir():
                path = path / "vq_vae.log"
            paths.append(path)
        return paths
    return [Path(p) for p in DEFAULT_LOGS]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count parameters for VQ-VAE experiments using their logs."
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Optional log files or experiment directories. Defaults to key experiments.",
    )
    args = parser.parse_args()

    results = []
    for log_path in resolve_logs(args.logs):
        cfg = parse_log_config(log_path)
        model = build_model(cfg)
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        results.append((cfg, total_params, trainable_params, log_path))

    for cfg, total_params, trainable_params, log_path in results:
        print(
            f"{cfg.name}: total params = {total_params:,} "
            f"(trainable = {trainable_params:,}) from log {log_path}"
        )

    # Highlight direct comparison when both key experiments are present.
    by_name = {cfg.name: (total, trainable) for cfg, total, trainable, _ in results}
    target_a = "vq_vae_v5.10"
    target_b = "vq_vae_multi_scale_codebook84_emb8"
    if target_a in by_name and target_b in by_name:
        total_a, train_a = by_name[target_a]
        total_b, train_b = by_name[target_b]
        delta_total = total_a - total_b
        delta_train = train_a - train_b
        print(
            "\nComparison:\n"
            f"- {target_a} vs {target_b}: Δtotal = {delta_total:+,} params, "
            f"Δtrainable = {delta_train:+,} params"
        )


if __name__ == "__main__":
    main()
