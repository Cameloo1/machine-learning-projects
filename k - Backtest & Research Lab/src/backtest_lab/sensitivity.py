from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from backtest_lab import config as cfg_mod
from backtest_lab.run import _run_pipeline


def _build_default_settings() -> List[Dict[str, Any]]:
    return [
        {
            "label": "low",
            "execution": {"cost_bps": 0.0, "slippage_model": "none", "slippage_params": {}},
        },
        {
            "label": "mid",
            "execution": {"cost_bps": 5.0, "slippage_model": "bps", "slippage_params": {"slip_bps": 1.0}},
        },
        {
            "label": "high",
            "execution": {"cost_bps": 20.0, "slippage_model": "bps", "slippage_params": {"slip_bps": 5.0}},
        },
    ]


def _config_from_dict(cfg_dict: Dict[str, Any]) -> cfg_mod.Config:
    if cfg_mod._PYDANTIC_V2:
        return cfg_mod.Config.model_validate(cfg_dict)
    return cfg_mod.Config.from_dict(cfg_dict)


def run_sensitivity(
    config_path: Path,
    *,
    output_dir: Path | None = None,
    run_id: str | None = None,
    settings: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    base_cfg = cfg_mod.load_config(config_path)
    base_dict = base_cfg.to_dict()

    if output_dir is not None:
        base_dict["output_dir"] = str(output_dir)
    if run_id is not None:
        base_dict["run_id"] = str(run_id)

    settings_list = list(settings) if settings is not None else _build_default_settings()

    summary_rows: List[Dict[str, Any]] = []
    artifacts_dirs: List[Path] = []

    for setting in settings_list:
        label = str(setting.get("label", "setting"))
        cfg_dict = dict(base_dict)
        cfg_dict["execution"] = dict(base_dict.get("execution", {}) or {})
        cfg_dict["execution"].update(setting.get("execution", {}))
        cfg_dict["run_id"] = f"{cfg_dict['run_id']}_{label}"

        cfg_obj = _config_from_dict(cfg_dict)
        cfg_obj = cfg_mod.resolve_config(cfg_obj, Path(config_path))

        artifacts_dir = cfg_obj.output_dir / cfg_obj.run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        config_sha256 = cfg_mod.write_resolved_config(artifacts_dir, cfg_obj)
        cfg_mod.write_run_metadata(artifacts_dir, cfg_obj, Path(config_path), config_sha256)
        config_integrity = cfg_mod.check_config_integrity(cfg_obj)
        (artifacts_dir / "config_integrity.json").write_text(
            json.dumps(config_integrity, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        _run_pipeline(cfg_obj, Path(config_path), artifacts_dir)
        artifacts_dirs.append(artifacts_dir)

        metrics_path = artifacts_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        summary_rows.append(
            {
                "label": label,
                "run_id": cfg_obj.run_id,
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "turnover_avg": metrics.get("turnover_avg"),
                "costs_avg": metrics.get("costs_avg"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(base_dict["output_dir"]) / f"{base_dict['run_id']}_sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    return {"summary_path": summary_path, "artifacts_dirs": artifacts_dirs}
