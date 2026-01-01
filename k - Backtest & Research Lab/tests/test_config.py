from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from backtest_lab import config as cfg_mod


def _base_config(prices_path: Path) -> dict:
    return {
        "run_id": "test_run",
        "output_dir": "artifacts",
        "data": {
            "mode": "csv",
            "prices_path": str(prices_path),
            "cache_dir": "data/raw",
        },
        "universe": {
            "symbols": ["SPY"],
            "min_history_days": 252,
            "missing_data_policy": "drop_symbol",
        },
        "features": {
            "sma_fast": 20,
            "sma_slow": 50,
            "rsi_window": 14,
            "rsi_low": 30,
            "rsi_high": 70,
        },
        "strategy": {"name": "sma_trend", "params": {}},
        "execution": {
            "cost_bps": 2.0,
            "slippage_model": "none",
            "slippage_params": {},
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
    }


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_config_validation_rejects_bad_sma(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.csv"
    prices_path.write_text("date,close\n2020-01-01,100\n", encoding="utf-8")
    cfg = _base_config(prices_path)
    cfg["features"]["sma_fast"] = 50
    cfg["features"]["sma_slow"] = 50
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, cfg)

    with pytest.raises(cfg_mod.ConfigError):
        cfg_mod.load_config(config_path)


def test_path_resolution_relative_to_yaml_dir(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    cfg_dir.mkdir()
    raw_dir.mkdir(parents=True)
    prices_path = data_dir / "prices.csv"
    prices_path.write_text("date,close\n2020-01-01,100\n", encoding="utf-8")

    cfg = _base_config(Path("../data/prices.csv"))
    cfg["output_dir"] = "../artifacts"
    cfg["data"]["cache_dir"] = "../data/raw"
    config_path = cfg_dir / "config.yaml"
    _write_yaml(config_path, cfg)

    loaded = cfg_mod.load_config(config_path)
    assert loaded.data.prices_path == prices_path.resolve()
    assert loaded.data.cache_dir == raw_dir.resolve()
    assert loaded.output_dir == (cfg_dir / "../artifacts").resolve()


def test_config_json_deterministic(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.csv"
    prices_path.write_text("date,close\n2020-01-01,100\n", encoding="utf-8")
    cfg = _base_config(prices_path)
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, cfg)

    cfg_first = cfg_mod.load_config(config_path)
    cfg_second = cfg_mod.load_config(config_path)

    json_first = json.dumps(cfg_first.to_dict(), indent=2, sort_keys=True)
    json_second = json.dumps(cfg_second.to_dict(), indent=2, sort_keys=True)
    assert json_first == json_second


def test_csv_mode_requires_prices_path_exists(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"
    cfg = _base_config(missing_path)
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, cfg)

    with pytest.raises(cfg_mod.ConfigError):
        cfg_mod.load_config(config_path)
