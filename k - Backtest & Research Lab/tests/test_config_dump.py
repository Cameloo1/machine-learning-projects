from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from backtest_lab import config as cfg_mod


def _base_config(prices_path: Path) -> dict:
    return {
        "run_id": "dump_test",
        "data": {
            "mode": "csv",
            "prices_path": str(prices_path),
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
        "strategy": {"name": "sma_trend"},
        "execution": {
            "cost_bps": 2.0,
            "slippage_model": "none",
            "max_leverage": 1.0,
            "max_weight_per_asset": 1.0,
        },
    }


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_config_dump_deterministic_includes_defaults(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.csv"
    prices_path.write_text("date,close\n2020-01-01,100\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _base_config(prices_path))

    cfg = cfg_mod.load_config(config_path)
    out_dir = cfg.output_dir / cfg.run_id
    cfg_hash = cfg_mod.write_resolved_config(out_dir, cfg)
    first_bytes = (out_dir / "config.json").read_bytes()

    cfg_second = cfg_mod.load_config(config_path)
    cfg_mod.write_resolved_config(out_dir, cfg_second)
    second_bytes = (out_dir / "config.json").read_bytes()

    assert first_bytes == second_bytes
    assert cfg_hash == hashlib.sha256(first_bytes).hexdigest()

    payload = json.loads(first_bytes.decode("utf-8"))
    expected_output_dir = (config_path.parent / "artifacts").resolve()
    expected_cache_dir = (config_path.parent / "data" / "raw").resolve()

    assert payload["output_dir"] == str(expected_output_dir)
    assert payload["data"]["cache_dir"] == str(expected_cache_dir)
    assert payload["strategy"]["params"] == {}
    assert payload["execution"]["slippage_params"] == {}
    assert "walkforward" in payload


def test_run_metadata_contains_fields_and_hash(tmp_path: Path, monkeypatch) -> None:
    prices_path = tmp_path / "prices.csv"
    prices_path.write_text("date,close\n2020-01-01,100\n", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, _base_config(prices_path))

    cfg = cfg_mod.load_config(config_path)
    out_dir = cfg.output_dir / cfg.run_id
    cfg_hash = cfg_mod.write_resolved_config(out_dir, cfg)

    monkeypatch.setattr(cfg_mod, "get_git_commit", lambda: None)
    cfg_mod.write_run_metadata(out_dir, cfg, config_path, cfg_hash)

    metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    required = {"created_at_utc", "python_version", "platform", "git_commit", "config_sha256"}
    assert required.issubset(metadata.keys())
    assert metadata["config_sha256"] == cfg_hash
    assert metadata["git_commit"] is None
