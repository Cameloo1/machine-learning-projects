from __future__ import annotations

import pytest

from backtest_lab.config import ConfigError, check_config_integrity


def test_config_integrity_min_history_guard() -> None:
    cfg = {
        "features": {"sma_slow": 20, "rsi_window": 14},
        "universe": {"min_history_days": 10},
        "walkforward": {"enabled": False},
    }
    with pytest.raises(ConfigError):
        check_config_integrity(cfg)
