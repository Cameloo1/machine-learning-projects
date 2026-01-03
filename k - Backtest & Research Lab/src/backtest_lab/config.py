from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    pass


_DATA_MODES = {"csv", "yfinance", "stooq"}
_MISSING_DATA_POLICIES = {"drop_symbol", "drop_rows", "keep_gaps"}
_STRATEGY_NAMES = {"sma_trend", "rsi_mr", "vol_target_trend", "ml_gated", "equal_weight"}
_SLIPPAGE_MODELS = {"none", "bps", "vol_prop"}
_RENORM_POLICIES = {"scale_down_if_exceeded", "error_if_exceeded"}
_UNIVERSE_SELECTION_MODES = {"train_only", "window_full"}


def _coerce_date_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    try:
        return str(value)
    except Exception:
        return None


def _is_pydantic_v2() -> bool:
    try:
        import pydantic
    except Exception:
        return False
    version = getattr(pydantic, "__version__", "")
    try:
        return int(version.split(".")[0]) >= 2
    except Exception:
        return False


_PYDANTIC_V2 = _is_pydantic_v2()

if _PYDANTIC_V2:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError
    from pydantic import field_validator, model_validator


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _ensure_value(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """
    Resolve a path relative to a base directory, handling user home expansion.
    
    Expands user home directory (~) and converts relative paths to absolute paths
    by resolving them against the base_dir. Absolute paths are returned as-is.
    
    Args:
        path: Path to resolve (can be relative or absolute)
        base_dir: Base directory to resolve relative paths against
    
    Returns:
        Resolved absolute path
    """
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return resolved.resolve(strict=False)


def _check_unknown_keys(data: Dict[str, Any], allowed: set[str], label: str) -> None:
    """
    Validate that a dictionary contains only allowed keys.
    
    Checks for unexpected keys in configuration dictionaries and raises an error
    if any unknown keys are found. This helps catch typos and invalid config options.
    
    Args:
        data: Dictionary to validate
        allowed: Set of allowed key names
        label: Label for error messages (e.g., "data", "universe")
    
    Raises:
        ConfigError: If any keys in data are not in the allowed set
    """
    unknown = set(data.keys()) - allowed
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ConfigError(f"Unknown {label} keys: {unknown_list}")


if _PYDANTIC_V2:

    class DataConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        mode: str
        prices_path: Optional[Path] = None
        cache_dir: Path = Path("data/raw")
        start: Optional[str] = None
        end: Optional[str] = None

        @field_validator("mode")
        @classmethod
        def _validate_mode(cls, value: str) -> str:
            _ensure_value(value in _DATA_MODES, f"data.mode must be one of: {sorted(_DATA_MODES)}")
            return value

        @field_validator("start", "end", mode="before")
        @classmethod
        def _coerce_date_fields(cls, value: Any) -> Optional[str]:
            return _coerce_date_str(value)

        @model_validator(mode="after")
        def _validate_csv_prices_path(self) -> "DataConfig":
            if self.mode == "csv" and self.prices_path is None:
                raise ValueError("data.prices_path is required when data.mode == 'csv'")
            return self

    class UniverseConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        symbols: List[str]
        min_history_days: int
        missing_data_policy: str

        @field_validator("symbols")
        @classmethod
        def _validate_symbols(cls, value: List[str]) -> List[str]:
            _ensure_value(bool(value), "universe.symbols must be a non-empty list")
            if not all(isinstance(item, str) for item in value):
                raise ValueError("universe.symbols must contain only strings")
            if len(set(value)) != len(value):
                raise ValueError("universe.symbols must be unique")
            return value

        @field_validator("min_history_days")
        @classmethod
        def _validate_min_history(cls, value: int) -> int:
            _ensure_value(value >= 1, "universe.min_history_days must be >= 1")
            return value

        @field_validator("missing_data_policy")
        @classmethod
        def _validate_policy(cls, value: str) -> str:
            _ensure_value(
                value in _MISSING_DATA_POLICIES,
                f"universe.missing_data_policy must be one of: {sorted(_MISSING_DATA_POLICIES)}",
            )
            return value

    class FeaturesConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        sma_fast: int
        sma_slow: int
        rsi_window: int
        rsi_low: float
        rsi_high: float

        @field_validator("sma_fast", "sma_slow", "rsi_window")
        @classmethod
        def _validate_min_two(cls, value: int, info: Any) -> int:
            _ensure_value(value >= 2, f"features.{info.field_name} must be >= 2")
            return value

        @field_validator("rsi_low", "rsi_high")
        @classmethod
        def _validate_rsi_bounds(cls, value: float, info: Any) -> float:
            _ensure_value(0 < value < 100, f"features.{info.field_name} must be between 0 and 100")
            return value

        @model_validator(mode="after")
        def _validate_relationships(self) -> "FeaturesConfig":
            if self.sma_fast >= self.sma_slow:
                raise ValueError("features.sma_fast must be < features.sma_slow")
            if self.rsi_low >= self.rsi_high:
                raise ValueError("features.rsi_low must be < features.rsi_high")
            return self

    class StrategyConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        name: str
        params: Dict[str, Any] = Field(default_factory=dict)

        @field_validator("name")
        @classmethod
        def _validate_name(cls, value: str) -> str:
            _ensure_value(value in _STRATEGY_NAMES, f"strategy.name must be one of: {sorted(_STRATEGY_NAMES)}")
            return value

        @field_validator("params")
        @classmethod
        def _validate_params(cls, value: Dict[str, Any]) -> Dict[str, Any]:
            if value is None:
                return {}
            if not isinstance(value, dict):
                raise ValueError("strategy.params must be a mapping")
            return value

    class ExecutionConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        cost_bps: float
        slippage_model: str
        slippage_params: Dict[str, Any] = Field(default_factory=dict)
        max_leverage: float
        max_weight_per_asset: float
        strict_weight_alignment: bool = False
        renorm_policy: str = "scale_down_if_exceeded"

        @field_validator("cost_bps")
        @classmethod
        def _validate_cost(cls, value: float) -> float:
            _ensure_value(value >= 0, "execution.cost_bps must be >= 0")
            return value

        @field_validator("slippage_model")
        @classmethod
        def _validate_slippage_model(cls, value: str) -> str:
            _ensure_value(
                value in _SLIPPAGE_MODELS,
                f"execution.slippage_model must be one of: {sorted(_SLIPPAGE_MODELS)}",
            )
            return value

        @field_validator("slippage_params")
        @classmethod
        def _validate_slippage_params(cls, value: Dict[str, Any]) -> Dict[str, Any]:
            if value is None:
                return {}
            if not isinstance(value, dict):
                raise ValueError("execution.slippage_params must be a mapping")
            return value

        @field_validator("max_leverage")
        @classmethod
        def _validate_max_leverage(cls, value: float) -> float:
            _ensure_value(value > 0, "execution.max_leverage must be > 0")
            return value

        @field_validator("max_weight_per_asset")
        @classmethod
        def _validate_max_weight(cls, value: float) -> float:
            _ensure_value(value > 0, "execution.max_weight_per_asset must be > 0")
            return value

        @field_validator("renorm_policy")
        @classmethod
        def _validate_renorm_policy(cls, value: str) -> str:
            _ensure_value(
                value in _RENORM_POLICIES,
                f"execution.renorm_policy must be one of: {sorted(_RENORM_POLICIES)}",
            )
            return value

        @model_validator(mode="after")
        def _validate_model(self) -> "ExecutionConfig":
            if self.max_weight_per_asset > self.max_leverage:
                raise ValueError("execution.max_weight_per_asset must be <= execution.max_leverage")
            if self.slippage_model == "vol_prop":
                slip_mult = self.slippage_params.get("slip_mult")
                vol_window = self.slippage_params.get("vol_window")
                if slip_mult is None or vol_window is None:
                    raise ValueError(
                        "execution.slippage_params must include slip_mult and vol_window when slippage_model == 'vol_prop'"
                    )
                if float(slip_mult) < 0:
                    raise ValueError("execution.slippage_params.slip_mult must be >= 0")
                if int(vol_window) < 2:
                    raise ValueError("execution.slippage_params.vol_window must be >= 2")
            return self

    class WalkForwardConfig(BaseModel):
        model_config = ConfigDict(extra="forbid")

        enabled: bool = False
        train_days: Optional[int] = None
        test_days: Optional[int] = None
        step_days: Optional[int] = None
        val_days: Optional[int] = None

        @field_validator("train_days", "test_days", "step_days")
        @classmethod
        def _validate_positive_days(cls, value: Optional[int], info: Any) -> Optional[int]:
            if value is None:
                return value
            _ensure_value(value >= 1, f"walkforward.{info.field_name} must be >= 1")
            return value

        @field_validator("val_days")
        @classmethod
        def _validate_val_days(cls, value: Optional[int]) -> Optional[int]:
            if value is None:
                return value
            _ensure_value(value >= 0, "walkforward.val_days must be >= 0")
            return value

        @model_validator(mode="after")
        def _validate_walkforward(self) -> "WalkForwardConfig":
            if self.enabled:
                missing = [
                    name
                    for name in ("train_days", "test_days", "step_days")
                    if getattr(self, name) is None
                ]
                if missing:
                    raise ValueError(
                        "walkforward.train_days, walkforward.test_days, and walkforward.step_days are required when walkforward.enabled is true"
                    )
                if self.step_days is not None and self.test_days is not None:
                    if self.step_days > self.test_days:
                        logger.warning("walkforward.step_days is greater than walkforward.test_days")
            return self

    class Config(BaseModel):
        model_config = ConfigDict(extra="forbid")

        run_id: str
        output_dir: Path = Path("artifacts")
        data: DataConfig
        universe: UniverseConfig
        features: FeaturesConfig
        strategy: StrategyConfig
        execution: ExecutionConfig
        walkforward: Optional[WalkForwardConfig] = None
        strategy_internal_risk_controls: bool = False
        universe_selection_mode: str = "train_only"

        @field_validator("run_id")
        @classmethod
        def _validate_run_id(cls, value: str) -> str:
            _ensure_value(bool(str(value).strip()), "run_id is required and cannot be empty")
            return value

        @field_validator("strategy_internal_risk_controls")
        @classmethod
        def _validate_internal_controls(cls, value: bool) -> bool:
            return bool(value)

        @field_validator("universe_selection_mode")
        @classmethod
        def _validate_universe_mode(cls, value: str) -> str:
            _ensure_value(
                value in _UNIVERSE_SELECTION_MODES,
                f"universe_selection_mode must be one of: {sorted(_UNIVERSE_SELECTION_MODES)}",
            )
            return value

        def to_dict(self) -> Dict[str, Any]:
            return self.model_dump(mode="json")

else:

    @dataclass
    class DataConfig:
        mode: str
        prices_path: Optional[Path] = None
        cache_dir: Path = Path("data/raw")
        start: Optional[str] = None
        end: Optional[str] = None

        def __post_init__(self) -> None:
            self.mode = str(self.mode)
            if self.prices_path is not None:
                self.prices_path = Path(self.prices_path)
            self.cache_dir = Path(self.cache_dir)
            self.start = _coerce_date_str(self.start)
            self.end = _coerce_date_str(self.end)
            _ensure(self.mode in _DATA_MODES, f"data.mode must be one of: {sorted(_DATA_MODES)}")
            if self.mode == "csv" and self.prices_path is None:
                raise ConfigError("data.prices_path is required when data.mode == 'csv'")

    @dataclass
    class UniverseConfig:
        symbols: List[str]
        min_history_days: int
        missing_data_policy: str

        def __post_init__(self) -> None:
            _ensure(bool(self.symbols), "universe.symbols must be a non-empty list")
            if not all(isinstance(item, str) for item in self.symbols):
                raise ConfigError("universe.symbols must contain only strings")
            if len(set(self.symbols)) != len(self.symbols):
                raise ConfigError("universe.symbols must be unique")
            _ensure(self.min_history_days >= 1, "universe.min_history_days must be >= 1")
            _ensure(
                self.missing_data_policy in _MISSING_DATA_POLICIES,
                f"universe.missing_data_policy must be one of: {sorted(_MISSING_DATA_POLICIES)}",
            )

    @dataclass
    class FeaturesConfig:
        sma_fast: int
        sma_slow: int
        rsi_window: int
        rsi_low: float
        rsi_high: float

        def __post_init__(self) -> None:
            _ensure(self.sma_fast >= 2, "features.sma_fast must be >= 2")
            _ensure(self.sma_slow >= 2, "features.sma_slow must be >= 2")
            _ensure(self.rsi_window >= 2, "features.rsi_window must be >= 2")
            _ensure(self.sma_fast < self.sma_slow, "features.sma_fast must be < features.sma_slow")
            _ensure(0 < float(self.rsi_low) < 100, "features.rsi_low must be between 0 and 100")
            _ensure(0 < float(self.rsi_high) < 100, "features.rsi_high must be between 0 and 100")
            _ensure(self.rsi_low < self.rsi_high, "features.rsi_low must be < features.rsi_high")

    @dataclass
    class StrategyConfig:
        name: str
        params: Dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            _ensure(self.name in _STRATEGY_NAMES, f"strategy.name must be one of: {sorted(_STRATEGY_NAMES)}")
            if self.params is None:
                self.params = {}
            if not isinstance(self.params, dict):
                raise ConfigError("strategy.params must be a mapping")

    @dataclass
    class ExecutionConfig:
        cost_bps: float
        slippage_model: str
        max_leverage: float
        max_weight_per_asset: float
        strict_weight_alignment: bool = False
        slippage_params: Dict[str, Any] = field(default_factory=dict)
        renorm_policy: str = "scale_down_if_exceeded"

        def __post_init__(self) -> None:
            self.cost_bps = float(self.cost_bps)
            self.max_leverage = float(self.max_leverage)
            self.max_weight_per_asset = float(self.max_weight_per_asset)
            self.strict_weight_alignment = bool(self.strict_weight_alignment)
            _ensure(self.cost_bps >= 0, "execution.cost_bps must be >= 0")
            _ensure(
                self.slippage_model in _SLIPPAGE_MODELS,
                f"execution.slippage_model must be one of: {sorted(_SLIPPAGE_MODELS)}",
            )
            if self.slippage_params is None:
                self.slippage_params = {}
            if not isinstance(self.slippage_params, dict):
                raise ConfigError("execution.slippage_params must be a mapping")
            _ensure(self.max_leverage > 0, "execution.max_leverage must be > 0")
            _ensure(self.max_weight_per_asset > 0, "execution.max_weight_per_asset must be > 0")
            if self.max_weight_per_asset > self.max_leverage:
                raise ConfigError("execution.max_weight_per_asset must be <= execution.max_leverage")
            _ensure(
                self.renorm_policy in _RENORM_POLICIES,
                f"execution.renorm_policy must be one of: {sorted(_RENORM_POLICIES)}",
            )
            if self.slippage_model == "vol_prop":
                slip_mult = self.slippage_params.get("slip_mult")
                vol_window = self.slippage_params.get("vol_window")
                if slip_mult is None or vol_window is None:
                    raise ConfigError(
                        "execution.slippage_params must include slip_mult and vol_window when slippage_model == 'vol_prop'"
                    )
                _ensure(float(slip_mult) >= 0, "execution.slippage_params.slip_mult must be >= 0")
                _ensure(int(vol_window) >= 2, "execution.slippage_params.vol_window must be >= 2")

    @dataclass
    class WalkForwardConfig:
        enabled: bool = False
        train_days: Optional[int] = None
        test_days: Optional[int] = None
        step_days: Optional[int] = None
        val_days: Optional[int] = None

        def __post_init__(self) -> None:
            for name in ("train_days", "test_days", "step_days"):
                value = getattr(self, name)
                if value is None:
                    continue
                _ensure(value >= 1, f"walkforward.{name} must be >= 1")
            if self.val_days is not None:
                _ensure(self.val_days >= 0, "walkforward.val_days must be >= 0")
            if self.enabled:
                missing = [
                    name
                    for name in ("train_days", "test_days", "step_days")
                    if getattr(self, name) is None
                ]
                if missing:
                    raise ConfigError(
                        "walkforward.train_days, walkforward.test_days, and walkforward.step_days are required when walkforward.enabled is true"
                    )
                if self.step_days is not None and self.test_days is not None:
                    if self.step_days > self.test_days:
                        logger.warning("walkforward.step_days is greater than walkforward.test_days")

    @dataclass
    class Config:
        run_id: str
        data: DataConfig
        universe: UniverseConfig
        features: FeaturesConfig
        strategy: StrategyConfig
        execution: ExecutionConfig
        output_dir: Path = Path("artifacts")
        walkforward: Optional[WalkForwardConfig] = None
        strategy_internal_risk_controls: bool = False
        universe_selection_mode: str = "train_only"

        def __post_init__(self) -> None:
            _ensure(bool(str(self.run_id).strip()), "run_id is required and cannot be empty")
            self.output_dir = Path(self.output_dir)
            self.strategy_internal_risk_controls = bool(self.strategy_internal_risk_controls)
            _ensure(
                self.universe_selection_mode in _UNIVERSE_SELECTION_MODES,
                f"universe_selection_mode must be one of: {sorted(_UNIVERSE_SELECTION_MODES)}",
            )

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "Config":
            if not isinstance(data, dict):
                raise ConfigError("Config root must be a mapping")
            _check_unknown_keys(
                data,
                {
                    "run_id",
                    "output_dir",
                    "data",
                    "universe",
                    "features",
                    "strategy",
                    "execution",
                    "walkforward",
                    "strategy_internal_risk_controls",
                    "universe_selection_mode",
                },
                "config",
            )
            try:
                run_id = data["run_id"]
            except KeyError as exc:
                raise ConfigError("run_id is required") from exc

            data_cfg = data.get("data")
            universe_cfg = data.get("universe")
            features_cfg = data.get("features")
            strategy_cfg = data.get("strategy")
            execution_cfg = data.get("execution")

            if not isinstance(data_cfg, dict):
                raise ConfigError("data must be a mapping")
            if not isinstance(universe_cfg, dict):
                raise ConfigError("universe must be a mapping")
            if not isinstance(features_cfg, dict):
                raise ConfigError("features must be a mapping")
            if not isinstance(strategy_cfg, dict):
                raise ConfigError("strategy must be a mapping")
            if not isinstance(execution_cfg, dict):
                raise ConfigError("execution must be a mapping")

            _check_unknown_keys(
                data_cfg,
                {"mode", "prices_path", "cache_dir", "start", "end"},
                "data",
            )
            _check_unknown_keys(
                universe_cfg,
                {"symbols", "min_history_days", "missing_data_policy"},
                "universe",
            )
            _check_unknown_keys(
                features_cfg,
                {"sma_fast", "sma_slow", "rsi_window", "rsi_low", "rsi_high"},
                "features",
            )
            _check_unknown_keys(strategy_cfg, {"name", "params"}, "strategy")
            _check_unknown_keys(
                execution_cfg,
                {
                    "cost_bps",
                    "slippage_model",
                    "slippage_params",
                    "max_leverage",
                    "max_weight_per_asset",
                    "strict_weight_alignment",
                    "renorm_policy",
                },
                "execution",
            )

            walkforward_cfg = data.get("walkforward")
            walkforward = None
            if walkforward_cfg is not None:
                if not isinstance(walkforward_cfg, dict):
                    raise ConfigError("walkforward must be a mapping")
                _check_unknown_keys(
                    walkforward_cfg,
                    {"enabled", "train_days", "test_days", "step_days", "val_days"},
                    "walkforward",
                )
                walkforward = WalkForwardConfig(**walkforward_cfg)

            try:
                return cls(
                    run_id=run_id,
                    data=DataConfig(**data_cfg),
                    universe=UniverseConfig(**universe_cfg),
                    features=FeaturesConfig(**features_cfg),
                    strategy=StrategyConfig(**strategy_cfg),
                    execution=ExecutionConfig(**execution_cfg),
                    output_dir=data.get("output_dir", "artifacts"),
                    walkforward=walkforward,
                    strategy_internal_risk_controls=data.get("strategy_internal_risk_controls", False),
                    universe_selection_mode=data.get("universe_selection_mode", "train_only"),
                )
            except TypeError as exc:
                raise ConfigError(str(exc)) from exc

        def to_dict(self) -> Dict[str, Any]:
            return _serialize_dataclass(self)


def _serialize_dataclass(obj: Any) -> Any:
    """
    Recursively serialize a dataclass to a JSON-compatible dictionary.
    
    Converts dataclass instances to dictionaries, Path objects to strings,
    and recursively processes nested structures (dicts, lists, dataclasses).
    Used for converting Config objects to JSON format.
    
    Args:
        obj: Object to serialize (can be dataclass, Path, dict, list, or primitive)
    
    Returns:
        JSON-compatible representation of the object
    """
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return {field_info.name: _serialize_dataclass(getattr(obj, field_info.name)) for field_info in fields(obj)}
    if isinstance(obj, dict):
        return {key: _serialize_dataclass(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialize_dataclass(value) for value in obj]
    return obj


def load_config(yaml_path: str | Path) -> "Config":
    cfg_path = Path(yaml_path)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")

    if _PYDANTIC_V2:
        try:
            cfg = Config.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(str(exc)) from exc
    else:
        cfg = Config.from_dict(raw)

    cfg = resolve_config(cfg, cfg_path)
    logger.info("Loaded config: %s", cfg_path.resolve(strict=False))
    return cfg


def resolve_config(cfg: "Config", yaml_path: Path) -> "Config":
    base_dir = yaml_path.parent.resolve(strict=False)
    cfg.output_dir = _resolve_path(Path(cfg.output_dir), base_dir)
    cfg.data.cache_dir = _resolve_path(Path(cfg.data.cache_dir), base_dir)
    if cfg.data.prices_path is not None:
        cfg.data.prices_path = _resolve_path(Path(cfg.data.prices_path), base_dir)

    if cfg.data.mode == "csv":
        if cfg.data.prices_path is None:
            raise ConfigError("data.prices_path is required when data.mode == 'csv'")
        if not cfg.data.prices_path.exists():
            raise ConfigError(f"data.prices_path does not exist: {cfg.data.prices_path}")

    if cfg.strategy.name == "ml_gated":
        params = dict(getattr(cfg.strategy, "params", {}) or {})
        preds_path = params.get("preds_path") or params.get("predictions_path")
        if preds_path is not None:
            params["preds_path"] = _resolve_path(Path(preds_path), base_dir)
            cfg.strategy.params = params

    return cfg


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def write_resolved_config(out_dir: Path, cfg: "Config") -> str:
    """
    Write the resolved configuration to a JSON file and compute its hash.
    
    Serializes the configuration object to JSON format with resolved absolute paths,
    writes it to the output directory, and computes a SHA256 hash for reproducibility
    tracking.
    
    Args:
        out_dir: Directory to write the config.json file to (created if doesn't exist)
        cfg: Configuration object to serialize
    
    Returns:
        SHA256 hash of the serialized configuration (hex string)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "config.json"
    payload = _json_bytes(cfg.to_dict())
    cfg_path.write_bytes(payload)
    config_sha256 = hashlib.sha256(payload).hexdigest()
    logger.info("Wrote resolved config: %s", cfg_path)
    return config_sha256


def get_git_commit() -> Optional[str]:
    """
    Retrieve the current Git commit hash for reproducibility tracking.
    
    Attempts to get the HEAD commit hash from the repository. Fails gracefully
    if Git is not available, the directory is not a Git repo, or any error occurs.
    
    Returns:
        Git commit hash (40-character hex string) if successful, None otherwise
    """
    try:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except Exception:
        return None


def write_run_metadata(
    out_dir: Path,
    cfg: "Config",
    yaml_path: Path,
    config_sha256: Optional[str] = None,
) -> None:
    """
    Write comprehensive run metadata for reproducibility and tracking.
    
    Creates a run_metadata.json file containing information about the execution
    environment, configuration, and timing. This enables full reproducibility
    of backtest runs and helps track experiment history.
    
    Metadata includes:
    - Timestamp of run creation
    - Python version and platform information
    - Git commit hash (if available)
    - Configuration file hash and path
    - Run ID
    
    Args:
        out_dir: Directory to write run_metadata.json to (created if doesn't exist)
        cfg: Configuration object containing run settings
        yaml_path: Path to the original YAML config file
        config_sha256: Optional pre-computed config hash (computed if not provided)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if config_sha256 is None:
        cfg_path = out_dir / "config.json"
        try:
            payload = cfg_path.read_bytes()
        except FileNotFoundError:
            payload = _json_bytes(cfg.to_dict())
        config_sha256 = hashlib.sha256(payload).hexdigest()
    metadata = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit(),
        "config_sha256": config_sha256,
        "config_path": str(Path(yaml_path).resolve(strict=False)),
        "run_id": cfg.run_id,
    }
    meta_path = out_dir / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info("Wrote run metadata: %s", meta_path)


def check_config_integrity(cfg: "Config | Dict[str, Any]") -> Dict[str, Any]:
    """
    Perform objective config integrity checks (non-heuristic).

    Raises ConfigError if a required invariant is violated.
    """
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    features = cfg_dict.get("features", {})
    universe = cfg_dict.get("universe", {})
    walkforward = cfg_dict.get("walkforward") or {}

    min_history = int(universe.get("min_history_days", 0))
    sma_slow = int(features.get("sma_slow", 0))
    rsi_window = int(features.get("rsi_window", 0))
    required_history = max(sma_slow, rsi_window)

    min_history_ok = min_history >= required_history
    if not min_history_ok:
        raise ConfigError(
            "universe.min_history_days must be >= max(features.sma_slow, features.rsi_window)"
        )

    wf_enabled = bool(walkforward.get("enabled"))
    step_days = walkforward.get("step_days")
    test_days = walkforward.get("test_days")
    wf_step_ok = True
    if wf_enabled and step_days is not None and test_days is not None:
        wf_step_ok = int(step_days) <= int(test_days)

    return {
        "min_history_days": min_history,
        "required_history_days": required_history,
        "min_history_ok": bool(min_history_ok),
        "walkforward_enabled": wf_enabled,
        "walkforward_step_le_test_ok": bool(wf_step_ok),
    }
