"""Data loading and splitting module for SPY intraday data.

This module handles loading CSV data, parsing timestamps, and creating
train/validation/test splits based on date ranges.
"""

from pathlib import Path

import pandas as pd

from rl_trading.config import DataConfig


# Required columns that must be present in the CSV
REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


class SPYDataLoader:
    """Loads and splits SPY intraday bar data from CSV.
    
    The loader expects a CSV file with at minimum the following columns:
    - timestamp: datetime column for each bar
    - open, high, low, close: OHLC prices
    - volume: trading volume
    
    Attributes:
        data_config: DataConfig instance with paths and date ranges.
        df: The loaded DataFrame sorted by timestamp.
    
    Example:
        >>> from rl_trading.config import DataConfig
        >>> config = DataConfig(csv_path="data/spy_30m_2019_2025.csv")
        >>> loader = SPYDataLoader(config)
        >>> train_df, val_df, test_df = loader.get_splits()
    """
    
    def __init__(self, data_config: DataConfig) -> None:
        """Initialize the data loader and load the CSV.
        
        Args:
            data_config: Configuration with CSV path and date ranges.
        
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If required columns are missing from the CSV.
        """
        self.data_config = data_config
        self.df = self._load_csv()
    
    def _load_csv(self) -> pd.DataFrame:
        """Load and prepare the CSV data.
        
        Returns:
            DataFrame with parsed timestamps, sorted ascending.
        
        Raises:
            FileNotFoundError: If CSV file does not exist.
            ValueError: If required columns are missing.
        """
        csv_path = Path(self.data_config.csv_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_path.absolute()}. "
                f"Please place your SPY intraday data CSV at this location."
            )
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        missing_cols = REQUIRED_COLUMNS - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. "
                f"Required columns are: {REQUIRED_COLUMNS}"
            )
        
        # Parse timestamp and sort
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df
    
    def _filter_by_date_range(
        self, 
        start: str, 
        end: str | None
    ) -> pd.DataFrame:
        """Filter DataFrame to a date range.
        
        Args:
            start: Start date string (inclusive), e.g., "2019-01-01".
            end: End date string (inclusive), or None for latest available.
        
        Returns:
            Filtered DataFrame within the specified date range.
        """
        start_dt = pd.to_datetime(start)
        
        if end is None:
            end_dt = self.df["timestamp"].max()
        else:
            end_dt = pd.to_datetime(end)
        
        mask = (self.df["timestamp"] >= start_dt) & (self.df["timestamp"] <= end_dt)
        return self.df.loc[mask].copy().reset_index(drop=True)
    
    def get_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Get train, validation, and test DataFrames based on config date ranges.
        
        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames.
        
        Raises:
            ValueError: If any split results in an empty DataFrame.
        
        Example:
            >>> loader = SPYDataLoader(DataConfig())
            >>> train_df, val_df, test_df = loader.get_splits()
            >>> print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        """
        train_df = self._filter_by_date_range(
            self.data_config.train_start,
            self.data_config.train_end
        )
        val_df = self._filter_by_date_range(
            self.data_config.val_start,
            self.data_config.val_end
        )
        test_df = self._filter_by_date_range(
            self.data_config.test_start,
            self.data_config.test_end
        )
        
        # Validate non-empty splits
        if train_df.empty:
            raise ValueError(
                f"Training split is empty for date range "
                f"[{self.data_config.train_start}, {self.data_config.train_end}]. "
                f"Check your data and date configuration."
            )
        if val_df.empty:
            raise ValueError(
                f"Validation split is empty for date range "
                f"[{self.data_config.val_start}, {self.data_config.val_end}]. "
                f"Check your data and date configuration."
            )
        if test_df.empty:
            raise ValueError(
                f"Test split is empty for date range "
                f"[{self.data_config.test_start}, {self.data_config.test_end}]. "
                f"Check your data and date configuration."
            )
        
        return train_df, val_df, test_df
    
    def get_full_df(self) -> pd.DataFrame:
        """Return a copy of the full loaded DataFrame.
        
        Returns:
            Copy of the complete DataFrame.
        """
        return self.df.copy()
    
    def get_date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Return the date range of the loaded data.
        
        Returns:
            Tuple of (min_timestamp, max_timestamp).
        """
        return self.df["timestamp"].min(), self.df["timestamp"].max()
    
    def __repr__(self) -> str:
        min_ts, max_ts = self.get_date_range()
        return (
            f"SPYDataLoader(rows={len(self.df)}, "
            f"range=[{min_ts.date()}, {max_ts.date()}])"
        )

