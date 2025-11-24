import pandas as pd
import logging
import os

logger = logging.getLogger("market_regime.loader")

def load_and_merge_regimes(features_df: pd.DataFrame, regime_path: str = "data/regimes/regime_labels.csv") -> pd.DataFrame:
    """
    Merge features with pre-computed regime labels.
    """
    logger.info(f"Loading regimes from {regime_path}...")
    
    if not os.path.exists(regime_path):
        logger.warning(f"Regime file not found at {regime_path}.")
        # For robust pipeline demonstration, we raise error, 
        # but main.py will handle generation if missing.
        raise FileNotFoundError(f"Regime labels file not found: {regime_path}")
        
    regimes = pd.read_csv(regime_path)
    
    # Parse dates
    if 'Date' not in regimes.columns:
        raise ValueError("Regime file must have 'Date' column.")
    
    regimes['Date'] = pd.to_datetime(regimes['Date'])
    regimes.set_index('Date', inplace=True)
    
    # Ensure 'regime' column exists
    if 'regime' not in regimes.columns:
        raise ValueError("Regime file must have 'regime' column.")
    
    # Merge on index (Date)
    # features_df should have Date index
    merged = features_df.join(regimes[['regime']], how='inner')
    
    # Drop rows without regime label (already done by inner join, but safety check)
    merged.dropna(subset=['regime'], inplace=True)
    
    # Cast regime to int
    merged['regime'] = merged['regime'].astype(int)
    
    output_path = "data/features/features_with_regime.csv"
    merged.to_csv(output_path)
    logger.info(f"Merged features and regimes. Shape: {merged.shape}")
    logger.info(f"Saved to {output_path}")
    
    return merged

