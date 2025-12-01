from typing import List
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from autotab.config import ExperimentConfig

def build_numeric_pipeline(config: ExperimentConfig) -> Pipeline:
    """
    Build preprocessing pipeline for numeric features.
    """
    steps = []
    
    # Add imputer
    steps.append(('imputer', SimpleImputer(strategy=config.preprocessing.impute_strategy_numeric)))
    
    # Add scaler if configured
    if config.preprocessing.scale_numeric:
        steps.append(('scaler', StandardScaler()))
    
    return Pipeline(steps)

def build_categorical_pipeline(config: ExperimentConfig) -> Pipeline:
    """
    Build preprocessing pipeline for categorical features.
    """
    steps = []
    
    # Add imputer
    steps.append(('imputer', SimpleImputer(strategy=config.preprocessing.impute_strategy_categorical)))
    
    # Add one-hot encoder if configured
    if config.preprocessing.one_hot_encode_categorical:
        steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    
    return Pipeline(steps)

def build_preprocessor(
    config: ExperimentConfig,
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Build a ColumnTransformer combining numeric and categorical pipelines.
    """
    transformers = []
    
    # Add numeric transformer if there are numeric features
    if numeric_features:
        numeric_pipeline = build_numeric_pipeline(config)
        transformers.append(('num', numeric_pipeline, numeric_features))
    
    # Add categorical transformer if there are categorical features
    if categorical_features:
        categorical_pipeline = build_categorical_pipeline(config)
        transformers.append(('cat', categorical_pipeline, categorical_features))
    
    return ColumnTransformer(transformers=transformers, remainder='drop')

def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get feature names after preprocessing transformation.
    """
    return preprocessor.get_feature_names_out().tolist()
