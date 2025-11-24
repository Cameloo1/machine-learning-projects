"""Data loading and feature preparation utilities."""

from __future__ import annotations

import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from .config import LABELED_DATA_PATH

RE_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
RE_MULTI_SPACE = re.compile(r"\s+")
REQUIRED_COLUMNS = {"ticker", "headline", "sentiment"}


def load_labeled_data(path: str | os.PathLike[str] = LABELED_DATA_PATH) -> pd.DataFrame:
    """Load manually labeled data and apply basic cleaning."""
    path = os.fspath(path)
    df = pd.read_csv(path)
    missing_cols = REQUIRED_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Labeled dataset missing columns: {missing_cols}")

    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df.drop_duplicates(subset=["headline"])
    df["ticker"] = df["ticker"].str.upper()
    df["sentiment"] = df["sentiment"].str.title()
    df["cleaned"] = df["headline"].astype(str).apply(clean_text)
    df = df[df["cleaned"].str.len() >= 3]
    df = df.reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """Normalize a headline for modeling."""
    text = text.lower()
    text = RE_NON_ALNUM.sub(" ", text)
    text = RE_MULTI_SPACE.sub(" ", text).strip()
    return text


def build_vectorizers(
    df: pd.DataFrame,
) -> Tuple[TfidfVectorizer, OneHotEncoder]:
    """Fit TF-IDF and ticker encoders."""
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_vectorizer.fit(df["cleaned"])

    ticker_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )
    ticker_encoder.fit(df[["ticker"]])
    return tfidf_vectorizer, ticker_encoder


def transform_features(
    df: pd.DataFrame,
    tfidf_vectorizer: TfidfVectorizer,
    ticker_encoder: OneHotEncoder,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Transform dataframe into model-ready feature matrix and labels."""
    text_features = tfidf_vectorizer.transform(df["cleaned"])
    ticker_features = ticker_encoder.transform(df[["ticker"]])
    ticker_sparse = sparse.csr_matrix(ticker_features)
    X = sparse.hstack([text_features, ticker_sparse]).tocsr()
    y = df["sentiment"].values
    return X, y


