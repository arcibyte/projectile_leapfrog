import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

from data.dataset import FEATURES, TARGETS
from config import (
    TEST_SIZE, N_ESTIMATORS, MAX_DEPTH,
    MIN_SAMPLES_SPLIT, MIN_SAMPLES_LEAF,
    CV_FOLDS, RANDOM_SEED, OUTPUT_DIR,
)

MODEL_PATH = os.path.join(OUTPUT_DIR, "random_forest.joblib")


def split_data(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df[TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    print(f"  train: {len(X_train):,} muestras | test: {len(X_test):,} muestras")
    return X_train, X_test, y_train, y_test


def build_model(n_estimators: int = N_ESTIMATORS,
                max_depth: int = MAX_DEPTH) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def train(df: pd.DataFrame) -> tuple:

    X_train, X_test, y_train, y_test = split_data(df)

    model = build_model()
    print("entrenando Random Forest por pasos temporales (aproximador dinamico)")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    _print_metrics(y_test, y_pred)

    return model, X_test, y_test, y_pred


def evaluate(model: RandomForestRegressor,
             X_test: np.ndarray,
             y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "mae_x": mean_absolute_error(y_test[:, 0], y_pred[:, 0]),
        "mae_y": mean_absolute_error(y_test[:, 1], y_pred[:, 1]),
        "r2_x":  r2_score(y_test[:, 0], y_pred[:, 0]),
        "r2_y":  r2_score(y_test[:, 1], y_pred[:, 1]),
    }


def cross_validate(df: pd.DataFrame, n_est: int = 50) -> None:
    X = df[FEATURES].values
    y = df[TARGETS].values
    model = build_model(n_estimators=n_est)

    for i, target in enumerate(TARGETS):
        scores = cross_val_score(
            model, X, y[:, i],
            cv=CV_FOLDS, scoring="r2", n_jobs=-1
        )
        print(f"  CV R² {target}: {scores.mean():.5f} ± {scores.std():.5f}")


def save_model(model: RandomForestRegressor,
               path: str = MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"modelo guardado en: {path}")


def load_model(path: str = MODEL_PATH) -> RandomForestRegressor:
    model = joblib.load(path)
    print(f"modelo cargado en: {path}")
    return model


def _print_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    mae_x = mean_absolute_error(y_test[:, 0], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test[:, 1], y_pred[:, 1])
    r2_x  = r2_score(y_test[:, 0], y_pred[:, 0])
    r2_y  = r2_score(y_test[:, 1], y_pred[:, 1])

    sep = "=" * 45
    print(f"\n{sep}")
    print("  métricas de prueba")
    print(sep)
    print(f"  MAE  x_next : {mae_x:.6f} m")
    print(f"  MAE  y_next : {mae_y:.6f} m")
    print(f"  R²   x_next : {r2_x:.6f}")
    print(f"  R²   y_next : {r2_y:.6f}")
    print(sep)