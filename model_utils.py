from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

FEATURE_COLUMNS: list[str] = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
TARGET_COLUMN: str = "HeartDisease"
ZERO_IMPUTE_COLUMNS: list[str] = ["RestingBP", "Cholesterol"]

SEX_MAP: dict[int, str] = {
    0: "หญิง (0)",
    1: "ชาย (1)",
}

CHEST_PAIN_MAP: dict[int, str] = {
    1: "ASY — ไม่มีอาการเจ็บหน้าอกชัดเจน (1)",
    2: "ATA — เจ็บหน้าอกไม่จำเพาะ (2)",
    3: "NAP — เจ็บหน้าอกที่ไม่ใช่โรคหัวใจ (3)",
    4: "TA — เจ็บหน้าอกแบบจำเพาะ (4)",
}

FASTING_BS_MAP: dict[int, str] = {
    0: "ปกติ ≤ 120 mg/dl (0)",
    1: "สูง > 120 mg/dl (1)",
}

RESTING_ECG_MAP: dict[int, str] = {
    1: "LVH — กล้ามเนื้อหัวใจห้องล่างซ้ายหนา (1)",
    2: "Normal — คลื่นไฟฟ้าหัวใจปกติ (2)",
    3: "ST — คลื่น ST-T ผิดปกติ (3)",
}

EXERCISE_ANGINA_MAP: dict[int, str] = {
    0: "ไม่มีอาการเจ็บหน้าอกขณะออกกำลังกาย (0)",
    1: "มีอาการเจ็บหน้าอกขณะออกกำลังกาย (1)",
}

ST_SLOPE_MAP: dict[int, str] = {
    1: "Down — ลาดลง (1)",
    2: "Flat — แนวราบ (2)",
    3: "Up — ลาดขึ้น (3)",
}

TARGET_MAP: dict[int, str] = {
    0: "ปกติ (ไม่พบความเสี่ยง)",
    1: "มีความเสี่ยงโรคหัวใจ",
}

FEATURE_THAI_NAMES: dict[str, str] = {
    "Age": "อายุ (ปี)",
    "Sex": "เพศ",
    "ChestPainType": "ประเภทอาการเจ็บหน้าอก",
    "RestingBP": "ความดันโลหิตขณะพัก (mm Hg)",
    "Cholesterol": "คอเลสเตอรอล (mg/dl)",
    "FastingBS": "น้ำตาลในเลือดหลังอดอาหาร",
    "RestingECG": "คลื่นไฟฟ้าหัวใจขณะพัก",
    "MaxHR": "อัตราการเต้นหัวใจสูงสุด (bpm)",
    "ExerciseAngina": "เจ็บหน้าอกขณะออกกำลังกาย",
    "Oldpeak": "ค่า ST Depression (Oldpeak)",
    "ST_Slope": "ความชันของคลื่น ST",
}


@dataclass
class KNNModelBundle:
    pipeline: Pipeline
    medians: dict[str, float]
    impute_zeros: bool
    k: int
    weights: str
    metric: str
    scaler_type: str
    metrics: dict[str, Any]
    X_train_raw: pd.DataFrame
    X_train_clean: pd.DataFrame
    y_train: pd.Series
    X_test_clean: pd.DataFrame
    y_test: pd.Series
    y_pred: np.ndarray
    y_prob: np.ndarray


def load_data(csv_path: str | Path = "Heart3.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"ไม่พบคอลัมน์ที่จำเป็นในไฟล์ข้อมูล: {sorted(missing)}")
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def compute_nonzero_medians(df: pd.DataFrame) -> dict[str, float]:
    medians: dict[str, float] = {}
    for col in ZERO_IMPUTE_COLUMNS:
        valid = df.loc[df[col] > 0, col]
        medians[col] = float(valid.median()) if len(valid) > 0 else float(df[col].median())
    return medians


def apply_imputation(
    df: pd.DataFrame,
    impute_zeros: bool,
    medians: dict[str, float],
) -> pd.DataFrame:
    out = df[FEATURE_COLUMNS].copy()
    if impute_zeros:
        for col in ZERO_IMPUTE_COLUMNS:
            out.loc[out[col] == 0, col] = medians[col]
    return out


def build_scaler(scaler_type: str):
    if scaler_type == "standard":
        return StandardScaler()
    if scaler_type == "minmax":
        return MinMaxScaler()
    if scaler_type == "robust":
        return RobustScaler()
    raise ValueError(f"ไม่รองรับวิธีสเกลข้อมูล: {scaler_type}")


def train_knn_model(
    df: pd.DataFrame,
    k: int = 9,
    weights: str = "distance",
    metric: str = "manhattan",
    scaler_type: str = "standard",
    impute_zeros: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> KNNModelBundle:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    medians = compute_nonzero_medians(X_train_raw)
    X_train_clean = apply_imputation(X_train_raw, impute_zeros, medians)
    X_test_clean = apply_imputation(X_test_raw, impute_zeros, medians)

    pipeline = Pipeline(
        steps=[
            ("scaler", build_scaler(scaler_type)),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=k,
                    weights=weights,
                    metric=metric,
                ),
            ),
        ]
    )
    pipeline.fit(X_train_clean, y_train)

    y_pred = pipeline.predict(X_test_clean)
    y_prob = pipeline.predict_proba(X_test_clean)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipeline, X_train_clean, y_train, cv=cv, scoring="accuracy")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=[TARGET_MAP[0], TARGET_MAP[1]],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "confusion_matrix": cm,
        "fpr": fpr,
        "tpr": tpr,
        "classification_report": report_dict,
    }

    return KNNModelBundle(
        pipeline=pipeline,
        medians=medians,
        impute_zeros=impute_zeros,
        k=k,
        weights=weights,
        metric=metric,
        scaler_type=scaler_type,
        metrics=metrics,
        X_train_raw=X_train_raw,
        X_train_clean=X_train_clean,
        y_train=y_train,
        X_test_clean=X_test_clean,
        y_test=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
    )


def evaluate_k_range(
    df: pd.DataFrame,
    k_values: list[int],
    weights: str = "distance",
    metric: str = "manhattan",
    scaler_type: str = "standard",
    impute_zeros: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    medians = compute_nonzero_medians(X_train_raw)
    X_train_clean = apply_imputation(X_train_raw, impute_zeros, medians)
    X_test_clean = apply_imputation(X_test_raw, impute_zeros, medians)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows = []
    for k_val in k_values:
        pipe = Pipeline(
            steps=[
                ("scaler", build_scaler(scaler_type)),
                (
                    "knn",
                    KNeighborsClassifier(
                        n_neighbors=k_val,
                        weights=weights,
                        metric=metric,
                    ),
                ),
            ]
        )
        pipe.fit(X_train_clean, y_train)
        test_acc = accuracy_score(y_test, pipe.predict(X_test_clean))
        cv_acc = cross_val_score(pipe, X_train_clean, y_train, cv=cv, scoring="accuracy").mean()
        rows.append(
            {
                "K": k_val,
                "Test Accuracy": float(test_acc),
                "5-Fold CV Accuracy": float(cv_acc),
            }
        )
    return pd.DataFrame(rows)


def predict_single(
    bundle: KNNModelBundle,
    input_data: dict[str, float | int],
) -> dict[str, Any]:
    row_df = pd.DataFrame([input_data])[FEATURE_COLUMNS]
    clean_df = apply_imputation(row_df, bundle.impute_zeros, bundle.medians)

    pred_class = int(bundle.pipeline.predict(clean_df)[0])
    prob_disease = float(bundle.pipeline.predict_proba(clean_df)[0, 1])

    scaler = bundle.pipeline.named_steps["scaler"]
    knn = bundle.pipeline.named_steps["knn"]
    scaled_point = scaler.transform(clean_df)
    distances, indices = knn.kneighbors(scaled_point, n_neighbors=bundle.k)

    neighbors_df = bundle.X_train_raw.iloc[indices[0]].copy()
    neighbors_df.insert(0, "ระยะห่าง (Distance)", np.round(distances[0], 4))
    neighbors_df["ผลจริง (HeartDisease)"] = bundle.y_train.iloc[indices[0]].map(TARGET_MAP).values

    return {
        "prediction": pred_class,
        "label_th": TARGET_MAP[pred_class],
        "prob_disease": prob_disease,
        "prob_normal": 1.0 - prob_disease,
        "imputed_values": clean_df.iloc[0].to_dict(),
        "neighbors": neighbors_df.reset_index(drop=True),
    }


def predict_batch(bundle: KNNModelBundle, new_df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [col for col in FEATURE_COLUMNS if col not in new_df.columns]
    if missing_cols:
        raise ValueError(
            "ไฟล์ CSV ขาดคอลัมน์ที่จำเป็น: " + ", ".join(missing_cols)
        )

    work_df = new_df.copy()
    for col in FEATURE_COLUMNS:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    if work_df[FEATURE_COLUMNS].isnull().any().any():
        bad_rows = int(work_df[FEATURE_COLUMNS].isnull().any(axis=1).sum())
        raise ValueError(
            f"พบข้อมูลที่ไม่ใช่ตัวเลขหรือค่าว่างในฟีเจอร์จำนวน {bad_rows} แถว กรุณาตรวจสอบไฟล์ก่อนทำนาย"
        )

    clean_df = apply_imputation(work_df, bundle.impute_zeros, bundle.medians)
    preds = bundle.pipeline.predict(clean_df)
    probs = bundle.pipeline.predict_proba(clean_df)[:, 1]

    result_df = new_df.copy()
    result_df["ผลการทำนาย (HeartDisease)"] = preds
    result_df["คำอธิบายผล"] = [TARGET_MAP[int(p)] for p in preds]
    result_df["ความน่าจะเป็นโรคหัวใจ (%)"] = np.round(probs * 100, 2)
    return result_df
