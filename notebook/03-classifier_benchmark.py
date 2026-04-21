"""
Benchmark classical classifiers against the feed-forward neural network used by
MegaPlantTF binary family classifiers.

Run from the repository root or from the notebook/ directory.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

# On Windows, TensorFlow can crash natively if it is imported after parts of the
# scientific stack. Import it first only for FNN runs; classical benchmarks still
# work in environments without TensorFlow.
if any("fnn" in arg for arg in sys.argv):
    import tensorflow as _tensorflow  # noqa: F401

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

PROTEIN_DOMAIN = "ACDEFGHIKLMNPQRSTVWYX"
DEFAULT_MODELS = [
    "knn",
    "xgb",
    "random_forest",
    "adaboost",
    "gaussian_nb",
    "svc_linear",
    "svc_rbf",
    "fnn",
]
ALL_MODELS = DEFAULT_MODELS + ["gaussian_process"]
ALL_MODELS = ALL_MODELS + ["fnn_pretrained"]


def repo_root() -> Path:
    here = Path(__file__).resolve()
    return here.parents[1]


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def kmer_frequency(sequence: str, k: int, domain: str = PROTEIN_DOMAIN) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    sequence = str(sequence).upper()
    allowed = set(domain)

    for i in range(0, len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        if all(char in allowed for char in kmer):
            counts[kmer] += 1
            total += 1

    if total == 0:
        return {}
    return {kmer: count / total for kmer, count in counts.items()}


def fit_kmer_vectorizer(df: pd.DataFrame, k: int):
    kmers = [kmer_frequency(seq, k) for seq in df["sequence"]]
    vectorizer = DictVectorizer(sparse=True)
    x = vectorizer.fit_transform(kmers)
    y = df["class"].astype(int).to_numpy()
    return x, y, vectorizer


def transform_kmers(df: pd.DataFrame, k: int, vectorizer: DictVectorizer):
    kmers = [kmer_frequency(seq, k) for seq in df["sequence"]]
    x = vectorizer.transform(kmers)
    y = df["class"].astype(int).to_numpy()
    return x, y


def sample_dataset(df: pd.DataFrame, max_rows: int | None, random_state: int) -> pd.DataFrame:
    if not max_rows or len(df) <= max_rows:
        return df.reset_index(drop=True)

    _, sampled = train_test_split(
        df,
        test_size=max_rows,
        stratify=df["class"],
        random_state=random_state,
    )
    return sampled.reset_index(drop=True)


def select_features(x_train, y_train, x_test, k_features: int):
    k_features = min(k_features, x_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_features)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)
    return x_train_selected, x_test_selected, selector


def select_features_by_names(x_train, x_test, feature_names, selected_feature_names):
    feature_to_index = {name: index for index, name in enumerate(feature_names)}
    columns_train = []
    columns_test = []
    missing = 0

    for feature_name in selected_feature_names:
        index = feature_to_index.get(feature_name)
        if index is None:
            missing += 1
            columns_train.append(sparse.csr_matrix((x_train.shape[0], 1), dtype=x_train.dtype))
            columns_test.append(sparse.csr_matrix((x_test.shape[0], 1), dtype=x_test.dtype))
        else:
            columns_train.append(x_train[:, index])
            columns_test.append(x_test[:, index])

    return sparse.hstack(columns_train).tocsr(), sparse.hstack(columns_test).tocsr(), missing


def load_pretrained_feature_mask(root: Path, family: str, k: int) -> list[str]:
    meta_path = root / "models" / "Binary-Classifier" / family / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing pretrained meta file: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)

    family_meta = meta.get(family)
    if family_meta is None and meta:
        family_meta = next(iter(meta.values()))
    if family_meta is None:
        raise ValueError(f"No metadata found in {meta_path}")

    model_key = f"FEEDFORWARD_k{k}"
    model_meta = family_meta.get(model_key)
    if model_meta is None:
        raise ValueError(f"No {model_key} metadata found in {meta_path}")

    features_mask = model_meta.get("features_mask")
    if not features_mask:
        raise ValueError(f"No features_mask found for {model_key} in {meta_path}")

    return [features_mask[key] for key in sorted(features_mask, key=lambda value: int(value))]


def build_sklearn_model(name: str, random_state: int, n_jobs: int):
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "adaboost":
        return AdaBoostClassifier(n_estimators=200, random_state=random_state)
    if name == "gaussian_nb":
        return GaussianNB()
    if name == "svc_linear":
        return SVC(kernel="linear", random_state=random_state)
    if name == "svc_rbf":
        return SVC(kernel="rbf", random_state=random_state)
    if name == "gaussian_process":
        return GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=random_state)
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is not installed in this environment") from exc
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )
    raise ValueError(f"Unknown model: {name}")


def build_fnn(input_dim: int, k: int, random_state: int):
    try:
        import tensorflow as tf
        from keras.layers import Dense, Dropout, Input
        from keras.models import Sequential
    except ImportError as exc:
        raise RuntimeError("tensorflow/keras is not installed in this environment") from exc

    tf.keras.utils.set_random_seed(random_state)
    model = Sequential(name=f"FEEDFORWARD_k{k}_benchmark")
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_sklearn_model(name, x_train, y_train, x_test, y_test, args):
    model = build_sklearn_model(name, args.random_state, args.n_jobs)
    train_x = x_train.toarray() if name == "gaussian_nb" else x_train
    test_x = x_test.toarray() if name == "gaussian_nb" else x_test

    start = time.perf_counter()
    model.fit(train_x, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_pred = model.predict(test_x)
    predict_time = time.perf_counter() - start
    return evaluate_predictions(y_test, y_pred), train_time, predict_time


def run_fnn(x_train, y_train, x_test, y_test, k, args):
    x_train_dense = x_train.toarray().astype("float32")
    x_test_dense = x_test.toarray().astype("float32")

    x_t, x_v, y_t, y_v = train_test_split(
        x_train_dense,
        y_train,
        train_size=0.8,
        stratify=y_train,
        random_state=args.random_state,
    )

    model = build_fnn(x_train_dense.shape[1], k, args.random_state)
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("tensorflow/keras is not installed in this environment") from exc

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.fnn_patience,
            restore_best_weights=True,
        )
    ]

    start = time.perf_counter()
    model.fit(
        x_t,
        y_t,
        validation_data=(x_v, y_v),
        epochs=args.fnn_epochs,
        batch_size=args.fnn_batch_size,
        callbacks=callbacks,
        verbose=args.keras_verbose,
    )
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    y_prob = model.predict(x_test_dense, verbose=0).reshape(-1)
    predict_time = time.perf_counter() - start
    y_pred = (y_prob >= 0.5).astype(int)
    return evaluate_predictions(y_test, y_pred), train_time, predict_time


def run_fnn_pretrained(root: Path, family: str, x_test, y_test, k: int, batch_size: int):
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise RuntimeError("tensorflow/keras is not installed in this environment") from exc

    model_path = root / "models" / "Binary-Classifier" / family / f"FEEDFORWARD_k{k}.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing pretrained model file: {model_path}")

    x_test_dense = x_test.toarray().astype("float32")

    start = time.perf_counter()
    model = load_model(model_path, compile=False)
    load_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = []
    for start_index in range(0, x_test_dense.shape[0], batch_size):
        batch = x_test_dense[start_index : start_index + batch_size]
        predictions.append(model(batch, training=False).numpy().reshape(-1))
    y_prob = np.concatenate(predictions)
    predict_time = time.perf_counter() - start
    y_pred = (y_prob >= 0.5).astype(int)
    return evaluate_predictions(y_test, y_pred), load_time, predict_time


def discover_families(data_dir: Path) -> list[str]:
    return sorted(path.stem for path in data_dir.glob("*.csv"))


def load_family_data(root: Path, family: str, data_mode: str, test_size: float, random_state: int):
    if data_mode == "paper_split":
        train_file = root / "data" / "mix_data" / "trainset" / f"{family}.csv"
        test_file = root / "data" / "mix_data" / "testset" / f"{family}.csv"
        if not train_file.exists():
            raise FileNotFoundError(f"Missing train file: {train_file}")
        if not test_file.exists():
            raise FileNotFoundError(f"Missing test file: {test_file}")
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        return train_df, test_df

    if data_mode == "one_vs_other_split":
        family_file = root / "data" / "one_vs_other" / f"{family}.csv"
        if not family_file.exists():
            raise FileNotFoundError(f"Missing family file: {family_file}")
        df = pd.read_csv(family_file)
        df = df.dropna(subset=["sequence", "class"])
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df["class"],
            random_state=random_state,
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    raise ValueError(f"Unknown data mode: {data_mode}")


def row_with_error(family, k, model, error, n_rows, n_train, n_test, n_features, selected_features):
    return {
        "family": family,
        "k": k,
        "model": model,
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "f1": np.nan,
        "train_time_sec": np.nan,
        "predict_time_sec": np.nan,
        "n_rows": n_rows,
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "selected_features": selected_features,
        "status": "error",
        "error": str(error),
    }


def build_pairwise_fnn_comparison(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[results["status"] == "ok"].copy()
    model_set = set(ok["model"])
    reference_model = "fnn" if "fnn" in model_set else "fnn_pretrained" if "fnn_pretrained" in model_set else None
    if ok.empty or reference_model is None:
        return pd.DataFrame()

    metric_names = ["accuracy", "precision", "recall", "f1"]
    key_cols = ["repeat", "seed", "family", "k"]
    comparison_rows = []

    for k, k_df in ok.groupby("k"):
        pivot = k_df.pivot_table(index=key_cols, columns="model", values=metric_names)
        if ("f1", reference_model) not in pivot.columns:
            continue

        model_names = sorted({model for _, model in pivot.columns if model != reference_model})
        for model_name in model_names:
            row = {"k": k, "baseline_model": model_name, "reference_model": reference_model}
            valid_count = 0
            for metric in metric_names:
                fnn_col = (metric, reference_model)
                baseline_col = (metric, model_name)
                if fnn_col not in pivot.columns or baseline_col not in pivot.columns:
                    continue
                delta = (pivot[fnn_col] - pivot[baseline_col]).dropna()
                if delta.empty:
                    continue
                valid_count = len(delta)
                row[f"mean_delta_{metric}"] = delta.mean()
                row[f"median_delta_{metric}"] = delta.median()
                row[f"fnn_wins_{metric}"] = int((delta > 0).sum())
                row[f"fnn_ties_{metric}"] = int((delta == 0).sum())
                row[f"fnn_losses_{metric}"] = int((delta < 0).sum())
                if metric == "f1":
                    try:
                        from scipy.stats import wilcoxon

                        non_zero_delta = delta[delta != 0]
                        if len(non_zero_delta) > 0:
                            row["wilcoxon_p_f1"] = wilcoxon(non_zero_delta).pvalue
                        else:
                            row["wilcoxon_p_f1"] = np.nan
                    except Exception:
                        row["wilcoxon_p_f1"] = np.nan
            row["n_pairs"] = valid_count
            comparison_rows.append(row)

    if not comparison_rows:
        return pd.DataFrame()

    return pd.DataFrame(comparison_rows).sort_values(
        ["k", "mean_delta_f1"], ascending=[True, False]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--families", default="AP2", help="Comma-separated families, e.g. AP2,bHLH,MYB")
    parser.add_argument("--all-families", action="store_true", help="Run all available family CSVs")
    parser.add_argument(
        "--data-mode",
        choices=["paper_split", "one_vs_other_split"],
        default="paper_split",
        help=(
            "paper_split uses data/mix_data/trainset and data/mix_data/testset, matching the "
            "80/20 split described in the paper. one_vs_other_split keeps the old behavior: "
            "read data/one_vs_other and create a new stratified split."
        ),
    )
    parser.add_argument("--k", default="3", help="Comma-separated k-mer sizes, e.g. 2,3,4,5")
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=f"Comma-separated models. Available: {','.join(ALL_MODELS)}",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional stratified sample per split and family; 0 means full data",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated stratified samples/splits")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--k2-features", type=int, default=200)
    parser.add_argument("--k345-features", type=int, default=1000)
    parser.add_argument("--fnn-epochs", type=int, default=100)
    parser.add_argument("--fnn-patience", type=int, default=10)
    parser.add_argument("--fnn-batch-size", type=int, default=64)
    parser.add_argument("--keras-verbose", type=int, default=0)
    parser.add_argument(
        "--output",
        default="test/metrics/model_benchmark/binary_classifier_benchmark.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="test/metrics/model_benchmark/binary_classifier_benchmark_summary.csv",
    )
    parser.add_argument(
        "--comparison-output",
        default="test/metrics/model_benchmark/binary_classifier_benchmark_fnn_comparison.csv",
    )
    args = parser.parse_args()

    root = repo_root()
    data_dir = root / "data" / "mix_data" / "trainset" if args.data_mode == "paper_split" else root / "data" / "one_vs_other"
    output = root / args.output
    summary_output = root / args.summary_output
    comparison_output = root / args.comparison_output
    output.parent.mkdir(parents=True, exist_ok=True)

    families = discover_families(data_dir) if args.all_families else parse_csv_list(args.families)
    k_values = [int(value) for value in parse_csv_list(args.k)]
    models = parse_csv_list(args.models)
    max_rows = args.max_rows if args.max_rows > 0 else None

    rows = []
    for repeat in range(args.repeats):
        seed = args.random_state + repeat
        print(f"\n=== repeat={repeat + 1}/{args.repeats} seed={seed} ===")
        for family in families:
            try:
                train_df, test_df = load_family_data(root, family, args.data_mode, args.test_size, seed)
            except FileNotFoundError as exc:
                print(f"[skip] {exc}")
                continue

            train_df = train_df.dropna(subset=["sequence", "class"]).reset_index(drop=True)
            test_df = test_df.dropna(subset=["sequence", "class"]).reset_index(drop=True)
            train_df = sample_dataset(train_df, max_rows=max_rows, random_state=seed)
            test_df = sample_dataset(test_df, max_rows=max_rows, random_state=seed)
            print(
                f"\n[family={family}] "
                f"train_rows={len(train_df)} train_class_counts={dict(Counter(train_df['class']))} "
                f"test_rows={len(test_df)} test_class_counts={dict(Counter(test_df['class']))}"
            )

            for k in k_values:
                print(f"  [k={k}] vectorizing")
                x_train, y_train, vectorizer = fit_kmer_vectorizer(train_df, k)
                x_test, y_test = transform_kmers(test_df, k, vectorizer)
                feature_names = vectorizer.get_feature_names_out()
                requested_features = args.k2_features if k == 2 else args.k345_features
                x_train_sel, x_test_sel, _ = select_features(x_train, y_train, x_test, requested_features)
                selected_features = x_train_sel.shape[1]
                print(f"  [k={k}] features={len(feature_names)} selected={selected_features}")

                for model_name in models:
                    print(f"    [model={model_name}] running")
                    old_random_state = args.random_state
                    args.random_state = seed
                    missing_pretrained_features = 0
                    try:
                        if model_name == "fnn":
                            metrics, train_time, predict_time = run_fnn(
                                x_train_sel, y_train, x_test_sel, y_test, k, args
                            )
                        elif model_name == "fnn_pretrained":
                            pretrained_features = load_pretrained_feature_mask(root, family, k)
                            _, x_test_pretrained, missing_pretrained_features = select_features_by_names(
                                x_train,
                                x_test,
                                feature_names,
                                pretrained_features,
                            )
                            selected_features = x_test_pretrained.shape[1]
                            metrics, train_time, predict_time = run_fnn_pretrained(
                                root, family, x_test_pretrained, y_test, k, args.fnn_batch_size
                            )
                        else:
                            metrics, train_time, predict_time = run_sklearn_model(
                                model_name, x_train_sel, y_train, x_test_sel, y_test, args
                            )

                        row = {
                            "repeat": repeat,
                            "seed": seed,
                            "family": family,
                            "k": k,
                            "model": model_name,
                            **metrics,
                            "train_time_sec": train_time,
                            "predict_time_sec": predict_time,
                            "data_mode": args.data_mode,
                            "n_rows": len(train_df) + len(test_df),
                            "n_train": len(y_train),
                            "n_test": len(y_test),
                            "n_features": len(feature_names),
                            "selected_features": selected_features,
                            "missing_pretrained_features": missing_pretrained_features,
                            "status": "ok",
                            "error": "",
                        }
                        print(
                            "      "
                            + json.dumps(
                                {
                                    "accuracy": round(row["accuracy"], 4),
                                    "precision": round(row["precision"], 4),
                                    "recall": round(row["recall"], 4),
                                    "f1": round(row["f1"], 4),
                                    "train_sec": round(row["train_time_sec"], 2),
                                }
                            )
                        )
                    except Exception as exc:
                        print(f"      error: {exc}")
                        row = row_with_error(
                            family,
                            k,
                            model_name,
                            exc,
                            len(train_df) + len(test_df),
                            len(y_train),
                            len(y_test),
                            len(feature_names),
                            selected_features,
                        )
                        row["repeat"] = repeat
                        row["seed"] = seed
                        row["data_mode"] = args.data_mode
                    finally:
                        args.random_state = old_random_state
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output, index=False)

    results = pd.DataFrame(rows)
    results.to_csv(output, index=False)
    if not results.empty:
        summary = (
            results[results["status"] == "ok"]
            .groupby(["k", "model"], as_index=False)[["accuracy", "precision", "recall", "f1", "train_time_sec"]]
            .mean()
            .sort_values(["k", "f1", "accuracy"], ascending=[True, False, False])
        )
        summary.to_csv(summary_output, index=False)
        comparison = build_pairwise_fnn_comparison(results)
        if not comparison.empty:
            comparison.to_csv(comparison_output, index=False)
        print(f"\nSaved detailed results: {output}")
        print(f"Saved summary results:  {summary_output}")
        if not comparison.empty:
            print(f"Saved FNN comparison:  {comparison_output}")
        print("\nSummary:")
        print(summary.to_string(index=False))
        if not comparison.empty:
            print("\nFNN pairwise comparison:")
            print(comparison.to_string(index=False))
    else:
        print("No results were produced.")


if __name__ == "__main__":
    main()
