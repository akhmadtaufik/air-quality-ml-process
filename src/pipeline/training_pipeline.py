"""
Training Pipeline - Model Training & Evaluation
"""

import sys
import joblib
from pathlib import Path
from typing import Optional, Tuple

# Add numpy compatibility fix
import numpy as np

if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

from src.modeling.baseline import get_model_list_for_experiment
from src.modeling.train import train_eval_model
from src.evaluation.metrics import compare_models


SEPARATOR = "=" * 70


def _load_processed_split(
    processed_data_dir: Path,
    prefix: str,
    require_feng: bool = False,
) -> Tuple[object, str]:
    """Load a dataset preferring *_feng.pkl and fallback to base files if allowed."""

    feng_name = f"{prefix}_feng.pkl"
    feng_path = processed_data_dir / feng_name
    base_name = f"{prefix}.pkl"
    base_path = processed_data_dir / base_name

    if feng_path.exists():
        return joblib.load(feng_path), feng_name

    if base_path.exists() and not require_feng:
        print(
            f"   ⚠️  {feng_name} not found. Falling back to {base_name}. "
            "Please re-run preprocessing to regenerate engineered features."
        )
        return joblib.load(base_path), base_name

    raise FileNotFoundError(
        f"Required dataset not found: {feng_name if require_feng else base_name}"
    )


def run_training(
    processed_data_dir: Path,
    log_path: Path,
    experiment_dir: Path,
    experiment_name: str = "baseline",
) -> Optional[dict]:
    """
    Training pipeline: load data, train models, save best model.

    Args:
        processed_data_dir: Directory containing processed data
        log_path: Path to save training logs
        experiment_dir: Directory to save experiment models
        experiment_name: Name for this experiment

    Returns:
        Dictionary with best model info, or None if failed
    """
    print("\n" + SEPARATOR)
    print(f"  TRAINING PIPELINE - {experiment_name.upper()}")
    print(SEPARATOR)

    try:
        # 1. Load processed data
        print(f"\n📊 Loading processed data from {processed_data_dir}...")

        X_rus, rus_filename = _load_processed_split(
            processed_data_dir, "X_rus"
        )
        y_rus, y_rus_filename = _load_processed_split(
            processed_data_dir, "y_rus"
        )
        X_valid, valid_filename = _load_processed_split(
            processed_data_dir, "X_valid"
        )
        y_valid, y_valid_filename = _load_processed_split(
            processed_data_dir, "y_valid"
        )

        print(f"   ✓ Training data (file: {rus_filename}): {X_rus.shape}")
        print(
            f"   ✓ Validation data (file: {valid_filename}): {X_valid.shape}"
        )

        # 2. Prepare models
        print("\n🤖 Preparing baseline models...")
        models = get_model_list_for_experiment(experiment_name)
        print(f"   ✓ {len(models)} models prepared:")
        for i, model_dict in enumerate(models, 1):
            print(f"      {i}. {model_dict['model_name']}")

        # 3. Train models
        print("\n🚀 Starting training...")
        print("   Configuration: undersampling")
        print("   This may take a moment...\n")

        log_path.parent.mkdir(parents=True, exist_ok=True)

        training_log, trained_models = train_eval_model(
            list_of_model=models,
            prefix_model_name=experiment_name,
            x_train=X_rus,
            y_train=y_rus,
            data_configuration_name="undersampling",
            x_valid=X_valid,
            y_valid=y_valid,
            log_path=str(log_path),
        )

        print("\n   ✓ Training complete!")
        print(f"   ✓ Logs saved to: {log_path}")

        # 4. Analyze results
        print("\n📈 Analyzing results...")
        comparison = compare_models(str(log_path))

        if not comparison.empty:
            print("\n   Top Models by F1-Score:")
            print("   " + "-" * 65)
            top_models = comparison.head(min(5, len(comparison)))
            for idx, row in top_models.iterrows():
                # Handle both string and list types
                model_name = row["model_name"]
                if isinstance(model_name, list):
                    model_name = model_name[0] if model_name else "Unknown"
                model_name = str(model_name)[:40]

                f1_score = row["f1_score_avg"]
                if isinstance(f1_score, list):
                    f1_score = f1_score[0] if f1_score else 0.0
                f1 = float(f1_score)

                print(f"   {idx+1}. {model_name:<40} F1: {f1:.4f}")
            print("   " + "-" * 65)

        # 5. Get best model from latest training
        print("\n🏆 Identifying best model from latest training...")

        # Get last training session
        import json

        with open(log_path, "r") as f:
            all_logs = json.load(f)
            last_log = all_logs[-1]

        # Find best in last session
        best_idx = max(
            range(len(last_log["f1_score_avg"])),
            key=lambda i: last_log["f1_score_avg"][i],
        )

        best_info = {
            "model_name": last_log["model_name"][best_idx],
            "model_uid": last_log["model_uid"][best_idx],
            "f1_score_avg": last_log["f1_score_avg"][best_idx],
            "training_time": last_log["training_time"][best_idx],
        }

        print(f"   ✓ Best Model: {best_info['model_name']}")
        print(f"     F1-Score: {best_info['f1_score_avg']:.4f}")
        print(f"     Training Time: {best_info['training_time']:.2f}s")
        print(f"     Model UID: {best_info['model_uid']}")

        # 6. Save best model
        print("\n💾 Saving best model...")
        experiment_dir.mkdir(parents=True, exist_ok=True)

        for model_dict in trained_models:
            if model_dict["model_uid"] == best_info["model_uid"]:
                best_model_path = experiment_dir / "best_model.pkl"
                joblib.dump(model_dict["model_object"], best_model_path)
                print(f"   ✓ Best model saved to: {best_model_path}")
                print(
                    f"   ✓ Size: {best_model_path.stat().st_size / 1024:.2f} KB"
                )
                break

        print("\n" + SEPARATOR)
        print("  ✅ TRAINING PIPELINE COMPLETE")
        print(SEPARATOR)

        return best_info

    except Exception as e:
        print(f"\n   ❌ Training pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return None
