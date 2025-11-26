"""Deployment Pipeline - Model Deployment to Production."""

import shutil
from pathlib import Path

from src.features.transformation import FEATURE_ARTIFACT_FILENAME


SEPARATOR = "=" * 70


def run_deployment(
    experiment_dir: Path,
    production_dir: Path,
    models_dir: Path,
    processed_data_dir: Path,
) -> bool:
    """
    Deployment pipeline: copy best model and artifacts to production.

    Args:
        experiment_dir: Directory containing experiment models
        production_dir: Production directory
        models_dir: Models directory (for encoders)

    Returns:
        True if successful
    """
    print("\n" + SEPARATOR)
    print("  DEPLOYMENT PIPELINE")
    print(SEPARATOR)

    try:
        production_dir.mkdir(parents=True, exist_ok=True)

        # 1. Find model to deploy
        print("\n🔍 Looking for model to deploy...")

        experiment_model = experiment_dir / "best_model.pkl"
        old_model = models_dir / "model.pkl"

        model_source = None
        if experiment_model.exists():
            model_source = experiment_model
            print(f"   ✓ Found: {experiment_model}")
        elif old_model.exists():
            model_source = old_model
            print(f"   ✓ Found: {old_model}")
        else:
            print("   ❌ No model found!")
            print(f"      Checked: {experiment_model}")
            print(f"      Checked: {old_model}")
            print("\n   Please run training first: python main.py train")
            return False

        # 2. Deploy model
        print("\n🚀 Deploying to production...")
        production_model = production_dir / "model.pkl"
        shutil.copy(model_source, production_model)
        print(f"   ✓ Model deployed: {production_model}")
        print(f"   ✓ Size: {production_model.stat().st_size / 1024:.2f} KB")

        # 3. Deploy artifacts (encoders)
        print("\n📦 Deploying artifacts...")
        legacy_artifacts = ["ohe_stasiun.pkl", "le_categori.pkl"]
        for artifact in legacy_artifacts:
            artifact_path = models_dir / artifact
            if artifact_path.exists():
                shutil.copy(artifact_path, production_dir / artifact)
                print(f"   ✓ Copied legacy artifact: {artifact}")
            else:
                print(
                    f"   ⚠️  Legacy artifact not found: {artifact} (optional)"
                )

        feature_artifact_src = processed_data_dir / FEATURE_ARTIFACT_FILENAME
        if feature_artifact_src.exists():
            shutil.copy(
                feature_artifact_src,
                production_dir / FEATURE_ARTIFACT_FILENAME,
            )
            print(
                f"   ✓ Copied feature artifacts: {FEATURE_ARTIFACT_FILENAME}"
            )
        else:
            print(
                f"   ⚠️  Feature artifacts missing at {feature_artifact_src}. "
                "API predictions may fail without them."
            )

        print("\n" + SEPARATOR)
        print("  ✅ DEPLOYMENT COMPLETE")
        print(SEPARATOR)
        print("\nNext steps:")
        print("  1. Restart API: make serve-api")
        print("  2. Test: curl http://localhost:8000/health")

        return True

    except Exception as e:
        print(f"\n   ❌ Deployment failed: {e}")
        import traceback

        traceback.print_exc()
        return False
