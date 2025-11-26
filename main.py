#!/usr/bin/env python3
"""
Main Entry Point - Air Quality Prediction ML Pipeline

Usage:
    python main.py preprocess      # Preprocess raw data
    python main.py train          # Train models
    python main.py deploy         # Deploy to production
    python main.py all            # Run full pipeline
    python main.py serve          # Start API server
"""

import sys
import argparse
from pathlib import Path

# Add numpy compatibility fix
import numpy as np

if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

from src.helper.config import load_config
from src.pipeline.preprocessing_pipeline import run_preprocessing
from src.pipeline.training_pipeline import run_training
from src.pipeline.deployment_pipeline import run_deployment


def setup_project_paths():
    """Setup and return project paths."""
    project_root = Path(__file__).parent

    return {
        "root": project_root,
        "config": project_root / "config" / "config.yaml",
        "processed_data": project_root / "data" / "processed",
        "log": project_root / "log" / "training_log.json",
        "experiment": project_root / "models" / "experiment",
        "production": project_root / "models" / "production",
        "models": project_root / "models",
    }


def cmd_preprocess(args):
    """Run preprocessing pipeline."""
    print(f"\n{'='*70}")
    print(f"  PREPROCESSING PIPELINE")
    print(f"{'='*70}")

    paths = setup_project_paths()
    config = load_config(paths["config"])

    success = run_preprocessing(config, paths["processed_data"])

    if success:
        print(f"\n✅ Preprocessing complete!")
        print(f"   Next step: python main.py train")
    else:
        print(f"\n❌ Preprocessing failed!")
        sys.exit(1)


def cmd_train(args):
    """Run training pipeline."""
    paths = setup_project_paths()

    # Check if processed data exists
    required_files = [
        "X_rus_feng.pkl",
        "y_rus_feng.pkl",
        "X_valid_feng.pkl",
        "y_valid_feng.pkl",
        "feature_engineering_artifacts.pkl",
    ]
    missing = [f for f in required_files if not (paths["processed_data"] / f).exists()]
    if missing:
        print(f"\n⚠️  Missing engineered datasets or artifacts: {missing}")
        print(f"   Run preprocessing first: python main.py preprocess")
        sys.exit(1)

    best_model = run_training(
        processed_data_dir=paths["processed_data"],
        log_path=paths["log"],
        experiment_dir=paths["experiment"],
        experiment_name=args.experiment_name,
    )

    if best_model:
        print(f"\n✅ Training complete!")
        print(f"   Next step: python main.py deploy")
    else:
        print(f"\n❌ Training failed!")
        sys.exit(1)


def cmd_deploy(args):
    """Run deployment pipeline."""
    paths = setup_project_paths()

    success = run_deployment(
        experiment_dir=paths["experiment"],
        production_dir=paths["production"],
        models_dir=paths["models"],
        processed_data_dir=paths["processed_data"],
    )

    if success:
        print(f"\n✅ Deployment complete!")
        print(f"   Start API: python main.py serve")
        print(f"   Or: make serve-api")
    else:
        print(f"\n❌ Deployment failed!")
        sys.exit(1)


def cmd_all(args):
    """Run complete pipeline: preprocess -> train -> deploy."""
    print(f"\n{'='*70}")
    print(f"  COMPLETE ML PIPELINE")
    print(f"{'='*70}")

    # Step 1: Preprocess
    print(f"\n[Step 1/3] Preprocessing...")
    cmd_preprocess(args)

    # Step 2: Train
    print(f"\n[Step 2/3] Training...")
    cmd_train(args)

    # Step 3: Deploy
    print(f"\n[Step 3/3] Deployment...")
    cmd_deploy(args)

    print(f"\n{'='*70}")
    print(f"  ✅ COMPLETE PIPELINE FINISHED!")
    print(f"{'='*70}")
    print(f"\nYour model is ready!")
    print(f"  Start API: python main.py serve")
    print(f"  Or use make commands:")
    print(f"    make serve-api  # Start API")
    print(f"    make serve-ui   # Start UI")


def cmd_serve(args):
    """Start API server."""
    import subprocess

    print(f"\n🚀 Starting API server...")
    print(f"   URL: http://localhost:8000")
    print(f"   Docs: http://localhost:8000/docs")
    print(f"   Press Ctrl+C to stop\n")

    try:
        subprocess.run(
            [
                "uvicorn",
                "src.serving.api:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ]
        )
    except KeyboardInterrupt:
        print(f"\n\n✓ Server stopped")
    except FileNotFoundError:
        print(f"\n❌ uvicorn not found!")
        print(f"   Install: pip install uvicorn")
        print(f"   Or use: make serve-api")


def cmd_status(args):
    """Show project status."""
    paths = setup_project_paths()

    print(f"\n{'='*70}")
    print(f"  PROJECT STATUS")
    print(f"{'='*70}")

    # Check files
    checks = {
        "Raw Data": paths["root"] / "data" / "raw",
        "Processed Data": paths["processed_data"] / "X_rus.pkl",
        "Training Log": paths["log"],
        "Experiment Model": paths["experiment"] / "best_model.pkl",
        "Production Model": paths["production"] / "model.pkl",
        "Config": paths["config"],
    }

    print(f"\n📋 Component Status:")
    for name, path in checks.items():
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024
                status = f"✅ Found ({size:.1f} KB)"
            else:
                # For directories, count files
                if path.is_dir():
                    files = list(path.glob("*"))
                    status = f"✅ Found ({len(files)} files)"
                else:
                    status = "✅ Found"
        else:
            status = "❌ Not found"

        print(f"   {name:<20} : {status}")

    # Suggest next steps
    print(f"\n💡 Next Steps:")

    if not checks["Processed Data"].exists():
        print(f"   1. Run: python main.py preprocess")
    elif not checks["Experiment Model"].exists():
        print(f"   1. Run: python main.py train")
    elif not checks["Production Model"].exists():
        print(f"   1. Run: python main.py deploy")
    else:
        print(f"   ✅ System ready!")
        print(f"   • Start API: python main.py serve")
        print(f"   • Start UI: make serve-ui")
        print(f"   • Run tests: make test")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Air Quality Prediction ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py all              # Run complete pipeline
  python main.py preprocess       # Preprocess data only
  python main.py train            # Train models only
  python main.py deploy           # Deploy to production
  python main.py serve            # Start API server
  python main.py status           # Check project status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # Preprocess command
    parser_preprocess = subparsers.add_parser(
        "preprocess", help="Run preprocessing pipeline"
    )
    parser_preprocess.set_defaults(func=cmd_preprocess)

    # Train command
    parser_train = subparsers.add_parser("train", help="Run training pipeline")
    parser_train.add_argument(
        "--experiment-name",
        default="baseline",
        help="Name for this experiment (default: baseline)",
    )
    parser_train.set_defaults(func=cmd_train)

    # Deploy command
    parser_deploy = subparsers.add_parser("deploy", help="Deploy model to production")
    parser_deploy.set_defaults(func=cmd_deploy)

    # All command
    parser_all = subparsers.add_parser("all", help="Run complete pipeline")
    parser_all.add_argument(
        "--experiment-name",
        default="baseline",
        help="Name for this experiment (default: baseline)",
    )
    parser_all.set_defaults(func=cmd_all)

    # Serve command
    parser_serve = subparsers.add_parser("serve", help="Start API server")
    parser_serve.set_defaults(func=cmd_serve)

    # Status command
    parser_status = subparsers.add_parser("status", help="Show project status")
    parser_status.set_defaults(func=cmd_status)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
