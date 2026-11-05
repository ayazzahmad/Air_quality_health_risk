"""Optional MLflow experiment tracking for train.py.

MLflow is imported only when tracking is requested (`train.py --mlflow`), so
training without it, the API and the Docker images never need the package.

By default runs go to a local SQLite store (`mlflow.db`) with artifacts under
`mlruns/`, both at the project root. Set MLFLOW_TRACKING_URI to log to an
MLflow server instead. Browse runs with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import os
import subprocess

from . import config as cfg

DEFAULT_EXPERIMENT = "air-quality-health-risk"


def _git_state() -> tuple[str, bool]:
    """(commit SHA, whether tracked files have uncommitted changes)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=cfg.PROJECT_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def log_run(run_name: str, params: dict, metrics: dict, tags: dict,
            artifact_path, experiment: str = DEFAULT_EXPERIMENT) -> tuple[str, str]:
    """Log one training run. Returns (run_id, tracking_uri)."""
    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit(
            "--mlflow needs MLflow: pip install -r requirements-dev.txt"
        ) from exc

    local = "MLFLOW_TRACKING_URI" not in os.environ
    if local:
        mlflow.set_tracking_uri(f"sqlite:///{(cfg.PROJECT_ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(experiment) is None:
        # Keep local artifacts next to the store, whatever the working directory
        location = (cfg.PROJECT_ROOT / "mlruns").as_uri() if local else None
        mlflow.create_experiment(experiment, artifact_location=location)
    mlflow.set_experiment(experiment)

    commit, dirty = _git_state()
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({**tags, "git_commit": commit, "git_dirty": str(dirty)})
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(artifact_path), artifact_path="model")
        return run.info.run_id, mlflow.get_tracking_uri()
