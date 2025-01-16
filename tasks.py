import os

from invoke import Context, task
from dotenv import load_dotenv

WINDOWS = os.name == "nt"
PROJECT_NAME = "mlops_project"
PYTHON_VERSION = "3.11"

load_dotenv()


# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )


@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)


# Project commands
@task
def preprocess_data(ctx: Context, percentage: float = 1.0) -> None:
    """Preprocess data."""
    ctx.run(
        f"python src/{PROJECT_NAME}/data.py data/raw data/processed --percentage {percentage}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker image."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_train(ctx: Context) -> None:
    """Run docker train image."""

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in the environment. Make sure it's set in the .env file.")

    ctx.run(
        f"docker run --name train1 --rm "
        f"-v $(pwd)/models:/models/ "
        f"-v $(pwd)/reports/figures:/reports/figures/ "
        f"-e WANDB_API_KEY={wandb_api_key} "
        f"train:latest",
        echo=True,
        pty=not WINDOWS,
    )


@task
def ruff_format(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"ruff format .", echo=True, pty=not WINDOWS)


# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
