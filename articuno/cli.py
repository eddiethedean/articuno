import typer
from pathlib import Path
from typing import Optional

from .bootstrap import bootstrap_app

app = typer.Typer(help="Articuno CLI - Polars to Pydantic/Patito model tools")


@app.command()
def bootstrap(
    app_path: str = typer.Argument(..., help="Path to your FastAPI app file (e.g., app/main.py)"),
    models_path: Optional[str] = typer.Option(None, help="Output path for models.py"),
    dry_run: bool = typer.Option(False, help="Preview changes without writing files"),
):
    """
    Analyze your FastAPI app, infer models from polars.DataFrame responses,
    and generate Pydantic or Patito model files + update your routes.
    """
    bootstrap_app(app_path, models_path=models_path, dry_run=dry_run)


if __name__ == "__main__":
    app()
