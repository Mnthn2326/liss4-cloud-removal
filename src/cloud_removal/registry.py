"""MLflow Model Registry utilities and promotion workflows.

Provides functions and CLI commands to promote registered models across stages
(e.g., Staging, Production, Archived) and assign modern MLflow aliases
(e.g., champion, challenger).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Union

import mlflow
from mlflow.tracking import MlflowClient

# Ensure file store compatibility if default local store is used
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

logger = logging.getLogger("cloud_removal.registry")


def promote_model(
    model_name: str,
    version: Union[int, str],
    stage: Optional[str] = None,
    alias: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    archive_existing_versions: bool = False,
) -> Dict[str, Any]:
    """Promote a registered model version to a target stage or assign an alias.

    Args:
        model_name: Name of the registered model.
        version: Version number or identifier to promote.
        stage: Target stage (e.g. 'Staging', 'Production', 'Archived').
        alias: Target alias (e.g. 'champion', 'challenger', 'baseline').
        tracking_uri: Optional MLflow tracking URI. Defaults to active URI.
        archive_existing_versions: If True, moves existing versions in the stage to Archived.

    Returns:
        Dictionary summarizing the promotion status and details.

    Raises:
        ValueError: If neither stage nor alias is provided.
        Exception: If promotion fails due to registry backend error.
    """
    if not stage and not alias:
        raise ValueError(
            "At least one of 'stage' or 'alias' must be specified for model promotion."
        )

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    version_str = str(version)

    result: Dict[str, Any] = {
        "model_name": model_name,
        "version": version_str,
        "stage": None,
        "alias": None,
        "status": "success",
    }

    # Assign alias if requested
    if alias:
        clean_alias = alias.strip().lower()
        client.set_registered_model_alias(
            name=model_name,
            alias=clean_alias,
            version=version_str,
        )
        result["alias"] = clean_alias
        logger.info(
            "Assigned alias '%s' to model '%s' version %s", clean_alias, model_name, version_str
        )

    # Transition stage if requested
    if stage:
        clean_stage = stage.strip().capitalize()
        # MLflow standard stages: 'None', 'Staging', 'Production', 'Archived'
        if clean_stage.lower() == "production":
            clean_stage = "Production"
        elif clean_stage.lower() == "staging":
            clean_stage = "Staging"
        elif clean_stage.lower() == "archived":
            clean_stage = "Archived"

        client.transition_model_version_stage(
            name=model_name,
            version=version_str,
            stage=clean_stage,
            archive_existing_versions=archive_existing_versions,
        )
        result["stage"] = clean_stage
        logger.info(
            "Transitioned model '%s' version %s to stage '%s'", model_name, version_str, clean_stage
        )

    return result


def get_model_version_details(
    model_name: str,
    version_or_alias: Union[int, str],
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve details for a specific model version or alias.

    Args:
        model_name: Name of the registered model.
        version_or_alias: Version number or alias (e.g. 'champion').
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        Dictionary with model version metadata.
    """
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    # Check if target is an alias
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias=str(version_or_alias))
    except Exception:
        # Otherwise get by version number
        mv = client.get_model_version(name=model_name, version=str(version_or_alias))

    return {
        "name": mv.name,
        "version": mv.version,
        "current_stage": mv.current_stage,
        "aliases": getattr(mv, "aliases", []),
        "source": mv.source,
        "run_id": mv.run_id,
        "status": mv.status,
    }


def main() -> int:
    """CLI entry point for model promotion."""
    parser = argparse.ArgumentParser(
        description="Promote registered MLflow models across stages and aliases."
    )
    parser.add_argument("--model", type=str, required=True, help="Name of registered model")
    parser.add_argument("--version", type=str, required=True, help="Model version to promote")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["Staging", "Production", "Archived"],
        help="Target stage",
    )
    parser.add_argument(
        "--alias", type=str, default=None, help="Target alias (e.g., champion, challenger)"
    )
    parser.add_argument("--tracking-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument(
        "--archive-existing", action="store_true", help="Archive existing versions in target stage"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        res = promote_model(
            model_name=args.model,
            version=args.version,
            stage=args.stage,
            alias=args.alias,
            tracking_uri=args.tracking_uri,
            archive_existing_versions=args.archive_existing,
        )
        print("Model promotion completed successfully:")
        for k, v in res.items():
            print(f"  {k}: {v}")
        return 0
    except Exception as exc:
        print(f"Error during model promotion: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
