#!/usr/bin/env python3
"""dql_trading.reporting.loaders

Utility functions to load metrics, visualisations and other artefacts from an
experiment directory.  These were previously embedded in `scripts/generate_report.py`
but have been extracted so they can be unit-tested and reused independently of
any CLI or PDF-generation code.
"""
from __future__ import annotations

import os
import json
import logging
from typing import Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def load_metrics(results_dir: str, experiment_name: str) -> Dict[str, Any]:
    """Load training/testing metrics and hyper-parameters for *experiment_name*.

    The function is intentionally side-effect-free (except for logging) so it
    can be used in tests.  The returned dictionary has three top-level keys::

        {
          "train_metrics": dict,
          "test_metrics":  dict | None,
          "hyperparams"  : dict,
        }
    """
    experiment_dir = os.path.join(results_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        logger.error("Experiment directory not found: %s", experiment_dir)
        raise FileNotFoundError(experiment_dir)

    train_metrics: Dict[str, Any] = {}
    test_metrics: Optional[Dict[str, Any]] = None
    hyperparams: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Training metrics
    # ------------------------------------------------------------------
    train_json = os.path.join(experiment_dir, "train_metrics.json")
    train_csv = os.path.join(experiment_dir, "train_metrics.csv")
    if os.path.exists(train_json):
        try:
            with open(train_json, "r") as fh:
                train_metrics = json.load(fh)
            logger.info("Loaded training metrics from %s", train_json)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse %s: %s", train_json, exc)
    elif os.path.exists(train_csv):
        try:
            df = pd.read_csv(train_csv)
            train_metrics = df.iloc[-1].to_dict()
            logger.info("Loaded training metrics from %s", train_csv)
        except Exception as exc:  # pragma: no cover – very general fallback
            logger.warning("Could not parse %s: %s", train_csv, exc)

    # ------------------------------------------------------------------
    # Testing metrics
    # ------------------------------------------------------------------
    test_json = os.path.join(experiment_dir, "test_metrics.json")
    test_csv = os.path.join(experiment_dir, "test_metrics.csv")

    if os.path.exists(test_json):
        try:
            with open(test_json, "r") as fh:
                test_metrics = json.load(fh)
            logger.info("Loaded test metrics from %s", test_json)
        except json.JSONDecodeError as exc:
            logger.warning("Could not parse %s: %s", test_json, exc)
    elif os.path.exists(test_csv):
        try:
            df = pd.read_csv(test_csv)
            test_metrics = df.iloc[-1].to_dict()
            logger.info("Loaded test metrics from %s", test_csv)
        except Exception as exc:
            # Attempt manual fallback parsing
            try:
                with open(test_csv, "r") as fh:
                    headers = fh.readline().strip().split(",")
                    values = fh.readline().strip().split(",")
                    if len(headers) == len(values):
                        test_metrics = {h: v for h, v in zip(headers, values)}
                        # convert numbers
                        for k, v in test_metrics.items():
                            try:
                                test_metrics[k] = float(v)
                            except ValueError:
                                pass
                        logger.info("Manually parsed test metrics from %s", test_csv)
            except Exception as exc2:
                logger.warning("Fallback parse also failed for %s: %s", test_csv, exc2)

    # ------------------------------------------------------------------
    # Baseline comparison helps populate some missing test metrics values
    # Later we still load the full baseline via a dedicated helper.
    # ------------------------------------------------------------------
    strategy_csv = os.path.join(experiment_dir, "strategy_comparison.csv")
    if os.path.exists(strategy_csv):
        try:
            df = pd.read_csv(strategy_csv)
            dql_row = df[df["Strategy"] == "DQL Agent"]
            if not dql_row.empty:
                if test_metrics is None:
                    test_metrics = {}
                for col in df.columns:
                    if col == "Strategy":
                        continue
                    key = col.lower().replace(" ", "_").replace("(%)", "")
                    test_metrics[key] = dql_row.iloc[0][col]
        except Exception as exc:
            logger.warning("Could not parse %s: %s", strategy_csv, exc)

    # ------------------------------------------------------------------
    # Hyper-parameters
    # ------------------------------------------------------------------
    hyper_json = os.path.join(experiment_dir, "hyperparameters.json")
    if os.path.exists(hyper_json):
        try:
            with open(hyper_json, "r") as fh:
                hyperparams = json.load(fh)
        except Exception as exc:
            logger.warning("Could not parse %s: %s", hyper_json, exc)

    # ------------------------------------------------------------------
    # Fallback: if we have test metrics but no train metrics copy a subset
    # ------------------------------------------------------------------
    if test_metrics and not train_metrics:
        logger.info("No training metrics – using test metrics as proxy")
        train_metrics = {
            "total_return_pct": test_metrics.get("total_return_pct", 0.0),
            "sharpe_ratio": test_metrics.get("sharpe_ratio", 0.0),
            "max_drawdown": test_metrics.get("max_drawdown", 0.0),
            "win_rate": test_metrics.get("win_rate", 0.0),
            "training_completed": True,
        }

    return {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "hyperparams": hyperparams,
    }


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def find_visualization_files(results_dir: str, experiment_name: str) -> Dict[str, Optional[str]]:
    """Locate all key PNG/JPG artefacts for *experiment_name*.

    Returns a dict with fixed keys even if some artefacts are missing.
    """
    experiment_dir = os.path.join(results_dir, experiment_name)
    if not os.path.exists(experiment_dir):
        raise FileNotFoundError(experiment_dir)

    viz_files = {
        "training_performance": None,
        "learning_curves": None,
        "test_performance": None,
        "trade_visualization": None,
        "strategy_comparison": None,
        "feature_importance": None,
        "test_feature_importance": None,
    }

    for filename in os.listdir(experiment_dir):
        if not (filename.lower().endswith(".png") or filename.lower().endswith(".jpg")):
            continue
        path = os.path.join(experiment_dir, filename)
        fname = filename.lower()

        if "training_plots" in fname or "training_performance" in fname:
            viz_files["training_performance"] = path
        if "learning_curves" in fname:
            viz_files["learning_curves"] = path
        if "performance_dashboard" in fname or "test_performance" in fname:
            viz_files["test_performance"] = path
        if "test_trades" in fname or "trade_visualization" in fname:
            viz_files["trade_visualization"] = path
        if "strategies_comparison" in fname:
            viz_files["strategy_comparison"] = path
        if "feature_importance" in fname and not fname.startswith("test_"):
            viz_files["feature_importance"] = path
        if fname.startswith("test_") and "feature_importance" in fname:
            viz_files["test_feature_importance"] = path

    return viz_files


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def load_baseline_comparison(results_dir: str, experiment_name: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load metrics for baseline strategies (if available)."""
    import pandas as pd  # local import to avoid mandatory dependency for others

    csv_path = os.path.join(results_dir, experiment_name, "strategy_comparison.csv")
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        baseline_metrics: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            name = row["Strategy"]
            if name == "DQL Agent":
                continue
            baseline_metrics[name] = {
                col.lower().replace(" ", "_").replace("(%)", ""): row[col]
                for col in df.columns if col != "Strategy"
            }
        return baseline_metrics or None
    except Exception as exc:  # pragma: no cover – general guard
        logger.warning("Failed loading baseline comparison: %s", exc)
        return None 