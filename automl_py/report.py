"""The auditable run report and reproducibility manifest.

``render_report`` turns a run into the narrative the architecture promises:
objective, validation design, baseline, diagnosis, every experiment with its
paired evidence and decision, what was kept, the single final holdout number,
why the loop stopped, and (optionally) the Value-of-Information case.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha1()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()[:16]


def build_manifest(config, df: pd.DataFrame | None = None, **extra) -> dict:
    """Everything needed to reproduce a run: versions, seed, config, data fingerprint."""
    cfg = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
    if cfg.get("validation") is not None and dataclasses.is_dataclass(config.validation):
        cfg["validation"] = dataclasses.asdict(config.validation)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in ("automl-py", "numpy", "pandas", "scikit-learn", "scipy")},
        "random_state": cfg.get("random_state"),
        "config": cfg,
    }
    if df is not None:
        manifest["data"] = {"rows": len(df), "columns": int(df.shape[1]), "fingerprint": data_fingerprint(df)}
    manifest.update(extra)
    return manifest


def _fmt(x, nd=4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, int | np.integer):
        return str(int(x))
    if isinstance(x, float | np.floating):
        return f"{float(x):.{nd}f}"
    return str(x)


def _table(frame: pd.DataFrame, columns: list[str], nd: int = 4) -> str:
    cols = [c for c in columns if c in frame.columns]
    if frame.empty or not cols:
        return "_none_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
    body = "".join("| " + " | ".join(_fmt(row[c], nd) for c in cols) + " |\n" for _, row in frame[cols].iterrows())
    return head + body


def render_report(result, *, voi: pd.DataFrame | None = None, manifest: dict | None = None, title: str | None = None) -> str:
    """Markdown report for an AutonomousScientistResult, OrchestratorResult or ScientistResult."""
    loop = getattr(result, "loop", None) if not hasattr(result, "stop_reason") else result
    scientist = getattr(result, "scientist", None)
    if scientist is not None and hasattr(scientist, "scientist"):  # OrchestratorResult.scientist is a ScientistResult
        scientist = scientist
    if scientist is None and hasattr(result, "automl"):
        scientist = result
    automl = scientist.automl if scientist is not None else None
    lines: list[str] = []
    best = automl.best_results.iloc[0] if automl is not None and not automl.best_results.empty else None
    target = best.target if best is not None else getattr(loop, "target", "?")
    metric = best.metric if best is not None else getattr(loop, "metric", "?")
    lines.append(f"# {title or f'Predictive Discovery report: {target}'}\n")
    lines.append(f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}_\n")

    # ---- objective and validation design ---------------------------------------------------
    lines.append("## Objective and validation design\n")
    if best is not None:
        folds = automl.folds[target]
        lines.append(f"- Target: `{target}`; metric: `{metric}`")
        lines.append(
            f"- Validation strategy: `{best.validation_strategy}` with {int(best.n_folds)} folds on {int(best.development_rows)} development rows"
        )
        if folds.timestamps is not None:
            d = folds.describe()
            lines.append(
                f"- Windows: train ends {d.train_end.iloc[0]} .. {d.train_end.iloc[-1]}; validation {d.validation_start.iloc[0]} .. {d.validation_end.iloc[-1]}"
            )
        lines.append(f"- Final holdout: {int(best.holdout_rows)} rows, locked before search, evaluated once")
    lines.append("")

    # ---- baseline --------------------------------------------------------------------------
    lines.append("## Baseline champion\n")
    if best is not None:
        lines.append(
            f"- `{best.model}` / preprocess `{best.preprocess}` / impute `{best.imputation}`: development {metric} = {_fmt(best.cv_performance)} +/- {_fmt(best.cv_std)}"
        )
        stages = automl.results.stage.value_counts().to_dict() if "stage" in automl.results else {}
        if stages:
            lines.append(f"- Configurations: {', '.join(f'{v} {k}' for k, v in stages.items())}")
    lines.append("")

    # ---- diagnosis -------------------------------------------------------------------------
    if scientist is not None:
        lines.append("## Diagnosis\n")
        leak = scientist.leakage
        if leak is not None and not leak.empty and (leak.risk == "high").any():
            lines.append("- Leakage risk (high): " + ", ".join(leak.loc[leak.risk == "high", "feature"].head(5)))
        drift = scientist.drift
        if drift:
            lines.append(f"- Drift: adversarial AUC {_fmt(drift['auc'], 3)} ({drift.get('shift_severity')}; {drift.get('comparison', '')})")
        for t, d in (scientist.residual_diagnostics or {}).items():
            if d is not None and not d.empty:
                lines.append(
                    f"- Residual structure for `{t}`: strongest association with `{d.iloc[0].feature}` (MI {_fmt(d.iloc[0].residual_mutual_information, 3)})"
                )
        for r in scientist.recommendations:
            lines.append(f"- {r}")
        lines.append("")

    # ---- feature families -------------------------------------------------------------------
    info = getattr(result, "information_value", None)
    if info is not None and not info.empty:
        lines.append("## Feature-family experiments (paired, judged)\n")
        lines.append(
            _table(
                info,
                ["feature_family", "decision", "mean_uplift", "ci_low", "ci_high", "positive_share", "noise_threshold", "required_gain"],
            )
        )

    # ---- loop --------------------------------------------------------------------------------
    if loop is not None:
        lines.append("## Autonomous loop\n")
        lines.append(f"- Iterations: {loop.iterations}; experiments: {len(loop.experiments)}; stop reason: `{loop.stop_reason}`")
        lines.append(f"- Kept: {', '.join(loop.kept_candidates) or 'none'}")
        hist = loop.baseline_history
        if not hist.empty:
            lines.append(f"- Development {metric}: {_fmt(hist.baseline_mean.iloc[0])} -> {_fmt(hist.baseline_mean.iloc[-1])}")
        lines.append("")
        lines.append("### Hypotheses\n")
        lines.append(
            _table(loop.hypotheses_log, ["iteration", "hypothesis_id", "kind", "concept", "confidence", "statement", "evidence"], 2)
        )
        lines.append("### Experiments\n")
        trail = loop.audit_trail.copy()
        if not trail.empty:
            trail["ci"] = trail.confidence_interval.apply(lambda ci: "n/a" if not isinstance(ci, list) else f"[{ci[0]:.4f}, {ci[1]:.4f}]")
            trail["adopted"] = trail.extra.apply(lambda e: bool(e.get("adopted")) if isinstance(e, dict) else False)
        lines.append(
            _table(
                trail,
                [
                    "experiment_id",
                    "iteration",
                    "candidate",
                    "hypothesis_id",
                    "decision",
                    "adopted",
                    "mean_uplift",
                    "ci",
                    "positive_share",
                    "noise_threshold",
                    "required_gain",
                ],
            )
        )
        if not trail.empty:
            lines.append("\n**Decision reasons**\n")
            for _, r in trail.iterrows():
                lines.append(f"- {r.experiment_id} {r.candidate}: {r.decision}. " + " ".join(r.reasons or []))
        lines.append("")
        lines.append("## Final system\n")
        card = loop.model_card
        lines.append(f"- Feature columns: {', '.join(card.get('final_feature_columns', []))}")
        if loop.holdout_evaluation is not None:
            ev = loop.holdout_evaluation
            lines.append(
                f"- **Final holdout {metric} = {_fmt(ev.score)}** on {ev.n_rows} rows (evaluated {card.get('holdout_evaluations', 1)} time)"
            )
        lines.append("")
    elif best is not None and np.isfinite(best.holdout_performance):
        lines.append("## Final holdout\n")
        lines.append(f"- **{metric} = {_fmt(best.holdout_performance)}** on {int(best.holdout_rows)} rows (evaluated once)\n")

    # ---- value -----------------------------------------------------------------------------
    if voi is not None and not voi.empty:
        lines.append("## Value of Information\n")
        lines.append(
            _table(
                voi,
                [
                    "candidate",
                    "evidence_decision",
                    "improvement_absolute",
                    "expected_annual_business_value",
                    "first_year_cost",
                    "first_year_net_value",
                    "payback_months",
                    "recommendation",
                ],
                2,
            )
        )

    # ---- diagnostics and manifest ----------------------------------------------------------
    diag = getattr(result, "diagnostics", None)
    if diag is not None and len(diag):
        errs = diag.errors()
        lines.append("## Diagnostics\n")
        lines.append(f"- {len(diag)} events, {len(errs)} errors")
        for e in errs[:10]:
            lines.append(f"- ERROR [{e.component}] {e.message}")
        lines.append("")
    if manifest:
        lines.append("## Reproducibility\n")
        pk = manifest.get("packages", {})
        lines.append(f"- Python {manifest.get('python')}; " + ", ".join(f"{k} {v}" for k, v in pk.items()))
        if "data" in manifest:
            d = manifest["data"]
            lines.append(f"- Data: {d['rows']} rows x {d['columns']} columns, fingerprint `{d['fingerprint']}`")
        lines.append(f"- random_state = {manifest.get('random_state')}")
        lines.append("")
    return "\n".join(lines)


def save_report(result, directory, *, voi=None, manifest=None, title=None) -> Path:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(result, voi=voi, manifest=manifest, title=title))
    if manifest:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out / "report.md"
