"""Run t-tests to find the best raw-data display rule (points on/off)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "scipy is required for statistical tests. Install with `pip install scipy`."
    ) from exc

from analyze_responses import (
    DATA_PATH,
    TRIAL_LOG_PATH,
    load_question_groups,
    prepare_long_form,
)
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import PerfectSeparationError

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
logger = logging.getLogger(__name__)


def load_long_responses() -> pd.DataFrame:
    groups = load_question_groups(DATA_PATH)

    responses = pd.read_csv(DATA_PATH, skiprows=[1, 2])
    responses = responses[responses.get("Finished", 0) == 1]
    responses = responses[responses.get("Status", 0) == 0]
    responses.reset_index(drop=True, inplace=True)

    trial_meta = pd.read_csv(TRIAL_LOG_PATH)
    for col in ("Left_SD", "Right_SD"):
        trial_meta[col] = pd.to_numeric(trial_meta[col], errors="coerce")
    trial_meta["Jitter"] = trial_meta["Jitter"].fillna("").astype(str)

    return prepare_long_form(responses, groups, trial_meta)


def bonferroni_adjust(p_values: List[float]) -> List[float]:
    count = len(p_values)
    return [min(p * count, 1.0) for p in p_values]


def format_raw_display(points: str, jitter: str) -> str:
    """Human-friendly label for raw data overlay."""

    points = (points or "").strip()
    jitter = (jitter or "").strip()
    if points.lower() != "points on":
        return "Points Off"
    if jitter.lower() == "jitter on":
        return "Points On (Jitter On)"
    return "Points On (Jitter Off)"


def raw_display_tests(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    long_df = ensure_raw_display_rule(long_df)
    # Expand to three-level factor: off / on-no-jitter / on-jitter.
    pivot = (
        long_df.dropna(subset=["is_correct"])
        .groupby(["response_id", "raw_display_rule"], observed=True)["is_correct"]
        .mean()
        .unstack("raw_display_rule")
    )

    means = pivot.mean(axis=0, skipna=True).sort_values(ascending=False)
    if means.empty:
        raise RuntimeError("No raw-display accuracy data found for statistical testing.")

    best = means.index[0]
    results = []

    for display_rule in means.index:
        if display_rule == best:
            continue

        paired = pivot[[best, display_rule]].dropna()
        test_type = "paired"
        if not paired.empty:
            t_stat, p_val = stats.ttest_rel(paired[best], paired[display_rule])
            diff = paired[best] - paired[display_rule]
            sd_diff = diff.std(ddof=1)
            cohens_d = diff.mean() / sd_diff if sd_diff > 0 else np.nan
            n_pairs = len(paired)
        else:
            best_vals = pivot[best].dropna()
            other_vals = pivot[display_rule].dropna()
            if len(best_vals) < 2 or len(other_vals) < 2:
                continue
            test_type = "welch"
            t_stat, p_val = stats.ttest_ind(best_vals, other_vals, equal_var=False)
            pooled_sd = np.sqrt(
                ((best_vals.std(ddof=1) ** 2) + (other_vals.std(ddof=1) ** 2)) / 2
            )
            cohens_d = (best_vals.mean() - other_vals.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            n_pairs = min(len(best_vals), len(other_vals))

        results.append(
            {
                "best_raw_display": best,
                "comparison_raw_display": display_rule,
                "test_type": test_type,
                "n_pairs": n_pairs,
                "mean_accuracy_best": pivot[best].mean(),
                "mean_accuracy_other": pivot[display_rule].mean(),
                "mean_diff": pivot[best].mean() - pivot[display_rule].mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
            }
        )

    if not results:
        raise RuntimeError("No overlapping observations available to compare raw-display rules.")

    adjusted = bonferroni_adjust([r["p_value"] for r in results])
    for res, adj_p in zip(results, adjusted):
        res["p_value_bonferroni"] = adj_p

    results_df = pd.DataFrame(results).sort_values(
        ["p_value_bonferroni", "mean_diff"], ascending=[True, False]
    )
    return results_df, means, best


def ensure_raw_display_rule(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the raw_display_rule column exists on the DataFrame."""

    if "raw_display_rule" in df.columns:
        return df
    df = df.copy()
    df["raw_display_rule"] = df.apply(
        lambda r: format_raw_display(r.get("points", ""), r.get("jitter", "")), axis=1
    )
    return df


def _extract_level(param_name: str) -> str:
    if "T." in param_name:
        return param_name.split("T.", 1)[1]
    return param_name


def run_logit_vs_best(
    long_df: pd.DataFrame, factor_col: str, best_level: str, output_path: Path
) -> Optional[pd.DataFrame]:
    """Fit logistic regression with best level as reference; cluster SE by respondent."""

    df = long_df.dropna(subset=["is_correct", factor_col]).copy()
    if df.empty:
        return None

    df[factor_col] = df[factor_col].astype(str)

    formula = f"is_correct ~ C({factor_col}, Treatment(reference='{best_level}'))"
    try:
        model = smf.logit(formula=formula, data=df)
        fit = model.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df["response_id"]})
    except PerfectSeparationError:
        logger.warning("Perfect separation encountered for %s; skipping logit.", factor_col)
        return None

    params = fit.params.drop("Intercept", errors="ignore")
    conf = fit.conf_int().loc[params.index]
    odds = params.apply(np.exp)
    ci_lower = conf[0].apply(np.exp)
    ci_upper = conf[1].apply(np.exp)

    out = pd.DataFrame(
        {
            "level_vs_best": [_extract_level(name) for name in params.index],
            "odds_ratio": odds,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "p_value": fit.pvalues.loc[params.index],
        }
    ).reset_index(drop=True)

    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = ensure_raw_display_rule(load_long_responses())

    results_df, means, best = raw_display_tests(long_df)
    best_acc = means.loc[best]

    results_path = OUTPUT_DIR / "paired_ttests_best_raw_display.csv"
    ranking_path = OUTPUT_DIR / "raw_display_accuracy_ranking.csv"
    logit_path = OUTPUT_DIR / "logit_raw_display_rule_vs_best.csv"
    results_df.to_csv(results_path, index=False)
    means.to_csv(ranking_path, header=["mean_accuracy"])
    logit_results = run_logit_vs_best(
        long_df=long_df,
        factor_col="raw_display_rule",
        best_level=best,
        output_path=logit_path,
    )

    logger.info("Best raw-data display: %s (mean accuracy = %.3f)", best, best_acc)
    logger.info("Raw-display ranking saved to %s", ranking_path)
    logger.info("Pairwise test results saved to %s", results_path)
    if logit_results is not None:
        logger.info("Logit odds ratios vs. best saved to %s", logit_path)
        logger.info("Top logit rows:\n%s", logit_results.head().to_string(index=False))
    logger.info("Top comparisons:\n%s", results_df.head().to_string(index=False))


if __name__ == "__main__":
    main()

