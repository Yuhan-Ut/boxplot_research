"""Run paired t-tests to find the best boxplot design.

This script loads the survey export, builds the tidy long-form table used in
``analyze_responses.py``, aggregates participant accuracy by plot design, and
performs paired t-tests comparing the top-performing design against every other
design. Results (including Bonferroni-adjusted p-values) are written to
``analysis_outputs/paired_ttests_best_design.csv`` along with a design ranking
table for reference.
"""

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
try:
    import statsmodels.formula.api as smf
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "statsmodels is required for OLS tests. Install with `pip install statsmodels`."
    ) from exc

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_outputs"
W_ACC = 1.0
W_LOSS = 1.0
logger = logging.getLogger(__name__)


def load_long_responses() -> pd.DataFrame:
    """Load and clean the raw survey export into long-form responses."""

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
    """Apply a Bonferroni correction to a list of p-values."""

    count = len(p_values)
    return [min(p * count, 1.0) for p in p_values]


def ensure_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite_score = w_acc*is_correct - w_loss*ratio_l1_loss_norm."""

    if "composite_score" in df.columns:
        return df
    df = df.copy()
    if "ratio_l1_loss_norm" not in df.columns:
        return df
    df["composite_score"] = W_ACC * df["is_correct"] - W_LOSS * df["ratio_l1_loss_norm"]
    return df


def paired_design_tests(
    long_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Run paired t-tests comparing the best design to all others.

    Returns
    -------
    results : DataFrame
        One row per comparison (best vs. other) with raw and adjusted p-values.
    design_means : Series
        Mean participant accuracy for each design (descending).
    best_design : str
        Label of the top-performing design.
    """

    pivot = (
        long_df.dropna(subset=["is_correct"])
        .groupby(["response_id", "plot_type"], observed=True)["is_correct"]
        .mean()
        .unstack("plot_type")
    )

    design_means = pivot.mean(axis=0, skipna=True).sort_values(ascending=False)
    if design_means.empty:
        raise RuntimeError("No design accuracy data found for statistical testing.")

    best_design = design_means.index[0]
    results = []

    for design in design_means.index:
        if design == best_design:
            continue

        # First try paired t-test (requires the same respondents to answer both designs).
        paired = pivot[[best_design, design]].dropna()
        test_type = "paired"
        if not paired.empty:
            t_stat, p_val = stats.ttest_rel(paired[best_design], paired[design])
            diff = paired[best_design] - paired[design]
            sd_diff = diff.std(ddof=1)
            cohens_d = diff.mean() / sd_diff if sd_diff > 0 else np.nan
            n_pairs = len(paired)
        else:
            # Fallback: use Welch's unequal-variance t-test on all available responses.
            best_vals = pivot[best_design].dropna()
            other_vals = pivot[design].dropna()
            if len(best_vals) < 2 or len(other_vals) < 2:
                # Not enough data to compare; skip this design.
                continue
            test_type = "welch"
            t_stat, p_val = stats.ttest_ind(best_vals, other_vals, equal_var=False)
            # Cohen's d for independent samples (Hedges' g not critical here).
            pooled_sd = np.sqrt(
                ((best_vals.std(ddof=1) ** 2) + (other_vals.std(ddof=1) ** 2)) / 2
            )
            cohens_d = (best_vals.mean() - other_vals.mean()) / pooled_sd if pooled_sd > 0 else np.nan
            n_pairs = min(len(best_vals), len(other_vals))

        results.append(
            {
                "best_design": best_design,
                "comparison_design": design,
                "test_type": test_type,
                "n_pairs": n_pairs,
                "mean_accuracy_best": pivot[best_design].mean(),
                "mean_accuracy_other": pivot[design].mean(),
                "mean_diff": pivot[best_design].mean() - pivot[design].mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
            }
        )

    if not results:
        raise RuntimeError("No overlapping observations available to compare designs.")

    adjusted = bonferroni_adjust([r["p_value"] for r in results])
    for res, adj_p in zip(results, adjusted):
        res["p_value_bonferroni"] = adj_p

    results_df = pd.DataFrame(results).sort_values(
        ["p_value_bonferroni", "mean_diff"], ascending=[True, False]
    )
    return results_df, design_means, best_design


def ratio_loss_tests(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Run tests on ratio L1 loss (lower is better)."""

    df = long_df.dropna(subset=["ratio_l1_loss"]).copy()
    if df.empty:
        raise RuntimeError("No ratio loss data available for statistical testing.")

    pivot = (
        df.groupby(["response_id", "plot_type"], observed=True)["ratio_l1_loss"]
        .mean()
        .unstack("plot_type")
    )

    mean_losses = pivot.mean(axis=0, skipna=True).sort_values(ascending=True)
    if mean_losses.empty:
        raise RuntimeError("No ratio loss data available for statistical testing.")

    best = mean_losses.index[0]
    results = []

    for design in mean_losses.index:
        if design == best:
            continue

        paired = pivot[[best, design]].dropna()
        test_type = "paired"
        if not paired.empty:
            t_stat, p_val = stats.ttest_rel(paired[design], paired[best])
            mean_diff = paired[design].mean() - paired[best].mean()
            n_pairs = len(paired)
        else:
            best_vals = pivot[best].dropna()
            other_vals = pivot[design].dropna()
            if len(best_vals) < 2 or len(other_vals) < 2:
                continue
            test_type = "welch"
            t_stat, p_val = stats.ttest_ind(other_vals, best_vals, equal_var=False)
            mean_diff = other_vals.mean() - best_vals.mean()
            n_pairs = min(len(best_vals), len(other_vals))

        results.append(
            {
                "best_design": best,
                "comparison_design": design,
                "test_type": test_type,
                "n_pairs": n_pairs,
                "mean_loss_best": pivot[best].mean(),
                "mean_loss_other": pivot[design].mean(),
                "mean_diff": mean_diff,  # positive means other has higher loss (best is better)
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )

    if not results:
        raise RuntimeError("No overlapping observations available to compare designs (ratio loss).")

    adjusted = bonferroni_adjust([r["p_value"] for r in results])
    for res, adj_p in zip(results, adjusted):
        res["p_value_bonferroni"] = adj_p

    results_df = pd.DataFrame(results).sort_values(
        ["p_value_bonferroni", "mean_diff"], ascending=[True, False]
    )
    return results_df, mean_losses, best


def composite_score_tests(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Run tests on composite score: w_acc*is_correct - w_loss*ratio_l1_loss_norm (higher is better)."""

    df = ensure_composite_score(long_df).dropna(subset=["is_correct", "ratio_l1_loss_norm", "composite_score"]).copy()
    if df.empty:
        raise RuntimeError("No composite-score data available for statistical testing.")

    df["composite_score"] = W_ACC * df["is_correct"] - W_LOSS * df["ratio_l1_loss_norm"]

    pivot = (
        df.groupby(["response_id", "plot_type"], observed=True)["composite_score"]
        .mean()
        .unstack("plot_type")
    )

    mean_scores = pivot.mean(axis=0, skipna=True).sort_values(ascending=False)
    if mean_scores.empty:
        raise RuntimeError("No composite-score data available for statistical testing.")

    best = mean_scores.index[0]
    results = []

    for design in mean_scores.index:
        if design == best:
            continue

        paired = pivot[[best, design]].dropna()
        test_type = "paired"
        if not paired.empty:
            t_stat, p_val = stats.ttest_rel(paired[best], paired[design])
            mean_diff = paired[best].mean() - paired[design].mean()
            n_pairs = len(paired)
        else:
            best_vals = pivot[best].dropna()
            other_vals = pivot[design].dropna()
            if len(best_vals) < 2 or len(other_vals) < 2:
                continue
            test_type = "welch"
            t_stat, p_val = stats.ttest_ind(best_vals, other_vals, equal_var=False)
            mean_diff = best_vals.mean() - other_vals.mean()
            n_pairs = min(len(best_vals), len(other_vals))

        results.append(
            {
                "best_design": best,
                "comparison_design": design,
                "test_type": test_type,
                "n_pairs": n_pairs,
                "mean_score_best": pivot[best].mean(),
                "mean_score_other": pivot[design].mean(),
                "mean_diff": mean_diff,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )

    if not results:
        raise RuntimeError("No overlapping observations available to compare designs (composite score).")

    adjusted = bonferroni_adjust([r["p_value"] for r in results])
    for res, adj_p in zip(results, adjusted):
        res["p_value_bonferroni"] = adj_p

    results_df = pd.DataFrame(results).sort_values(
        ["p_value_bonferroni", "mean_diff"], ascending=[True, False]
    )
    return results_df, mean_scores, best


def run_ols_vs_best(
    long_df: pd.DataFrame, factor_col: str, outcome_col: str, best_level: str, output_path: Path
) -> Optional[pd.DataFrame]:
    """OLS with best level as reference; HC3 robust SEs."""

    df = long_df.dropna(subset=[outcome_col, factor_col]).copy()
    if df.empty:
        return None

    df[factor_col] = df[factor_col].astype(str)
    formula = f"{outcome_col} ~ C({factor_col}, Treatment(reference='{best_level}'))"
    model = smf.ols(formula=formula, data=df)
    fit = model.fit(cov_type="HC3")

    params = fit.params
    bse = fit.bse
    t_vals = fit.tvalues
    p_vals = fit.pvalues
    conf = fit.conf_int()

    def clean_term(name: str) -> str:
        if name == "Intercept":
            return "_cons"
        marker = f"C({factor_col})[T."
        if marker in name:
            return name.split("[T.", 1)[1].rstrip("]")
        return name

    out = pd.DataFrame(
        {
            "term": [clean_term(name) for name in params.index],
            "coef": params,
            "std_err": bse,
            "t": t_vals,
            "p_value": p_vals,
            "ci_lower": conf[0],
            "ci_upper": conf[1],
        }
    ).reset_index(drop=True)

    out.to_csv(output_path, index=False)
    return out




def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = load_long_responses()

    results_df, design_means, best_design = paired_design_tests(long_df)
    best_accuracy = design_means.loc[best_design]

    ratio_results_df, ratio_means, best_design_ratio = ratio_loss_tests(long_df)
    best_ratio = ratio_means.loc[best_design_ratio]

    composite_results_df, composite_means, best_design_composite = composite_score_tests(long_df)
    best_composite = composite_means.loc[best_design_composite]

    # OLS regressions (HC3 robust SEs) using the best level as reference
    ols_accuracy_path = OUTPUT_DIR / "ols_accuracy_plot_type_vs_best.csv"
    ols_ratio_path = OUTPUT_DIR / "ols_ratio_plot_type_vs_best.csv"
    ols_composite_path = OUTPUT_DIR / "ols_composite_plot_type_vs_best.csv"
    long_df = ensure_composite_score(long_df)
    ols_accuracy = run_ols_vs_best(
        long_df=long_df,
        factor_col="plot_type",
        outcome_col="is_correct",
        best_level=best_design,
        output_path=ols_accuracy_path,
    )
    ols_ratio = run_ols_vs_best(
        long_df=long_df,
        factor_col="plot_type",
        outcome_col="ratio_l1_loss",
        best_level=best_design_ratio,
        output_path=ols_ratio_path,
    )
    ols_composite = run_ols_vs_best(
        long_df=long_df,
        factor_col="plot_type",
        outcome_col="composite_score",
        best_level=best_design_composite,
        output_path=ols_composite_path,
    )

    results_path = OUTPUT_DIR / "paired_ttests_best_design.csv"
    design_ranking_path = OUTPUT_DIR / "design_accuracy_ranking.csv"
    ratio_results_path = OUTPUT_DIR / "paired_ttests_ratio_best_design.csv"
    ratio_ranking_path = OUTPUT_DIR / "design_ratio_loss_ranking.csv"
    composite_results_path = OUTPUT_DIR / "paired_ttests_composite_best_design.csv"
    composite_ranking_path = OUTPUT_DIR / "design_composite_score_ranking.csv"
    results_df.to_csv(results_path, index=False)
    design_means.to_csv(design_ranking_path, header=["mean_accuracy"])
    ratio_results_df.to_csv(ratio_results_path, index=False)
    ratio_means.to_csv(ratio_ranking_path, header=["mean_ratio_l1_loss"])
    composite_results_df.to_csv(composite_results_path, index=False)
    composite_means.to_csv(composite_ranking_path, header=["mean_composite_score"])

    logger.info("Best design: %s (mean accuracy = %.3f)", best_design, best_accuracy)
    logger.info("Best design (lowest ratio loss): %s (mean ratio L1 = %.3f)", best_design_ratio, best_ratio)
    logger.info("Best design (composite score): %s (mean score = %.3f)", best_design_composite, best_composite)
    logger.info("Design ranking saved to %s", design_ranking_path)
    logger.info("Pairwise test results saved to %s", results_path)
    logger.info("Ratio-loss ranking saved to %s", ratio_ranking_path)
    logger.info("Ratio-loss pairwise results saved to %s", ratio_results_path)
    logger.info("Composite ranking saved to %s", composite_ranking_path)
    logger.info("Composite pairwise results saved to %s", composite_results_path)
    if ols_accuracy is not None:
        logger.info("OLS accuracy vs. best saved to %s", ols_accuracy_path)
    if ols_ratio is not None:
        logger.info("OLS ratio loss vs. best saved to %s", ols_ratio_path)
    if ols_composite is not None:
        logger.info("OLS composite vs. best saved to %s", ols_composite_path)
    logger.info("Top comparisons:\n%s", results_df.head().to_string(index=False))


if __name__ == "__main__":
    main()

