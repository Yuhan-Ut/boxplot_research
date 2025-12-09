"""Run t-tests to find the best whisker rule."""

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


def whisker_tests(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    pivot = (
        long_df.dropna(subset=["is_correct"])
        .groupby(["response_id", "whisker"], observed=True, sort=False)["is_correct"]
        .mean()
        .unstack("whisker")
    )

    means = pivot.mean(axis=0, skipna=True)
    if means.empty:
        raise RuntimeError("No whisker accuracy data found for statistical testing.")

    best = means.idxmax()
    results = []

    for whisker in means.index:
        if whisker == best:
            continue

        paired = pivot[[best, whisker]].dropna()
        test_type = "paired"
        if not paired.empty:
            t_stat, p_val = stats.ttest_rel(paired[best], paired[whisker])
            diff = paired[best] - paired[whisker]
            sd_diff = diff.std(ddof=1)
            cohens_d = diff.mean() / sd_diff if sd_diff > 0 else np.nan
            n_pairs = len(paired)
        else:
            best_vals = pivot[best].dropna()
            other_vals = pivot[whisker].dropna()
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
                "best_whisker": best,
                "comparison_whisker": whisker,
                "test_type": test_type,
                "n_pairs": n_pairs,
                "mean_accuracy_best": pivot[best].mean(),
                "mean_accuracy_other": pivot[whisker].mean(),
                "mean_diff": pivot[best].mean() - pivot[whisker].mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
            }
        )

    if not results:
        raise RuntimeError("No overlapping observations available to compare whisker rules.")

    adjusted = bonferroni_adjust([r["p_value"] for r in results])
    for res, adj_p in zip(results, adjusted):
        res["p_value_bonferroni"] = adj_p

    results_df = pd.DataFrame(results)
    return results_df, means, best


def ratio_loss_tests(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Run tests on ratio L1 loss (lower is better) for whisker rules."""

    df = long_df.dropna(subset=["ratio_l1_loss"]).copy()
    if df.empty:
        raise RuntimeError("No ratio loss data available for statistical testing.")

    pivot = (
        df.groupby(["response_id", "whisker"], observed=True, sort=False)["ratio_l1_loss"]
        .mean()
        .unstack("whisker")
    )

    mean_losses = pivot.mean(axis=0, skipna=True)
    if mean_losses.empty:
        raise RuntimeError("No ratio loss data available for statistical testing.")

    best = mean_losses.idxmin()
    results = []

    for whisker in mean_losses.index:
        if whisker == best:
            continue

        paired = pivot[[best, whisker]].dropna()
        test_type = "paired"
        if not paired.empty:
            t_stat, p_val = stats.ttest_rel(paired[whisker], paired[best])
            mean_diff = paired[whisker].mean() - paired[best].mean()
            n_pairs = len(paired)
        else:
            best_vals = pivot[best].dropna()
            other_vals = pivot[whisker].dropna()
            if len(best_vals) < 2 or len(other_vals) < 2:
                continue
            test_type = "welch"
            t_stat, p_val = stats.ttest_ind(other_vals, best_vals, equal_var=False)
            mean_diff = other_vals.mean() - best_vals.mean()
            n_pairs = min(len(best_vals), len(other_vals))

        results.append(
            {
                "best_whisker": best,
                "comparison_whisker": whisker,
                "test_type": test_type,
                "n_pairs": n_pairs,
                "mean_loss_best": pivot[best].mean(),
                "mean_loss_other": pivot[whisker].mean(),
                "mean_diff": mean_diff,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )

    if not results:
        raise RuntimeError("No overlapping observations available to compare whisker rules (ratio loss).")

    adjusted = bonferroni_adjust([r["p_value"] for r in results])
    for res, adj_p in zip(results, adjusted):
        res["p_value_bonferroni"] = adj_p

    results_df = pd.DataFrame(results)
    return results_df, mean_losses, best


def composite_score_tests(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Run tests on composite score: w_acc*is_correct - w_loss*ratio_l1_loss_norm (higher is better)."""

    df = long_df.copy()
    if "composite_score" not in df.columns and "ratio_l1_loss_norm" in df.columns:
        df["composite_score"] = 1.0 * df["is_correct"] - 1.0 * df["ratio_l1_loss_norm"]
    df = df.dropna(subset=["is_correct", "ratio_l1_loss_norm", "composite_score"])
    if df.empty:
        raise RuntimeError("No composite-score data available for statistical testing.")

    df["composite_score"] = 1.0 * df["is_correct"] - 1.0 * df["ratio_l1_loss_norm"]

    pivot = (
        df.groupby(["response_id", "whisker"], observed=True, sort=False)["composite_score"]
        .mean()
        .unstack("whisker")
    )

    mean_scores = pivot.mean(axis=0, skipna=True)
    if mean_scores.empty:
        raise RuntimeError("No composite-score data available for statistical testing.")

    best = mean_scores.idxmax()
    results = []

    for whisker in mean_scores.index:
        if whisker == best:
            continue

        paired = pivot[[best, whisker]].dropna()
        test_type = "paired"
        if not paired.empty:
            t_stat, p_val = stats.ttest_rel(paired[best], paired[whisker])
            mean_diff = paired[best].mean() - paired[whisker].mean()
            n_pairs = len(paired)
        else:
            best_vals = pivot[best].dropna()
            other_vals = pivot[whisker].dropna()
            if len(best_vals) < 2 or len(other_vals) < 2:
                continue
            test_type = "welch"
            t_stat, p_val = stats.ttest_ind(best_vals, other_vals, equal_var=False)
            mean_diff = best_vals.mean() - other_vals.mean()
            n_pairs = min(len(best_vals), len(other_vals))

        results.append(
            {
                "best_whisker": best,
                "comparison_whisker": whisker,
                "test_type": test_type,
                "n_pairs": n_pairs,
                "mean_score_best": pivot[best].mean(),
                "mean_score_other": pivot[whisker].mean(),
                "mean_diff": mean_diff,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )

    if not results:
        raise RuntimeError("No overlapping observations available to compare whisker rules (composite score).")

    adjusted = bonferroni_adjust([r["p_value"] for r in results])
    for res, adj_p in zip(results, adjusted):
        res["p_value_bonferroni"] = adj_p

    results_df = pd.DataFrame(results)
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

    out.to_csv(output_path, index=False, float_format="%.2f")
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = load_long_responses()

    results_df, means, best = whisker_tests(long_df)
    best_acc = means.loc[best]

    ratio_results_df, ratio_means, best_ratio = ratio_loss_tests(long_df)
    best_ratio_loss = ratio_means.loc[best_ratio]

    composite_results_df, composite_means, best_composite = composite_score_tests(long_df)
    best_composite_score = composite_means.loc[best_composite]

    # OLS regressions (HC3 robust SEs) using the best level as reference
    ols_accuracy_path = OUTPUT_DIR / "ols_accuracy_whisker_vs_best.csv"
    ols_ratio_path = OUTPUT_DIR / "ols_ratio_whisker_vs_best.csv"
    ols_composite_path = OUTPUT_DIR / "ols_composite_whisker_vs_best.csv"
    long_df = long_df.copy()
    if "composite_score" not in long_df.columns and "ratio_l1_loss_norm" in long_df.columns:
        long_df["composite_score"] = 1.0 * long_df["is_correct"] - 1.0 * long_df["ratio_l1_loss_norm"]
    ols_accuracy = run_ols_vs_best(
        long_df=long_df,
        factor_col="whisker",
        outcome_col="is_correct",
        best_level=best,
        output_path=ols_accuracy_path,
    )
    ols_ratio = run_ols_vs_best(
        long_df=long_df,
        factor_col="whisker",
        outcome_col="ratio_l1_loss",
        best_level=best_ratio,
        output_path=ols_ratio_path,
    )
    ols_composite = run_ols_vs_best(
        long_df=long_df,
        factor_col="whisker",
        outcome_col="composite_score",
        best_level=best_composite,
        output_path=ols_composite_path,
    )

    results_path = OUTPUT_DIR / "paired_ttests_best_whisker.csv"
    ranking_path = OUTPUT_DIR / "whisker_accuracy_ranking.csv"
    ratio_results_path = OUTPUT_DIR / "paired_ttests_ratio_best_whisker.csv"
    ratio_ranking_path = OUTPUT_DIR / "whisker_ratio_loss_ranking.csv"
    composite_results_path = OUTPUT_DIR / "paired_ttests_composite_best_whisker.csv"
    composite_ranking_path = OUTPUT_DIR / "whisker_composite_score_ranking.csv"
    results_df.to_csv(results_path, index=False, float_format="%.2f")
    means.to_csv(ranking_path, header=["mean_accuracy"], float_format="%.2f")
    ratio_results_df.to_csv(ratio_results_path, index=False, float_format="%.2f")
    ratio_means.to_csv(ratio_ranking_path, header=["mean_ratio_l1_loss"], float_format="%.2f")
    composite_results_df.to_csv(composite_results_path, index=False, float_format="%.2f")
    composite_means.to_csv(composite_ranking_path, header=["mean_composite_score"], float_format="%.2f")

    logger.info("Best whisker rule: %s (mean accuracy = %.2f)", best, best_acc)
    logger.info("Best whisker (lowest ratio loss): %s (mean ratio L1 = %.2f)", best_ratio, best_ratio_loss)
    logger.info("Best whisker (composite score): %s (mean score = %.2f)", best_composite, best_composite_score)
    logger.info("Whisker ranking saved to %s", ranking_path)
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
    logger.info("Top comparisons:\n%s", results_df.head().to_string(index=False, float_format="%.2f"))


if __name__ == "__main__":
    main()

