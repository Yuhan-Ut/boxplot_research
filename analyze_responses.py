"""Analyze survey responses for the boxplot variance study.

Loads the Qualtrics export and metadata about each trial, evaluates
participant accuracy and ratio estimates for every plot type, and
produces summary tables plus diagnostic plots.
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# --- Paths and global config -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "proejct_data.csv"
TRIAL_LOG_PATH = PROJECT_ROOT / "boxplot_stimuli" / "trial_sd_log.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="colorblind")
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


# --- Helper structures -------------------------------------------------------

@dataclass
class QuestionGroup:
    """Logical grouping of the four columns that describe a single plot."""

    trial_idx: int
    which_col: str
    confidence_col: str
    ratio_col: str
    comment_col: str


def load_question_groups(csv_path: Path) -> List[QuestionGroup]:
    """Identify the repeating question blocks for each plot."""

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        headers = next(reader)
        question_texts = next(reader)

    groups: List[QuestionGroup] = []
    for idx, text in enumerate(question_texts):
        if not text:
            continue
        text = text.strip()
        if text.startswith("Which plot has higher variance"):
            cols = headers[idx : idx + 4]
            if len(cols) < 4:
                continue
            groups.append(
                QuestionGroup(
                    trial_idx=len(groups) + 1,
                    which_col=cols[0],
                    confidence_col=cols[1],
                    ratio_col=cols[2],
                    comment_col=cols[3],
                )
            )

    if not groups:
        raise RuntimeError("No plot question groups found in the CSV header.")

    return groups


CHOICE_MAP = {
    "1": "Left",
    "2": "Right",
    "3": "Equal",
    "left": "Left",
    "right": "Right",
    "a": "Left",
    "b": "Right",
    "plot a": "Left",
    "plot b": "Right",
}

# Preserve the canonical ordering for plot types when displaying results/plots.
PLOT_TYPE_ORDER = [
    "MinMax - Points Off",
    "MinMax - Points On (No Jitter)",
    "MinMax - Points On (Jitter)",
    "Tukey - Points Off",
    "Tukey - Points On (No Jitter)",
    "Tukey - Points On (Jitter)",
]


def normalize_choice(value: object) -> Optional[str]:
    """Reduce free-text multiple choice answers to Left/Right/Equal."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text in CHOICE_MAP:
        return CHOICE_MAP[text]

    try:
        numeric = float(text)
    except ValueError:
        numeric = None
    if numeric is not None:
        if math.isclose(numeric, 1.0):
            return "Left"
        if math.isclose(numeric, 2.0):
            return "Right"
        if math.isclose(numeric, 3.0):
            return "Equal"

    if "left" in text and "right" not in text:
        return "Left"
    if "right" in text and "left" not in text:
        return "Right"
    if "equal" in text or "same" in text or "both" in text:
        return "Equal"

    # Unknown free-text answer
    return None


NUM_PATTERN = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parts_to_float(parts: Iterable[str]) -> Optional[float]:
    tokens = [p for p in parts if p]
    if len(tokens) != 2:
        return None
    try:
        left = float(tokens[0])
        right = float(tokens[1])
    except ValueError:
        return None
    if right == 0:
        return None
    return left / right


def parse_ratio(value: object) -> Optional[float]:
    """Parse ratio strings such as '2', '2:1', '1/3', or '2x'."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    text = (
        text.replace(" ", "")
        .replace("left", "")
        .replace("right", "")
        .replace("ratio", "")
        .replace("approx", "")
    )

    if text.endswith("x") and text[:-1].replace(".", "", 1).isdigit():
        return float(text[:-1])

    if ":" in text:
        ratio = _parts_to_float(text.split(":"))
        if ratio is not None:
            return ratio
    if "/" in text:
        ratio = _parts_to_float(text.split("/"))
        if ratio is not None:
            return ratio

    match = NUM_PATTERN.search(text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None

    return None


def to_float(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def describe_plot_type(whisker: str, points: str, jitter: str) -> str:
    if points == "Points Off":
        return f"{whisker} - Points Off"
    jitter = (jitter or "Jitter Off").strip()
    jitter_label = "Jitter" if jitter.lower() == "jitter on" else "No Jitter"
    return f"{whisker} - Points On ({jitter_label})"


def prepare_long_form(
    responses: pd.DataFrame, groups: List[QuestionGroup], meta: pd.DataFrame
) -> pd.DataFrame:
    """Convert the wide Qualtrics export into a tidy long table."""

    if len(groups) != len(meta):
        raise ValueError(
            f"Question groups ({len(groups)}) do not match trial log rows ({len(meta)})."
        )

    meta_sorted = meta.sort_values("TrialIdx").reset_index(drop=True)
    records = []

    for _, resp in responses.iterrows():
        respondent_id = resp.get("ResponseId")
        email = resp.get("Email")

        for idx, group in enumerate(groups):
            trial_meta = meta_sorted.iloc[idx]
            which_val = resp.get(group.which_col)
            confidence_val = resp.get(group.confidence_col)
            ratio_val = resp.get(group.ratio_col)
            comment_val = resp.get(group.comment_col)

            if (
                pd.isna(which_val)
                and pd.isna(confidence_val)
                and pd.isna(ratio_val)
                and pd.isna(comment_val)
            ):
                continue

            answer_choice = normalize_choice(which_val)
            confidence = to_float(confidence_val)
            ratio_guess = parse_ratio(ratio_val)

            left_sd = float(trial_meta["Left_SD"])
            right_sd = float(trial_meta["Right_SD"])
            if right_sd == 0:
                true_ratio = np.nan
            else:
                true_ratio = left_sd / right_sd

            l1_loss = np.nan
            if (
                ratio_guess is not None
                and pd.notna(true_ratio)
                and true_ratio > 0
                and ratio_guess > 0
            ):
                l1_loss = abs(ratio_guess - true_ratio)

            more_variable = trial_meta["More_Variable"]
            is_correct = np.nan
            if answer_choice:
                if more_variable == "Equal":
                    is_correct = 1.0 if answer_choice in ["Equal", "Right"] else 0.0
                else:
                    is_correct = 1.0 if answer_choice == more_variable else 0.0

            records.append(
                {
                    "response_id": respondent_id,
                    "email": email,
                    "trial_idx": int(trial_meta["TrialIdx"]),
                    "trial_type": trial_meta["TrialType"],
                    "instance": trial_meta["Instance"],
                    "whisker": trial_meta["Whisker"],
                    "points": trial_meta["Points"],
                    "jitter": trial_meta["Jitter"] if isinstance(trial_meta["Jitter"], str) else "",
                    "plot_type": describe_plot_type(
                        trial_meta["Whisker"], trial_meta["Points"], trial_meta["Jitter"]
                    ),
                    "left_sd": left_sd,
                    "right_sd": right_sd,
                    "true_ratio": true_ratio,
                    "more_variable": more_variable,
                    "answer_raw": which_val,
                    "answer_choice": answer_choice,
                    "is_correct": is_correct,
                    "confidence": confidence,
                    "ratio_guess": ratio_guess,
                    "ratio_l1_loss": l1_loss,
                    "comment": comment_val,
                }
            )

    if not records:
        raise RuntimeError("No response records were produced; check the input data.")

    df = pd.DataFrame(records)
    df["plot_type"] = pd.Categorical(df["plot_type"], categories=PLOT_TYPE_ORDER, ordered=True)
    # Treat raw-data overlay as a binary factor independent of jitter.
    df["raw_display"] = np.where(df["points"].str.lower() == "points on", "Points On", "Points Off")

    if df["ratio_l1_loss"].notna().any():
        max_l1 = df["ratio_l1_loss"].max()
        if pd.notna(max_l1) and max_l1 > 0:
            df["ratio_l1_loss_norm"] = df["ratio_l1_loss"] / max_l1
        else:
            df["ratio_l1_loss_norm"] = 0.0
    else:
        df["ratio_l1_loss_norm"] = np.nan

    return df


def summarize(long_df: pd.DataFrame) -> dict:
    """Create aggregated tables used for reporting and plotting."""

    # Only treat respondents with at least one scorable answer as participants.
    valid_mask = long_df["is_correct"].notna()
    valid_participants = long_df.loc[valid_mask, "response_id"].nunique()

    design_summary = (
        long_df.groupby("plot_type", observed=True, sort=False)
        .agg(
            responses=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("confidence", "mean"),
            median_ratio_guess=("ratio_guess", "median"),
            mean_l1_loss=("ratio_l1_loss", "mean"),
            mean_norm_l1_loss=("ratio_l1_loss_norm", "mean"),
        )
        .reset_index()
    )
    design_summary["plot_type"] = pd.Categorical(
        design_summary["plot_type"], categories=PLOT_TYPE_ORDER, ordered=True
    )
    design_summary = design_summary.sort_values("plot_type")

    whisker_summary = (
        long_df.groupby("whisker", observed=True, sort=False)
        .agg(
            responses=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("confidence", "mean"),
            median_ratio_guess=("ratio_guess", "median"),
            mean_l1_loss=("ratio_l1_loss", "mean"),
            mean_norm_l1_loss=("ratio_l1_loss_norm", "mean"),
        )
        .reset_index()
    )

    raw_display_summary = (
        long_df.groupby("raw_display", observed=True, sort=False)
        .agg(
            responses=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("confidence", "mean"),
            median_ratio_guess=("ratio_guess", "median"),
            mean_l1_loss=("ratio_l1_loss", "mean"),
            mean_norm_l1_loss=("ratio_l1_loss_norm", "mean"),
        )
        .reset_index()
    )

    trial_type_summary = (
        long_df.groupby("trial_type", observed=True, sort=False)
        .agg(
            responses=("is_correct", "count"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("confidence", "mean"),
            mean_l1_loss=("ratio_l1_loss", "mean"),
            mean_norm_l1_loss=("ratio_l1_loss_norm", "mean"),
        )
        .reset_index()
    )

    overall = {
        "responses": int(long_df["is_correct"].count()),
        "participants": valid_participants,
        "overall_accuracy": float(long_df["is_correct"].mean()),
        "mean_confidence": float(long_df["confidence"].mean()),
    }

    return {
        "design_summary": design_summary,
        "whisker_summary": whisker_summary,
        "raw_display_summary": raw_display_summary,
        "trial_type_summary": trial_type_summary,
        "overall": overall,
    }


def build_accuracy_display(df: pd.DataFrame, label_col: str, label_name: str) -> pd.DataFrame:
    """Prepare a human-readable accuracy table for logging/output."""

    columns = [label_col, "responses", "accuracy"]
    if "mean_confidence" in df.columns:
        columns.append("mean_confidence")

    display = df[columns].copy()
    display.rename(columns={label_col: label_name}, inplace=True)
    display["accuracy (%)"] = (display["accuracy"] * 100).round(2)
    display.drop(columns="accuracy", inplace=True)

    if "mean_confidence" in display.columns:
        display["mean_confidence"] = display["mean_confidence"].round(2)
        display.rename(columns={"mean_confidence": "Mean Confidence"}, inplace=True)

    return display[[label_name, "responses", "accuracy (%)"] + (["Mean Confidence"] if "Mean Confidence" in display.columns else [])]


def save_tables(summaries: dict) -> None:
    summaries["design_summary"].to_csv(
        OUTPUT_DIR / "accuracy_by_plot_type.csv", index=False, float_format="%.2f"
    )
    summaries["whisker_summary"].to_csv(
        OUTPUT_DIR / "accuracy_by_whisker_rule.csv", index=False, float_format="%.2f"
    )
    summaries["raw_display_summary"].to_csv(
        OUTPUT_DIR / "accuracy_by_raw_display.csv", index=False, float_format="%.2f"
    )
    summaries["trial_type_summary"].to_csv(
        OUTPUT_DIR / "accuracy_by_trial_type.csv", index=False, float_format="%.2f"
    )


def make_plots(long_df: pd.DataFrame, summaries: dict) -> None:
    """Generate jpeg/png summaries describing the findings."""

    design_summary = summaries["design_summary"]
    whisker_summary = summaries["whisker_summary"]
    raw_display_summary = summaries["raw_display_summary"]
    trial_type_summary = summaries["trial_type_summary"]

    plot_accuracy_order = [
        pt for pt in PLOT_TYPE_ORDER if pt in design_summary["plot_type"].unique()
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=design_summary,
        y="plot_type",
        x="accuracy",
        order=plot_accuracy_order,
        ax=ax,
    )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Plot type")
    ax.set_xlim(0, 1)
    ax.set_title("Accuracy by Plot Type")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_by_plot_type.png", dpi=300)
    plt.close(fig)

    if not whisker_summary.empty:
        whisker_order = whisker_summary["whisker"].tolist()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=whisker_summary,
            x="whisker",
            y="accuracy",
            order=whisker_order,
            ax=ax,
        )
        ax.set_xlabel("Whisker rule")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy by Whisker Rule")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "accuracy_by_whisker_rule.png", dpi=300)
        plt.close(fig)

    if not raw_display_summary.empty:
        display_order = raw_display_summary["raw_display"].tolist()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=raw_display_summary,
            x="raw_display",
            y="accuracy",
            order=display_order,
            ax=ax,
        )
        ax.set_xlabel("Raw data display")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy by Raw-Data Display")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "accuracy_by_raw_display.png", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=design_summary,
        y="plot_type",
        x="mean_l1_loss",
        order=plot_accuracy_order,
        ax=ax,
    )
    ax.set_xlabel("Mean L1 loss")
    ax.set_ylabel("Plot type")
    ax.set_title("Mean L1 Loss by Plot Type")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l1_loss_by_plot_type.png", dpi=300)
    plt.close(fig)

    if not whisker_summary.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=whisker_summary,
            x="whisker",
            y="mean_l1_loss",
            order=whisker_summary["whisker"],
            ax=ax,
        )
        ax.set_xlabel("Whisker rule")
        ax.set_ylabel("Mean L1 loss")
        ax.set_title("Mean L1 Loss by Whisker Rule")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "l1_loss_by_whisker_rule.png", dpi=300)
        plt.close(fig)

    if not raw_display_summary.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=raw_display_summary,
            x="raw_display",
            y="mean_l1_loss",
            order=raw_display_summary["raw_display"],
            ax=ax,
        )
        ax.set_xlabel("Raw data display")
        ax.set_ylabel("Mean L1 loss")
        ax.set_title("Mean L1 Loss by Raw-Data Display")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "l1_loss_by_raw_display.png", dpi=300)
        plt.close(fig)

    trial_accuracy_order = trial_type_summary["trial_type"].tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=trial_type_summary,
        x="trial_type",
        y="accuracy",
        order=trial_accuracy_order,
        ax=ax,
    )
    ax.set_xlabel("Trial type")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Scenario")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "accuracy_by_trial_type.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=trial_type_summary,
        x="trial_type",
        y="mean_l1_loss",
        order=trial_type_summary["trial_type"],
        ax=ax,
    )
    ax.set_xlabel("Trial type")
    ax.set_ylabel("Mean L1 loss")
    ax.set_title("Mean L1 Loss by Scenario")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "l1_loss_by_trial_type.png", dpi=300)
    plt.close(fig)

    ratio_df = long_df.dropna(subset=["ratio_l1_loss"])
    if not ratio_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(
            data=ratio_df,
            x="trial_type",
            y="ratio_l1_loss",
            ax=ax,
            order=trial_type_summary["trial_type"],
        )
        ax.set_xlabel("Trial type")
        ax.set_ylabel("Ratio L1 loss")
        ax.set_title("Distribution of Ratio L1 Loss by Scenario")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "ratio_l1_loss_by_trial_type.png", dpi=300)
        plt.close(fig)


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Cannot find survey export at {DATA_PATH}")
    if not TRIAL_LOG_PATH.exists():
        raise FileNotFoundError(f"Cannot find trial log at {TRIAL_LOG_PATH}")

    groups = load_question_groups(DATA_PATH)

    responses = pd.read_csv(DATA_PATH, skiprows=[1, 2])
    responses = responses[responses.get("Finished", 0) == 1].copy()
    responses = responses[responses.get("Status", 0) == 0]
    responses.reset_index(drop=True, inplace=True)

    trial_meta = pd.read_csv(TRIAL_LOG_PATH)
    for col in ("Left_SD", "Right_SD"):
        trial_meta[col] = pd.to_numeric(trial_meta[col], errors="coerce")
    trial_meta["Jitter"] = trial_meta["Jitter"].fillna("").astype(str)

    long_df = prepare_long_form(responses, groups, trial_meta)
    long_df.to_csv(OUTPUT_DIR / "long_format_responses.csv", index=False)

    summaries = summarize(long_df)
    save_tables(summaries)
    make_plots(long_df, summaries)

    overall = summaries["overall"]
    logger.info("=== Overall Summary ===")
    logger.info(
        "Participants (valid): %s | Responses: %s | Accuracy: %.2f | Mean confidence: %.2f",
        overall["participants"],
        overall["responses"],
        overall["overall_accuracy"],
        overall["mean_confidence"],
    )

    design_accuracy_table = build_accuracy_display(
        summaries["design_summary"], "plot_type", "Plot Type"
    )
    whisker_accuracy_table = build_accuracy_display(
        summaries["whisker_summary"], "whisker", "Whisker Rule"
    )
    raw_display_accuracy_table = build_accuracy_display(
        summaries["raw_display_summary"], "raw_display", "Raw Data Display"
    )
    trial_accuracy_table = build_accuracy_display(
        summaries["trial_type_summary"], "trial_type", "Trial Type"
    )

    logger.info(
        "=== Accuracy by Plot Type ===\n%s",
        design_accuracy_table.to_string(index=False, float_format="{:.2f}".format),
    )
    logger.info(
        "=== Accuracy by Whisker Rule ===\n%s",
        whisker_accuracy_table.to_string(index=False, float_format="{:.2f}".format),
    )
    logger.info(
        "=== Accuracy by Raw Data Display ===\n%s",
        raw_display_accuracy_table.to_string(index=False, float_format="{:.2f}".format),
    )
    logger.info(
        "=== Accuracy by Trial Type ===\n%s",
        trial_accuracy_table.to_string(index=False, float_format="{:.2f}".format),
    )

    design_display = summaries["design_summary"][
        [
            "plot_type",
            "responses",
            "mean_norm_l1_loss",
            "mean_l1_loss",
        ]
    ]
    whisker_display = summaries["whisker_summary"][
        [
            "whisker",
            "responses",
            "mean_norm_l1_loss",
            "mean_l1_loss",
        ]
    ]
    raw_display_display = summaries["raw_display_summary"][
        [
            "raw_display",
            "responses",
            "mean_norm_l1_loss",
            "mean_l1_loss",
        ]
    ]
    trial_display = summaries["trial_type_summary"][
        [
            "trial_type",
            "responses",
            "mean_norm_l1_loss",
            "mean_l1_loss",
        ]
    ]
    logger.info(
        "=== Normalized L1 by Plot Type ===\n%s",
        design_display.to_string(index=False, float_format="{:.2f}".format),
    )
    logger.info(
        "=== Normalized L1 by Whisker Rule ===\n%s",
        whisker_display.to_string(index=False, float_format="{:.2f}".format),
    )
    logger.info(
        "=== Normalized L1 by Raw Data Display ===\n%s",
        raw_display_display.to_string(index=False, float_format="{:.2f}".format),
    )
    logger.info(
        "=== Normalized L1 by Trial Type ===\n%s",
        trial_display.to_string(index=False, float_format="{:.2f}".format),
    )


if __name__ == "__main__":
    main()

