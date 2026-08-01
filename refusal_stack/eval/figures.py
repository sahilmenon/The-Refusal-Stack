"""Plotly figure generation for the refusal evaluation results.

All functions accept the results DataFrame (built in metrics.py) and an
output directory, save HTML and PNG versions, and return the figure + saved path.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def _save(fig: go.Figure, output_dir: str, stem: str) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"{stem}.html"
    png_path = out / f"{stem}.png"
    fig.write_html(str(html_path))
    try:
        fig.write_image(str(png_path))
    except Exception as exc:
        logger.warning("Could not export %s as PNG (kaleido issue?): %s", stem, exc)
    return str(html_path)


def plot_refusal_rates(df: pd.DataFrame, output_dir: str) -> tuple[go.Figure, str]:
    rows = []
    for ds_name, group in df.groupby("dataset"):
        harmful = group[group["label"] == "harmful"]
        benign = group[group["label"] == "benign"]
        rr = harmful["is_refusal"].mean() if len(harmful) > 0 else 0.0
        frr = benign["is_refusal"].mean() if len(benign) > 0 else 0.0
        rows.append({"dataset": ds_name, "type": "Refusal Rate (Harmful)", "rate": rr})
        rows.append({"dataset": ds_name, "type": "False-Refusal Rate (Benign)", "rate": frr})
    plot_df = pd.DataFrame(rows)

    fig = px.bar(
        plot_df,
        x="dataset",
        y="rate",
        color="type",
        barmode="group",
        color_discrete_map={
            "Refusal Rate (Harmful)": "crimson",
            "False-Refusal Rate (Benign)": "seagreen",
        },
        title="Refusal Rate by Dataset",
        labels={"rate": "Rate", "dataset": "Dataset"},
    )
    fig.update_yaxes(range=[0, 1])
    fig.add_hline(
        y=0.05,
        line_dash="dash",
        line_color="gray",
        annotation_text="5% false-refusal threshold",
    )
    path = _save(fig, output_dir, "refusal_rates")
    return fig, path


def plot_judge_regex_confusion(df: pd.DataFrame, output_dir: str) -> tuple[go.Figure, str]:
    judged = df[df["judge_hit"].notna()]
    if len(judged) == 0:
        logger.warning("No judge scores available; skipping confusion matrix")
        fig = go.Figure()
        path = _save(fig, output_dir, "judge_regex_confusion")
        return fig, path

    # 2x2 matrix: rows=regex, cols=judge
    # [[regex=T & judge=T, regex=T & judge=F], [regex=F & judge=T, regex=F & judge=F]]
    tt = len(judged[(judged["regex_hit"]) & (judged["judge_hit"])])
    tf = len(judged[(judged["regex_hit"]) & (~judged["judge_hit"])])
    ft = len(judged[(~judged["regex_hit"]) & (judged["judge_hit"])])
    ff_ = len(judged[(~judged["regex_hit"]) & (~judged["judge_hit"])])

    n = len(judged)
    z = [[tt, tf], [ft, ff_]]
    text = [
        [f"{tt}<br>({tt/n:.0%})", f"{tf}<br>({tf/n:.0%})"],
        [f"{ft}<br>({ft/n:.0%})", f"{ff_}<br>({ff_/n:.0%})"],
    ]

    fig = ff.create_annotated_heatmap(
        z=z,
        annotation_text=text,
        x=["Judge=Refusal", "Judge=Compliance"],
        y=["Regex=Refusal", "Regex=Compliance"],
        colorscale="Blues",
    )
    fig.update_layout(title="Regex vs LLM-Judge Agreement")
    path = _save(fig, output_dir, "judge_regex_confusion")
    return fig, path


def plot_partial_compliance_breakdown(df: pd.DataFrame, output_dir: str) -> tuple[go.Figure, str]:
    harmful = df[df["label"] == "harmful"]
    if len(harmful) == 0:
        fig = go.Figure()
        path = _save(fig, output_dir, "partial_compliance")
        return fig, path

    full_refusal = harmful["is_refusal"] & ~harmful["partial_compliance"]
    partial = harmful["partial_compliance"]
    full_compliance = ~harmful["is_refusal"] & ~harmful["partial_compliance"]

    labels = ["Full Refusal", "Partial Compliance", "Full Compliance"]
    values = [full_refusal.sum(), partial.sum(), full_compliance.sum()]

    fig = px.pie(
        names=labels,
        values=values,
        title="Response Categories (Harmful Prompts)",
    )
    path = _save(fig, output_dir, "partial_compliance")
    return fig, path


def plot_score_distribution(df: pd.DataFrame, output_dir: str) -> tuple[go.Figure, str]:
    rows = []
    for ds_name, group in df.groupby("dataset"):
        refusals = group["is_refusal"].sum()
        partial = group["partial_compliance"].sum()
        compliance = (~group["is_refusal"]).sum()
        rows.append({"dataset": ds_name, "category": "Refusal", "count": refusals})
        rows.append({"dataset": ds_name, "category": "Partial Compliance", "count": partial})
        rows.append({"dataset": ds_name, "category": "Compliance", "count": compliance})
    plot_df = pd.DataFrame(rows)
    fig = px.bar(
        plot_df,
        x="dataset",
        y="count",
        color="category",
        barmode="stack",
        title="Response Distribution per Dataset",
    )
    path = _save(fig, output_dir, "score_distribution")
    return fig, path


def save_all_figures(df: pd.DataFrame, output_dir: str) -> list[str]:
    paths = []
    for fn in [
        plot_refusal_rates,
        plot_judge_regex_confusion,
        plot_partial_compliance_breakdown,
        plot_score_distribution,
    ]:
        try:
            _, path = fn(df, output_dir)
            paths.append(path)
        except Exception as exc:
            logger.warning("Figure %s failed: %s", fn.__name__, exc)
    return paths
