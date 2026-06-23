# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "pandas>=3.0.3",
#     "numpy>=2.4.5",
#     "matplotlib>=3.9.0",
#     "seaborn>=0.13.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="NBA Games EDA")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings("ignore")
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13,
                         "axes.labelsize": 11})
    return mo, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md("""
    # 🏀 NBA Games — Exploratory Data Analysis
    **Dataset:** `nba_games_clean.csv` — 2015-16 → 2024-25 Regular Seasons
    **One row = one game.** Produced by the Data Pipeline notebook.
    ---
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("nba_games_clean.csv", parse_dates=["GAME_DATE"])
    df["SEASON_YEAR"] = df["SEASON"].str[:4].astype(int)
    df["POINT_DIFF"] = df["HOME_PTS"] - df["AWAY_PTS"]
    return (df,)


@app.cell
def _(df, mo):
    mo.md(f"""
    ## 1 · Dataset Overview

    | Metric | Value |
    |---|---|
    | **Total games** | {len(df):,} |
    | **Seasons** | {df["SEASON"].nunique()} (2015-16 → 2024-25) |
    | **Teams** | {df["HOME_TEAM_ABBREVIATION"].nunique()} |
    | **Date range** | {df["GAME_DATE"].min().date()} → {df["GAME_DATE"].max().date()} |
    | **Columns** | {df.shape[1]} |
    | **Missing values** | {df.isnull().sum().sum()} |
    | **Home-win rate** | {df["TARGET"].mean():.1%} |
    """)
    return


@app.cell
def _(df, mo):
    mo.md("## 2 · Descriptive Statistics")
    _stat_cols = [
        "HOME_PTS", "AWAY_PTS", "HOME_FG_PCT", "AWAY_FG_PCT",
        "HOME_FG3_PCT", "AWAY_FG3_PCT", "HOME_REB", "AWAY_REB",
        "HOME_AST", "AWAY_AST", "HOME_TOV", "AWAY_TOV",
        "HOME_STL", "AWAY_STL", "HOME_BLK", "AWAY_BLK",
        "HOME_REST_DAYS", "AWAY_REST_DAYS",
        "HOME_WIN_PCT_L10", "AWAY_WIN_PCT_L10",
        "HOME_PTS_L5", "AWAY_PTS_L5",
    ]
    mo.ui.table(
        df[_stat_cols].describe().round(3).reset_index().rename(columns={"index": "stat"}),
        selection=None,
    )
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 3 · Target Class Balance")
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(11, 4))

    _counts = df["TARGET"].value_counts().sort_index()
    _ax0.bar(["Away Win (0)", "Home Win (1)"], _counts.values,
             color=["#e06c75", "#61afef"], edgecolor="white", linewidth=1.4)
    for _i, _v in enumerate(_counts.values):
        _ax0.text(_i, _v + 50, f"{_v:,}\n({_v/len(df):.1%})", ha="center", fontsize=10)
    _ax0.set_title("Home vs Away Wins")
    _ax0.set_ylabel("Games")
    _ax0.set_ylim(0, _counts.max() * 1.15)

    _season_wins = df.groupby("SEASON")["TARGET"].mean().reset_index()
    _ax1.bar(_season_wins["SEASON"], _season_wins["TARGET"],
             color="#98c379", edgecolor="white")
    _ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50%")
    _ax1.set_title("Home Win % by Season")
    _ax1.set_ylabel("Home Win Rate")
    _ax1.set_ylim(0.4, 0.7)
    _ax1.tick_params(axis="x", rotation=40)
    _ax1.legend()

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 4 · Points Distribution & Point Differential")
    _fig, (_ax0, _ax1, _ax2) = plt.subplots(1, 3, figsize=(15, 4))

    _ax0.hist(df["HOME_PTS"], bins=40, color="#61afef", edgecolor="white", alpha=0.85)
    _ax0.axvline(df["HOME_PTS"].mean(), color="red", linewidth=1.5,
                 linestyle="--", label=f"Mean {df['HOME_PTS'].mean():.1f}")
    _ax0.set_title("Home Points")
    _ax0.set_xlabel("Points")
    _ax0.legend()

    _ax1.hist(df["AWAY_PTS"], bins=40, color="#e5c07b", edgecolor="white", alpha=0.85)
    _ax1.axvline(df["AWAY_PTS"].mean(), color="red", linewidth=1.5,
                 linestyle="--", label=f"Mean {df['AWAY_PTS'].mean():.1f}")
    _ax1.set_title("Away Points")
    _ax1.set_xlabel("Points")
    _ax1.legend()

    _diff = df["POINT_DIFF"]
    _ax2.hist(_diff, bins=50, color="#c678dd", edgecolor="white", alpha=0.85)
    _ax2.axvline(0, color="black", linewidth=1.5, linestyle="--")
    _ax2.axvline(_diff.mean(), color="red", linewidth=1.5,
                 linestyle="--", label=f"Mean {_diff.mean():.1f}")
    _ax2.set_title("Point Differential (Home − Away)")
    _ax2.set_xlabel("Pts")
    _ax2.legend()

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 5 · Scoring Trends by Season")
    _season_pts = df.groupby("SEASON")[["HOME_PTS", "AWAY_PTS"]].mean().reset_index()
    _x = range(len(_season_pts))

    _fig, _ax = plt.subplots(figsize=(12, 4.5))
    _ax.plot(_x, _season_pts["HOME_PTS"], marker="o", linewidth=2,
             color="#61afef", label="Home Avg Pts")
    _ax.plot(_x, _season_pts["AWAY_PTS"], marker="s", linewidth=2,
             color="#e06c75", label="Away Avg Pts")
    _ax.set_xticks(list(_x))
    _ax.set_xticklabels(_season_pts["SEASON"], rotation=35)
    _ax.set_title("Average Points Scored per Season")
    _ax.set_ylabel("Points")
    _ax.legend()
    _ax.set_ylim(100, 120)

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 6 · Box-Score Stats — Home Wins vs Losses")
    _cols   = ["HOME_FG_PCT", "HOME_FG3_PCT", "HOME_REB", "HOME_AST",
               "HOME_TOV", "HOME_STL", "HOME_BLK"]
    _labels = ["FG%", "3P%", "Rebounds", "Assists", "Turnovers", "Steals", "Blocks"]
    _wins   = df[df["TARGET"] == 1]
    _losses = df[df["TARGET"] == 0]

    _fig, _axs = plt.subplots(2, 4, figsize=(16, 7))
    _axs_flat = _axs.flatten()
    for _col, _lbl, _ax in zip(_cols, _labels, _axs_flat):
        _ax.hist(_wins[_col],   bins=35, alpha=0.6, color="#98c379",
                 label="Win",  edgecolor="white")
        _ax.hist(_losses[_col], bins=35, alpha=0.6, color="#e06c75",
                 label="Loss", edgecolor="white")
        _ax.set_title(_lbl)
        _ax.legend(fontsize=8)
    _axs_flat[-1].axis("off")

    plt.suptitle("Home Team Stats Distribution: Wins vs Losses", fontsize=14, y=1.01)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 7 · Engineered Rolling Feature Distributions")
    _roll_cols = [
        ("HOME_WIN_PCT_L10", "AWAY_WIN_PCT_L10", "Rolling Win % (L10)"),
        ("HOME_PTS_L5",      "AWAY_PTS_L5",      "Rolling Avg Points (L5)"),
        ("HOME_TOV_L5",      "AWAY_TOV_L5",      "Rolling Avg Turnovers (L5)"),
        ("HOME_REB_L5",      "AWAY_REB_L5",      "Rolling Avg Rebounds (L5)"),
        ("HOME_FG_PCT_L5",   "AWAY_FG_PCT_L5",   "Rolling FG% (L5)"),
    ]
    _fig, _axs = plt.subplots(1, 5, figsize=(20, 4))
    for _ax, (_hc, _ac, _title) in zip(_axs, _roll_cols):
        _ax.hist(df[_hc].dropna(), bins=40, alpha=0.65,
                 color="#61afef", label="Home", edgecolor="white")
        _ax.hist(df[_ac].dropna(), bins=40, alpha=0.65,
                 color="#e06c75", label="Away", edgecolor="white")
        _ax.set_title(_title, fontsize=10)
        _ax.legend(fontsize=8)

    plt.suptitle("Rolling Feature Distributions — Home vs Away", fontsize=13, y=1.02)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 8 · Rest Days Analysis")
    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(13, 4.5))

    _ax0.hist(df["HOME_REST_DAYS"], bins=15, alpha=0.7,
              color="#61afef", label="Home", edgecolor="white")
    _ax0.hist(df["AWAY_REST_DAYS"], bins=15, alpha=0.7,
              color="#e06c75", label="Away", edgecolor="white")
    _ax0.set_title("Rest Days Distribution")
    _ax0.set_xlabel("Days Since Last Game")
    _ax0.legend()

    _rest_win = (df.groupby("HOME_REST_DAYS")["TARGET"]
                   .agg(["mean", "count"])
                   .reset_index()
                   .query("count >= 20"))
    _ax1.scatter(_rest_win["HOME_REST_DAYS"], _rest_win["mean"],
                 s=_rest_win["count"] / 5, alpha=0.75,
                 color="#98c379", edgecolors="white")
    _ax1.axhline(0.5, color="gray", linestyle="--")
    _ax1.set_title("Home Win Rate by Home Rest Days\n(bubble size = game count)")
    _ax1.set_xlabel("Home Rest Days")
    _ax1.set_ylabel("Home Win Rate")

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt, sns):
    mo.md("## 9 · Correlation Heatmap — Engineered Features vs Target")
    _feat_cols = [
        "HOME_REST_DAYS", "AWAY_REST_DAYS",
        "HOME_WIN_PCT_L10", "AWAY_WIN_PCT_L10",
        "HOME_PTS_L5", "AWAY_PTS_L5",
        "HOME_TOV_L5", "AWAY_TOV_L5",
        "HOME_REB_L5", "AWAY_REB_L5",
        "HOME_FG_PCT_L5", "AWAY_FG_PCT_L5",
        "TARGET",
    ]
    _corr = df[_feat_cols].corr()

    _fig, _ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(
        _corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.4, linecolor="white",
        ax=_ax, annot_kws={"size": 8},
    )
    _ax.set_title("Pearson Correlation — Engineered Features & Target", fontsize=13)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, np, plt):
    mo.md("## 10 · Win-Pct Spread vs Home Win Probability")
    _spread_df = df.copy()
    _spread_df["WIN_PCT_SPREAD"] = (
        _spread_df["HOME_WIN_PCT_L10"] - _spread_df["AWAY_WIN_PCT_L10"]
    )
    _spread_df["SPREAD_BIN"] = np.round(_spread_df["WIN_PCT_SPREAD"], 1)

    _binned = (
        _spread_df.groupby("SPREAD_BIN")
        .agg(home_win_rate=("TARGET", "mean"), count=("TARGET", "count"))
        .reset_index()
        .query("count >= 15")
    )

    _fig, _ax = plt.subplots(figsize=(13, 4.5))
    _sc = _ax.scatter(
        _binned["SPREAD_BIN"], _binned["home_win_rate"],
        s=_binned["count"] / 4, alpha=0.75, c=_binned["home_win_rate"],
        cmap="RdYlGn", edgecolors="white", linewidth=0.5,
    )
    _ax.axhline(0.5, color="gray", linestyle="--")
    _ax.axvline(0.0, color="gray", linestyle="--")
    plt.colorbar(_sc, ax=_ax, label="Home Win Rate")
    _ax.set_xlabel("HOME_WIN_PCT_L10 − AWAY_WIN_PCT_L10")
    _ax.set_ylabel("Observed Home Win Rate")
    _ax.set_title("Rolling Win-% Spread vs Home Win Rate  (bubble = sample size)")
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 11 · Home Games per Team")
    _home_counts = (
        df.groupby("HOME_TEAM_ABBREVIATION")
        .size()
        .sort_values(ascending=True)
        .reset_index(name="games")
    )
    _median_g = _home_counts["games"].median()

    _fig, _ax = plt.subplots(figsize=(14, 6))
    _ax.barh(
        _home_counts["HOME_TEAM_ABBREVIATION"],
        _home_counts["games"],
        color=["#98c379" if g > _median_g else "#e06c75"
               for g in _home_counts["games"]],
        edgecolor="white",
    )
    _ax.axvline(_median_g, color="navy", linestyle="--",
                linewidth=1.2, label="Median")
    _ax.set_xlabel("Home Games (2015–2025)")
    _ax.set_title("Total Home Games per Team")
    _ax.legend()
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 12 · Home Win Rate by Team (2015–2025)")
    _team_stats = (
        df.groupby("HOME_TEAM_ABBREVIATION")
        .agg(home_win_rate=("TARGET", "mean"), games=("TARGET", "count"))
        .sort_values("home_win_rate", ascending=True)
        .reset_index()
    )

    _fig, _ax = plt.subplots(figsize=(14, 6))
    _ax.barh(
        _team_stats["HOME_TEAM_ABBREVIATION"],
        _team_stats["home_win_rate"],
        color=["#98c379" if v >= 0.6 else "#e5c07b" if v >= 0.5 else "#e06c75"
               for v in _team_stats["home_win_rate"]],
        edgecolor="white",
    )
    _ax.axvline(0.5, color="gray", linestyle="--")
    _ax.axvline(df["TARGET"].mean(), color="navy", linestyle="--",
                linewidth=1.2, label=f"League avg {df['TARGET'].mean():.1%}")
    _ax.set_xlim(0.3, 0.85)
    _ax.set_xlabel("Home Win Rate")
    _ax.set_title("Home Win Rate by Team (all seasons)")
    _ax.legend()
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 13 · Home Win Rate by Month")
    _month_map  = {10: "Oct", 11: "Nov", 12: "Dec", 1: "Jan",
                   2: "Feb", 3: "Mar", 4: "Apr"}
    _month_order = [10, 11, 12, 1, 2, 3, 4]
    _df_month = df.copy()
    _df_month["MONTH"] = _df_month["GAME_DATE"].dt.month
    _monthly = (
        _df_month.groupby("MONTH")["TARGET"].mean()
        .reset_index()
        .pipe(lambda d: d[d["MONTH"].isin(_month_map)])
        .assign(MONTH_NAME=lambda d: d["MONTH"].map(_month_map))
        .set_index("MONTH").reindex(_month_order).reset_index()
    )

    _fig, _ax = plt.subplots(figsize=(9, 4))
    _ax.bar(_monthly["MONTH_NAME"], _monthly["TARGET"],
            color="#61afef", edgecolor="white")
    _ax.axhline(0.5, color="gray", linestyle="--")
    _ax.set_title("Home Win Rate by Calendar Month")
    _ax.set_ylabel("Home Win Rate")
    _ax.set_ylim(0.4, 0.7)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(df, mo, plt):
    mo.md("## 14 · Feature Correlations with TARGET")
    _exclude = {"TARGET", "GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID",
                "HOME_COURT", "SEASON_YEAR"}
    _num_cols = [c for c in df.select_dtypes(include="number").columns
                 if c not in _exclude]
    _corr_target = (df[_num_cols + ["TARGET"]]
                    .corr()["TARGET"].drop("TARGET").sort_values())

    _fig, _ax = plt.subplots(figsize=(10, 9))
    _ax.barh(
        _corr_target.index, _corr_target.values,
        color=["#e06c75" if v < 0 else "#98c379" for v in _corr_target.values],
        edgecolor="white",
    )
    _ax.axvline(0, color="black", linewidth=0.8)
    _ax.set_title("Pearson Correlation with TARGET (home win = 1)")
    _ax.set_xlabel("Correlation coefficient")
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## ✅ EDA Summary

    | Finding | Detail |
    |---|---|
    | **Home-court advantage is real** | Home teams win **56.6%** of games across all seasons |
    | **Stable class imbalance** | Home win rate is consistent across all 10 seasons (53–60%) |
    | **Scoring has trended up** | League avg points rose from ~102 (2015-16) to ~114 (2024-25) |
    | **FG% is the strongest box-score predictor** | Clearest separation in win/loss distributions |
    | **Rolling win-% spread is predictive** | Larger spread correlates with higher home win probability |
    | **Rest days have modest effect** | Slight home advantage with more rest, but effect is small |
    | **No missing values** | Dataset is fully clean and ready for modelling |

    ### Recommended next steps
    1. Drop raw box-score columns — they're post-game outcomes, not pre-game features
    2. Use engineered rolling features + rest days as primary ML inputs
    3. Address mild class imbalance (56/44) with class-weighted loss or stratified CV
    4. Logistic Regression is a strong baseline given the near-linear relationships observed
    """)
    return


if __name__ == "__main__":
    app.run()
