# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "nba-api>=1.11.4",
#     "numpy>=2.4.5",
#     "pandas>=3.0.3",
#     "tqdm>=4.67.3",
# ]
# ///

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="NBA Data Pipeline")


@app.cell
def _():
    import marimo as mo
    mo.md("""
    # 🏀 NBA Game Prediction — Data Acquisition & Preprocessing
    **Project:** Predicting NBA Game Winners Using Machine Learning  
    **Data Range:** 2015–2025 Regular Seasons  
    **Source:** NBA API (nba_api)
    """)
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 1 — Install & Import Libraries
    """)
    return


@app.cell
def _():
    import subprocess, sys

    def pip_install(pkg):
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

    for package in ["nba_api", "pandas", "numpy", "scikit-learn", "tqdm"]:
        pip_install(package)
    print("✅ All packages installed.")
    return


@app.cell
def _():
    import time
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    from nba_api.stats.endpoints import leaguegamefinder, boxscoreadvancedv2
    from nba_api.stats.static import teams as nba_teams_static

    print("✅ Libraries imported successfully.")
    return leaguegamefinder, np, pd, time


@app.cell
def _(mo):
    mo.md("""
    ## Step 2 — Fetch Raw Game Logs (2015–2025)
    We use `LeagueGameFinder` to pull every regular season game per season.
    Rate-limiting delays are applied to respect the NBA API's throttle limits.
    """)
    return


@app.cell
def _(leaguegamefinder, pd, time):
    SEASONS = [
        "2015-16", "2016-17", "2017-18", "2018-19", "2019-20",
        "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
    ]

    raw_frames = []

    for season in SEASONS:
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable="Regular Season",
                timeout=60,
            )
            df_season = finder.get_data_frames()[0]
            df_season["SEASON"] = season
            raw_frames.append(df_season)
            print(f"  ✅ {season}: {len(df_season):,} team-game rows fetched")
            time.sleep(1.2)   # be polite to the API
        except Exception as e:
            print(f"  ⚠️  {season}: FAILED — {e}")

    raw_df = pd.concat(raw_frames, ignore_index=True)
    print(f"\n📦 Total raw rows: {len(raw_df):,}")
    raw_df.head(3)
    return (raw_df,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 3 — Data Cleaning & Restructuring

    The NBA API returns **one row per team per game**. We pivot this into
    **one row per game** (home team vs. away team) so every observation
    represents a single contest with both sides' stats side by side.
    """)
    return


@app.cell
def _(pd, raw_df):
    # ── 3a. Keep only the columns we need ──────────────────────────────────────
    KEEP_COLS = [
        "SEASON", "GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION",
        "MATCHUP",                        # e.g. "LAL vs. BOS" or "LAL @ BOS"
        "WL",                             # Win / Loss
        "PTS",                            # Points scored
        "FGM", "FGA", "FG_PCT",          # Field goals
        "FG3M", "FG3A", "FG3_PCT",       # 3-pointers
        "FTM", "FTA", "FT_PCT",          # Free throws
        "OREB", "DREB", "REB",           # Rebounds
        "AST", "STL", "BLK", "TOV",      # Misc box-score
        "PLUS_MINUS",
    ]

    available = [c for c in KEEP_COLS if c in raw_df.columns]
    df = raw_df[available].copy()

    # ── 3b. Parse dates & types ────────────────────────────────────────────────
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["WL"]        = df["WL"].str.strip()
    df["WIN"]       = (df["WL"] == "W").astype(int)

    # ── 3c. Home / Away flag from MATCHUP string ───────────────────────────────
    #   "LAL vs. BOS"  → LAL is the HOME team
    #   "LAL @ BOS"    → LAL is the AWAY team
    df["IS_HOME"] = df["MATCHUP"].str.contains("vs\\.").astype(int)

    print(f"Cleaned shape: {df.shape}")
    df.head(3)
    return (df,)


@app.cell
def _(df, pd):
    # ── 3d. Pivot to one-row-per-game ─────────────────────────────────────────
    home = df[df["IS_HOME"] == 1].copy()
    away = df[df["IS_HOME"] == 0].copy()

    STAT_COLS = [
        "TEAM_ID", "TEAM_ABBREVIATION", "WL", "WIN", "PTS",
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
        "AST", "STL", "BLK", "TOV", "PLUS_MINUS",
    ]
    stat_cols_available = [c for c in STAT_COLS if c in home.columns]

    home_renamed = home[["GAME_ID", "GAME_DATE", "SEASON"] + stat_cols_available].copy()
    home_renamed.columns = (
        ["GAME_ID", "GAME_DATE", "SEASON"] +
        [f"HOME_{c}" for c in stat_cols_available]
    )

    away_renamed = away[["GAME_ID"] + stat_cols_available].copy()
    away_renamed.columns = ["GAME_ID"] + [f"AWAY_{c}" for c in stat_cols_available]

    games = pd.merge(home_renamed, away_renamed, on="GAME_ID")
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    print(f"Game-level dataset shape: {games.shape}")
    games.head(3)
    return (games,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 4 — Feature Engineering

    We derive the project's core predictive features:

    | Feature | Description |
    |---|---|
    | `HOME_REST_DAYS` | Days since last game for the home team |
    | `AWAY_REST_DAYS` | Days since last game for the away team |
    | `HOME_WIN_PCT_L10` | Home team rolling win % over last 10 games |
    | `AWAY_WIN_PCT_L10` | Away team rolling win % over last 10 games |
    | `HOME_PTS_L5` | Home team rolling avg points (last 5) |
    | `AWAY_PTS_L5` | Away team rolling avg points (last 5) |
    | `HOME_TOV_L5` | Home team rolling avg turnovers (last 5) |
    | `AWAY_TOV_L5` | Away team rolling avg turnovers (last 5) |
    | `HOME_REB_L5` | Home team rolling avg rebounds (last 5) |
    | `AWAY_REB_L5` | Away team rolling avg rebounds (last 5) |
    | `HOME_FG_PCT_L5` | Home team rolling FG% (last 5) |
    | `AWAY_FG_PCT_L5` | Away team rolling FG% (last 5) |
    | `HOME_COURT` | Always 1 (home-court indicator) |
    | `TARGET` | 1 = home team wins, 0 = away team wins |
    """)
    return


@app.cell
def _(df, games, np):
    # ── 4a. Rest-days per team ─────────────────────────────────────────────────
    team_dates = (
        df[["TEAM_ID", "GAME_DATE", "GAME_ID"]]
        .sort_values(["TEAM_ID", "GAME_DATE"])
        .copy()
    )
    team_dates["PREV_DATE"] = team_dates.groupby("TEAM_ID")["GAME_DATE"].shift(1)
    team_dates["REST_DAYS"] = (
        team_dates["GAME_DATE"] - team_dates["PREV_DATE"]
    ).dt.days.fillna(3).clip(upper=14)   # cap at 14; NaN (season opener) → 3

    rest_lookup = team_dates.set_index("GAME_ID")["REST_DAYS"].to_dict()

    # Map rest days using GAME_ID + team identity stored in games
    games["HOME_REST_DAYS"] = games.apply(
        lambda r: rest_lookup.get(r["GAME_ID"], np.nan), axis=1
    )
    games["AWAY_REST_DAYS"] = games.apply(
        lambda r: rest_lookup.get(r["GAME_ID"], np.nan), axis=1
    )

    # ── 4b. Rolling stats per team (computed on the raw team-game df) ──────────
    team_stats = (
        df[["TEAM_ID", "GAME_DATE", "GAME_ID", "WIN", "PTS", "TOV", "REB", "FG_PCT"]]
        .sort_values(["TEAM_ID", "GAME_DATE"])
        .copy()
    )

    for col, window in [("WIN", 10), ("PTS", 5), ("TOV", 5), ("REB", 5), ("FG_PCT", 5)]:
        label = f"ROLL_{col}_L{window}"
        team_stats[label] = (
            team_stats.groupby("TEAM_ID")[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    roll_lookup = team_stats.set_index(["TEAM_ID", "GAME_ID"])[
        ["ROLL_WIN_L10", "ROLL_PTS_L5", "ROLL_TOV_L5", "ROLL_REB_L5", "ROLL_FG_PCT_L5"]
    ].to_dict(orient="index")

    def get_roll(team_id, game_id, stat):
        key = (team_id, game_id)
        return roll_lookup.get(key, {}).get(stat, np.nan)

    # Home team rolling features
    games["HOME_WIN_PCT_L10"] = games.apply(
        lambda r: get_roll(r["HOME_TEAM_ID"], r["GAME_ID"], "ROLL_WIN_L10"), axis=1
    )
    games["HOME_PTS_L5"] = games.apply(
        lambda r: get_roll(r["HOME_TEAM_ID"], r["GAME_ID"], "ROLL_PTS_L5"), axis=1
    )
    games["HOME_TOV_L5"] = games.apply(
        lambda r: get_roll(r["HOME_TEAM_ID"], r["GAME_ID"], "ROLL_TOV_L5"), axis=1
    )
    games["HOME_REB_L5"] = games.apply(
        lambda r: get_roll(r["HOME_TEAM_ID"], r["GAME_ID"], "ROLL_REB_L5"), axis=1
    )
    games["HOME_FG_PCT_L5"] = games.apply(
        lambda r: get_roll(r["HOME_TEAM_ID"], r["GAME_ID"], "ROLL_FG_PCT_L5"), axis=1
    )

    # Away team rolling features
    games["AWAY_WIN_PCT_L10"] = games.apply(
        lambda r: get_roll(r["AWAY_TEAM_ID"], r["GAME_ID"], "ROLL_WIN_L10"), axis=1
    )
    games["AWAY_PTS_L5"] = games.apply(
        lambda r: get_roll(r["AWAY_TEAM_ID"], r["GAME_ID"], "ROLL_PTS_L5"), axis=1
    )
    games["AWAY_TOV_L5"] = games.apply(
        lambda r: get_roll(r["AWAY_TEAM_ID"], r["GAME_ID"], "ROLL_TOV_L5"), axis=1
    )
    games["AWAY_REB_L5"] = games.apply(
        lambda r: get_roll(r["AWAY_TEAM_ID"], r["GAME_ID"], "ROLL_REB_L5"), axis=1
    )
    games["AWAY_FG_PCT_L5"] = games.apply(
        lambda r: get_roll(r["AWAY_TEAM_ID"], r["GAME_ID"], "ROLL_FG_PCT_L5"), axis=1
    )

    # ── 4c. Home-court indicator & target ─────────────────────────────────────
    games["HOME_COURT"] = 1
    games["TARGET"]     = games["HOME_WIN"]   # 1 = home team won

    print(f"Features added. Dataset shape: {games.shape}")
    games[["GAME_ID", "HOME_TEAM_ABBREVIATION", "AWAY_TEAM_ABBREVIATION",
           "HOME_WIN_PCT_L10", "AWAY_WIN_PCT_L10",
           "HOME_REST_DAYS", "AWAY_REST_DAYS", "TARGET"]].head(5)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 5 — Handle Missing Values & Final Column Selection
    """)
    return


@app.cell
def _(games):
    # ── 5a. Select only the modelling-relevant columns ─────────────────────────
    FINAL_COLS = [
        "GAME_ID", "GAME_DATE", "SEASON",
        "HOME_TEAM_ID", "HOME_TEAM_ABBREVIATION",
        "AWAY_TEAM_ID", "AWAY_TEAM_ABBREVIATION",
        # Core box-score (last actual game values — useful as context)
        "HOME_PTS", "AWAY_PTS",
        "HOME_FG_PCT", "AWAY_FG_PCT",
        "HOME_FG3_PCT", "AWAY_FG3_PCT",
        "HOME_FT_PCT", "AWAY_FT_PCT",
        "HOME_REB", "AWAY_REB",
        "HOME_AST", "AWAY_AST",
        "HOME_TOV", "AWAY_TOV",
        "HOME_STL", "AWAY_STL",
        "HOME_BLK", "AWAY_BLK",
        "HOME_PLUS_MINUS", "AWAY_PLUS_MINUS",
        # Engineered features (the primary ML inputs)
        "HOME_REST_DAYS", "AWAY_REST_DAYS",
        "HOME_WIN_PCT_L10", "AWAY_WIN_PCT_L10",
        "HOME_PTS_L5", "AWAY_PTS_L5",
        "HOME_TOV_L5", "AWAY_TOV_L5",
        "HOME_REB_L5", "AWAY_REB_L5",
        "HOME_FG_PCT_L5", "AWAY_FG_PCT_L5",
        "HOME_COURT",
        "TARGET",
    ]

    available_final = [c for c in FINAL_COLS if c in games.columns]
    clean = games[available_final].copy()

    # ── 5b. Drop rows where ANY engineered feature is missing ──────────────────
    engineered = [
        "HOME_WIN_PCT_L10", "AWAY_WIN_PCT_L10",
        "HOME_PTS_L5", "AWAY_PTS_L5",
        "HOME_REST_DAYS", "AWAY_REST_DAYS",
    ]
    eng_available = [c for c in engineered if c in clean.columns]
    before = len(clean)
    clean = clean.dropna(subset=eng_available).reset_index(drop=True)
    print(f"Dropped {before - len(clean):,} rows with missing engineered features.")

    # ── 5c. Fill remaining numeric NaNs with column median ─────────────────────
    num_cols = clean.select_dtypes(include="number").columns
    clean[num_cols] = clean[num_cols].fillna(clean[num_cols].median())

    # ── 5d. Sort chronologically ───────────────────────────────────────────────
    clean = clean.sort_values("GAME_DATE").reset_index(drop=True)

    print(f"\n✅ Final dataset: {len(clean):,} games × {len(clean.columns)} columns")
    print(f"   Date range: {clean['GAME_DATE'].min().date()} → {clean['GAME_DATE'].max().date()}")
    print(f"   Target balance — Home wins: {clean['TARGET'].mean():.1%}")

    clean.head(5)
    return (clean,)


@app.cell
def _(mo):
    mo.md("""
    ## Step 6 — Summary Statistics & Missing Value Report
    """)
    return


@app.cell
def _(clean, pd):
    # Missing value report
    missing = clean.isnull().sum()
    missing_pct = (missing / len(clean) * 100).round(2)
    miss_report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": missing_pct,
    }).query("`Missing Count` > 0")

    if miss_report.empty:
        print("✅ No missing values in the final dataset.")
    else:
        print("⚠️  Remaining missing values:")
        print(miss_report)

    print("\n📊 Descriptive statistics (engineered features):")
    feat_cols = [
        "HOME_REST_DAYS", "AWAY_REST_DAYS",
        "HOME_WIN_PCT_L10", "AWAY_WIN_PCT_L10",
        "HOME_PTS_L5", "AWAY_PTS_L5",
        "HOME_TOV_L5", "AWAY_TOV_L5",
        "HOME_REB_L5", "AWAY_REB_L5",
        "HOME_FG_PCT_L5", "AWAY_FG_PCT_L5",
    ]
    feat_available = [c for c in feat_cols if c in clean.columns]
    clean[feat_available].describe().round(3)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Step 7 — Export to CSV
    """)
    return


@app.cell
def _(clean):
    OUTPUT_PATH = "/Users/eron/Downloads/MSDS692/nba_games_clean.csv"
    clean.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Dataset saved to: {OUTPUT_PATH}")
    print(f"   Rows: {len(clean):,}  |  Columns: {len(clean.columns)}")
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## ✅ Pipeline Complete

    The file **`nba_games_clean.csv`** is ready for EDA and model training.

    ### What was produced
    | | |
    |---|---|
    | **Rows** | One row per regular-season game (2015–2025) |
    | **Target** | `TARGET = 1` → home team wins |
    | **Key features** | Rest days, rolling win %, rolling pts/tov/reb/FG%, home-court indicator |

    ### Next steps
    1. **EDA** — correlation heatmaps, distribution plots, class balance check
    2. **Feature selection** — remove highly correlated raw box-score columns
    3. **Model training** — Logistic Regression with cross-validation
    4. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
    """)
    return


if __name__ == "__main__":
    app.run()
