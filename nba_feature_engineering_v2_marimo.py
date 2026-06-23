# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "pandas>=3.0.3",
#     "numpy>=2.4.5",
#     "scikit-learn>=1.5.0",
#     "matplotlib>=3.9.0",
#     "seaborn>=0.13.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="NBA Feature Engineering v2")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score
    warnings.filterwarnings("ignore")
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13, "axes.labelsize": 11})
    STATS         = ["PTS","FG_PCT","FG3_PCT","REB","AST","TOV","STL","BLK"]
    TRAIN_SEASONS = ["2015-16","2016-17","2017-18","2018-19","2019-20","2020-21"]
    VAL_SEASONS   = ["2021-22","2022-23"]
    TEST_SEASONS  = ["2023-24","2024-25"]
    return (
        LogisticRegression,
        STATS,
        StandardScaler,
        TEST_SEASONS,
        TRAIN_SEASONS,
        VAL_SEASONS,
        accuracy_score,
        mo,
        np,
        pd,
        plt,
        sns,
    )


@app.cell
def _(mo):
    mo.md("""
    # 🏀 NBA Feature Engineering v2
    **Input:** `nba_games_clean.csv` · **Output:** `nba_games_features_v2.csv`

    | Rule | Why |
    |---|---|
    | `shift(1)` on every rolling stat | Current game never in its own features |
    | Window-only: L5, L10 | No cumulative season win% — too strong a proxy for final standings |
    | No PLUS_MINUS rolling | Confounds team quality with opponent strength |
    | No HOME_COURT | Constant = 1, zero variance |
    | H2H: shift-based cumulative | Verified corr ≈ 0.08, not 0.83 |
    ---
    """)
    return


@app.cell
def _(mo, pd):
    df = pd.read_csv(
        "nba_games_clean.csv", parse_dates=["GAME_DATE"]
    ).sort_values(["GAME_DATE","GAME_ID"]).reset_index(drop=True)

    mo.md(f"""
    ## Step 1 · Load
    **{len(df):,} games** · {df['SEASON'].nunique()} seasons ·
    {df['GAME_DATE'].min().date()} → {df['GAME_DATE'].max().date()} ·
    {df.shape[1]} columns
    """)
    return (df,)


@app.cell
def _(df, mo, pd):
    mo.md("## Step 2 · Per-Team Game Log")

    _home = df[[
        "GAME_ID","GAME_DATE","SEASON","HOME_TEAM_ID","HOME_TEAM_ABBREVIATION",
        "HOME_PTS","HOME_FG_PCT","HOME_FG3_PCT","HOME_FT_PCT",
        "HOME_REB","HOME_AST","HOME_TOV","HOME_STL","HOME_BLK","TARGET",
    ]].copy()
    _home.columns = [
        "GAME_ID","GAME_DATE","SEASON","TEAM_ID","TEAM_ABB",
        "PTS","FG_PCT","FG3_PCT","FT_PCT","REB","AST","TOV","STL","BLK","WIN",
    ]

    _away = df[[
        "GAME_ID","GAME_DATE","SEASON","AWAY_TEAM_ID","AWAY_TEAM_ABBREVIATION",
        "AWAY_PTS","AWAY_FG_PCT","AWAY_FG3_PCT","AWAY_FT_PCT",
        "AWAY_REB","AWAY_AST","AWAY_TOV","AWAY_STL","AWAY_BLK","TARGET",
    ]].copy()
    _away.columns = [
        "GAME_ID","GAME_DATE","SEASON","TEAM_ID","TEAM_ABB",
        "PTS","FG_PCT","FG3_PCT","FT_PCT","REB","AST","TOV","STL","BLK","WIN",
    ]
    _away["WIN"] = 1 - _away["WIN"]

    team_log = (
        pd.concat([_home, _away], ignore_index=True)
        .sort_values(["TEAM_ID","GAME_DATE","GAME_ID"])
        .reset_index(drop=True)
    )
    print(f"Team log: {len(team_log):,} rows × {team_log.shape[1]} cols")
    return (team_log,)


@app.cell
def _(STATS, mo, team_log):
    mo.md("## Step 3 · Rolling L5 & L10 Stats")

    def _roll(series, window, min_p):
        return series.shift(1).rolling(window, min_periods=min_p).mean()

    for _s in STATS:
        team_log[f"ROLL_{_s}_L5"]  = team_log.groupby("TEAM_ID")[_s].transform(
            lambda x: _roll(x, 5, 3)
        )
        team_log[f"ROLL_{_s}_L10"] = team_log.groupby("TEAM_ID")[_s].transform(
            lambda x: _roll(x, 10, 5)
        )

    team_log["WIN_L5"]  = team_log.groupby("TEAM_ID")["WIN"].transform(
        lambda x: _roll(x, 5, 3)
    )
    team_log["WIN_L10"] = team_log.groupby("TEAM_ID")["WIN"].transform(
        lambda x: _roll(x, 10, 5)
    )
    print(f"Added {len(STATS)*2 + 2} rolling columns")
    return


@app.cell
def _(mo, team_log):
    mo.md("## Step 4 · Win / Loss Streaks")

    def _win_streak(wins):
        result, _n = [], 0
        for v in wins.shift(1):
            _n = _n + 1 if v == 1 else 0
            result.append(_n)
        return result

    def _loss_streak(wins):
        result, _n = [], 0
        for v in wins.shift(1):
            _n = _n + 1 if v == 0 else 0
            result.append(_n)
        return result

    team_log["WIN_STREAK"]  = team_log.groupby("TEAM_ID")["WIN"].transform(_win_streak)
    team_log["LOSS_STREAK"] = team_log.groupby("TEAM_ID")["WIN"].transform(_loss_streak)
    print("WIN_STREAK and LOSS_STREAK added")
    return


@app.cell
def _(STATS, mo, np, team_log):
    mo.md("## Step 5 · Build Lookup Table")

    ROLL_COLS = (
        [f"ROLL_{s}_L5"  for s in STATS] +
        [f"ROLL_{s}_L10" for s in STATS] +
        ["WIN_L5","WIN_L10","WIN_STREAK","LOSS_STREAK"]
    )
    lookup = (
        team_log.set_index(["TEAM_ID","GAME_ID"])[ROLL_COLS]
        .to_dict(orient="index")
    )

    def get_feat(team_id, game_id, col):
        return lookup.get((team_id, game_id), {}).get(col, np.nan)

    print(f"Lookup: {len(lookup):,} entries × {len(ROLL_COLS)} features")
    return ROLL_COLS, get_feat


@app.cell
def _(ROLL_COLS, STATS, df, get_feat, mo, np):
    mo.md("## Steps 6–10 · Map → Schedule → H2H → Diffs → Cleanup")

    # 6. Map rolling features onto game-level rows
    # Note: dropna for core history check is chained at end so feat is assigned once
    _mapped = df.copy()
    for _col in ROLL_COLS:
        _mapped[f"HOME_{_col}"] = [
            get_feat(t, g, _col)
            for t, g in zip(_mapped["HOME_TEAM_ID"], _mapped["GAME_ID"])
        ]
        _mapped[f"AWAY_{_col}"] = [
            get_feat(t, g, _col)
            for t, g in zip(_mapped["AWAY_TEAM_ID"], _mapped["GAME_ID"])
        ]
    print(f"Mapped rolling features → shape {_mapped.shape}")

    # 7. Scheduling: back-to-back flag
    _mapped["HOME_B2B"] = (_mapped["HOME_REST_DAYS"] == 1).astype(int)
    _mapped["AWAY_B2B"] = (_mapped["AWAY_REST_DAYS"] == 1).astype(int)
    print(f"B2B flags added  |  B2B rate: {_mapped['HOME_B2B'].mean():.1%}")

    # 8. H2H cumulative win% (shift-based, verified corr ≈ 0.08)
    _h2h = (
        _mapped[["GAME_DATE","HOME_TEAM_ID","AWAY_TEAM_ID","TARGET"]]
        .sort_values("GAME_DATE").reset_index(drop=True)
    )
    _h2h["H2H_WINS"] = _h2h.groupby(
        ["HOME_TEAM_ID","AWAY_TEAM_ID"]
    )["TARGET"].transform(lambda x: x.shift(1).expanding().sum())
    _h2h["H2H_GAMES"] = _h2h.groupby(
        ["HOME_TEAM_ID","AWAY_TEAM_ID"]
    )["TARGET"].transform(lambda x: x.shift(1).expanding().count())
    _mapped["H2H_WIN_PCT"] = (
        _h2h["H2H_WINS"] / _h2h["H2H_GAMES"].replace(0, np.nan)
    ).fillna(0.5).values
    print(f"H2H_WIN_PCT corr with TARGET: {_mapped['H2H_WIN_PCT'].corr(_mapped['TARGET']):.4f}  (safe ≈ 0.08)")

    # 9. Differential features (Home − Away)
    _diff_bases = (
        [f"ROLL_{s}_L5"  for s in STATS] +
        [f"ROLL_{s}_L10" for s in STATS] +
        ["WIN_L5","WIN_L10","WIN_STREAK"]
    )
    for _base in _diff_bases:
        _hc, _ac = f"HOME_{_base}", f"AWAY_{_base}"
        if _hc in _mapped.columns and _ac in _mapped.columns:
            _mapped[f"DIFF_{_base}"] = _mapped[_hc] - _mapped[_ac]
    print(f"Added {len(_diff_bases)} DIFF_ columns")

    # 10a. Build raw feature list
    _raw_feature_cols = (
        [f"HOME_{c}" for c in ROLL_COLS] +
        [f"AWAY_{c}" for c in ROLL_COLS] +
        ["HOME_REST_DAYS","AWAY_REST_DAYS","HOME_B2B","AWAY_B2B","H2H_WIN_PCT"] +
        [f"DIFF_{b}" for b in _diff_bases if f"DIFF_{b}" in _mapped.columns]
    )
    _raw_feature_cols = [c for c in _raw_feature_cols if c in _mapped.columns]

    # 10b. Drop rows missing core rolling history, fill NaNs — assigned once as feat
    _core = ["HOME_WIN_L10","AWAY_WIN_L10","HOME_ROLL_PTS_L5","AWAY_ROLL_PTS_L5"]
    feat  = _mapped.dropna(subset=_core).reset_index(drop=True)
    print(f"Dropped rows with missing core history → {len(feat):,} games remain")

    # 10c. Fill remaining edge-case NaNs with column median
    _num = feat[_raw_feature_cols].select_dtypes(include="number").columns
    feat[_num] = feat[_num].fillna(feat[_num].median())

    # 10d. Remove zero-variance columns
    FEATURE_COLS = [c for c in _raw_feature_cols if feat[c].std() > 0]
    _dropped_zv  = [c for c in _raw_feature_cols if feat[c].std() == 0]
    if _dropped_zv:
        print(f"Dropped zero-variance: {_dropped_zv}")

    print(f"Final: {len(feat):,} games × {len(FEATURE_COLS)} features | NaNs: {feat[FEATURE_COLS].isnull().sum().sum()}")
    return FEATURE_COLS, feat


@app.cell
def _(FEATURE_COLS, feat, mo, plt):
    mo.md("## Step 11 · Leakage Audit")

    _corr = feat[FEATURE_COLS + ["TARGET"]].corr()["TARGET"].drop("TARGET").abs()
    _top  = _corr.sort_values(ascending=False).head(20)

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.barh(
        _top.index[::-1], _top.values[::-1],
        color=["#e06c75" if v > 0.40 else "#61afef" for v in _top.values[::-1]],
        edgecolor="white",
    )
    _ax.axvline(0.40, color="red", linestyle="--", linewidth=1.2,
                label="Leakage threshold (0.40)")
    _ax.set_xlabel("|Pearson r| with TARGET")
    _ax.set_title("Top 20 Feature Correlations with TARGET")
    _ax.legend()
    plt.tight_layout()

    _status = (
        "✅  All features within safe range (max corr < 0.40)"
        if _corr.max() <= 0.40
        else f"⚠️  SUSPICIOUS: {_corr.idxmax()} = {_corr.max():.4f}"
    )
    print(_status)
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(feat, mo, plt):
    mo.md("## Step 12 · Key Feature Distributions")

    _pairs = [
        ("HOME_WIN_L10",        "AWAY_WIN_L10",        "Rolling Win % L10"),
        ("HOME_ROLL_PTS_L5",    "AWAY_ROLL_PTS_L5",    "Rolling Pts L5"),
        ("HOME_ROLL_FG_PCT_L5", "AWAY_ROLL_FG_PCT_L5", "Rolling FG% L5"),
        ("HOME_ROLL_TOV_L5",    "AWAY_ROLL_TOV_L5",    "Rolling TOV L5"),
    ]
    _fig, _axs = plt.subplots(1, 4, figsize=(18, 4))
    for _ax, (_hc, _ac, _title) in zip(_axs, _pairs):
        _ax.hist(feat[_hc].dropna(), bins=35, alpha=0.65,
                 color="#61afef", label="Home", edgecolor="white")
        _ax.hist(feat[_ac].dropna(), bins=35, alpha=0.65,
                 color="#e06c75", label="Away", edgecolor="white")
        _ax.set_title(_title, fontsize=11)
        _ax.legend(fontsize=8)
    plt.suptitle("Home vs Away Rolling Feature Distributions", fontsize=13, y=1.02)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(feat, mo, plt, sns):
    mo.md("## Step 13 · DIFF Features Correlation Heatmap")

    _diff_cols = [c for c in feat.columns if c.startswith("DIFF_")] + ["TARGET"]
    _fig, _ax  = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        feat[_diff_cols].corr(),
        annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.3, linecolor="white",
        ax=_ax, annot_kws={"size": 7},
    )
    _ax.set_title("DIFF Features — Correlation Matrix")
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(
    FEATURE_COLS,
    LogisticRegression,
    StandardScaler,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    accuracy_score,
    feat,
    mo,
):
    mo.md("## Step 14 · Logistic Regression Baseline (Honest Ceiling)")

    _train = feat[feat["SEASON"].isin(TRAIN_SEASONS)]
    _val   = feat[feat["SEASON"].isin(VAL_SEASONS)]
    _test  = feat[feat["SEASON"].isin(TEST_SEASONS)]

    _sc  = StandardScaler()
    _Xtr = _sc.fit_transform(_train[FEATURE_COLS])
    _Xva = _sc.transform(_val[FEATURE_COLS])
    _Xte = _sc.transform(_test[FEATURE_COLS])

    _lr  = LogisticRegression(max_iter=1000, random_state=42)
    _lr.fit(_Xtr, _train["TARGET"])

    _tr_acc = accuracy_score(_train["TARGET"], _lr.predict(_Xtr))
    _va_acc = accuracy_score(_val["TARGET"],   _lr.predict(_Xva))
    _te_acc = accuracy_score(_test["TARGET"],  _lr.predict(_Xte))

    mo.md(f"""
    ### Results

    | Split | Accuracy |
    |---|---|
    | **Train** | {_tr_acc:.3f} |
    | **Validation** | {_va_acc:.3f} |
    | **Test** | {_te_acc:.3f} |

    ➜ Any deep learning model scoring above **{_te_acc + 0.05:.3f}** test accuracy
    is likely overfitting or has residual leakage.
    """)
    return


@app.cell
def _(FEATURE_COLS, TEST_SEASONS, TRAIN_SEASONS, VAL_SEASONS, feat, mo):
    _ID_COLS = [
        "GAME_ID","GAME_DATE","SEASON",
        "HOME_TEAM_ID","HOME_TEAM_ABBREVIATION",
        "AWAY_TEAM_ID","AWAY_TEAM_ABBREVIATION","TARGET",
    ]
    _out  = feat[_ID_COLS + FEATURE_COLS]
    _path = "nba_games_features_v2.csv"
    _out.to_csv(_path, index=False)

    _tr = len(feat[feat["SEASON"].isin(TRAIN_SEASONS)])
    _va = len(feat[feat["SEASON"].isin(VAL_SEASONS)])
    _te = len(feat[feat["SEASON"].isin(TEST_SEASONS)])

    mo.md(f"""
    ## ✅ Step 15 · Export Complete

    Saved **`{_path}`**

    | | |
    |---|---|
    | **Games** | {len(_out):,} |
    | **Features** | {len(FEATURE_COLS)} |
    | **Train** | {_tr:,} (2015-16 → 2020-21) |
    | **Validation** | {_va:,} (2021-22 → 2022-23) |
    | **Test** | {_te:,} (2023-24 → 2024-25) |
    | **Target** | {_out['TARGET'].mean():.1%} home wins |

    ### Feature groups
    | Group | Count |
    |---|---|
    | Rolling L5 — 8 stats × home + away | {len([c for c in FEATURE_COLS if 'L5' in c and 'DIFF' not in c and 'WIN_L' not in c])} |
    | Rolling L10 — 8 stats × home + away | {len([c for c in FEATURE_COLS if 'L10' in c and 'DIFF' not in c and 'WIN_L' not in c])} |
    | Win rate L5/L10 — home + away | {len([c for c in FEATURE_COLS if 'WIN_L' in c and 'DIFF' not in c and 'STREAK' not in c])} |
    | Streaks — win + loss × home + away | {len([c for c in FEATURE_COLS if 'STREAK' in c and 'DIFF' not in c])} |
    | Rest / schedule | {len([c for c in FEATURE_COLS if 'REST' in c or 'B2B' in c])} |
    | H2H | {len([c for c in FEATURE_COLS if 'H2H' in c])} |
    | Differentials (DIFF_) | {len([c for c in FEATURE_COLS if c.startswith('DIFF_')])} |
    | **Total** | **{len(FEATURE_COLS)}** |

    ### Explicitly excluded
    - `HOME_COURT` — constant 1, zero variance
    - `SEASON_WIN_PCT` — cumulative proxy for final standings
    - `PLUS_MINUS` rolling — confounds team quality with opponent strength
    - `REST_ADVANTAGE` — zero variance in this dataset

    """)
    return


if __name__ == "__main__":
    app.run()
