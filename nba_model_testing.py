# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
#     "pandas>=3.0.3",
#     "numpy>=2.4.5",
#     "torch>=2.3.0",
#     "scikit-learn>=1.5.0",
#     "matplotlib>=3.9.0",
#     "seaborn>=0.13.0",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full", app_title="NBA Deep Learning")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, classification_report,
        roc_auc_score, confusion_matrix, roc_curve,
    )
    from sklearn.model_selection import train_test_split
    warnings.filterwarnings("ignore")
    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13, "axes.labelsize": 11})
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        DEVICE,
        DataLoader,
        StandardScaler,
        TensorDataset,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        nn,
        np,
        pd,
        plt,
        roc_auc_score,
        roc_curve,
        sns,
        torch,
    )


@app.cell
def _(DEVICE, mo):
    mo.md(f"""
    # 🏀 NBA Game Prediction — Deep Learning
    **Input:** `nba_games_features_v2.csv`  
    **Device:** `{DEVICE}`

    Three architectures are trained and compared:

    | Model | Architecture | Notes |
    |---|---|---|
    | **MLP** | 3-layer fully connected | Baseline deep model |
    | **MLP + Dropout** | Same + aggressive dropout | Regularised variant |
    | **Wide & Deep** | Parallel wide + deep paths | Captures linear + non-linear signals |

    All models use **time-based train/test split** (2015-22 train · 2022-25 test) to prevent leakage.
    ---
    """)
    return


@app.cell
def _(mo, pd):
    raw = pd.read_csv("nba_games_features_v2.csv", parse_dates=["GAME_DATE"]).sort_values(
        "GAME_DATE"
    ).reset_index(drop=True)
    mo.md(f"""
    ## Step 1 · Load Data
    **{len(raw):,} games** · {raw['SEASON'].nunique()} seasons · target balance {raw['TARGET'].mean():.1%} home wins
    """)
    return (raw,)


@app.cell
def _(StandardScaler, mo, np, raw):
    mo.md("## Step 2 · Feature Selection & Train / Test Split")

    ID_COLS    = ["GAME_ID", "GAME_DATE", "SEASON",
                  "HOME_TEAM_ID", "HOME_TEAM_ABBREVIATION",
                  "AWAY_TEAM_ID", "AWAY_TEAM_ABBREVIATION"]
    TARGET_COL = "TARGET"
    FEATURE_COLS = [c for c in raw.columns if c not in ID_COLS + [TARGET_COL]]

    # Time-based split: train on seasons before 2022-23, test on the rest
    _cutoff    = "2022-10-01"
    train_df   = raw[raw["GAME_DATE"] < _cutoff].copy()
    test_df    = raw[raw["GAME_DATE"] >= _cutoff].copy()

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(
        train_df[FEATURE_COLS].values.astype(np.float32)
    ).astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)
    X_test  = scaler.transform(
        test_df[FEATURE_COLS].values.astype(np.float32)
    ).astype(np.float32)
    y_test  = test_df[TARGET_COL].values.astype(np.float32)

    N_FEATURES = X_train.shape[1]

    print(f"Train: {len(X_train):,} games | Test: {len(X_test):,} games | Features: {N_FEATURES}")
    return FEATURE_COLS, N_FEATURES, X_test, X_train, scaler, y_test, y_train


@app.cell
def _(
    DEVICE,
    DataLoader,
    TensorDataset,
    X_test,
    X_train,
    mo,
    torch,
    y_test,
    y_train,
):
    mo.md("## Step 3 · Build PyTorch Datasets")

    _Xtr = torch.tensor(X_train).to(DEVICE)
    _ytr = torch.tensor(y_train).to(DEVICE)
    _Xte = torch.tensor(X_test).to(DEVICE)
    _yte = torch.tensor(y_test).to(DEVICE)

    train_loader = DataLoader(TensorDataset(_Xtr, _ytr), batch_size=256, shuffle=True)
    test_loader  = DataLoader(TensorDataset(_Xte, _yte), batch_size=256, shuffle=False)

    # Store tensors for direct eval
    X_train_t = _Xtr
    y_train_t = _ytr
    X_test_t  = _Xte
    y_test_t  = _yte

    print(f"Train loader: {len(train_loader)} batches | Test loader: {len(test_loader)} batches")
    return X_test_t, train_loader


@app.cell
def _(N_FEATURES, mo, nn, torch):
    mo.md("## Step 4 · Model Definitions")

    # ── MLP ───────────────────────────────────────────────────────────────────
    class MLP(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(),
                nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(),
                nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(1)

    # ── MLP with Dropout ──────────────────────────────────────────────────────
    class MLPDropout(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64),   nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(1)

    # ── Wide & Deep ───────────────────────────────────────────────────────────
    class WideAndDeep(nn.Module):
        def __init__(self, n_in):
            super().__init__()
            # Wide path: direct linear projection
            self.wide = nn.Linear(n_in, 32)
            # Deep path: 3-layer MLP
            self.deep = nn.Sequential(
                nn.Linear(n_in, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128),  nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 32),
            )
            self.out = nn.Linear(64, 1)

        def forward(self, x):
            _w = self.wide(x)
            _d = self.deep(x)
            return self.out(torch.cat([_w, _d], dim=1)).squeeze(1)

    mlp         = MLP(N_FEATURES)
    mlp_dropout = MLPDropout(N_FEATURES)
    wnd         = WideAndDeep(N_FEATURES)

    _total_params = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"MLP params         : {_total_params(mlp):,}")
    print(f"MLP+Dropout params : {_total_params(mlp_dropout):,}")
    print(f"Wide & Deep params : {_total_params(wnd):,}")
    return mlp, mlp_dropout, wnd


@app.cell
def _(DEVICE, mo, nn, torch):
    mo.md("## Step 5 · Training")

    def train_model(model, train_loader, n_epochs=60, lr=1e-3, weight_decay=1e-4):
        model = model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.BCEWithLogitsLoss()
        history = {"train_loss": [], "train_acc": []}

        for _epoch in range(n_epochs):
            model.train()
            _total_loss, _correct, _total = 0.0, 0, 0
            for _xb, _yb in train_loader:
                optimizer.zero_grad()
                _logits = model(_xb)
                _loss   = criterion(_logits, _yb)
                _loss.backward()
                optimizer.step()
                _total_loss += _loss.item() * len(_xb)
                _preds   = (_logits.sigmoid() > 0.5).float()
                _correct += (_preds == _yb).sum().item()
                _total   += len(_xb)
            scheduler.step()
            history["train_loss"].append(_total_loss / _total)
            history["train_acc"].append(_correct / _total)

        return model, history

    print("train_model() defined — ready to train.")
    return (train_model,)


@app.cell
def _(mlp, mlp_dropout, mo, train_loader, train_model, wnd):
    mo.md("### Training all three models (60 epochs each)…")

    mlp_trained,     mlp_history     = train_model(mlp,         train_loader, n_epochs=60)
    dropout_trained, dropout_history = train_model(mlp_dropout, train_loader, n_epochs=60)
    wnd_trained,     wnd_history     = train_model(wnd,         train_loader, n_epochs=60)

    print("Training complete.")
    return (
        dropout_history,
        dropout_trained,
        mlp_history,
        mlp_trained,
        wnd_history,
        wnd_trained,
    )


@app.cell
def _(DEVICE, accuracy_score, mo, roc_auc_score, torch):
    mo.md("## Step 6 · Evaluation")

    def evaluate(model, X_t, y_np):
        model.eval()
        with torch.no_grad():
            _logits = model(X_t.to(DEVICE))
            _probs  = _logits.sigmoid().cpu().numpy()
        _preds = (_probs > 0.5).astype(int)
        return {
            "accuracy":  accuracy_score(y_np, _preds),
            "roc_auc":   roc_auc_score(y_np, _probs),
            "probs":     _probs,
            "preds":     _preds,
        }

    print("evaluate() defined.")
    return (evaluate,)


@app.cell
def _(
    X_test_t,
    dropout_trained,
    evaluate,
    mlp_trained,
    mo,
    wnd_trained,
    y_test,
):
    mlp_res     = evaluate(mlp_trained,     X_test_t, y_test)
    dropout_res = evaluate(dropout_trained, X_test_t, y_test)
    wnd_res     = evaluate(wnd_trained,     X_test_t, y_test)

    mo.md(f"""
    ### Test-Set Results

    | Model | Accuracy | ROC-AUC |
    |---|---|---|
    | **MLP** | {mlp_res['accuracy']:.3f} | {mlp_res['roc_auc']:.3f} |
    | **MLP + Dropout** | {dropout_res['accuracy']:.3f} | {dropout_res['roc_auc']:.3f} |
    | **Wide & Deep** | {wnd_res['accuracy']:.3f} | {wnd_res['roc_auc']:.3f} |
    """)
    return dropout_res, mlp_res, wnd_res


@app.cell
def _(dropout_history, mlp_history, mo, plt, wnd_history):
    mo.md("## Step 7 · Training Curves")

    _fig, (_ax0, _ax1) = plt.subplots(1, 2, figsize=(14, 5))

    for _name, _hist, _color in [
        ("MLP",          mlp_history,     "#61afef"),
        ("MLP+Dropout",  dropout_history, "#e06c75"),
        ("Wide & Deep",  wnd_history,     "#98c379"),
    ]:
        _ax0.plot(_hist["train_loss"], label=_name, color=_color, linewidth=2)
        _ax1.plot(_hist["train_acc"],  label=_name, color=_color, linewidth=2)

    _ax0.set_title("Training Loss (BCE)")
    _ax0.set_xlabel("Epoch")
    _ax0.set_ylabel("Loss")
    _ax0.legend()

    _ax1.set_title("Training Accuracy")
    _ax1.set_xlabel("Epoch")
    _ax1.set_ylabel("Accuracy")
    _ax1.set_ylim(0.5, 0.9)
    _ax1.legend()

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(dropout_res, mlp_res, mo, plt, roc_curve, wnd_res, y_test):
    mo.md("## Step 8 · ROC Curves")

    _fig, _ax = plt.subplots(figsize=(8, 6))

    for _name, _res, _color in [
        ("MLP",          mlp_res,     "#61afef"),
        ("MLP+Dropout",  dropout_res, "#e06c75"),
        ("Wide & Deep",  wnd_res,     "#98c379"),
    ]:
        _fpr, _tpr, _ = roc_curve(y_test, _res["probs"])
        _auc = _res["roc_auc"]
        _ax.plot(_fpr, _tpr, label=f"{_name} (AUC={_auc:.3f})", color=_color, linewidth=2)

    _ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    _ax.set_title("ROC Curves — Test Set")
    _ax.set_xlabel("False Positive Rate")
    _ax.set_ylabel("True Positive Rate")
    _ax.legend()
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(confusion_matrix, dropout_res, mlp_res, mo, plt, sns, wnd_res, y_test):
    mo.md("## Step 9 · Confusion Matrices")

    _fig, _axs = plt.subplots(1, 3, figsize=(15, 4))

    for _ax, _name, _res in zip(
        _axs,
        ["MLP", "MLP + Dropout", "Wide & Deep"],
        [mlp_res, dropout_res, wnd_res],
    ):
        _cm = confusion_matrix(y_test, _res["preds"])
        sns.heatmap(
            _cm, annot=True, fmt="d", cmap="Blues", ax=_ax,
            xticklabels=["Away Win", "Home Win"],
            yticklabels=["Away Win", "Home Win"],
        )
        _ax.set_title(f"{_name}\nAcc={_res['accuracy']:.3f}")
        _ax.set_xlabel("Predicted")
        _ax.set_ylabel("Actual")

    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(classification_report, dropout_res, mlp_res, mo, wnd_res, y_test):
    _reports = {}
    for _name, _res in [("MLP", mlp_res), ("MLP+Dropout", dropout_res), ("Wide & Deep", wnd_res)]:
        _reports[_name] = classification_report(
            y_test, _res["preds"],
            target_names=["Away Win", "Home Win"],
        )

    mo.md(f"""
    ## Step 10 · Classification Reports

    **MLP**
    ```
    {_reports['MLP']}
    ```
    **MLP + Dropout**
    ```
    {_reports['MLP+Dropout']}
    ```
    **Wide & Deep**
    ```
    {_reports['Wide & Deep']}
    ```
    """)
    return


@app.cell
def _(dropout_res, mlp_res, mo, plt, wnd_res, y_test):
    mo.md("## Step 11 · Prediction Confidence Distribution")

    _fig, _axs = plt.subplots(1, 3, figsize=(15, 4))

    for _ax, _name, _res in zip(
        _axs,
        ["MLP", "MLP + Dropout", "Wide & Deep"],
        [mlp_res, dropout_res, wnd_res],
    ):
        _ax.hist(
            _res["probs"][y_test == 1], bins=40,
            alpha=0.6, color="#98c379", label="Home Win", edgecolor="white",
        )
        _ax.hist(
            _res["probs"][y_test == 0], bins=40,
            alpha=0.6, color="#e06c75", label="Away Win", edgecolor="white",
        )
        _ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        _ax.set_title(_name)
        _ax.set_xlabel("Predicted Probability (Home Win)")
        _ax.legend(fontsize=8)

    plt.suptitle("Predicted Probability Distributions by True Outcome", fontsize=13, y=1.02)
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(
    FEATURE_COLS,
    dropout_res,
    dropout_trained,
    mlp_res,
    mlp_trained,
    mo,
    scaler,
    torch,
    wnd_res,
    wnd_trained,
):
    # Pick the best model by ROC-AUC
    _candidates = [
        ("MLP",          mlp_trained,     mlp_res),
        ("MLP+Dropout",  dropout_trained, dropout_res),
        ("Wide & Deep",  wnd_trained,     wnd_res),
    ]
    _best_name, _best_model, _best_res = max(_candidates, key=lambda x: x[2]["roc_auc"])

    torch.save({
        "model_state": _best_model.state_dict(),
        "model_name":  _best_name,
        "scaler_mean": scaler.mean_,
        "scaler_std":  scaler.scale_,
        "feature_cols": FEATURE_COLS,
        "accuracy":    _best_res["accuracy"],
        "roc_auc":     _best_res["roc_auc"],
    }, "nba_best_model.pt")

    mo.md(f"""
    ## ✅ Step 12 · Best Model Saved

    **Winner: {_best_name}**  
    Accuracy: **{_best_res['accuracy']:.3f}** · ROC-AUC: **{_best_res['roc_auc']:.3f}**

    Saved to `nba_best_model.pt` (includes model weights, scaler, and feature list).

    ### Summary

    | Model | Accuracy | ROC-AUC |
    |---|---|---|
    | MLP | {mlp_res['accuracy']:.3f} | {mlp_res['roc_auc']:.3f} |
    | MLP + Dropout | {dropout_res['accuracy']:.3f} | {dropout_res['roc_auc']:.3f} |
    | Wide & Deep | {wnd_res['accuracy']:.3f} | {wnd_res['roc_auc']:.3f} |

    ### Next steps
    - Tune hyperparameters (learning rate, hidden sizes, dropout rate)
    - Add LSTM/GRU to model sequential game history
    - Ensemble the three models for improved robustness
    - Compare against the Logistic Regression baseline from the proposal
    """)
    return


if __name__ == "__main__":
    app.run()
