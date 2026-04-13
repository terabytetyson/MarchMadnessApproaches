"""Train v9: Paris features + market odds, ExtraTreesClassifier. No BERT."""

import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from deps import load_data as v4_load_data, load_odds, apply_power_debias, fit_market_mapping

from src import config
from src.build_training_data import build_training_data, print_samples


def train(
    data_dir: Optional[str] = None,
    odds_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    exclude_seasons: Optional[list] = None,
    val_seasons: Optional[list] = None,
) -> object:
    """Train Paris + market model. ExtraTreesClassifier."""
    output_dir = output_dir or os.path.join(config.MODEL_DIR, "paris-v9")
    os.makedirs(output_dir, exist_ok=True)

    data_dir = data_dir or config.DATA_DIR
    odds_path = odds_path or config.ODDS_PATH

    data = v4_load_data(data_dir)
    odds_lookup = load_odds(odds_path, data_dir) if odds_path and os.path.exists(odds_path) else {}
    seed_map = data.get("seed_map", {})
    debiased_lookup, c, intercept = None, 1.0, 0.0
    if odds_lookup:
        tourney = data["m_tourney"]
        subset = tourney[(tourney["Season"] >= config.TRAIN_START) & (tourney["Season"] <= config.TRAIN_END)]
        if exclude_seasons:
            subset = subset[~subset["Season"].isin(exclude_seasons)]
        games_for_fit = []
        for _, r in subset.iterrows():
            w_id, l_id = r["WTeamID"], r["LTeamID"]
            t1, t2 = min(w_id, l_id), max(w_id, l_id)
            actual = 1.0 if w_id == t1 else 0.0
            games_for_fit.append((int(r["Season"]), t1, t2, actual))
        alpha, c, intercept = fit_market_mapping(odds_lookup, seed_map, games_for_fit)
        debiased_lookup = apply_power_debias(odds_lookup, alpha)
        with open(os.path.join(output_dir, "market_params.pkl"), "wb") as f:
            pickle.dump({"alpha": alpha, "c": c, "intercept": intercept}, f)
        print(f"Fitted market: α={alpha:.4f}, c={c:.4f}, intercept={intercept:.4f}")
    else:
        with open(os.path.join(output_dir, "market_params.pkl"), "wb") as f:
            pickle.dump({"alpha": 1.0, "c": 1.0, "intercept": 0.0}, f)
        print("No odds file; using seed prior for men, 0.5 for women")

    train_df = build_training_data(
        data_dir=data_dir,
        odds_path=odds_path,
        exclude_seasons=exclude_seasons,
        debiased_lookup=debiased_lookup,
        c=c,
        intercept=intercept,
        include_women=True,
    )
    print_samples(train_df, n=2)

    if val_seasons:
        val_df = build_training_data(
            data_dir=data_dir,
            odds_path=odds_path,
            include_only_seasons=val_seasons,
            debiased_lookup=debiased_lookup,
            c=c,
            intercept=intercept,
            include_women=True,
        )
        print(f"Train: {len(train_df)} (excl. {exclude_seasons}), Val: {len(val_df)} (seasons {val_seasons})")
    else:
        seasons = sorted(train_df["season"].unique())
        n_val = max(1, int(len(seasons) * config.VAL_RATIO))
        val_season_set = set(seasons[-n_val:])
        val_df = train_df[train_df["season"].isin(val_season_set)]
        train_df = train_df[~train_df["season"].isin(val_season_set)]
        print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    X_train = np.vstack(train_df["features"].values)
    y_train = train_df["label"].values
    X_val = np.vstack(val_df["features"].values)
    y_val = val_df["label"].values

    model = ExtraTreesClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_proba = model.predict_proba(X_val)[:, 1]
    val_proba = np.clip(val_proba, 0.05, 0.95)
    val_brier = np.mean((val_proba - y_val) ** 2)
    val_acc = np.mean((val_proba > 0.5) == y_val)
    print(f"Val Brier: {val_brier:.4f}, Val Accuracy: {val_acc:.4f}")

    with open(os.path.join(output_dir, "paris_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {output_dir}")

    return model
