"""
02_modelling.py
────────────────────────────────────────────────────────────────────────────
Data Science layer: feature engineering → model comparison → evaluation.

Predicts log(gross box office) from budget, genre, rating, year, and
audience engagement signals.  Compares four model families and reports
R², cross-validated R², and MAE in original dollars.

Inputs  : data/movies_clean.csv  (produced by 01_data_cleaning.py)
Outputs : data/model_results.csv
          data/feature_importance.csv

Run: python 02_modelling.py
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

SEED = 42
sns.set_theme(style="whitegrid", palette="muted")


# ── 1. Load & feature engineering ─────────────────────────────────────────────

print("[MODEL] Loading cleaned data …")
df = pd.read_csv("data/movies_clean.csv")
print(f"  {len(df):,} rows loaded")

# Keep top-10 genres; bucket rest as 'Other'
top_genres = df["primary_genre"].value_counts().head(10).index
df["genre_bucket"] = df["primary_genre"].where(df["primary_genre"].isin(top_genres), "Other")

# Encode categoricals
genre_dummies   = pd.get_dummies(df["genre_bucket"],   prefix="genre")
content_dummies = pd.get_dummies(df["content_rating"].fillna("Unknown"), prefix="rating")

feature_cols = ["log_budget", "imdb_score", "duration", "title_year",
                "num_voted_users", "num_critic_for_reviews"]

X = pd.concat([
    df[feature_cols].fillna(0),
    genre_dummies,
    content_dummies,
], axis=1)

y = df["log_gross"]

print(f"  Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features")


# ── 2. Train / test split ──────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


# ── 3. Model zoo ──────────────────────────────────────────────────────────────

models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LinearRegression()),
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  Ridge(alpha=10)),
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, random_state=SEED
    ),
}


# ── 4. Evaluate ────────────────────────────────────────────────────────────────

print("\n── Model Comparison ─────────────────────────────────────────────────")
results = []
fitted  = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds   = model.predict(X_test)
    r2      = r2_score(y_test, preds)
    mae     = mean_absolute_error(np.expm1(y_test), np.expm1(preds))
    cv_r2   = cross_val_score(model, X, y, cv=kf, scoring="r2").mean()

    results.append({
        "Model"  : name,
        "R²"     : round(r2, 3),
        "CV_R²"  : round(cv_r2, 3),
        "MAE_$M" : round(mae / 1e6, 1),
    })
    fitted[name] = (model, preds)
    print(f"  {name:25s}  R²={r2:.3f}  CV-R²={cv_r2:.3f}  MAE=${mae/1e6:.1f}M")

results_df = pd.DataFrame(results).sort_values("CV_R²", ascending=False)
results_df.to_csv("data/model_results.csv", index=False)


# ── 5. Feature importance (best tree model) ───────────────────────────────────

best_name = "Gradient Boosting"
best_model = fitted[best_name][0]

importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

top_imp = importances.head(15)
top_imp.to_csv("data/feature_importance.csv", header=["importance"])

print(f"\n── Top 15 Feature Importances ({best_name}) ──────────────────────────")
for feat, imp in top_imp.items():
    bar = "█" * int(imp * 100)
    print(f"  {feat:35s} {imp:.3f}  {bar}")


# ── 6. Residual analysis ───────────────────────────────────────────────────────

preds_best = fitted[best_name][1]
residuals  = y_test - preds_best
actual_usd = np.expm1(y_test)
pred_usd   = np.expm1(preds_best)

print(f"\n── Residual Analysis ({best_name}) ───────────────────────────────────")
print(f"  Mean residual (log scale):  {residuals.mean():.4f}")
print(f"  Std  residual (log scale):  {residuals.std():.4f}")
print(f"  % predictions within $20M: {((abs(actual_usd - pred_usd)) < 20e6).mean()*100:.1f}%")
print(f"  % predictions within $50M: {((abs(actual_usd - pred_usd)) < 50e6).mean()*100:.1f}%")


# ── 7. Plots ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Gradient Boosting — Model Diagnostics", fontsize=14, fontweight="bold")

# 7a. Actual vs predicted
ax = axes[0]
ax.scatter(actual_usd / 1e6, pred_usd / 1e6, alpha=0.3, s=12, color="#4C72B0")
lim = max(actual_usd.max(), pred_usd.max()) / 1e6
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Gross ($M)")
ax.set_ylabel("Predicted Gross ($M)")
ax.set_title("Actual vs Predicted")
ax.legend(fontsize=8)

# 7b. Residuals distribution
ax = axes[1]
ax.hist(residuals, bins=40, color="#55A868", edgecolor="white", linewidth=0.5)
ax.axvline(0, color="red", linestyle="--", lw=1.5)
ax.set_xlabel("Residual (log scale)")
ax.set_ylabel("Count")
ax.set_title("Residual Distribution")

# 7c. Feature importances
ax = axes[2]
top_imp.head(10).sort_values().plot(kind="barh", ax=ax, color="#C44E52")
ax.set_xlabel("Importance")
ax.set_title("Top 10 Feature Importances")

plt.tight_layout()
plt.savefig("data/model_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n[MODEL] Saved → data/model_diagnostics.png")


# ── 8. Genre-level ROI analysis ───────────────────────────────────────────────

print("\n── Genre ROI Summary ─────────────────────────────────────────────────")
genre_roi = (
    df.groupby("primary_genre")
      .agg(
          n_films=("roi", "count"),
          median_roi=("roi", "median"),
          pct_profitable=("profitable", "mean"),
          median_gross_M=("gross", lambda x: x.median() / 1e6),
      )
      .query("n_films >= 20")
      .sort_values("median_roi", ascending=False)
      .round(2)
)
genre_roi["pct_profitable"] = (genre_roi["pct_profitable"] * 100).round(1)
print(genre_roi.to_string())
genre_roi.to_csv("data/genre_roi_summary.csv")

print("\n[MODEL] Done ✓")
print("  → data/model_results.csv")
print("  → data/feature_importance.csv")
print("  → data/genre_roi_summary.csv")
print("  → data/model_diagnostics.png")
