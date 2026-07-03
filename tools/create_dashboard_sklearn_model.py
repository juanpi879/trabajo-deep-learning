from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "artifacts" / "data"
MODEL_DIR = ROOT / "artifacts" / "models"

df = pd.read_csv(DATA_DIR / "feature_dataset.csv", parse_dates=["date"])
feature_cols = pd.read_csv(DATA_DIR / "feature_columns.csv")["feature"].tolist()

train_df = df[(df["is_world_cup_2022_test"] == 0) & (df["date"] < "2022-11-20")].copy()
X = train_df[feature_cols].astype(float)
y_result = train_df["target_result"].astype(int)
y_home_goals = train_df["target_home_goals"].astype(float).clip(0, 8)
y_away_goals = train_df["target_away_goals"].astype(float).clip(0, 8)

classifier = make_pipeline(
    StandardScaler(),
    HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, random_state=42),
)
home_goals = make_pipeline(
    StandardScaler(),
    HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, random_state=42),
)
away_goals = make_pipeline(
    StandardScaler(),
    HistGradientBoostingRegressor(max_iter=80, learning_rate=0.05, random_state=43),
)

classifier.fit(X, y_result)
home_goals.fit(X, y_home_goals)
away_goals.fit(X, y_away_goals)

bundle = {
    "classifier": classifier,
    "home_goals": home_goals,
    "away_goals": away_goals,
    "feature_cols": feature_cols,
    "kind": "sklearn_dashboard_no_tensorflow",
}
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, MODEL_DIR / "dashboard_sklearn.pkl")
print("saved", MODEL_DIR / "dashboard_sklearn.pkl")
