from sklearn.ensemble import RandomForestRegressor

from common import load_fob_minimal_dataset, print_metrics, train_regression


if __name__ == "__main__":
    metrics = train_regression(
        "01_fob_random_forest",
        RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1),
        load_fob_minimal_dataset,
        time_split=True,
    )
    print_metrics(metrics)
