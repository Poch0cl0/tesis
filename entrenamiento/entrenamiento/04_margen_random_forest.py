from sklearn.ensemble import RandomForestRegressor

from common import load_margin_minimal_dataset, print_metrics, train_regression


if __name__ == "__main__":
    metrics = train_regression(
        "04_margen_random_forest",
        RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42, n_jobs=-1),
        load_margin_minimal_dataset,
    )
    print_metrics(metrics)
