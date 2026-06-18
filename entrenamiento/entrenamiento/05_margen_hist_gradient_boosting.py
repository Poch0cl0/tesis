from sklearn.ensemble import HistGradientBoostingRegressor

from common import load_margin_minimal_dataset, print_metrics, train_regression


if __name__ == "__main__":
    metrics = train_regression(
        "05_margen_hist_gradient_boosting",
        HistGradientBoostingRegressor(max_iter=350, learning_rate=0.05, l2_regularization=0.05, random_state=42),
        load_margin_minimal_dataset,
    )
    print_metrics(metrics)
