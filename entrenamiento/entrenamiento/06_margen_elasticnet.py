from sklearn.linear_model import ElasticNet

from common import load_margin_minimal_dataset, print_metrics, train_regression


if __name__ == "__main__":
    metrics = train_regression(
        "06_margen_elasticnet",
        ElasticNet(alpha=0.01, l1_ratio=0.25, max_iter=20000, random_state=42),
        load_margin_minimal_dataset,
        scale_numeric=True,
    )
    print_metrics(metrics)
