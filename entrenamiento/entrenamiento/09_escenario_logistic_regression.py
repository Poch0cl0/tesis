from sklearn.linear_model import LogisticRegression

from common import load_scenario_minimal_dataset, print_metrics, train_classifier


if __name__ == "__main__":
    metrics = train_classifier(
        "09_escenario_logistic_regression",
        LogisticRegression(max_iter=20000, class_weight="balanced", solver="lbfgs"),
        load_scenario_minimal_dataset,
        scale_numeric=True,
    )
    print_metrics(metrics)
