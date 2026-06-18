from sklearn.ensemble import HistGradientBoostingClassifier

from common import load_scenario_minimal_dataset, print_metrics, train_classifier


if __name__ == "__main__":
    metrics = train_classifier(
        "08_escenario_hist_gradient_boosting",
        HistGradientBoostingClassifier(max_iter=350, learning_rate=0.05, l2_regularization=0.05, random_state=42),
        load_scenario_minimal_dataset,
    )
    print_metrics(metrics)
