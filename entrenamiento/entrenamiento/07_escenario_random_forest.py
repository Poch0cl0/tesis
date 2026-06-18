from sklearn.ensemble import RandomForestClassifier

from common import load_scenario_minimal_dataset, print_metrics, train_classifier


if __name__ == "__main__":
    metrics = train_classifier(
        "07_escenario_random_forest",
        RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        load_scenario_minimal_dataset,
    )
    print_metrics(metrics)
