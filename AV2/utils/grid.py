param_grid = {
    'decision_tree': {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [5, 10, 15, 20],
        'classifier__min_samples_split': [10, 15, 20]
    },
    'random_forest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 20, 30],
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__min_samples_split': [10, 15, 20]
    },
    'adaboost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__learning_rate': [0.01, 0.1],
    },
    'gradient_boost': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [5, 10, 20, 30],
        'classifier__learning_rate': [0.01, 0.1]
    },
    'svc': {
        'classifier__kernel': ['linear', 'sigmoid', 'rbf'],
        'classifier__C': [0.01, 1, 10],
    },
    'HistGradientBoosting': {
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_iter': [100, 1000],
        'classifier__max_depth': [5, 10, 25, 20]
    }
}
