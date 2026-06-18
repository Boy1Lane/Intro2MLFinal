LABELS = [0, 1, 2]
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]

# internal sklearn keys, ordered for showdown
SKLEARN_ORDER = [
    "LogisticRegression", "LinearSVC", "MultinomialNB",
    "RandomForest", "SGDClassifier", "VotingEnsemble",
]
# full showdown order incl. PhoBERT (best first, matching report table)
MODEL_ORDER = ["PhoBERT"] + SKLEARN_ORDER

# human-facing names matching the report metric table
DISPLAY_NAMES = {
    "PhoBERT": "PhoBERT-base-v2",
    "LogisticRegression": "Logistic Regression",
    "LinearSVC": "Linear SVC",
    "MultinomialNB": "Multinomial NB",
    "RandomForest": "Random Forest",
    "SGDClassifier": "SGD Classifier",
    "VotingEnsemble": "Voting Ensemble",
}
