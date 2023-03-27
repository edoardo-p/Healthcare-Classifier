import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.base import clone
from scores import print_metrics, cross_validate


def cross_validation(
    estimator, X: pd.DataFrame, y: pd.DataFrame, neg: int, k=5, seed=42
):
    np.random.seed(seed)
    idxs = X.index.values
    np.random.shuffle(idxs)
    models = [clone(estimator) for _ in range(k)]
    for i in range(k):
        print(f"Starting fold {i} for 1v{neg}")
        start = i * len(X) // k
        end = (i + 1) * len(X) // k
        fold = idxs[start:end]
        # X_test = X.iloc[fold]
        # y_test = y.iloc[fold]
        X_train = X.iloc[np.setdiff1d(idxs, fold)]
        y_train = y.iloc[np.setdiff1d(idxs, fold)]
        print("Fitting model...")
        models[i].fit(X_train, y_train)
        print("Model fitted")
        dump(models[i], f".\\models\\svc_1v{neg}_fold{i}.joblib")

    return models


def pca(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    l, v = np.linalg.eig(X.corr())
    eigen = pd.DataFrame(v.real.T)
    eigen["l"] = l.real
    eigen.sort_values("l", ascending=False, inplace=True)
    cumulative = np.cumsum(eigen["l"] / sum(eigen["l"]))
    n_comp = sum(cumulative <= threshold) + 1
    return eigen.head(n_comp).drop("l", axis=1).T


def reduce(X: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    transformed = np.dot(X, components)
    return pd.DataFrame(
        transformed - np.mean(transformed, axis=0) / np.std(transformed, axis=0)
    )


def svm_classification(
    df: pd.DataFrame, labels: pd.Series, neg_class: int, k=5, **kwargs
) -> None:
    # Model params
    seed = kwargs.get("seed", 42)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.25, random_state=seed
    )

    # PCA on train
    if kwargs.get("pca", False):
        components = pca(X_train)
        X_train = reduce(X_train, components)
        X_test = reduce(X_test, components)

    # Train the models
    svc = SVC(kernel="linear", random_state=seed, C=5, max_iter=50)
    svc_models = cross_validation(svc, X_train, y_train, neg_class, k=k, seed=seed)
    predictions = np.array([model.predict(X_test) for model in svc_models])
    final = np.count_nonzero(predictions, axis=0) > (k // 2)
    print(f"SVC 1v{neg_class}")
    accuracy = sum(final == y_test) / len(y_test)
    print(accuracy)
    print()
