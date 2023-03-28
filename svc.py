import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.base import clone
from scores import print_metrics


def cross_validation(
    estimator, X: pd.DataFrame, y: pd.DataFrame, neg: int, k=5, seed=42
) -> list:
    np.random.seed(seed)
    idxs = X.index.values
    np.random.shuffle(idxs)
    models = [clone(estimator) for _ in range(k)]
    for i in range(k):
        start = i * len(X) // k
        end = (i + 1) * len(X) // k
        test_fold = idxs[start:end]
        train_fold = np.setdiff1d(idxs, test_fold)
        # X_test = X.loc[test_fold]
        # y_test = y.loc[test_fold]
        X_train = X.loc[train_fold]
        y_train = y.loc[train_fold]
        models[i].fit(X_train, y_train)
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
    components.index = X.columns
    return X @ components


def svm_classification(
    df: pd.DataFrame, labels: pd.Series, neg_class: int, **kwargs
) -> None:
    seed = kwargs.get("seed", 42)
    k = kwargs.get("k", 1)

    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.25, random_state=seed
    )

    if kwargs.get("pca", False):
        components = pca(X_train)
        X_train = reduce(X_train, components)
        X_test = reduce(X_test, components)

    if kwargs.get("scale", False):
        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
        X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

    # Training k-folds CV
    svc = SVC(kernel="linear", random_state=seed, C=5, max_iter=1000)
    svc_models = cross_validation(svc, X_train, y_train, neg_class, k=k, seed=seed)
    predictions = np.array([model.predict(X_test) for model in svc_models])
    y_pred = np.count_nonzero(predictions, axis=0) > (k // 2)
    print_metrics(y_test, y_pred, f"SVC 1v{neg_class}")
