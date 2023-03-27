import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from scores import print_metrics


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
    return (transformed - np.mean(transformed, axis=0)) / np.std(transformed, axis=0)


def svm_classification(
    df: pd.DataFrame, labels: pd.Series, neg_class: int, **kwargs
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
    svc = SVC(kernel="linear", random_state=seed, C=5)
    svc.fit(X_train, y_train)
    dump(svc, f".\\models\\svc_1v{neg_class}.joblib")
    print_metrics(y_test, svc.predict(X_test), f"SVC 1v{neg_class}")
    print()
