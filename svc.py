import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from scores import cross_validate


def get_principal_components(X: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    l, v = np.linalg.eig(X.corr())
    eigen = pd.DataFrame(v.real.T)
    eigen["l"] = l.real
    eigen.sort_values("l", ascending=False, inplace=True)
    cumulative = np.cumsum(eigen["l"] / sum(eigen["l"]))
    n_comp = sum(cumulative <= threshold) + 1
    return eigen.head(n_comp)


def pca(X: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    transformed = np.dot(X, components.drop("l", axis=1).T)
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
        components = get_principal_components(X_train)
        X_train = pca(X_train, components)
        X_test = pca(X_test, components)

    # Train the models
    for c in (1, 5, 10):
        svc = SVC(kernel="linear", random_state=seed, C=c)
        # svc.fit(X_train, y_train)

        # Cross Validation
        print(f"Linear standardized SVC (C={c}) 1v{neg_class}")
        cross_validate(svc, X_train, y_train, neg_class)
        # print(svc.score(X_test, y_test))
        print()
