import pandas as pd

import svc

seed = 42


def main():
    df = pd.read_csv("data.csv", index_col=0)
    df1 = df[df["y"] == 1]
    df2 = df[df["y"] == 2]
    df3 = df[df["y"] == 3]
    df4 = df[df["y"] == 4]
    df5 = df[df["y"] == 5]

    df12 = pd.concat((df1, df2), axis=0)
    df13 = pd.concat((df1, df3), axis=0)
    df14 = pd.concat((df1, df4), axis=0)
    df15 = pd.concat((df1, df5), axis=0)

    labels12 = df12.pop("y")
    labels13 = df13.pop("y")
    labels14 = df14.pop("y")
    labels15 = df15.pop("y")

    svc.svm_classification(df12, labels12, 2, C=5, pca=True, seed=seed)
    svc.svm_classification(df13, labels13, 3, C=5, pca=True, seed=seed)
    svc.svm_classification(df14, labels14, 4, C=5, pca=True, seed=seed)
    svc.svm_classification(df15, labels15, 5, C=5, pca=True, seed=seed)


if __name__ == "__main__":
    main()
