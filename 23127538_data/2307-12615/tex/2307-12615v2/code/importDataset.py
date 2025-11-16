from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelBinarizer,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import scanpy
from functorch import vmap

torch.manual_seed(0)
np.random.seed(0)


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == "binary":
            return np.hstack((Y, 1 - Y))
        else:
            return Y


def getCovTypeXY(scale=True, n=200, dim=54):
    if dim > 54:
        dim = 54
    X, y = fetch_openml("covertype", return_X_y=True)
    X = X.values
    y = y.values
    toTake = np.random.choice(np.arange(X.shape[0]), n)
    X = X[toTake, :dim]
    y = y[toTake]
    onehot = OneHotEncoder()
    y = onehot.fit_transform(y.reshape(-1, 1)).toarray()
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    scaler = MinMaxScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return (
        torch.from_numpy(Xtrain),
        torch.from_numpy(Xtest),
        torch.from_numpy(ytrain),
        torch.from_numpy(ytest),
    )


def getBinaryXY(n=200, dim=4):
    X = torch.randn(n, dim)
    trueBeta = 10 * torch.randn(dim) + torch.bernoulli(torch.ones(dim) / 2)
    prob = 1 / (1 + torch.exp(-X @ trueBeta))
    y = torch.bernoulli(prob)
    encoder = LabelBinarizer()
    y = torch.from_numpy(encoder.fit_transform(y))
    return X, y


def getGaussianXY(n=200, dim=4):
    # toAdd = torch.bernoulli(torch.ones(dim) / 2) * 6 - 3
    X = torch.randn(n, dim)
    trueBeta = torch.randn(dim) * 10  # + torch.bernoulli(torch.ones(dim)/2)*4
    scaler = MinMaxScaler()
    X = torch.from_numpy(scaler.fit_transform(X))
    y = X @ trueBeta + torch.randn(n)
    return X, y


def getMnistXY(n=200, dim=50, oneHot=True, scale=True):
    data = iter(torchvision.datasets.MNIST(root="./MNIST/", train=True, download=True))
    n = min(n, 60000)
    X = torch.empty(n, np.minimum(28 * 28, dim))
    Y = torch.empty(n)
    i = 0
    while i < n:
        (x, y) = next(data)
        XFirst = transforms.ToTensor()(x).reshape(-1, 28 * 28)
        X[i] = XFirst[:, : min(28 * 28, dim)]
        Y[i] = y
        i += 1
    if oneHot:
        encoder = LabelBinarizer()
        Y = torch.from_numpy(encoder.fit_transform(Y))
    if scale:
        scaler = MinMaxScaler()
        X = torch.from_numpy(scaler.fit_transform(X))
    return X, Y


def getAustralian(scale=True, n=689, dim=14):
    data = pd.read_csv("data/australian.dat", sep=" ").to_numpy()
    data = data[
        :n,
    ]
    X, y = data[:, :-1], data[:, -1]
    if dim < 15:
        X = X[:, :dim]
    if scale:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    encoder = MyLabelBinarizer()
    y = encoder.fit_transform(y)
    return (
        torch.from_numpy(X.astype("float64")),
        torch.from_numpy(y),
    )


def getSCRNA(n=200, dim=50, max_class=28):
    data = scanpy.read_h5ad("data/2k_cell_per_study_10studies.h5ad")
    X = data.X.toarray()
    y = data.obs["standard_true_celltype_v5"]
    le = LabelEncoder()
    y = le.fit_transform(y)
    filter = y < max_class
    y = y[filter]
    X = X[filter]
    not_only_zeros = np.sum(X, axis=0) > 0
    X = X[:, not_only_zeros]
    X = X[
        :n,
    ]
    y = y[
        :n,
    ]
    if y.shape[0] < n:
        print(f"NOT ENOUGH CLASSES, n= {n} but will get only {y.shape[0]} samples")
    var = np.var(X, axis=0)
    most_variables = np.argsort(var)[-dim:]
    X = X[:, most_variables]
    onehot = OneHotEncoder()
    y = onehot.fit_transform(y.reshape(-1, 1)).toarray()
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )
    scaler = MinMaxScaler()
    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)
    return (
        torch.from_numpy(Xtrain.astype("float64")),
        torch.from_numpy(Xtest.astype("float64")),
        torch.from_numpy(ytrain),
        torch.from_numpy(ytest),
    )


def getDataset(datasetName, pbChosen, n, dim, scale=True, max_class=28):
    if datasetName == "covType":
        Xtrain, Xtest, Ytrain, Ytest = getCovTypeXY(n=n, dim=dim, scale=True)
    elif datasetName == "scRNA":
        Xtrain, Xtest, Ytrain, Ytest = getSCRNA(n, dim, max_class)
    # elif datasetName == "australian":
    #     X, Y = getAustralian(n=n, dim=dim, scale=scale)
    # elif datasetName == "MNIST":
    #     X, Y = getMnistXY(n, dim=dim, oneHot=True, scale=scale)
    return Xtrain, Xtest, Ytrain, Ytest


def getTheoreticL(X, pbChosen):
    if pbChosen == "linearRegression":
        return getLinearL(X)
    elif pbChosen == "logisticRegression" or pbChosen == "MultiLogisticRegression":
        return getLogisticL(X)
    else:
        print("no theoretic L for this pb, return 1")
        return 1
