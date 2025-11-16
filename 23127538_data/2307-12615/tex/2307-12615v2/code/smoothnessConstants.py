import torch
import numpy as np
from functorch import vmap


def getLinearL(X, LFORMULA):
    def getL(x):
        if len(x.shape) > 1:
            xtx = 1 / x.shape[0] * x.T @ x
        else:
            xtx = x.unsqueeze(1) @ (x.unsqueeze(0))
        eigvals = torch.linalg.eigvalsh(xtx)
        return torch.max(eigvals)

    return LfromFormula(getL, X, LFORMULA)


def getLogisticL(X, nbClasses, LFORMULA):
    IdMinusOuter = torch.eye(nbClasses) - 1 / nbClasses * torch.full(
        (nbClasses, nbClasses), 1, dtype=torch.double
    )
    IdMinusOuterEigvals = torch.linalg.eigvalsh(IdMinusOuter)

    def getL(x):
        if len(x.shape) > 1:
            xtx = 1 / x.shape[0] * x.T @ x
        else:
            xtx = x.unsqueeze(1) @ (x.unsqueeze(0))
        xtxEigvals = torch.linalg.eigvalsh(xtx)
        return torch.max(IdMinusOuterEigvals) * torch.max(xtxEigvals)

    return LfromFormula(getL, X, LFORMULA)


def LfromFormula(getL, X, LFORMULA):
    if LFORMULA == "max":
        return torch.max(vmap(getL)(X))
    elif LFORMULA == "mean":
        return torch.mean(vmap(getL)(X))
    elif LFORMULA == "full":
        return getL(X)
    else:
        print("Not a good formula, returning None")
        return None
