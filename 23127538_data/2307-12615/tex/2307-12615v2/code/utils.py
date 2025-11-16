import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import torch
import random

FENETREMAX = 100000


def generatorXYbatch(X, Y, batchSize, indices):
    """
    get batches of size batchSize, except the last one if the division is not zero.

    args :
            'X' torch.tensor
            'Y' torch.tensor
            'batchSize' int.  the batch size you want.

    returns : a generator. Will generate n/batchSize samples of size batchSize
    """
    # np.random.seed(2)
    n = Y.shape[0]
    # get the number of batches and the size of the last one.
    nb_full_batch, last_batchSize = n // batchSize, n % batchSize

    for i in range(nb_full_batch):
        yield (
            X[indices[i * batchSize : (i + 1) * batchSize]],
            Y[indices[i * batchSize : (i + 1) * batchSize]],
            i,
        )
    if last_batchSize != 0:
        yield (
            X[indices[-last_batchSize:]],
            Y[indices[-last_batchSize:]],
            i,
        )


def buildXYBatchesList(X, Y, batchSize):
    np.random.seed(0)
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    batchesXList = list()
    batchesYList = list()
    nbFullBatch, lastBatchSize = n // batchSize, n % batchSize

    for i in range(nbFullBatch):
        batchesXList.append(X[indices[i * batchSize : (i + 1) * batchSize]])
        batchesYList.append(Y[indices[i * batchSize : (i + 1) * batchSize]])
    if lastBatchSize > 0:
        batchesXList.append(X[indices[-lastBatchSize:]])
        batchesYList.append(Y[indices[-lastBatchSize:]])
    return batchesXList, batchesYList


def pseudoInverse1d(x):
    return torch.nan_to_num(1 / x, posinf=0)


class UnknownProblemException(Exception):
    def __init__(self, pbChosen):
        self.message = pbChosen + " is unknown. Please select a known problem."
        super().__init__(self.message)


def getSomeSamples(percentageToTake, X, Y):
    lengthOfX = X.shape[0]
    numberOfSamplesToTake = int(lengthOfX * percentageToTake)
    SamplesToTake = np.random.choice(
        np.arange(lengthOfX), numberOfSamplesToTake, replace=False
    )
    someSamplesOfX = X[SamplesToTake]
    someSamplesOfY = Y[SamplesToTake]
    return someSamplesOfX, someSamplesOfY


def plotOptim(
    optim_class_and_plotparam,
    axes,
    fEtoileTrain,
    legend_names=False,
    set_additional_legend=False,
):
    toPlotTrain = np.array(optim_class_and_plotparam.train_loss) / fEtoileTrain - 1
    toPlotTrain = np.minimum(toPlotTrain, np.full(toPlotTrain.shape[0], 2 * FENETREMAX))
    toPlotTrain = np.log(toPlotTrain)
    print("to plot train", toPlotTrain)
    toPlotTest = np.array(optim_class_and_plotparam.test_metric)
    if legend_names is True and set_additional_legend is True:
        label = optim_class_and_plotparam.label
    else:
        label = None
    nb_to_keep = 20
    # nb_to_keep = 500
    axes[0].plot(
        optim_class_and_plotparam.abscisse,
        toPlotTrain,
        linestyle=optim_class_and_plotparam.linestyle,
        color=optim_class_and_plotparam.color,
        label=label,
        # marker=optim_class_and_plotparam.marker,
        # markevery=0.14,
    )
    axes[0].set_xlim(0, nb_to_keep)
    axes[1].plot(
        optim_class_and_plotparam.abscisse,
        toPlotTest,
        linestyle=optim_class_and_plotparam.linestyle,
        color=optim_class_and_plotparam.color,
        # marker=optim_class_and_plotparam.marker,
        # markevery=0.14,
    )
    axes[1].set_xlim(0, nb_to_keep)


def plot_complete_list(
    complete_list,
    l,
    axes,
    supplement_title="",
    save=True,
    legend_names=False,
    set_additional_legend=False,
    is_last_graph=False,
):
    nbRows = 2
    for optim_class_and_plotparam in complete_list.list_optim_and_plotparam:
        plotOptim(
            optim_class_and_plotparam,
            axes,
            complete_list.f_etoile_train,
            legend_names,
            set_additional_legend,
        )
    # axes[0].set_yscale("log")
    if set_additional_legend is True:
        axes[1].set_xlabel(r"Number of gradients computed$/n$")
        if is_last_graph is True:
            # axes[0].legend(bbox_to_anchor=(-0.92, 4.15), ncol = 3, prop={'size':20}) ## adapt vs each
            # axes[0].legend(bbox_to_anchor=(-0.55, 4.15), ncol = 4, prop={'size':20}) ## accel
            axes[0].legend(
                bbox_to_anchor=(-1.45, 4.15), ncol=2, prop={"size": 20}
            )  ## adapt vs adapt
            # axes[0].legend(bbox_to_anchor=(-1.98, 4), ncol = 2) ## adapt vs accel
    else:
        axes[0].set_title(rf"$L ={l}$")
    # axes[0].set_ylim(0,10)
    # axes[1].set_ylim(-1,10)
    if save is True:
        complete_list.save()
