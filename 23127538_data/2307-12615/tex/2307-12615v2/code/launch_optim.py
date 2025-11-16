import numpy as np
import torch
from importDataset import getDataset
from sklearn.model_selection import train_test_split
import myoptim
from tqdm import tqdm
from utils import plot_complete_list
import matplotlib.pyplot as plt

optimFEtoile = myoptim.LBFGS


def launch_optim(Xtrain, Xtest, Ytrain, Ytest, complete_list, l):
    pbChosen = complete_list.param_args.pb
    reg = complete_list.param_args.reg
    nbEpoch = complete_list.param_args.nbEpoch
    batchSize = complete_list.param_args.batchSize
    for optim_class_and_plot_param in complete_list.list_optim_and_plotparam:
        optimClass = optim_class_and_plot_param.optim_class
        if complete_list.check_if_optim_is_launched(optimClass) == False:
            print("Training :", optimClass.NAME)
            optim = optimClass(Xtrain, Xtest, Ytrain, Ytest, pbChosen, reg)
            optim.train(nbEpoch * optim.n, batchSize, l)
            optim.LINESTYLE = "dotted"
            optim_class_and_plot_param.train_loss = optim.trainLossList
            optim_class_and_plot_param.test_metric = optim.test_metric_list
            optim_class_and_plot_param.abscisse = optim.cumsumComputedGradList / optim.n
            optim_class_and_plot_param.real_dim = optim.dim
            complete_list.param_args.real_dim = optim.dim
        else:
            print(f"getting back data for optim {optimClass.NAME} L {l}")
            (
                trainLoss,
                test_metric,
                abscisse,
                lr,
                real_dim,
            ) = complete_list.get_back_optim(optimClass)
            optim_class_and_plot_param.train_loss = trainLoss
            optim_class_and_plot_param.test_metric = test_metric
            optim_class_and_plot_param.abscisse = abscisse
            optim_class_and_plot_param.lr = lr
            optim_class_and_plot_param.real_dim = real_dim
            complete_list.param_args.real_dim = real_dim
    return complete_list


def launchGridLListOptim(
    complete_list,
    lList,
    number_of_grad_for_f_etoile,
    axes,
    name_fig=None,
    set_additional_legend=True,
):
    Xtrain, Xtest, Ytrain, Ytest = complete_list.get_dataset()
    fEtoileTrain = complete_list.get_f_etoile(
        Xtrain, Xtest, Ytrain, Ytest, number_of_grad_for_f_etoile
    )
    complete_list.f_etoile_train = fEtoileTrain
    print("fEtoile train ", fEtoileTrain)
    listOptim = []
    minimum_train = 100
    maximum_train = 0
    minimum_test = 100
    maximum_test = 0
    for i in tqdm(range(len(lList))):
        l = lList[i]
        complete_list.L = l
        complete_list = launch_optim(Xtrain, Xtest, Ytrain, Ytest, complete_list, l)
        l_axes = axes[:, i]
        if i == len(lList) - 1:
            legend_names = True
        else:
            legend_names = False
        if i == len(lList)-1:
            is_last_graph = True
        else:
            is_last_graph = False
        plot_complete_list(
            complete_list,
            l,
            l_axes,
            save=True,
            legend_names=legend_names,
            set_additional_legend=set_additional_legend,
            is_last_graph= is_last_graph
        )
        y_min_train, y_max_train = l_axes[0].get_ylim()
        y_min_test, y_max_test = l_axes[1].get_ylim()
        if y_min_train < minimum_train:
            minimum_train = y_min_train
        if y_max_train > maximum_train:
            maximum_train = y_max_train
        if y_min_test < minimum_test:
            minimum_test = y_min_test
        if y_max_test > maximum_test:
            maximum_test = y_max_test

    for i in range(len(lList)):
        l_axes = axes[:, i]
        # l_axes[0].set_yticks(minimum, maximum)
        l_axes[0].set_ylim(bottom=minimum_train, top=3)
        l_axes[1].set_ylim(bottom=minimum_test, top=maximum_test)

    axes[0, 0].set_ylabel(
        r"$\operatorname{log}\left(\frac {f_{train}}{f_{train}^{\star}}- 1\right)$ ",
        fontsize=16,
    )
    axes[1, 0].set_ylabel("Balanced accuracy", fontsize=16)
    if set_additional_legend is True:
        axes[0, 0].set_title(
            "(b) scMARK", rotation="vertical", x=-0.6, y=-0.35, fontsize=20
        )
    else:
        axes[1, 0].set_title(
            "(a) Covtype", rotation="vertical", x=-0.6, y=0.75, fontsize=20
        )
    # axes[0].set_title(title)
    # axes[1].set_title(title)
    # if name_fig is not None:
    # plt.savefig(f"{name_fig}.pdf", format="pdf")


def getFEtoile(
    Xtrain,
    Xtest,
    Ytrain,
    Ytest,
    pbChosen,
    reg,
    number_of_grad,
    lList=[1000, 100, 10, 1],
):
    optimTrainSet = trainOptimizerWithGridSearch(
        optimFEtoile,
        Xtrain,
        Xtest,
        Ytrain,
        Ytest,
        number_of_grad,
        10,
        pbChosen,
        reg,
        Xtrain.shape[0],
        lList,
    )
    fEtoileTrain = np.min(optimTrainSet.trainLossList)
    print("fEtoileTrain", fEtoileTrain)
    print("last train losses:", optimTrainSet.trainLossList[-10:])
    return fEtoileTrain


def launchFixedOptimAndGridLListOptim(
    datasetName,
    pbChosen,
    fixedOptimClassList,
    nonFixedOptimClassList,
    nbEpoch,
    lList,
    lFixed,
    n,
    dim,
    reg,
    batchSize,
):
    X, Y = getDataset(datasetName, pbChosen, n, dim)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)
    fEtoileTrain = getFEtoile(Xtrain, Xtest, Ytrain, Ytest, pbChosen, reg)
    print("fEtoileTrain :", fEtoileTrain)
    for lNonFixed in lList:
        listOptim = launchFixedOptimAndNonFixedOptim(
            Xtrain,
            Xtest,
            Ytrain,
            Ytest,
            pbChosen,
            fixedOptimClassList,
            nonFixedOptimClassList,
            nbEpoch,
            lNonFixed,
            lFixed,
            reg,
            batchSize,
        )
        supplement_title = (
            " dotted=Lfixed,"
            + " epochs="
            + str(nbEpoch)
            + ",batchSize="
            + str(batchSize)
        )
        plotListOptim(
            listOptim,
            fEtoileTrain,
            datasetName,
            supplement_title,
            lFixed,
            lNonFixed,
            save=True,
        )


def trainOptimizerWithGridSearch(
    optimizer,
    Xtrain,
    Xtest,
    Ytrain,
    Ytest,
    nbEpochTrueOptim,
    nbEpochGridSearch,
    pbChosen,
    reg,
    batchSize,
    lList,
):
    optim = optimizer(Xtrain, Xtest, Ytrain, Ytest, pbChosen, reg)
    bestL, listOptim = optim.gridSearch(lList, nbEpochGridSearch * optim.n, batchSize)
    if bestL == "unknown":
        raise ValueError
    print("True optimization beginning ----------")
    optim.train(nbEpochTrueOptim * (optim.n), batchSize, bestL)
    return optim


def launchFixedOptimAndNonFixedOptim(
    Xtrain,
    Xtest,
    Ytrain,
    Ytest,
    pbChosen,
    fixedOptimClassList,
    nonFixedOptimClassList,
    nbEpoch,
    lNonFixed,
    lFixed,
    reg,
    batchSize,
):
    listOptim = []
    for fixedOptimClass in fixedOptimClassList:
        print("training :", fixedOptimClass.NAME)
        fixedOptim = fixedOptimClass(Xtrain, Xtest, Ytrain, Ytest, pbChosen, reg)
        fixedOptim.train(nbEpoch * fixedOptim.n, batchSize, lFixed)
        fixedOptim.LINESTYLE = "dotted"
        listOptim.append(fixedOptim)
    for nonFixedOptimClass in nonFixedOptimClassList:
        print(" Optimizer :", nonFixedOptimClass.NAME)
        nonFixedOptim = nonFixedOptimClass(Xtrain, Xtest, Ytrain, Ytest, pbChosen, reg)
        nonFixedOptim.train(nbEpoch * (nonFixedOptim.n), batchSize, lNonFixed)
        listOptim.append(nonFixedOptim)
    return listOptim


def launchProblemWithGridSearch(
    datasetName, pbChosen, Optimizers, nbEpoch, nbEpochGridSearch, n, dim, reg
):
    X, Y = getDataset(datasetName, pbChosen, n, dim)
    theoreticL = getTheoreticL(X, pbChosen)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)
    listOptim = list()
    print("theoretic l", theoreticL)
    for optimizer in Optimizers:
        print("Optimizer : ", optimizer.NAME)
        optim = trainOptimizerWithGridSearch(
            optimizer,
            Xtrain,
            Xtest,
            Ytrain,
            Ytest,
            nbEpoch,
            nbEpochGridSearch,
            pbChosen,
            reg,
        )
        listOptim.append(optim)
    plotListOptim(listOptim, fEtoileTrain)
