import myoptim
import matplotlib.pyplot as plt
from launch_optim import launchGridLListOptim
from containers import Param_args, Complete_list_optim_and_plotparam


number_of_grad_for_f_etoile = 500
problems = ["MultiLogisticRegression", "linearRegression"]
datasetNames = ["covType", "MNIST", "australian", "scRNA"]
name_fig = "AdaptVRvsEach"
list_init_optim = [
    # [myoptim.SAGAdaDiagRestart, "orange", "dotted"],
    # [myoptim.SAGARMSprop, "green", "solid"],
    # [myoptim.SAGAdam, "blue", "solid"],
    [myoptim.SAGAdaDiag, "orange", "solid", "None"],
    [myoptim.SAGAdaNorm, "orange", "dashed", "o"],
    [myoptim.AdaLSVRGDiag, "orange", "dotted", "v"],
    [myoptim.AdaLSVRGNorm, "orange", (0, (1, 10)), "s"],
    # [myoptim.AdaSVRGRestart, "orange", (0, (3, 10, 1, 10, 1, 10))],
    [myoptim.AdaDiag, "purple", "solid", "*"],
    [myoptim.AdaNorm, "purple", "dashed", "+"],
    [myoptim.LSVRG, "red", "dashed", "P"],
    [myoptim.SAGA, "red", "solid", 9],
    # [myoptim.Adam, "black", "solid"],
    # [myoptim.RMSprop, "pink", "solid"]
]

lList = [0.001, 0.01, 0.1, 1, 10, 100]
# lList = [0.0001, 0.01]
pb = problems[0]
reg = 0

nbRows = 4
nbCols = len(lList)
fig, axes = plt.subplots(nbRows, nbCols, figsize=(30, 15))
title = ""

####### for covtype data
n = 600000
# n = 500
dim = 54
nbEpoch = 22
batchSize = 10
dataset_name = datasetNames[0]
param_args = Param_args(dataset_name, pb, n, dim, nbEpoch, reg, batchSize)

complete_list = Complete_list_optim_and_plotparam(list_init_optim, param_args)
launchGridLListOptim(
    complete_list,
    lList,
    number_of_grad_for_f_etoile,
    axes=axes[:2, :],
    name_fig="covtype_adaptVSeach",
    set_additional_legend=False,
)

title = complete_list.param_args.pbDescriptor

###### scrna

n = 20000
# n = 500
dim = 300
# nbEpoch = 300
batchSize = 1
nbEpoch = 22
# batchSize = 1
# lList = [0.001, 0.01,0.1,1, 10, 100, 1000, 10000]
dataset_name = datasetNames[-1]

param_args = Param_args(dataset_name, pb, n, dim, nbEpoch, reg, batchSize)

complete_list = Complete_list_optim_and_plotparam(list_init_optim, param_args)
launchGridLListOptim(
    complete_list,
    lList,
    number_of_grad_for_f_etoile,
    axes=axes[2:, :],
    name_fig="adaptVRVSeach",
    set_additional_legend=True,
)


title += f"\n{complete_list.param_args.pbDescriptor}"
# fig.suptitle(title)

if name_fig is not None:
    format = "png"
    plt.savefig(f"{name_fig}.{format}", format=format)
plt.show()
