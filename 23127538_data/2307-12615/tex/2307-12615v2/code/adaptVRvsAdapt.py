import myoptim
import matplotlib.pyplot as plt
from launch_optim import launchGridLListOptim
from containers import Param_args, Complete_list_optim_and_plotparam


number_of_grad_for_f_etoile = 500
problems = ["MultiLogisticRegression", "linearRegression"]
datasetNames = ["covType", "MNIST", "australian", "scRNA"]

list_init_optim = [
    # [myoptim.SAGAdaDiagRestart, "orange", "dotted"],
    [myoptim.SAGARMSprop, "orange", "dashdot", "None"],
    [myoptim.SAGAdam, "orange", "solid", "o"],
    [myoptim.AdamLSVRG, "orange", "dashed", "v"],
    [myoptim.RMSpropLSVRG, "orange", "dotted", "s"],
    # [myoptim.SAGAdaDiag, "orange", "solid"],
    # [myoptim.AdaLSVRGDiag, "orange", (0, (1, 10))],
    # [myoptim.AdaSVRGRestart, "orange", (0, (3, 10, 1, 10, 1, 10))],
    [myoptim.RMSprop, "purple", "dashdot", "*"],
    [myoptim.Adam, "purple", "solid", "P"],
]

lList = [0.001,0.01, 0.1, 1, 10, 100]
nbEpoch = 22
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
# nbEpoch = 300
batchSize = 10
# batchSize = 1
# lList = [0.001, 0.01,0.1,1, 10, 100, 1000, 10000]
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
batchSize = 1
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
    name_fig="covtype_adaptVSeach",
    set_additional_legend=True,
)

# title += f"\n{complete_list.param_args.pbDescriptor}"
# fig.suptitle(title)
name_fig = "adaptVRvsAdapt"
fig.savefig(name_fig + ".png", format = "png")
plt.show()
