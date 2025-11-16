import myoptim
import matplotlib.pyplot as plt
from launch_optim import launchGridLListOptim
from containers import Param_args, Complete_list_optim_and_plotparam


number_of_grad_for_f_etoile = 5000
problems = ["MultiLogisticRegression", "linearRegression"]
datasetNames = ["covType", "MNIST", "australian", "scRNA"]

list_init_optim = [
    # [myoptim.AdaDiag, "orange", "dashdot"],
    [myoptim.SAGAdaDiag, "orange", "solid", "None"],
    [myoptim.SAGAdaNorm, "orange", "dashed", "o"],
    [myoptim.AdaLSVRGDiag, "orange", "dotted", "v"],
    [myoptim.AdaLSVRGNorm, "orange", (0, (1, 10)), "s"],
    # [myoptim.SAGAdaDiagRestart, "orange", "dotted"],
    # [myoptim.AdaSVRGRestart, "orange", (0, (3, 10, 1, 10, 1, 10))],
    # [myoptim.SAGARMSprop, "orange", "solid"],
    # [myoptim.SAGAdam, "orange", "dashed"],
    [myoptim.Katyusha, "purple", "solid", "*"],
    [myoptim.VRADA, "purple", "dashdot", "+"],
    [myoptim.Varag, "purple", "dashed", "P"],
    [myoptim.LKatyusha, "purple", "dotted", "X"],
    [myoptim.MiG, "purple", (0, (1, 10)), 9],
    [myoptim.DAVIS, "purple", (0, (3, 5, 1, 5, 1, 5)), "D"],
]

lList = [0.001, 0.01, 0.1, 1, 10, 100]
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
    set_additional_legend=False,
)
title = complete_list.param_args.pbDescriptor

###### scrna
name_fig = "AdaptVRvsAccel"
n = 20000
# n = 500
dim = 300
nbEpoch = 22
batchSize = 1
# batchSize = 1
dataset_name = datasetNames[-1]

param_args = Param_args(dataset_name, pb, n, dim, nbEpoch, reg, batchSize)

complete_list = Complete_list_optim_and_plotparam(list_init_optim, param_args)
launchGridLListOptim(
    complete_list,
    lList,
    number_of_grad_for_f_etoile,
    axes=axes[2:, :],
    set_additional_legend=True,
)

title += f"\n{complete_list.param_args.pbDescriptor}"
# fig.suptitle(title)
if name_fig is not None:
    format = "png"
    plt.savefig(f"{name_fig}.{format}", format=format)
plt.show()
