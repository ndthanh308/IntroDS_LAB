import myoptim
import matplotlib.pyplot as plt
from launch_optim import launchGridLListOptim
from containers import Param_args, Complete_list_optim_and_plotparam


number_of_grad_for_f_etoile = 500
problems = ["MultiLogisticRegression", "linearRegression"]
datasetNames = ["covType", "MNIST", "australian", "scRNA"]

list_init_optim = [
    # [myoptim.DAVIS, "orange", "dotted"],
    [myoptim.SAGAdaDiagRestart, "orange", "dotted"],
    [myoptim.SAGARMSprop, "orange", "solid"],
    [myoptim.SAGAdam, "orange", "dashed"],
    [myoptim.SAGAdaDiag, "orange", "dashdot"],
    [myoptim.AdaSVRG, "orange", (0, (1, 10))],
    [myoptim.AdaSVRGRestart, "orange", (0, (3, 10, 1, 10, 1, 10))],
    [myoptim.AdaLSVRGDiag, "orange", "dotted"]
    [myoptim.AdaLSVRGNorm, "orange", "dotted"]
    # [myoptim.SAGA,"purple", "solid"],
    [myoptim.LSVRG, "purple", "dashed"],
    [myoptim.SAGA, "purple", "dotted"],
]

lList = [0.01, 0.1, 1, 10, 100]
nbEpoch = 80
pb = problems[0]
reg = 0

nbRows = 4
nbCols = len(lList)
fig, axes = plt.subplots(nbRows, nbCols, figsize=(30, 15))

####### for covtype data
n = 2000
# n = 500
dim = 300
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

n = 2000
# n = 500
dim = 300
# nbEpoch = 300
batchSize = 10
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

title += f"\n{complete_list.param_args.pbDescriptor}"
fig.suptitle(title)
plt.show()
