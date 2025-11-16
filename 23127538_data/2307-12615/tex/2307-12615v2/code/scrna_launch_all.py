import myoptim
import matplotlib.pyplot as plt
from launch_optim import launchGridLListOptim
from containers import Param_args, Complete_list_optim_and_plotparam


number_of_grad_for_f_etoile = 10000
problems = ["MultiLogisticRegression", "linearRegression"]
datasetNames = ["covType", "MNIST", "australian", "scRNA"]
## for scRNA data
n = 20000
dim = 300
nbEpoch = 22
reg = 0
batchSize = 1
lList = [0.001, 0.01, 0.1, 1, 10, 100]
dataset_name = datasetNames[-1]
pb = problems[0]
fig, axes = plt.subplots(2, len(lList))

list_init_optim = [
    [myoptim.SAGAdaDiag, "black", "solid", "None"],
    [myoptim.SAGAdaNorm, "black", "dashed", "None"],
    [myoptim.AdaLSVRGDiag, "black", "dotted", "None"],
    [myoptim.AdaLSVRGNorm, "black", (0, (1, 10)), "None"],
    [myoptim.AdaDiag, "dimgrey", "solid", "None"],
    [myoptim.AdaNorm, "dimgrey", "dashed", "None"],
    [myoptim.LSVRG, "lightgrey", "dashed", "None"],
    [myoptim.SAGA, "lightgrey", "solid", "None"],
    [myoptim.Katyusha, "purple", "solid", "*"],
    [myoptim.VRADA, "purple", "dashdot", "+"],
    [myoptim.Varag, "purple", "dashed", "P"],
    [myoptim.LKatyusha, "purple", "dotted", "X"],
    [myoptim.MiG, "purple", (0, (1, 10)), 9],
    [myoptim.DAVIS, "purple", (0, (3, 5, 1, 5, 1, 5)), "D"],
    [myoptim.SAGARMSprop, "orange", "dashdot", "None"],
    [myoptim.SAGAdam, "orange", "solid", "o"],
    [myoptim.AdamLSVRG, "orange", "dashed", "v"],
    [myoptim.RMSpropLSVRG, "orange", "dotted", "s"],
    [myoptim.RMSprop, "purple", "dashdot", "*"],
    [myoptim.Adam, "purple", "solid", "P"],
]

param_args = Param_args(dataset_name, pb, n, dim, nbEpoch, reg, batchSize)
complete_list = Complete_list_optim_and_plotparam(list_init_optim, param_args)

launchGridLListOptim(complete_list, lList, number_of_grad_for_f_etoile, axes)

plt.show()
