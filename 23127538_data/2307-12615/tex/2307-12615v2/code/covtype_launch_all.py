import myoptim
import matplotlib.pyplot as plt
from launch_optim import launchGridLListOptim
from containers import Param_args, Complete_list_optim_and_plotparam


number_of_grad_for_f_etoile = 5000
problems = ["MultiLogisticRegression", "linearRegression"]
datasetNames = ["covType", "MNIST", "australian", "scRNA"]
## for covtype data
n = 600
# n = 2000
dim = 54
# nbEpoch = 70
nbEpoch = 9
reg = 0
# batchSize = 10
batchSize = 52
# lList = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10,100]
lList = [0.001, 0.01, 0.1, 1, 10, 100]
# lList = [1,"theoretical",10,100]#, 1, 100]
dataset_name = datasetNames[0]
pb = problems[0]

list_init_optim = [
    [myoptim.AdaVRAE, "black", "solid", "None"],
    [myoptim.AdaVRAG, "orange", "solid", "None"],
    [myoptim.SAGAdaDiag, "black", "solid", "None"],
    # [myoptim.SAGAdaNorm, "black", "dashed", "None"],
    # [myoptim.AdaLSVRGDiag, "black", "dotted", "None"],
    # [myoptim.AdaLSVRGNorm, "black", (0, (1, 10)), "None"],
    # [myoptim.AdaDiag, "dimgrey", "solid", "None"],
    # [myoptim.AdaNorm, "dimgrey", "dashed", "None"],
    # [myoptim.LSVRG, "lightgrey", "dashed", "None"],
    # [myoptim.SAGA, "lightgrey", "solid", "None"],
    # [myoptim.Katyusha, "purple", "solid", "*"],
    # [myoptim.VRADA, "purple", "dashdot", "+"],
    # [myoptim.Varag, "purple", "dashed", "P"],
    # [myoptim.LKatyusha, "purple", "dotted", "X"],
    # [myoptim.MiG, "purple", (0, (1, 10)), 9],
    # [myoptim.DAVIS, "purple", (0, (3, 5, 1, 5, 1, 5)), "D"],
    # [myoptim.SAGARMSprop, "orange", "dashdot", "None"],
    # [myoptim.SAGAdam, "orange", "solid", "o"],
    # [myoptim.AdamLSVRG, "orange", "dashed", "v"],
    # [myoptim.RMSpropLSVRG, "orange", "dotted", "s"],
    # [myoptim.RMSprop, "purple", "dashdot", "*"],
    # [myoptim.Adam, "purple", "solid", "P"],
]


param_args = Param_args(dataset_name, pb, n, dim, nbEpoch, reg, batchSize)
complete_list = Complete_list_optim_and_plotparam(list_init_optim, param_args)
fig, axes = plt.subplots(2, len(lList))
launchGridLListOptim(complete_list, lList, number_of_grad_for_f_etoile, axes=axes)
plt.show()
