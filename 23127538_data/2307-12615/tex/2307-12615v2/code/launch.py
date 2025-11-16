import myoptim
import matplotlib.pyplot as plt
from launch_optim import launchGridLListOptim
from containers import Param_args, Complete_list_optim_and_plotparam
import argparse

number_of_grad_for_f_etoile = 3000


list_init_optim_ada_and_vr = [
    [myoptim.SAGAdaDiag, "orange", "solid", "None"],
    [myoptim.SAGAdaNorm, "orange", "dashed", "None"],
    [myoptim.AdaLSVRGDiag, "orange", "dotted", "None"],
    [myoptim.AdaLSVRGNorm, "orange", (0, (1, 10)), "None"],
    [myoptim.AdaDiag, "purple", "solid", "None"],
    [myoptim.AdaNorm, "purple", "dashed", "None"],
    [myoptim.LSVRG, "red", "dashed", "None"],
    [myoptim.SAGA, "red", "solid", "None"],
]
list_init_optim_accel = [
    [myoptim.SAGAdaDiag, "orange", "solid", "None"],
    [myoptim.SAGAdaNorm, "orange", "dashed", "o"],
    [myoptim.AdaLSVRGDiag, "orange", "dotted", "v"],
    [myoptim.AdaLSVRGNorm, "orange", (0, (1, 10)), "s"],
    [myoptim.Katyusha, "purple", "solid", "*"],
    [myoptim.VRADA, "purple", "dashdot", "+"],
    [myoptim.Varag, "purple", "dashed", "P"],
    [myoptim.LKatyusha, "purple", "dotted", "X"],
    [myoptim.MiG, "purple", (0, (1, 10)), 9],
    [myoptim.DAVIS, "purple", (0, (3, 5, 1, 5, 1, 5)), "D"],
]

list_init_optim_adam_and_rms = [
    [myoptim.SAGARMSprop, "orange", "dashdot", "None"],
    [myoptim.SAGAdam, "orange", "solid", "o"],
    [myoptim.AdamLSVRG, "orange", "dashed", "v"],
    [myoptim.RMSpropLSVRG, "orange", "dotted", "s"],
    [myoptim.RMSprop, "purple", "dashdot", "*"],
    [myoptim.Adam, "purple", "solid", "P"],
]


list_init_optim_dict = {
    "ada_and_vr": list_init_optim_ada_and_vr,
    "accel": list_init_optim_accel,
    "adam_and_rms": list_init_optim_adam_and_rms,
}
problem = "MultiLogisticRegression"

dataset_scrna = "scRNA"
dataset_covtype = "covType"

lList = [0.001, 0.01, 0.1, 1, 10, 100]
# lList = [0.001, 0.001]
# nbEpoch = 90
nbEpoch = 20

nbRows = 4
nbCols = len(lList)
title = ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", required=True)
    args = parser.parse_args()
    if str(args.plot) in list_init_optim_dict.keys():
        list_init_optim = list_init_optim_dict[args.plot]
    else:
        raise ValueError(
            f"argument --plot ({args.plot}) not in {list(list_init_optim_dict.keys())}."
        )
    fig, axes = plt.subplots(nbRows, nbCols, figsize=(30, 15))
    #### covtype
    n = 600000
    # n = 800
    dim = 54
    # dim = 20
    batchSize = 10
    # batchSize = n//5
    reg = 0
    param_args = Param_args(dataset_covtype, problem, n, dim, nbEpoch, reg, batchSize)

    complete_list = Complete_list_optim_and_plotparam(list_init_optim, param_args)

    launchGridLListOptim(
        complete_list,
        lList,
        number_of_grad_for_f_etoile,
        axes=axes[:2, :],
        name_fig="covtype_adaptVSeach",
        set_additional_legend=False,
    )

    n = 20000
    dim = 300
    nbEpoch = 20
    batchSize = 1

    param_args = Param_args(dataset_scrna, problem, n, dim, nbEpoch, reg, batchSize)

    complete_list = Complete_list_optim_and_plotparam(list_init_optim, param_args)
    launchGridLListOptim(
        complete_list,
        lList,
        number_of_grad_for_f_etoile,
        axes=axes[2:, :],
        name_fig="adaptVRVSeach",
        set_additional_legend=True,
    )
    format = "png"
    plt.savefig(f"{args.plot}.{format}", format=format)
    plt.show()
