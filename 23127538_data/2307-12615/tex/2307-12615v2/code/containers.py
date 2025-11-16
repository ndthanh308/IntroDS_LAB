from importDataset import getDataset
from os.path import exists
from launch_optim import getFEtoile
import numpy as np
import pickle
import os


class Param_args:
    @property
    def pbDescriptor(self):
        return f"dataset={self.dataset_name},pb={self.pb},n_train={self.ntrain},n_test={self.ntest},real_dim={self.real_dim},reg={self.dim},batchSize={self.batchSize}"

    def get_complete_unique_name_f_etoile(self):
        return "f_etoiles/" + self.get_unique_name_f_etoile() + ".csv"

    def get_unique_name_f_etoile(self):
        return f"dataset_{self.dataset_name}_pb_{self.pb}_n_{self.n}_d_{self.dim}_reg_{self.reg}"

    def get_unique_name_trial(self):
        return (
            self.get_unique_name_f_etoile()
            + f"_nbEpoch_{self.nbEpoch}_batchSize_{self.batchSize}"
        )

    def __init__(self, dataset_name, pb, n, dim, nbEpoch, reg, batchSize):
        self.dataset_name = dataset_name
        self.pb = pb
        self.n = n
        self.dim = dim
        self.nbEpoch = nbEpoch
        self.reg = reg
        self.batchSize = batchSize
        self.ntrain = int(self.n * 0.8)
        self.ntest = self.n - self.ntrain
        self.real_dim = None

    @property
    def saveable_dict(self):
        return {
            "dataset": self.dataset_name,
            "problem": self.pb,
            "n": self.n,
            "dim": self.dim,
            "nbEpoch": self.nbEpoch,
            "regularization": self.reg,
            "batch size": self.batchSize,
        }


class Optim_and_plot_params:
    def __init__(self, optimClass, color, linestyle, marker):
        self.optim_class = optimClass
        self.color = color
        self.linestyle = linestyle
        self.marker = marker
        self.train_loss = []
        self.test_metric = []
        self.abscisse = []
        self.label = self.optim_class.NAME
        self.lr = None
        self.real_dim = None

    @property
    def saveable_dict(self):
        return {
            "color": self.color,
            "linestyle": self.linestyle,
            "train_loss": self.train_loss,
            "test_metric": self.test_metric,
            "abscisse": self.abscisse,
            "label": self.label,
            "lr": self.lr,
            "real_dim": self.real_dim,
        }

    def __repr__(self):
        return self.label


class Complete_list_optim_and_plotparam:
    def get_data_from_file(self):
        name_doss = self.get_name_doss()
        for i, optim_and_plot_params in enumerate(self.list_optim_and_plotparam):
            name_file = f"{name_doss}/{optim_and_plot_params.optim_class.NAME}"
            with open(name_file, "rb") as fp:
                data = pickle.load(fp)
            optim_and_plot_params.train_loss = data["train_loss"]
            optim_and_plot_params.test_metric = data["test_metric"]
            optim_and_plot_params.abscisse = data["abscisse"]
            optim_and_plot_params.lr = data["lr"]
            optim_and_plot_params.real_dim = data["real_dim"]
        self.param_args.real_dim = data["real_dim"]

    def get_back_optim(self, optimClass):
        name_doss = self.get_name_doss()
        name_file = optimClass.NAME
        with open(f"{name_doss}/{name_file}", "rb") as fp:
            data = pickle.load(fp)
        return (
            data["train_loss"],
            data["test_metric"],
            data["abscisse"],
            data["lr"],
            data["real_dim"],
        )

    def check_if_optim_is_launched(self, optim_class):
        name_doss = self.get_name_doss()
        return exists(f"{name_doss}/{optim_class.NAME}")

    def is_already_launch(self):
        name_doss = self.get_name_doss()
        # name_file = self.get_name_file()
        name_file = ""
        if not exists(name_doss + name_file):
            return False
        else:
            with open(name_doss + name_file, "rb") as fp:
                pickled_data = pickle.load(fp)
            return self.check_if_all_optim_is_launched(pickled_data)

    def check_if_all_optim_is_launched(self, pickled_data):
        set_of_optim_names = get_set_of_optim_names(pickled_data)
        for i, optim_and_plot_params in enumerate(self.list_optim_and_plotparam):
            if optim_and_plot_params.optim_class.NAME not in set_of_optim_names:
                return False
        return True

    def __init__(self, list_of_init_list, param_args):
        self.list_optim_and_plotparam = get_list_optim_and_plotparam(list_of_init_list)
        self.param_args = param_args
        self.f_etoile_train = None
        self.L = None

    def get_dataset(self):
        return getDataset(
            self.param_args.dataset_name,
            self.param_args.pb,
            self.param_args.n,
            self.param_args.dim,
        )

    def get_f_etoile(self, Xtrain, Xtest, Ytrain, Ytest, number_of_grad_for_f_etoile):
        if self.is_f_etoile_already_computed() is True:
            print("Get f_etoile back from file")
            f_etoile_train = self.get_f_etoile_from_data()
        else:
            print("Computing f_etoile")
            f_etoile_train = getFEtoile(
                Xtrain,
                Xtest,
                Ytrain,
                Ytest,
                self.param_args.pb,
                self.param_args.reg,
                number_of_grad_for_f_etoile,
            )
        self.save_f_etoile(f_etoile_train)
        return f_etoile_train

    def is_f_etoile_already_computed(self):
        return exists(self.param_args.get_complete_unique_name_f_etoile())

    def get_f_etoile_from_data(self):
        f_etoile_train = np.genfromtxt(
            self.param_args.get_complete_unique_name_f_etoile(), delimiter=""
        )
        return f_etoile_train

    def save_f_etoile(self, f_etoile_train):
        f_etoile_train = np.array([f_etoile_train])
        np.savetxt(
            self.param_args.get_complete_unique_name_f_etoile(),
            f_etoile_train,
            delimiter=",",
        )

    @property
    def pbDescriptor(self):
        return self.param_args.pbDescriptor + f"L={self.L}"

    def save(self):
        name_doss = self.get_name_doss()
        os.makedirs(name_doss, exist_ok=True)
        for optim in self.list_optim_and_plotparam:
            name_file = f"{name_doss}/{optim.optim_class.NAME}"
            with open(name_file, "wb") as fp:
                pickle.dump(optim.saveable_dict, fp)

    def get_name_doss(self):
        name_doss = f"results_simu/{self.param_args.get_unique_name_trial()}/L_{self.L}"
        return name_doss


def get_list_optim_and_plotparam(list_of_init_list):
    list_optim_and_plotparam = []
    for init_args in list_of_init_list:
        list_optim_and_plotparam.append(
            Optim_and_plot_params(
                init_args[0], init_args[1], init_args[2], init_args[3]
            )
        )
    return list_optim_and_plotparam


def get_set_of_optim_names(pickled_data):
    return set(pickled_data.keys())
