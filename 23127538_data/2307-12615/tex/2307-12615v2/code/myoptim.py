from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
import math
from smoothnessConstants import getLinearL, getLogisticL
from functorch import vmap, grad, hessian
from sklearn.metrics import balanced_accuracy_score
import time

# from torch.autograd.functional import hessian
from utils import (
    buildXYBatchesList,
    UnknownProblemException,
    pseudoInverse1d,
)
import numpy as np

torch.set_printoptions(12)
torch.set_default_tensor_type(torch.DoubleTensor)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device : ", device)
torch.set_default_tensor_type(torch.DoubleTensor)

CONSTANT_TIMES_1_OVER_L = 1 / 5
CONSTANT_PROP_AND_ADAM = 1 / 20
CONSTANT_NORM = 1
CONSTANT_VR_CLASSIC = 1 / 10


class myOptim(ABC):
    @property
    def pbDescriptor(self):
        return (
            self.pbChosen
            + ",ntrain="
            + str(self.n)
            + "ntest="
            + str(self.ntest)
            + ",dim="
            + str(self.dim)
            + ",reg="
            + str(self.reg)
        )

    LINESTYLE = "-"

    def initLinearRegressionLossAndParam(self):
        self.dim = self.Xtrain[0].shape
        self.param = 0 * torch.randn(self.dim, requires_grad=False, device=device).to(
            torch.double
        )
        self.reg = 0

        def linearRegressionLoss(beta, x, y):
            return 1 / 2 * torch.norm(x @ beta - y, p=2) ** 2

        self.singleLoss = linearRegressionLoss
        self.theoreticalL =1 # getLinearL(self.Xtrain, self.LFORMULA)

    def __init__(self, Xtrain, Xtest, Ytrain, Ytest, pbChosen, reg):
        self.initDataset(Xtrain, Xtest, Ytrain, Ytest)
        self.initLossAndParamAndL(pbChosen)
        self.pbChosen = pbChosen
        self.reg = reg

    def initDataset(self, Xtrain, Xtest, Ytrain, Ytest):
        self.n = Xtrain.shape[0]
        self.ntest = Xtest.shape[0]
        self.Xtrain = Xtrain.to(torch.double).to(device)
        self.Xtest = Xtest.to(torch.double).to(device)
        self.Ytrain = Ytrain.to(device)
        self.Ytest = Ytest.to(device)

    def initLossAndParamAndL(self, pbChosen):
        if pbChosen == "linearRegression":
            self.initLinearRegressionLossAndParam()
        elif pbChosen == "MultiLogisticRegression":
            self.initLogisticRegressionLossAndParam()
        else:
            raise UnknownProblemException(pbChosen)

    def initLogisticRegressionLossAndParam(self):
        self.nbClasses = len(self.Ytrain[0])
        if self.nbClasses > 2:
            self.initMultLogisticRegressionLossAndParam()
        else:
            self.init2dLogisticRegressionLossAndParam()

    def init2dLogisticRegressionLossAndParam(self):
        self.Ytrain = torch.argmax(self.Ytrain, axis=1)
        self.Ytest = torch.argmax(self.Ytest, axis=1)
        if torch.max(self.Ytrain) > 1 or torch.min(self.Ytrain) < 0:
            raise Exception("Y must be between 0 and 1.")
        self.dim = self.Xtrain.shape[1]
        self.param = torch.zeros(self.dim, requires_grad=True, device=device)
        self.reg = 0.001

        def logisticRegressionLoss(beta, x, y):
            xbeta = x @ beta
            return -y * xbeta + torch.log(1 + torch.exp(xbeta))

        self.singleLoss = logisticRegressionLoss

        def predict(beta, Xtest):
            xbeta = Xtest @ beta
            return (torch.nn.Sigmoid()(xbeta) > 0.5).float()

        self.predict = predict

        def accuracy(beta, Xtest, Ytest):
            prediction = self.predict(beta, Xtest)
            return balanced_accuracy_score(Ytest.cpu(), prediction.cpu())

        self.accuracy = accuracy

    def initMultLogisticRegressionLossAndParam(self):
        self.nbClasses = len(self.Ytrain[0])
        self.dim = self.Xtrain.shape[1] * self.nbClasses
        self.param = torch.zeros(
            self.nbClasses, self.Xtrain.shape[1], device=device
        ).flatten()

        def multLogisticRegressionLoss(vectBeta, x, y):
            beta = vectBeta.reshape(self.nbClasses, self.Xtrain.shape[1])
            xbeta = torch.mm(beta, x.unsqueeze(1)).squeeze()
            logsoftmax = torch.nn.LogSoftmax(dim=-1)(xbeta)
            logsoftmaxTimesy = logsoftmax * y
            return -torch.sum(logsoftmaxTimesy) + self.reg * torch.norm(beta) ** 2

        self.singleLoss = multLogisticRegressionLoss
        self.theoreticalL = 1 #getLogisticL(self.Xtrain, self.nbClasses, self.LFORMULA)

        def predict(vectBeta, Xtest):
            beta = vectBeta.reshape(self.nbClasses, self.Xtrain.shape[1])
            xbeta = torch.mm(Xtest, beta.t()).squeeze()
            logsoftmax = torch.nn.LogSoftmax(dim=-1)(xbeta)
            return torch.argmax(logsoftmax, axis=1)

        self.predict = predict

        def accuracy(beta, Xtest, Ytest):
            prediction = self.predict(beta, Xtest)
            true_value = torch.argmax(Ytest, axis=1)
            return balanced_accuracy_score(true_value.cpu(), prediction.cpu())

        self.accuracy = accuracy

    @property
    def paramForPlotLoss(self):
        return self.meaningfullMeanParam

    def train(self, nbGradToCompute, batchSize, L):
        self.batchSize = batchSize
        if L == "theoretical":
            print("Setting the theoretical L = ", self.theoreticalL)
            self.L = self.theoreticalL
        else:
            print("setting L :", L)
            self.L = L
        self.initGlobalVariables()
        self.setLrFromL()
        self.initArguments()
        self.meaningfullMeanParam = torch.zeros(self.param.shape, device=device)
        nbOuterLoopsDone = 0
        while self.totalNbComputedGrad < nbGradToCompute:
            self.meaningfullMeanParam *= nbOuterLoopsDone / (nbOuterLoopsDone + 1)
            self.meaningfullMeanParam += self.meaningfullParam / (nbOuterLoopsDone + 1)
            newIndexes = np.arange(self.nbBatches)
            np.random.shuffle(newIndexes)
            trainLoss = self.computeLoss(
                self.paramForPlotLoss, self.Xtrain, self.Ytrain
            )
            test_metric = self.compute_test_metric(
                self.paramForPlotLoss, self.Xtest, self.Ytest
            )
            try:
                test_metric = test_metric.cpu()
            except:
                pass
            self.bug = False
            if torch.isnan(trainLoss):
                self.bug = True
            self.nbComputedGrad = 0
            self.stopEpoch = False
            t1 = time.time()
            if self.needFullGradAndAnchorUpdate:
                self.updateFullGradAndAnchor()
                self.nbComputedGrad += self.Xtrain.shape[0]
                self.updateSomeVariables()
            t0 = time.time()
            for i, batchIndex in enumerate(newIndexes):
                self.i = i + 1
                batchX, batchY = self.getBatchXbatchY(batchIndex)
                self.trainStep(batchX, batchY, batchIndex)
                self.nbComputedGrad += self.getNbGradComputedAndUpdateNbSeemSamples(
                    batchX
                )
                if self.bug:
                    break
            if self.bug:
                break
            self.keepTrackLossAndNbComputedGrad(
                trainLoss, test_metric, self.nbComputedGrad
            )
            nbOuterLoopsDone += 1

    def updateSomeVariables(self):
        pass

    def sample_uniform(self):
        prev_state = torch.random.get_rng_state()
        torch.random.manual_seed(self.seed_number)
        self.seed_number += 1
        uniform = torch.rand(1)
        torch.random.set_rng_state(prev_state)
        return uniform

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)

    def initGlobalVariables(self):
        self.nbComputedGradList = list()
        self.trainLossList = list()
        self.test_metric_list = list()
        self.totalNbComputedGrad = 0
        self.nbSamplesSeen = 0

    def computePersampleLoss(self, param, X, Y):
        return vmap(self.singleLoss, in_dims=(None, 0, 0))(param, X, Y)

    def computeLoss(self, param, X, Y):
        return torch.mean(self.computePersampleLoss(param, X, Y), axis=0)

    def computeFullTrainLoss(self, param):
        return self.computeLoss(param, self.Xtrain, self.Ytrain)

    def compute_test_metric(self, beta, X_test, Y_test):
        return self.accuracy(beta, X_test, Y_test)

    def computeFullTestLoss(self, param):
        return self.computeLoss(param, self.Xtest, self.Ytest)

    ## should change meaningfullParam to meaningfullMeanParam, same for test loss
    def computeMeaningfullTrainLoss(self):
        return self.computeFullTrainLoss(self.meaningfullParam)

    def computeMeaningfullTestLoss(self):
        return (
            self.computeFullTestLoss(self.meaningfullParam)
            - self.reg * torch.norm(self.meaningfullParam) ** 2
        )

    @property
    @abstractmethod
    def meaningfullParam(self):
        return self.param

    def computeSingleGrad(self, param, X, Y):
        return grad(self.singleLoss)(param, X, Y)

    def computeFullGrad(self, param):
        return self.computeMeanOfBatchGrad(param, self.Xtrain, self.Ytrain)

    def computeMeanOfBatchGrad(self, param, X, Y):
        return grad(self.computeLoss)(param, X, Y)

    def zero_grad(self):
        self.param.grad.zero_()

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        pass

    @property
    def needFullGradAndAnchorUpdate(self):
        return False

    def initArguments(self):
        self.batchesXList, self.batchesYList = buildXYBatchesList(
            self.Xtrain, self.Ytrain, self.batchSize
        )
        self.nbBatches = len(self.batchesXList)

    def trainStep(self, batchX, batchY, batchIndex):
        self.computeAndSetGrad(batchX, batchY, batchIndex)
        self.step()
        self.zero_grad()

    def getBatchXbatchY(self, batchIndex):
        batchX = self.batchesXList[batchIndex]
        batchY = self.batchesYList[batchIndex]
        return batchX, batchY

    def keepTrackLossAndNbComputedGrad(self, trainLoss, test_metric, nbComputedGrad):
        self.nbComputedGradList.append(nbComputedGrad)
        self.totalNbComputedGrad += nbComputedGrad
        self.trainLossList.append(trainLoss.cpu().detach().numpy())
        self.test_metric_list.append(test_metric)

    @property
    def cumsumComputedGradList(self):
        return np.cumsum(np.array(self.nbComputedGradList))

    def getNbGradComputedAndUpdateNbSeemSamples(self, X):
        self.nbSamplesSeen += X.shape[0]
        return X.shape[0]

    def updateFullGradAndAnchor(self):
        self.anchor = torch.clone(self.param)
        self.fullGradAnchor = self.computeFullGrad(self.anchor)

    def step(self):
        with torch.no_grad():
            self.param -= self.lr * self.param.grad

    def gridSearch(self, lList, nbGradToCompute, batchSize):
        initializationParameter = torch.clone(self.param)
        self.initDataset(self.Xtrain, self.Xtest, self.Ytrain, self.Ytest)
        criteriaList = list()
        listOptim = list()
        for l in tqdm(lList):
            self.train(nbGradToCompute, batchSize, l)
            criteriaList.append(self.trainLossList[-1])
            copiedOptim = copyOptim(self)
            copiedOptim.COLOR = None
            copiedOptim.NAME = str(np.round(l, 8)) + " " + copiedOptim.NAME
            listOptim.append(copiedOptim)
            self.setParam(initializationParameter)
        criteriaListWithoutNan = np.array(np.nan_to_num(criteriaList, nan=np.inf))

        print(
            "criteria list lr:criteria",
            dict(zip(lList, np.round(criteriaListWithoutNan))),
        )
        if np.max(criteriaListWithoutNan) > 1e9:
            bestL = "unknown"
        else:
            bestL = lList[np.argmin(criteriaListWithoutNan)]
        print("BEST L IS:", bestL)
        return bestL, listOptim

    def setParam(self, initializationParameter):
        with torch.no_grad():
            self.param = torch.clone(initializationParameter)


class VRADA(myOptim):
    NAME = "VRADA"
    COLOR = "lightgreen"
    LINESTYLE = "solid"
    LFORMULA = "max"

    def initArguments(self):
        super().initArguments()
        self.A_s = 1 / self.L
        self.A_smoins1 = 0
        self.vradaInit()

    @property
    def meaningfullParam(self):
        return self.xTilde

    def doOneFullGradUpdate(self):
        FullGrad = self.computeFullGrad(self.param)
        with torch.no_grad():
            self.param -= (
                1 / self.L * FullGrad
            )  # one step of gradient descent for initialization

    def updateSomeVariables(self):
        self.updateA_s()
        self.zTilde *= 0

    @property
    def needFullGradAndAnchorUpdate(self):
        return True

    def trainStep(self, batchX, batchY, batchIndex):
        self.computeAndSetGrad(batchX, batchY)
        self.step()
        self.incrementZTilde()

    def updateFullGradAndAnchor(self):
        self.xTilde = (
            self.A_smoins1 / self.A_s * self.xTilde + self.a_s / self.A_s * self.zTilde
        )
        self.fullGradAnchor = self.computeFullGrad(self.xTilde)

    def updateA_s(self):
        self.A_smoins1 = self.A_s
        self.A_s = self.A_smoins1 + math.sqrt(
            self.nbBatches * self.A_smoins1 / (2 * self.L)
        )

    def computeAndSetGrad(self, batchX, batchY):
        y_sk = (
            self.A_smoins1 / self.A_s * self.xTilde + self.a_s / self.A_s * self.param
        )
        yComputedGrad = self.computeMeanOfBatchGrad(y_sk, batchX, batchY)
        xTildeComputedGrad = self.computeMeanOfBatchGrad(self.xTilde, batchX, batchY)
        self.param.grad = yComputedGrad - xTildeComputedGrad + self.fullGradAnchor

    def vradaInit(self):
        # we don't use self.computeMeaningfullTrainLoss in the initialization
        # since it requires xTilde that will be computed after the first gradient upgrade.
        trainLoss = self.computeLoss(self.param, self.Xtrain, self.Ytrain)
        test_metric = self.compute_test_metric(self.param, self.Xtest, self.Ytest)
        try:
            test_metric = test_metric.cpu()
        except:
            pass
        self.doOneFullGradUpdate()
        nbGradComputed = self.Xtrain.shape[0]
        self.keepTrackLossAndNbComputedGrad(trainLoss, test_metric, nbGradComputed)
        self.xTilde = torch.clone(self.param)
        self.fullGradAnchor = self.computeFullGrad(self.xTilde)
        self.zTilde = torch.clone(self.param)

    def setLrFromL(self):
        self.L *= CONSTANT_VR_CLASSIC
        print("L", self.LFORMULA, " = ", self.L)
        print("Lr changing, initialized at ", 1 / (self.L))

    def getNbGradComputedAndUpdateNbSeemSamples(self, X):
        self.nbSamplesSeen += X.shape[0]
        return 2 * X.shape[0]

    @property
    def a_s(self):
        return self.A_s - self.A_smoins1

    def step(self):
        with torch.no_grad():
            self.param -= self.a_s / self.nbBatches * self.param.grad

    def incrementZTilde(self):
        self.zTilde += self.param / self.nbBatches


class SVRG(myOptim):
    NAME = "SVRG"
    COLOR = "lightcoral"
    LINESTYLE = "solid"
    LFORMULA = "max"

    @property
    def meaningfullParam(self):
        return self.param

    def initArguments(self):
        super().initArguments()
        self.anchor = None
        self.previous_grad = self.param * 0
        # self.meaningFullParam = torch.clone(self.param)

    def setLrFromL(self):
        print("L", self.LFORMULA, " = ", self.L)
        self.lr = 1 / (7 * self.L) * CONSTANT_VR_CLASSIC
        print("lr = ", self.lr)

    @property
    def needFullGradAndAnchorUpdate(self):
        return True

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        computedGrad = self.computeMeanOfBatchGrad(self.param, batchX, batchY)
        computedGradSnapshot = self.computeMeanOfBatchGrad(self.anchor, batchX, batchY)
        self.param.grad = computedGrad - computedGradSnapshot + self.fullGradAnchor
        # print('diff SVRG', torch.norm(self.param.grad - self.previous_grad)**2)
        self.previous_grad = self.param.grad

    def getNbGradComputedAndUpdateNbSeemSamples(self, X):
        self.nbSamplesSeen += X.shape[0]
        return 2 * X.shape[0]


class DAVIS(myOptim):
    NAME = "DAVIS"
    COLOR = "blue"
    LINESTYLE = "dashed"
    LFORMULA = "max"

    def initArguments(self):
        super().initArguments()
        self.anchor_x_tilde = torch.clone(self.param).to(device)
        self.nb_outer_loop_done = 0
        self.z = torch.zeros(self.dim).to(device)
        self.next_anchor_x_tilde = torch.clone(self.param).to(device)
        self.m = self.nbBatches
        # self.fullGradAnchor_x_tilde = self.computeFullGrad(self.param)

    @property
    def theta(self):
        ##should be + 2 at the denominator but we are updating the nb of outer loop done at the beginning of the outer loop.
        ## so that it should be 2/(s-1+2)
        return 2 / (self.nb_outer_loop_done + 1)

    def needFullGradAndAnchorUpdate(self):
        return True

    def getNbGradComputedAndUpdateNbSeemSamples(self, X):
        return 2 * X.shape[0]

    def updateFullGradAndAnchor(self):
        self.anchor_x_tilde = torch.clone(self.next_anchor_x_tilde).to(device)
        self.resetNextAnchor()
        self.fullGradAnchor_x_tilde = self.computeFullGrad(self.anchor_x_tilde)
        self.anchor_z_bar = (
            self.anchor_x_tilde
            - self.m * self.lr / self.theta * self.fullGradAnchor_x_tilde
        )
        self.anchor_x_bar = (
            self.theta * self.anchor_z_bar + (1 - self.theta) * self.anchor_x_tilde
        )
        self.fullGradAnchor_x_bar = self.computeFullGrad(self.anchor_x_bar)
        self.nbComputedGrad += self.Xtrain.shape[0]

    def trainStep(self, batchX, batchY, batchIndex):
        # super().trainStep(batchX, batchY, batchIndex)
        self.computeAndSetGrad(batchX, batchY, batchIndex)
        self.step()
        # self.zero_grad()
        self.incrementNextAnchor_x_tilde()

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        # problem here, we should be computed the grad wrt y and not param. How can w compute y.
        self.y = (
            self.theta / self.m * self.z + (1 - self.theta / self.m) * self.anchor_x_bar
        )
        computedGrad = self.computeMeanOfBatchGrad(self.y, batchX, batchY)
        computedGradSnapshot_x_bar = self.computeMeanOfBatchGrad(
            self.anchor_x_bar, batchX, batchY
        )
        compensated_estimator = (
            self.m * self.theta * (self.anchor_z_bar - self.anchor_x_tilde) / self.lr
        )
        grad_tilde = (
            computedGrad
            - computedGradSnapshot_x_bar
            + self.fullGradAnchor_x_bar
            + compensated_estimator
        )
        self.new_z = (
            self.z
            + 2 * (self.anchor_z_bar - self.anchor_x_tilde)
            - self.lr / (self.m * self.theta) * grad_tilde
        )

    def step(self):
        with torch.no_grad():
            self.param = self.theta / self.m * (self.new_z - self.z) + self.y
        self.z = self.new_z

    def incrementNextAnchor_x_tilde(self):
        self.next_anchor_x_tilde += self.param / self.nbBatches

    def resetNextAnchor(self):
        self.next_anchor_x_tilde *= 0

    def updateSomeVariables(self):
        self.nb_outer_loop_done += 1

    @property
    def meaningfullParam(self):
        return self.anchor_x_tilde

    def setLrFromL(self):
        # no formula given in the paper, setting as LSVRG
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = 1 / (10000 * self.L)
        print("lr = ", self.lr)


class AdaSVRG(SVRG):
    NAME = "AdaSVRG"
    LINESTYLE = "solid"
    COLOR = "red"
    LFORMULA = "mean"
    GLOBALSTEPSIZE = 0.01

    def initArguments(self):
        super().initArguments()
        self.Gt = torch.zeros(self.param.shape, device=device)

    def step(self):
        self.updateGt()
        with torch.no_grad():
            self.param -= self.lr * torch.multiply(
                pseudoInverse1d(torch.sqrt(self.Gt)), self.param.grad
            )

    def updateGt(self):
        self.Gt += (self.param.grad) ** 2

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class AdaSVRGRestart(AdaSVRG):
    NAME = "AdaSVRGRestart"
    LINESTYLE = "dashed"
    COLOR = "red"
    LFORMULA = "mean"
    ADAPTIVE = True

    @property
    def needFullGradAndAnchorUpdate(self):
        self.Gt *= 0
        return True

    def initGlobalVariables(self):
        super().initGlobalVariables()
        self.nbOuterLoopsDone = 0
        self.meanOfLastEpochParam = torch.clone(self.param)

    def updateFullGradAndAnchor(self):
        self.anchor = torch.clone(self.param)
        self.fullGradAnchor = self.computeFullGrad(self.anchor)
        self.nbOuterLoopsDone += 1
        self.meanOfLastEpochParam = (
            1
            / self.nbOuterLoopsDone
            * ((self.nbOuterLoopsDone - 1) * self.meanOfLastEpochParam + self.param)
        )

    @property
    def meaningfullParam(self):
        return self.param


class MiG(SVRG):
    NAME = "MiG"
    COLOR = "lightgreen"
    LINESTYLE = "dashed"
    LFORMULA = "max"
    ADAPTIVE = False

    def initArguments(self):
        super().initArguments()
        self.xTilde = torch.clone(self.param)
        self.xTildeNext = torch.clone(self.param)
        self.outerLoopDone = 0
        self.tempSumOfParams = torch.clone(self.xTilde)

    def setLrFromL(self):
        self.L *= 1 / CONSTANT_VR_CLASSIC
        print("L", self.LFORMULA, " = ", self.L)
        print("Changing learning rate initialized at ", 1 / (8 * self.L))

    def resetXTildeNext(self):
        self.xTildeNext *= 0

    @property
    def theta(self):
        return 2 / (self.outerLoopDone + 1)

    @property
    def eta(self):
        return 1 / (4 * self.L * self.theta)

    def incrementXTilde(self):
        self.xTildeNext += self.param / self.nbBatches

    @property
    def meaningfullParam(self):
        return self.xTilde

    def step(self):
        with torch.no_grad():
            self.param -= self.eta * self.param.grad
        self.incrementXTilde()

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        y = self.theta * self.param + (1 - self.theta) * self.xTilde
        yComputedGrad = self.computeMeanOfBatchGrad(y, batchX, batchY)
        xTildeComputedGrad = self.computeMeanOfBatchGrad(self.xTilde, batchX, batchY)
        self.param.grad = yComputedGrad - xTildeComputedGrad + self.fullGradAnchor

    def updateFullGradAndAnchor(self):
        self.xTilde = torch.clone(
            self.theta * self.xTildeNext + (1 - self.theta) * self.xTilde
        )
        self.fullGradAnchor = self.computeFullGrad(self.xTilde)
        self.resetXTildeNext()
        self.outerLoopDone += 1

    @property
    def needFullGradAndAnchorUpdate(self):
        return True


class AdaVRAE(SVRG):
    NAME = "AdaVRAE"
    COLOR = "lightcoral"
    LINESTYLE = "solid"
    LFORMULA = "mean"

    def initArguments(self):
        super().initArguments()
        self.c = 3 / 2
        self.s0 = np.floor(np.log2(np.log2(4 * self.n))) + 1
        self.gamma_t = 0.01
        self.x_haut_t = torch.clone(self.param).to(device)
        self.z_t = torch.clone(self.param).to(device)
        self.anchor = torch.clone(self.param).to(device)
        self.fullGradAnchor = self.computeFullGrad(self.anchor)
        self.previous_grad = torch.clone(self.fullGradAnchor)
        self.nb_outer_loop_done = 0
        self.A_s = 5 / 4

    def setLrFromL(self):
        print("L", self.LFORMULA, " = ", self.L)
        self.lr = 1000 / (self.L)
        print("lr = ", self.lr)

    @property
    def meaningfullParam(self):
        return self.anchor

    def step(self):
        self.z_t = torch.clone(
            1
            / self.gamma_t
            * (
                self.previous_gamma_t * self.z_t
                + (self.gamma_t - self.previous_gamma_t) * self.param
                - self.a_s * self.param.grad
            )
        )

    @property
    def s(self):
        return self.nb_outer_loop_done

    def updateFullGradAndAnchor(self):
        self.nb_outer_loop_done += 1
        self.A_s -= self.n * (self.a_s) ** 2
        self.anchor = torch.clone(self.x_haut_t)

    @property
    def a_s(self):
        if self.s > self.s0:
            return (self.s - self.s0 - 1 + self.c) / (2 * self.c)
        return (4 * self.n) ** (-(0.5**self.s))

    @property
    def meaningfullParam(self):
        return self.anchor

    @property
    def paramForPlotLoss(self):
        return self.meaningfullParam

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        self.param = torch.clone(
            self.z_t - 1 / self.gamma_t * self.a_s * self.previous_grad
        )
        previous_A_s = self.A_s
        self.A_s += self.a_s * (1 + self.a_s)
        self.x_haut_t = torch.clone(
            1
            / (self.A_s)
            * (
                previous_A_s * self.x_haut_t
                + self.a_s * self.param
                + self.a_s**2 * self.anchor
            )
        )
        if self.i != self.nbBatches:
            batch_grad_x_t = self.computeMeanOfBatchGrad(self.x_haut_t, batchX, batchY)
            batch_grad_anchor = self.computeMeanOfBatchGrad(self.anchor, batchX, batchY)
            self.param.grad = torch.clone(
                batch_grad_x_t - batch_grad_anchor + self.fullGradAnchor
            )
        else:
            self.param.grad = self.computeFullGrad(self.x_haut_t)
            self.fullGradAnchor = torch.clone(self.param.grad)
        self.previous_gamma_t = self.gamma_t
        diff_grad = torch.norm(self.param.grad - self.previous_grad) ** 2
        self.gamma_t = (
            1
            / self.lr
            * torch.sqrt(self.lr**2 * self.gamma_t**2 + self.a_s**2 * diff_grad)
        )
        self.previous_grad = torch.clone(self.param.grad)


class AdaVRAG(SVRG):
    NAME = "AdaVRAG"
    COLOR = "lightcoral"
    LINESTYLE = "solid"
    LFORMULA = "mean"

    def setLrFromL(self):
        print("L", self.LFORMULA, " = ", self.L)
        self.lr = 1000 / self.L
        print("lr = ", self.lr)

    def initArguments(self):
        super().initArguments()
        self.c = (3 + math.sqrt(33)) / 4
        self.s0 = np.floor(np.log2(np.log2(4 * self.n))) + 1
        self.nb_outer_loop_done = 0
        self.gamma_t = 1e-2
        self.anchor = torch.clone(self.param).to(device)
        self.next_anchor = torch.clone(self.param).to(device)
        self.fullGrandAnchor = self.computeFullGrad(self.param)

    @property
    def s(self):
        return self.nb_outer_loop_done

    @property
    def a_s(self):
        if self.s > self.s0:
            return self.c / (self.s - self.s0 + 2 * self.c)
        return 1 - (4 * self.n) ** (-(0.5**self.s))

    @property
    def q_s(self):
        if self.s > self.s0:
            return 8 * (2 - self.a_s) * self.a_s / (3 * (1 - self.a_s))
        return 1 / ((1 - self.a_s) * self.a_s)

    def updateFullGradAndAnchor(self):
        self.anchor = torch.clone(self.next_anchor).to(device)
        self.x_haut_t = self.a_s * self.param + (1 - self.a_s) * self.anchor
        self.next_anchor *= 0
        self.fullGradAnchor = self.computeFullGrad(self.anchor)
        self.nb_outer_loop_done += 1

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        grad_x_haut_t = self.computeMeanOfBatchGrad(self.x_haut_t, batchX, batchY)
        grad_anchor = self.computeMeanOfBatchGrad(self.anchor, batchX, batchY)
        self.param.grad = grad_x_haut_t - grad_anchor + self.fullGradAnchor

    def step(self):
        self.param -= 1 / (self.gamma_t * self.q_s) * self.param.grad
        self.x_haut_t = self.a_s * self.param + (1 - self.a_s) * self.anchor
        self.gamma_t += torch.norm(
            1 / (self.gamma_t * self.q_s) * self.param.grad
        ) ** 2 / (self.lr**2)
        self.next_anchor += 1 / (self.n) * self.x_haut_t

    @property
    def meaningfullParam(self):
        return self.anchor

    @property
    def paramForPlotLoss(self):
        return self.next_anchor


class Varag(SVRG):
    NAME = "VARAG"
    COLOR = "lightgreen"
    LINESTYLE = "dashdot"
    LFORMULA = "mean"
    ADAPTIVE = False

    def initArguments(self):
        super().initArguments()
        self.initConstants()
        self.paramHaut = torch.clone(
            self.param
        )  # paramHaut is for the x with a bar above it.
        self.anchor = torch.clone(self.param)
        self.nextAnchor = torch.zeros(self.anchor.shape, device=device)
        self.fullGradAnchor = self.computeFullGrad(self.param)

    def initConstants(self):
        self.s0 = np.floor(np.log(self.n)) + 1
        self.nbOuterLoopsDone = 1
        self.alpha_s = 1 / 2
        self.p_s = 1 / 2
        self.nbInnerLoopsDone = 0  # t in the paper
        self.nbInnerLoopToDo = 1  # T_s in the paper
        self.vectorOfThetas, self.sumThetas = self.getVectorOfThetasAndSumTheta()

    def setLrFromL(self):
        self.L *= 1 / CONSTANT_VR_CLASSIC
        print("L", self.LFORMULA, " = ", self.L)
        print("lr changing, intialized at ", 4 / (3 * self.L))

    @property
    def gamma_s(self):
        return 1 / (3 * self.L * self.alpha_s)

    def getVectorOfThetasAndSumTheta(self):
        vectorOfThetas = np.repeat(
            self.gamma_s / self.alpha_s * (self.alpha_s + self.p_s),
            self.nbInnerLoopToDo - 1,
        )
        vectorOfThetas = torch.from_numpy(
            np.concatenate((vectorOfThetas, np.array([self.gamma_s / self.alpha_s])))
        )
        sumThetas = torch.sum(vectorOfThetas)
        return vectorOfThetas, sumThetas

    def beginningAnOuterLoop(self):
        return self.nbInnerLoopsDone > self.nbInnerLoopToDo - 1

    @property
    def needFullGradAndAnchorUpdate(self):
        if self.beginningAnOuterLoop():
            self.resetNbInnerLoopsDone()
            self.updateOuterLoopParameters()
            return True
        else:
            return False

    def resetNbInnerLoopsDone(self):
        self.nbInnerLoopsDone = 0

    def updateOuterLoopParameters(self):
        self.nbOuterLoopsDone += 1
        if self.nbOuterLoopsDone < self.s0 + 1:
            self.nbInnerLoopToDo *= 2
        else:
            self.alpha_s = 2 / (self.nbOuterLoopsDone - self.s0 + 4)
        self.vectorOfThetas, self.sumThetas = self.getVectorOfThetasAndSumTheta()

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        paramBas = (
            (1 - self.alpha_s - self.p_s) * self.paramHaut
            + self.alpha_s * self.param
            + self.p_s * self.anchor
        )
        gradParamBas = self.computeMeanOfBatchGrad(paramBas, batchX, batchY)
        gradAnchor = self.computeMeanOfBatchGrad(self.anchor, batchX, batchY)
        self.param.grad = gradParamBas - gradAnchor + self.fullGradAnchor

    def trainStep(self, batchX, batchY, batchIndex):
        if self.beginningAnOuterLoop() == True:
            self.stopEpoch = True
        if self.stopEpoch == True:
            return None
        else:
            self.computeAndSetGrad(batchX, batchY, batchIndex)
            self.step()
            self.updateParamHaut()
            self.incrementNextAnchor()
            self.nbInnerLoopsDone += 1
            self.zero_grad()

    def updateParamHaut(self):
        self.paramHaut = (
            (1 - self.alpha_s - self.p_s) * self.paramHaut
            + self.alpha_s * self.param
            + self.p_s * self.anchor
        )

    def incrementNextAnchor(self):
        self.nextAnchor += (
            (self.vectorOfThetas[self.nbInnerLoopsDone])
            * self.paramHaut
            / self.sumThetas
        )

    def step(self):
        self.lr = 2 * self.gamma_s
        super().step()

    def updateFullGradAndAnchor(self):
        self.anchor = torch.clone(self.nextAnchor)
        self.fullGradAnchor = self.computeFullGrad(self.anchor)
        self.resetNextAnchor()

    def resetNextAnchor(self):
        self.nextAnchor *= 0

    @property
    def meaningfullParam(self):
        return self.anchor

    def getNbGradComputedAndUpdateNbSeemSamples(self, batchX):
        if self.stopEpoch == True:
            return 0
        else:
            return super().getNbGradComputedAndUpdateNbSeemSamples(batchX)


class LSVRG(SVRG):
    NAME = "L-SVRG"
    COLOR = "lightcoral"
    LINESTYLE = "dashed"
    LFORMULA = "mean"
    ADAPTIVE = False

    def initArguments(self):
        super().initArguments()
        self.seed_number = 0

    @property
    def needFullGradAndAnchorUpdate(self):
        return False

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        toss = self.sample_uniform()

        if toss < self.batchSize / (self.n):
            self.updateFullGradAndAnchor()
            self.nbComputedGrad += self.Xtrain.shape[0]
        if (
            self.anchor is None
        ):  # until we have not computed an anchor point, we do SGD with fixed learning rate.
            self.param.grad = self.computeMeanOfBatchGrad(self.param, batchX, batchY)
        else:
            super().computeAndSetGrad(batchX, batchY, batchIndex)

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = 1 / (6 * self.L) * CONSTANT_VR_CLASSIC
        print("lr = ", self.lr)


class AdaLSVRGDiag(LSVRG):
    NAME = "AdaL-SVRG-Diagonal"
    COLOR = "blue"
    LINESTYLE = "dashed"
    LFORMULA = "max"

    def initArguments(self):
        super().initArguments()
        self.Gt = torch.zeros(self.param.shape, device=device)

    def step(self):
        self.updateGt()
        with torch.no_grad():
            self.param -= self.lr * torch.multiply(
                pseudoInverse1d(torch.sqrt(self.Gt)), self.param.grad
            )

    def updateGt(self):
        self.Gt += (self.param.grad) ** 2

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class AdaLSVRGNorm(AdaLSVRGDiag):
    NAME = "AdaL-SVRG-Norm"
    COLOR = "blue"
    LINESTYLE = "dashed"
    LFORMULA = "max"

    def initArguments(self):
        super().initArguments()
        self.Gt = 0

    def updateGt(self):
        self.Gt += torch.norm(self.param.grad) ** 2

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_NORM / (self.L)
        print("lr = ", self.lr)


class Katyusha(SVRG):
    NAME = "Katyusha"
    COLOR = "green"
    LINESTYLE = "solid"
    LFORMULA = "max"
    ADAPTIVE = False

    def initArguments(self):
        super().initArguments()
        self.nbOuterLoopDone = 0
        self.tau_2 = 1 / (2 * self.batchSize)
        self.nextAnchor = torch.clone(self.param)
        self.yk = torch.clone(self.param)
        self.zk = torch.clone(self.param)
        self.anchor = torch.clone(self.param)
        self.fullGradAnchor = None

    def setLrFromL(self):
        print("L", self.LFORMULA, " = ", self.L)
        self.lr = 1 / (3 * self.L) * CONSTANT_VR_CLASSIC
        print("lr = ", self.lr)

    @property
    def tau_1s(self):
        return 2 / (self.nbOuterLoopDone + 4)

    @property
    def alpha_s(self):
        return 1 / (3 * self.tau_1s * self.L)

    @property
    def meaningfullParam(self):
        return self.anchor

    def updateParamWithConvexCombination(self):
        with torch.no_grad():
            self.param = (
                self.tau_1s * self.zk
                + self.tau_2 * self.anchor
                + (1 - self.tau_1s - self.tau_2) * self.yk
            )

    def getGrad(self, batchX, batchY):
        oldGrad = self.computeMeanOfBatchGrad(self.anchor, batchX, batchY)
        computedGrad = self.computeMeanOfBatchGrad(self.param, batchX, batchY)
        return computedGrad - oldGrad + self.fullGradAnchor

    def updateZk(self, computedGrad):
        self.zk -= self.alpha_s * computedGrad

    def updateYk(self, computedGrad):
        self.yk = torch.clone(self.param - self.lr * computedGrad)

    def incrementNextAnchor(self):
        self.nextAnchor += self.yk

    def trainStep(self, batchX, batchY, batchIndex):
        self.updateParamWithConvexCombination()
        computedGrad = self.getGrad(batchX, batchY)
        self.updateZk(computedGrad)
        self.updateYk(computedGrad)
        self.incrementNextAnchor()

    def updateFullGradAndAnchor(self):
        self.updateAnchor()
        self.fullGradAnchor = self.computeFullGrad(self.anchor)
        self.resetNextAnchor()
        self.nbOuterLoopDone += 1

    def updateAnchor(self):
        self.anchor = torch.clone(self.nextAnchor / self.nbBatches)

    def resetNextAnchor(self):
        self.nextAnchor *= 0


## The formula is completely different for l katysha, it does not have a next anchor etc ...
class LKatyusha(SVRG):
    NAME = "L-Katyusha"
    COLOR = "green"
    LINESTYLE = "dashed"
    LFORMULA = "full"
    ADAPTIVE = False

    THETA1 = 1 / 2
    THETA2 = 1 / 2

    def initGlobalVariables(self):
        super().initGlobalVariables()
        self.eta = self.THETA1 / ((1 + self.THETA2) * self.THETA1)

    # param = y in the algorithm
    def initArguments(self):
        super().initArguments()
        self.anchor = torch.clone(self.param)
        self.z = torch.clone(self.param)
        self.fullGradAnchor = self.computeFullGrad(self.anchor)
        self.seed_number = 0

    @property
    def needFullGradAndAnchorUpdate(self):
        return False

    def trainStep(self, batchX, batchY, batchIndex):
        # if we have not yet computed a full gradient, we just perform SGD
        self.x = (
            self.THETA1 * self.z
            + self.THETA2 * self.anchor
            + (1 - self.THETA1 - self.THETA2) * self.param
        )
        computedGrad = self.computeMeanOfBatchGrad(self.param, batchX, batchY)
        computedGradSnapshot = self.computeMeanOfBatchGrad(self.anchor, batchX, batchY)
        oldZ = torch.clone(self.z)
        self.z -= self.lr * (computedGrad + self.fullGradAnchor - computedGradSnapshot)
        self.param = torch.clone(self.x + self.THETA1 * (self.z - oldZ))
        toss = self.sample_uniform()
        if toss < self.batchSize / (self.n):
            self.updateFullGradAndAnchor()
            self.nbComputedGrad += self.Xtrain.shape[0]

    def setLrFromL(self):
        print("L ", self.LFORMULA, " = ", self.L)
        self.lr = self.eta / (self.L) * CONSTANT_VR_CLASSIC
        print("lr = ", self.lr)


class SGD(myOptim):
    COLOR = "lightgrey"
    NAME = "SGD"
    LINESTYLE = "solid"
    LFORMULA = "max"
    ADAPTIVE = False

    def initArguments(self):
        super().initArguments()
        self.t = 0

    @property
    def meaningfullParam(self):
        return super().meaningfullParam

    def computeAndSetGrad(self, batchX, batchY, batchIndex=None):
        self.param.grad = self.computeMeanOfBatchGrad(self.param, batchX, batchY)

    def step(self):
        with torch.no_grad():
            self.param -= self.lr / math.sqrt(self.t + 1) * self.param.grad
        self.t += 1

    def setLrFromL(self):
        print("L", self.LFORMULA, " = ", self.L)
        self.lr = 1 / (2 * self.L)
        print(" lr = ", self.lr)


class SAGA(myOptim):
    NAME = "SAGA"
    COLOR = "lightskyblue"
    LINESTYLE = "solid"
    LFORMULA = "max"
    ADAPTIVE = False

    @property
    def meaningfullParam(self):
        return super().meaningfullParam

    def initArguments(self):
        super().initArguments()
        self.bias = 1
        self.initTable()

    def trainStep(self, batchX, batchY, batchIndex):
        self.computeAndSetGradAndUpdateTableAndFullGrad(batchX, batchY, batchIndex)
        self.step()
        self.zero_grad()

    def computeAndSetGradAndUpdateTableAndFullGrad(self, batchX, batchY, batchIndex):
        computedGrad = self.computeMeanOfBatchGrad(self.param, batchX, batchY)
        OldGrad = self.param.table[batchIndex]
        newMinusOldGrad = computedGrad - OldGrad
        if self.everySampleSeen():
            self.SAGAGradUpdate(newMinusOldGrad)
        else:
            self.param.grad = computedGrad
            # self.addComputedGradToFullGrad(computedGrad, batchIndex)
        self.updateFullGrad(batchX.shape[0], newMinusOldGrad)
        self.updateTable(batchIndex, computedGrad)

    def SAGAGradUpdate(self, newMinusOldGrad):
        self.param.grad = (
            self.bias * newMinusOldGrad
            + self.param.UnnormalizedFullGrad / self.nbBatches
        )

    def addComputedGradToFullGrad(self, computedGrad):
        self.param.UnnormalizedFullGrad += computedGrad
        ## si on tombe deux fois sur le meme, ca l'ajoute 2 fois

    def everySampleSeen(self):
        isFirstEpochDone = len(self.nbComputedGradList) > 0
        return isFirstEpochDone  # return torch.sum(self.param.table == 0) == 0

    def updateFullGrad(self, batchXshape, newMinusOldGrad):
        # one can check that updating the (unnormalized) full Gradient for SAGA
        # can be done by adding the new computed gradients and removing the old computed gradients.
        self.param.UnnormalizedFullGrad += (newMinusOldGrad).detach()

    def updateTable(self, batchIndex, computedGrad):
        self.param.table[batchIndex] = computedGrad.detach()

    def initTable(self):
        shape = list(self.param.shape)
        shape.insert(0, self.nbBatches)
        self.param.table = torch.zeros(shape, device=device)
        self.param.UnnormalizedFullGrad = torch.zeros(self.param.shape, device=device)

    def setLrFromL(self):
        print("L ", self.LFORMULA, " = ", self.L)
        self.lr = 1 / (3 * self.L) * CONSTANT_VR_CLASSIC
        print(" lr = ", self.lr)


class RMSprop(SGD):
    NAME = "RMSprop"
    COLOR = "darkslateblue"
    LINESTYLE = "dashed"
    LFORMULA = "mean"
    GLOBALSTEPSIZE = 0.01

    BETA = 0.9
    EPS = 0.000001

    def initArguments(self):
        super().initArguments()
        self.MA = torch.zeros(self.param.shape, device=device)

    def step(self):
        self.MA = self.BETA * self.MA + (1 - self.BETA) * (self.param.grad) ** 2
        with torch.no_grad():
            self.param -= self.lr / (torch.sqrt(self.MA + self.EPS)) * self.param.grad

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_PROP_AND_ADAM * CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class SAGARMSprop(SAGA):
    NAME = "RMSpropSAGA"
    COLOR = "blue"
    LINESTYLE = "solid"
    LFORMULA = "max"
    ADAPTIVE = True
    BETA = 0.99
    EPS = 0.000001
    GLOBALSTEPSIZE = 0.01

    def initArguments(self):
        super().initArguments()
        self.MA = torch.zeros(self.param.shape, device=device)

    def step(self):
        self.MA = self.BETA * self.MA + (1 - self.BETA) * (self.param.grad) ** 2
        with torch.no_grad():
            self.param -= self.lr / (torch.sqrt(self.MA + self.EPS)) * self.param.grad

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_PROP_AND_ADAM * CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class Newton(myOptim):
    COLOR = "black"
    LINESTYLE = "solid"
    NAME = "Newton"
    LFORMULA = "None"
    ADAPTIVE = False

    @property
    def paramForPlotLoss(self):
        return self.param

    @property
    def meaningfullParam(self):
        return super().meaningfullParam

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        print("using Newton")
        computedGrad = grad(self.computeLoss)(self.param, self.Xtrain, self.Ytrain)
        hess = hessian(self.computeLoss)(self.param, self.Xtrain, self.Ytrain)
        try:
            self.param.grad = torch.linalg.solve(hess, computedGrad)
            self.managed_to_invers_hessian = True
        except:
            self.param.grad = self.lr * computedGrad
            self.managed_to_invers_hessian = False
            print(
                f"Can't inverse the hessian, taking 1/100*grad. Dimension ={self.dim},n={self.n}"
            )

    def getNbGradComputedAndUpdateNbSeemSamples(self, X):
        return self.Xtrain.shape[0]

    def initGlobalVariables(self):
        super().initGlobalVariables()
        self.managed_to_invers_hessian = False

    def train(self, nbGradToCompute, batchSize, L):
        super().train(nbGradToCompute, batchSize, L)
        if self.managed_to_invers_hessian is False:
            self.trainLossList[-1] = 1e10
        print("Managed to invert the hessian = ", self.managed_to_invers_hessian)


class LBFGS(myOptim):
    COLOR = "black"
    LINESTYLE = "dashed"
    NAME = "LBFGS"
    LFORMULA = "mean"
    ADAPTIVE = False
    BETA = 0.9
    BETAPRIME = 10e-4
    M = 3

    @property
    def meaningfullParam(self):
        return super().meaningfullParam

    def initGlobalVariables(self):
        super().initGlobalVariables()
        self.alphas = [0] * (self.M)
        self.ss = list()
        self.rhos = list()
        self.ys = list()

    def computeAndSetGrad(self, batchX, batchY, batchIndex):
        computedGrad = grad(self.computeLoss)(self.param, self.Xtrain, self.Ytrain)
        ## compute the required variables if we have done more than one iteration
        if len(self.ss) > 0:
            self.ys.append(torch.clone(computedGrad - self.oldGrad))
            self.rhos.append(
                torch.nan_to_num(
                    1 / (self.ys[-1] @ (self.ss[-1])),
                    posinf=3.7e33,
                    nan=3.7e33,
                    neginf=-3.7e33,
                )
            )
        self.removeUnwantedVariables()
        self.oldGrad = torch.clone(computedGrad)
        q = self.getQAndalphas(computedGrad)
        H0 = self.gammak * torch.eye(self.param.shape[0], device=device)
        z = H0 @ q
        self.param.grad = self.computeDescentDirection(z)

    def removeUnwantedVariables(self):
        if len(self.ss) > self.M:
            self.ys.pop(0)
            self.rhos.pop(0)
            self.ss.pop(0)

    def computeDescentDirection(self, z):
        for i in range(0, len(self.ss)):
            beta = self.rhos[i] * (self.ys[i] @ z)
            z += self.ss[i] * (self.alphas[i] - beta)
        return z

    def getQAndalphas(self, q):
        for j in range(0, len(self.ss)):
            i = len(self.ss) - 1 - j
            self.alphas[i] = self.rhos[i] * (self.ss[i] @ q)
            q -= self.alphas[i] * self.ys[i]
        return q

    @property
    def gammak(self):
        try:
            gamma_k = (self.ss[-1] @ self.ys[-1]) / (torch.norm(self.ys[-1]) ** 2)
            return torch.nan_to_num(gamma_k, nan=1)
            return (self.ss[-1] @ self.ys[-1]) / (torch.norm(self.ys[-1]) ** 2)
        except:
            print("Error, returning 1 for gammak")
            return 1

    def getNbGradComputedAndUpdateNbSeemSamples(self, X):
        return self.Xtrain.shape[0]

    def step(self):
        self.oldParam = torch.clone(self.param)
        with torch.no_grad():
            self.param -= self.param.grad
        self.ss.append(self.param - self.oldParam)


class AdaDiag(SGD):
    NAME = "AdaGrad-Diagonal"
    COLOR = "darkslateblue"
    LINESTYLE = "dashed"
    LFORMULA = "mean"
    ADAPTIVE = True
    GLOBALSTEPSIZE = 0.01

    def initArguments(self):
        super().initArguments()
        self.Gt = torch.zeros(self.param.shape, device=device)

    def step(self):
        self.updateGt()
        with torch.no_grad():
            self.param -= self.lr * torch.multiply(
                pseudoInverse1d(torch.sqrt(self.Gt)), self.param.grad
            )

    def updateGt(self):
        self.Gt += (self.param.grad) ** 2

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class SAGAdaDiag(SAGA):
    NAME = "AdaSAGA-Diagonal"
    LINESTYLE = "dashdot"
    COLOR = "orange"
    LFORMULA = "max"
    ADAPTIVE = True
    GLOBALSTEPSIZE = 0.01

    def initArguments(self):
        super().initArguments()
        self.Gt = torch.zeros(self.param.shape, device=device)

    def step(self):
        self.updateGt()
        with torch.no_grad():
            self.param -= self.lr * torch.multiply(
                pseudoInverse1d(torch.sqrt(self.Gt)), self.param.grad
            )

    def updateGt(self):
        self.Gt += (self.param.grad) ** 2

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class SAGAdaDiagRestart(SAGAdaDiag):
    NAME = "AdaSAGArestart"
    LFORMULA = "mean"

    @property
    def needFullGradAndAnchorUpdate(self):
        self.Gt *= 0
        return False


class Adam(SGD):
    NAME = "Adam"
    COLOR = "darkslateblue"
    LINESTYLE = "dashdot"
    LFORMULA = "mean"
    GLOBALSTEPSIZE = 0.001

    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 10 ** (-8)

    def initArguments(self):
        super().initArguments()
        self.firstMoment = 0
        self.secondMoment = 0
        self.t = 0

    def step(self):
        self.t += 1
        self.firstMoment = (
            self.BETA1 * self.firstMoment + (1 - self.BETA1) * self.param.grad
        )
        self.secondMoment = (
            self.BETA2 * self.secondMoment + (1 - self.BETA2) * self.param.grad**2
        )
        biasCorrectedFirstMoment = self.firstMoment / (1 - self.BETA1 ** (self.t))
        biasCorrectedSecondMoment = self.secondMoment / (1 - self.BETA2 ** (self.t))
        with torch.no_grad():
            self.param -= (
                self.lr
                * biasCorrectedFirstMoment
                / (torch.sqrt(biasCorrectedSecondMoment) + self.EPS)
            )

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_PROP_AND_ADAM * CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class RMSpropLSVRG(LSVRG):
    NAME = "RMSpropL-SVRG"
    COLOR = "darkslateblue"
    LINESTYLE = "dashed"
    LFORMULA = "mean"
    GLOBALSTEPSIZE = 0.01

    BETA = 0.9
    EPS = 0.000001

    def initArguments(self):
        super().initArguments()
        self.MA = torch.zeros(self.param.shape, device=device)

    def step(self):
        self.MA = self.BETA * self.MA + (1 - self.BETA) * (self.param.grad) ** 2
        with torch.no_grad():
            self.param -= self.lr / (torch.sqrt(self.MA + self.EPS)) * self.param.grad

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_PROP_AND_ADAM * CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class AdamLSVRG(LSVRG):
    NAME = "AdamL-SVRG"
    COLOR = "darkslateblue"
    LINESTYLE = "dashdot"
    LFORMULA = "mean"
    GLOBALSTEPSIZE = 0.001

    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 10 ** (-8)

    def initArguments(self):
        super().initArguments()
        self.firstMoment = 0
        self.secondMoment = 0
        self.t = 0

    def step(self):
        self.t += 1
        self.firstMoment = (
            self.BETA1 * self.firstMoment + (1 - self.BETA1) * self.param.grad
        )
        self.secondMoment = (
            self.BETA2 * self.secondMoment + (1 - self.BETA2) * self.param.grad**2
        )
        biasCorrectedFirstMoment = self.firstMoment / (1 - self.BETA1 ** (self.t))
        biasCorrectedSecondMoment = self.secondMoment / (1 - self.BETA2 ** (self.t))
        with torch.no_grad():
            self.param -= (
                self.lr
                * biasCorrectedFirstMoment
                / (torch.sqrt(biasCorrectedSecondMoment) + self.EPS)
            )

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_PROP_AND_ADAM * CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class SAGAdam(SAGA):
    NAME = "AdamSAGA"
    COLOR = "darkslateblue"
    LINESTYLE = "solid"
    LFORMULA = "max"
    GLOBALSTEPSIZE = 0.001

    BETA1 = 0.9
    BETA2 = 0.999
    EPS = 10 ** (-8)

    def initArguments(self):
        super().initArguments()
        self.firstMoment = 0
        self.secondMoment = 0
        self.t = 0

    def step(self):
        self.t += 1
        self.firstMoment = (
            self.BETA1 * self.firstMoment + (1 - self.BETA1) * self.param.grad
        )
        self.secondMoment = (
            self.BETA2 * self.secondMoment + (1 - self.BETA2) * self.param.grad**2
        )
        biasCorrectedFirstMoment = self.firstMoment / (1 - self.BETA1 ** (self.t))
        biasCorrectedSecondMoment = self.secondMoment / (1 - self.BETA2 ** (self.t))
        with torch.no_grad():
            self.param -= (
                self.lr
                * biasCorrectedFirstMoment
                / (torch.sqrt(biasCorrectedSecondMoment) + self.EPS)
            )

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_PROP_AND_ADAM * CONSTANT_TIMES_1_OVER_L / (self.L)
        print("lr = ", self.lr)


class SAG(SAGA):
    NAME = "SAG"
    COLOR = "lightskyblue"
    LINESTYLE = "dashed"
    LFORMULA = "max"
    ADAPTIVE = False

    def initArguments(self):
        super().initArguments()
        self.bias = 1 / (
            self.n
        )  # SAG differs from SAGA only in the update rule, where the bias is more important.

    def setLrFromL(self):
        print(" L ", self.LFORMULA, " = ", self.L)
        self.lr = 1 / (16 * self.L)
        print("lr = ", self.lr)


class AdaNorm(AdaDiag):
    NAME = "AdaGrad-Norm"
    COLOR = "olive"
    ADAPTIVE = True

    def initArguments(self):
        super().initArguments()
        self.Gt = 0

    def updateGt(self):
        self.Gt += torch.norm(self.param.grad) ** 2

    def setLrFromL(self):
        print(" L ", self.LFORMULA, " = ", self.L)
        self.lr = CONSTANT_NORM / self.L
        print("lr = ", self.lr)


class SAGAdaNorm(SAGAdaDiag):
    NAME = "AdaSAGA-Norm"
    COLOR = "blueviolet"
    LINESTYLE = "dashed"
    LFORMULA = "mean"
    ADAPTIVE = True
    GLOBALSTEPSIZE = 0.01

    def initArguments(self):
        super().initArguments()
        self.Gt = 0

    def updateGt(self):
        self.Gt += torch.norm(self.param.grad) ** 2

    def setLrFromL(self):
        print("Formula for L:", self.LFORMULA, "L = ", self.L)
        self.lr = CONSTANT_NORM / (self.L)
        print("lr = ", self.lr)


class copyOptim:
    def __init__(self, optim: myOptim):
        self.LINESTYLE = optim.LINESTYLE
        self.COLOR = optim.COLOR
        self.trainLossList = optim.trainLossList
        self.test_metric_list = optim.test_metric_list
        self.cumsumComputedGradList = optim.cumsumComputedGradList
        self.NAME = optim.NAME
        self.pbDescriptor = optim.pbDescriptor
