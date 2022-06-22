from abc import ABC, abstractmethod
from typing import Dict, Iterable

import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neighbors
import sklearn.neural_network
import sklearn.svm
import sklearn.tree


class AbstractGridSearchParamsFactory(ABC):
    @abstractmethod
    def get_model_class(self):
        pass

    @abstractmethod
    def get_param_dict(self) -> Dict[str, Iterable]:
        pass


class L2RegularizedLRParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.linear_model.LogisticRegression

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "C": list(range(11)),
        }
        return param_dict


class ENRegularizedLRParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.linear_model.LogisticRegression

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "penalty": ["elasticnet"],
            "solver": ["saga"],
            "C": list(range(11)),
            "l1_ratio": list(np.linspace(0, 1, num=5)),
        }
        return param_dict


class RFCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.ensemble.RandomForestClassifier

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "max_depth": list(range(4, 10)),
            "n_estimators": [2**i for i in range(3, 7)],
            "min_samples_split": [2, 3, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        return param_dict


class SVCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.svm.SVC

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "kernel": ["rbf", "poly", "sigmoid"],
            "gamma": np.logspace(-3, 1, num=5),
            "C": np.logspace(-1, 2, num=4),
        }
        return param_dict


class LinSVCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.svm.LinearSVC

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "penalty": ["l1", "l2"],
            "dual": [True, False],
            "C": np.logspace(-5, 5, num=11),
        }
        return param_dict


class MLPCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.neural_network.MLPClassifier

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "hidden_layer_sizes": [
                (10,),
                (20,),
                (100,),
                (10, 30, 10),
                (50, 50, 50),
                (50, 100, 50),
            ],
            "activation": ["tanh", "relu", "logistic"],
            "alpha": np.logspace(-4, -2, num=7),
            "learning_rate": ["constant", "adaptive"],
        }
        return param_dict


class DTCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.tree.DecisionTreeClassifier

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "criterion": ["gini", "entropy"],
            "max_depth": list(range(3, 10)),
            "min_samples_split": list(range(2, 10)),
            "min_samples_leaf": list(range(1, 5)),
        }
        return param_dict


class KNCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.neighbors.KNeighborsClassifier

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "n_neighbors": list(range(1, 30, 3)),
            "p": list(range(1, 5)),
            "leaf_size": list(range(1, 50, 5)),
            "weights": ["uniform", "distance"],
        }
        return param_dict
