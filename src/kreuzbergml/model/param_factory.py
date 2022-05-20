from abc import ABC, abstractmethod
from typing import Dict, Iterable

import numpy as np
import sklearn


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
            "solver": ["lbfgs", "liblinear"],
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
            "max_depth": list(range(4,10)),
            "n_estimator": [8, 16, 32, 64, 100, 200, 500, 1000],
            "min_samples_split": [2, 3, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"],
        }
        return param_dict

class SVCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.svm.SVC

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "kernel": ["rbf"],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
            "C": [0.1, 1, 10, 100, 1000],
            "probability": [True, False],
        }
        return param_dict

class LinSVCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.svm.LinearSVC

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "penalty": ["l1", "l2"],
            "dual": [True, False],
            "C": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000, 10000, 100000],
        }
        return param_dict

class MLPCParamsFactory(AbstractGridSearchParamsFactory):
    def get_model_class(self):
        return sklearn.neural_network.MLPClassifier

    def get_param_dict(self) -> Dict[str, Iterable]:
        param_dict = {
            "hidden_layer_sizes": [(10,), (20,), (100,), (10, 30, 10), (50,50,50), (50,100,50)],
            "activation": ["tanh", "relu", "logistic"],
            "solver": ["sgd", "adam"],
            "alpha": [0.0001, 0.001, 0.01, 0.05],
            "learning_rate": ["constant", "adaptive"],
        }
        return param_dict
