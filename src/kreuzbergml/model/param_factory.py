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
