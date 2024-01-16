import typing

from nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel

# Type variable of AbstractOpponentModel class to declare a type for a variable
OpponentModelClass = typing.TypeVar('OpponentModelClass', bound=AbstractOpponentModel.__class__)

from nenv.OpponentModel.EstimatedPreference import EstimatedPreference
from nenv.OpponentModel.ClassicFrequencyOpponentModel import ClassicFrequencyOpponentModel
from nenv.OpponentModel.FrequencyWindowOpponentModel import FrequencyWindowOpponentModel
from nenv.OpponentModel.BayesianOpponentModel import BayesianOpponentModel
from nenv.OpponentModel.UncertaintyOpponentModel import UncertaintyOpponentModel
