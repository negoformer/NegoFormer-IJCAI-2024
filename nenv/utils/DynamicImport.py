import importlib
from nenv.Agent import AgentClass, AbstractAgent
from nenv.OpponentModel import OpponentModelClass, AbstractOpponentModel
from nenv.logger import LoggerClass, AbstractLogger


def load_agent_class(class_path: str) -> AgentClass:
    if "." in class_path:
        modules = class_path.split(".")

        path = ".".join(modules[:-1])
        class_name = modules[-1]

        agent_class = getattr(importlib.import_module(path), class_name)

        assert issubclass(agent_class, AbstractAgent), f"{class_path} is not a subclass of AbstractAgent class."

        return agent_class
    else:
        agent_class = getattr(importlib.import_module("agents"), class_path)

        assert issubclass(agent_class, AbstractAgent), f"agents.{class_path} is not a subclass of AbstractAgent class."

        return agent_class


def load_estimator_class(class_path: str) -> OpponentModelClass:
    if "." in class_path:
        modules = class_path.split(".")

        path = ".".join(modules[:-1])
        class_name = modules[-1]

        opponent_model_class = getattr(importlib.import_module(path), class_name)

        assert issubclass(opponent_model_class, AbstractOpponentModel), f"{class_path} is not a subclass of AbstractOpponentModel class."

        return opponent_model_class
    else:
        opponent_model_class = getattr(importlib.import_module("nenv.OpponentModel"), class_path)

        assert issubclass(opponent_model_class, AbstractOpponentModel), f"nenv.OpponentModel.{class_path} is not a subclass of AbstractOpponentModel class."

        return opponent_model_class


def load_logger_class(class_path: str) -> LoggerClass:
    if "." in class_path:
        modules = class_path.split(".")

        path = ".".join(modules[:-1])
        class_name = modules[-1]

        logger_class = getattr(importlib.import_module(path), class_name)

        assert issubclass(logger_class, AbstractLogger), f"{class_path} is not a subclass of AbstractLogger class."

        return logger_class
    else:
        logger_class = getattr(importlib.import_module("nenv.logger"), class_path)

        assert issubclass(logger_class, AbstractLogger), f"nenv.logger.{class_path} is not a subclass of AbstractLogger class."

        return logger_class
