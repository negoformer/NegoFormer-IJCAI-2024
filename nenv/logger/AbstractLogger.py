from typing import List, Union
from nenv.Session import Session
from nenv.SessionEstimator import SessionEstimator
from nenv.Preference import Bid
from nenv.utils import ExcelLog
from nenv.utils.ExcelLog import LogRow
from abc import ABC
import os
import pandas as pd


class AbstractLogger(ABC):
    """
        The loggers work as event handler.
        In each event, it should return the corresponding log as dictionary to append into the log file.
    """
    log_dir: str  # The log directory

    def __init__(self, log_dir: str):
        """
            Constructor
        :param log_dir: The log directory 
        """
        self.log_dir = log_dir

        self.initiate()

    def initiate(self):
        """
            This method is for initiating the logger before the tournament starts.
        :return: Nothing
        """

        pass

    def before_session_start(self, session: Union[Session, SessionEstimator]) -> List[str]:
        """
            This method is for initiating the logger with the given negotiation session information.
            Also, this method provides the required sheet names
        :param session: Current negotiation session
        :return: List of sheet names
        """
        return []

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        """
            This method will be called when an agent offers.
        :param agent: The agent who offered
        :param offer: The offered bid
        :param time: Current negotiation time
        :param session: Current negotiation session
        :return: LogRow to append into the session log file
        """
        return {}

    def on_accept(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        """
            This method will be called when an agent accept an offer.
        :param agent: The agent who accepted
        :param offer: The accepted bid
        :param time: Acceptance time
        :param session: Current negotiation session
        :return: LogRow to append into the session log file
        """
        return {}

    def on_fail(self, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        """
            This method will be called when the negotiation ends without any acceptance.
        :param time: End time
        :param session: Current negotiation session
        :return: LogRow to append into the session log file
        """
        return {}

    def on_session_end(self, final_row: LogRow, session: Union[Session, SessionEstimator]) -> LogRow:
        """
            This method will be called when the negotiation session ends.
            This method generate a log for tournament log file.
        :param final_row: The final log file. It includes the failure or acceptance logs.
        :param session: Current negotiation session
        :return: LogRow to append into the tournament log file
        """
        return {}

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str], estimator_names: List[str]):
        """
            This method will be called when the tournament ends.
        :param tournament_logs: Whole tournament logs
        :param agent_names: List of agent name in the tournament
        :param domain_names: List of domain names in the tournament
        :param estimator_names: List of estimator name (i.e., opponent model) in the tournament
        :return: Nothing
        """
        pass

    def get_path(self, file_name: str) -> str:
        """
            This method generates the full path for given file name.
        :param file_name: File name in log directory
        :return: The full path
        """
        return os.path.join(self.log_dir, file_name)
