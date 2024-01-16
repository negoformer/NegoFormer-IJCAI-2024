import datetime
import math
import os
import random
import shutil
import time
import warnings
from typing import Union, Set, List, Tuple
import numpy as np
import pandas as pd
from nenv.Agent import AgentClass
from nenv.logger import AbstractLogger, LoggerClass
from nenv.OpponentModel import OpponentModelClass
from nenv.SessionRunner import SessionRunner
from nenv.utils import ExcelLog


AGENT_STORAGE_DIR = "agent_storage/"


class Tournament:
    agent_classes: Set[AgentClass]
    loggers: List[AbstractLogger]
    domains: List[str]
    estimators: Set[OpponentModelClass]
    deadline_time: Union[int, None]
    deadline_round: Union[int, None]
    result_dir: str
    seed: Union[int, None]
    shuffle: bool
    repeat: int
    self_negotiation: bool

    def __init__(self, agent_classes: Union[List[AgentClass], Set[AgentClass]],
                 domains: List[str],
                 logger_classes: Union[List[LoggerClass], Set[LoggerClass]],
                 estimator_classes: Union[List[OpponentModelClass], Set[OpponentModelClass]],
                 deadline_time: Union[int, None],
                 deadline_round: Union[int, None],
                 self_negotiation: bool = False,
                 repeat: int = 1,
                 result_dir: str = "results/",
                 seed: Union[int, None] = None,
                 shuffle: bool = False
                 ):
        """
            This class conducts a negotiation tournament.
        :param agent_classes: List of agent classes (i.e., subclass of AbstractAgent class)
        :param domains: List of domains
        :param logger_classes: List of loggers classes (i.e., subclass of AbstractLogger class)
        :param estimator_classes: List of estimator classes (i.e, subclass of AbstractOpponentModel class)
        :param deadline_time: Time-based deadline in terms of seconds
        :param deadline_round: Round-based deadline in terms of number of rounds
        :param self_negotiation: Whether the agents negotiate with itself. Default false.
        :param repeat: Number of repetition. Default 1.
        :param result_dir: The result directory that the tournament logs will be created. Default 'results/'
        :param seed: Setting seed for whole tournament. Default None.
        :param shuffle: Whether shuffle negotiation combinations. Default False
        """

        assert deadline_time is not None or deadline_round is not None, "No deadline type is specified."
        assert deadline_time is None or deadline_time > 0, "Deadline must be positive."
        assert deadline_round is None or deadline_round > 0, "Deadline must be positive."

        if repeat <= 0:
            warnings.warn("repeat is set to 1.")
            repeat = 1

        assert len(agent_classes) > 0, "Empty list of agent classes."
        assert len(domains) > 0, "Empty list of domains."

        self.agent_classes = agent_classes
        self.domains = domains
        self.estimators = estimator_classes
        self.deadline_time = deadline_time
        self.deadline_round = deadline_round
        self.loggers = [logger_class(result_dir) for logger_class in set(logger_classes)]
        self.result_dir = result_dir
        self.seed = seed
        self.repeat = repeat
        self.self_negotiation = self_negotiation
        self.shuffle = shuffle

    def run(self):
        """
            This method starts the tournament
        :return: Nothing
        """
        # Set seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Create directory
        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir)

        os.mkdir(self.result_dir)
        os.mkdir(os.path.join(os.path.join(self.result_dir, "sessions/")))

        # Clean agent storage directory
        if os.path.exists(AGENT_STORAGE_DIR):
            shutil.rmtree(AGENT_STORAGE_DIR)

        os.mkdir(AGENT_STORAGE_DIR)

        # Extract domain information into the result directory
        self.extract_domains()

        # Get all combinations
        negotiations = self.generate_combinations()

        # Names for logger
        agent_names = []
        estimator_names = []

        # Tournament log file
        tournament_logs = ExcelLog(["TournamentResults"])

        tournament_logs.save(os.path.join(self.result_dir, "results.xlsx"))

        tournament_start_time = time.time()

        print(f'Started at {str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}.')
        print("Total negotiation:", len(negotiations))

        print("*" * 50)

        for i, (agent_class_1, agent_class_2, domain_name) in enumerate(negotiations):
            # Start session
            session_runner = SessionRunner(agent_class_1, agent_class_2, domain_name, self.deadline_time, self.deadline_round, list(self.estimators), self.loggers)

            session_path = "%s_%s_Domain%s.xlsx" % \
                           (session_runner.agentA.name, session_runner.agentB.name, domain_name)

            session_start_time = time.time()
            tournament_logs.append(session_runner.run(os.path.join(self.result_dir, "sessions/", session_path)))
            session_end_time = time.time()

            # Update total elapsed time
            session_elapsed_time = session_end_time - session_start_time

            tournament_logs.update({"TournamentResults": {"SessionRealTime": session_elapsed_time}})

            # Get list of name for loggers
            if len(estimator_names) == 0:
                estimator_names = [estimator.name for estimator in session_runner.agentA.estimators]

            if session_runner.agentA.name not in agent_names:
                agent_names.append(session_runner.agentA.name)

            if session_runner.agentB.name not in agent_names:
                agent_names.append(session_runner.agentB.name)

            # Remaining time estimation
            completed_percentage = (i + 1) / len(negotiations)

            elapsed_time = time.time() - tournament_start_time

            remaining_time = math.ceil((1 - completed_percentage) * elapsed_time / completed_percentage)

            print(session_runner.agentA.name, "vs.", session_runner.agentB.name, "in Domain:", domain_name,
                  "\t-\tSession Real Time:", str(datetime.timedelta(seconds=math.ceil(session_elapsed_time))),
                  "\t-\tProcess: %.2f %%" % (completed_percentage * 100.),
                  "\t-\tEstimated Remaining Time:", str(datetime.timedelta(seconds=remaining_time)),
                  "\t-\tElapsed Time:", str(datetime.timedelta(seconds=math.ceil(elapsed_time))),
                  "\t-\tLast Update:", str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

        print("*" * 50)
        print("Tournament has been done. Please, wait for analysis...")

        # Backup
        tournament_logs.save(os.path.join(self.result_dir, "results_backup.xlsx"))

        # On tournament end
        for logger in self.loggers:
            logger.on_tournament_end(tournament_logs, agent_names, self.domains, estimator_names)

        # Save tournament logs
        tournament_logs.save(os.path.join(self.result_dir, "results.xlsx"))

        print("Analysis have been completed.")
        print("*" * 50)

        print("Total Elapsed Time:", str(datetime.timedelta(seconds=math.ceil(time.time() - tournament_start_time))))

    def generate_combinations(self) -> List[Tuple[AgentClass, AgentClass, str]]:
        """
            This method generates all combinations of negotiations.
        :return: Nothing
        """
        combinations = []

        for domain in self.domains:
            for agent_class_1 in self.agent_classes:
                for agent_class_2 in self.agent_classes:
                    if not self.self_negotiation and agent_class_1.__name__ == agent_class_2.__name__:
                        continue

                    for i in range(self.repeat):
                        combinations.append((agent_class_1, agent_class_2, domain))

        if self.shuffle:
            random.shuffle(combinations)

        return combinations

    def extract_domains(self):
        """
            This method extracts the domain information into the result directory.
        :return: Nothing
        """
        full_domains = pd.read_csv("domains/domains.csv", sep=";")

        domains = pd.DataFrame(columns=full_domains.columns[1:])

        for i, row in full_domains.iterrows():
            if str(row["DomainID"]) in self.domains:
                domains = domains.append(row)

        domains.to_csv(os.path.join(self.result_dir, "domains.csv"), sep=";")
