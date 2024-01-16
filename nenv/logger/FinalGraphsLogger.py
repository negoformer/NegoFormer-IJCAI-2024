import pandas as pd

from nenv.logger.AbstractLogger import *
from tournament_graphs import draw_heatmap, plt
from typing import List
import numpy as np


class FinalGraphsLogger(AbstractLogger):
    """
        FinalGraphsLogger draw plots and Confusion Matrix for the evaluation of the Agent performance at the end of the
        negotiation.
    """
    domain_sep: int

    def __init__(self, log_dir: str, domain_sep: int = 15):
        super().__init__(log_dir)

        self.domain_sep = domain_sep

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str],
                          estimator_names: List[str]):
        tournament_results = tournament_logs.to_data_frame("TournamentResults")

        if not os.path.exists(self.get_path("tournament_graphs/")):
            os.mkdir(self.get_path("tournament_graphs/"))

        self.draw_agent_agent(tournament_results, agent_names, self.get_path("tournament_graphs/"))

        for i in range(0, len(domain_names), self.domain_sep):
            end = min(len(domain_names), i + self.domain_sep)

            idx = domain_names[i:end]

            self.draw_agent_domain(tournament_results, agent_names, idx, self.get_path("tournament_graphs/"),
                                   f"({i}_{end})")

    def draw_agent_agent(self, tournament_results: pd.DataFrame, agent_names: list, directory: str):
        data_utility = [[0. for j in range(len(agent_names))] for i in range(len(agent_names))]
        data_opp_utility = [[0. for j in range(len(agent_names))] for i in range(len(agent_names))]
        data_nash_product = [[0. for j in range(len(agent_names))] for i in range(len(agent_names))]
        data_nash_distances = [[0. for j in range(len(agent_names))] for i in range(len(agent_names))]
        data_social_welfare = [[0. for j in range(len(agent_names))] for i in range(len(agent_names))]
        data_time = [[0. for j in range(len(agent_names))] for i in range(len(agent_names))]
        data_acceptance_rate = [[0. for j in range(len(agent_names))] for i in range(len(agent_names))]

        for i, agent_name_a in enumerate(agent_names):
            for j, agent_name_b in enumerate(agent_names):
                rows_utility = tournament_results.loc[(tournament_results["AgentA"] == agent_name_a) &
                                                      (tournament_results["AgentB"] == agent_name_b),
                                                      "AgentAUtility"].to_list()

                rows_utility.extend(tournament_results.loc[(tournament_results["AgentA"] == agent_name_b) &
                                                           (tournament_results["AgentB"] == agent_name_a),
                                                           "AgentBUtility"].to_list())

                rows_opp_utility = tournament_results.loc[(tournament_results["AgentA"] == agent_name_a) &
                                                          (tournament_results["AgentB"] == agent_name_b),
                                                          "AgentBUtility"].to_list()

                rows_opp_utility.extend(tournament_results.loc[(tournament_results["AgentA"] == agent_name_b) &
                                                               (tournament_results["AgentB"] == agent_name_a),
                                                               "AgentAUtility"].to_list())

                row_nash_distances = tournament_results.loc[(tournament_results["AgentA"] == agent_name_a) &
                                                            (tournament_results["AgentB"] == agent_name_b),
                                                            "NashDistance"].to_list()

                row_nash_distances.extend(tournament_results.loc[(tournament_results["AgentA"] == agent_name_b) &
                                                                 (tournament_results["AgentB"] == agent_name_a),
                                                                 "NashDistance"].to_list())

                rows_acceptance = tournament_results.loc[(tournament_results["AgentA"] == agent_name_a) &
                                                         (tournament_results["AgentB"] == agent_name_b) &
                                                         (tournament_results["Result"] == "Acceptance"),
                                                         "Time"].to_list()

                rows_acceptance.extend(tournament_results.loc[(tournament_results["AgentA"] == agent_name_b) &
                                                              (tournament_results["AgentB"] == agent_name_a) &
                                                              (tournament_results["Result"] == "Acceptance"),
                                                              "Time"].to_list())

                if len(rows_utility) == 0:
                    continue
                else:
                    data_utility[i][j] = np.mean(rows_utility)

                    data_opp_utility[i][j] = np.mean(rows_opp_utility)

                    data_nash_product[i][j] = np.mean(
                        [rows_utility[k] * rows_opp_utility[k] for k in range(len(rows_utility))])

                    data_social_welfare[i][j] = np.mean(
                        [rows_utility[k] + rows_opp_utility[k] for k in range(len(rows_utility))])

                    data_nash_distances[i][j] = np.mean(row_nash_distances)

                    data_time[i][j] = np.mean(rows_acceptance)

                    data_acceptance_rate[i][j] = len(rows_acceptance) / len(rows_utility)

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, agent_names, data_utility)

        draw_heatmap(sorted_map, sorted_agent_names, sorted_agent_names,
                     os.path.join(directory, "agent_agent_utility.png"), "Opponent", "Agent")

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, agent_names, data_opp_utility)

        draw_heatmap(sorted_map, sorted_agent_names, sorted_agent_names,
                     os.path.join(directory, "agent_opponent_utility.png"), "Opponent", "Agent")

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, agent_names, data_nash_product)

        draw_heatmap(sorted_map, sorted_agent_names, sorted_agent_names,
                     os.path.join(directory, "agent_agent_nash_product.png"), "Opponent", "Agent")

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, agent_names, data_social_welfare)

        draw_heatmap(sorted_map, sorted_agent_names, sorted_agent_names,
                     os.path.join(directory, "agent_agent_social_welfare.png"), "Opponent", "Agent", vmax=True)

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, agent_names, data_nash_distances, descending=False)

        draw_heatmap(sorted_map, sorted_agent_names, sorted_agent_names,
                     os.path.join(directory, "agent_agent_nash_distances.png"), "Opponent", "Agent", reverse=True,
                     vmax=True)

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, agent_names, data_time)

        draw_heatmap(sorted_map, sorted_agent_names, sorted_agent_names,
                     os.path.join(directory, "agent_agent_time.png"), "Opponent", "Agent")

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, agent_names, data_acceptance_rate)

        draw_heatmap(sorted_map, sorted_agent_names, sorted_agent_names,
                     os.path.join(directory, "agent_agent_acceptance_rate.png"), "Opponent", "Agent", fmt=".0%")

    def draw_agent_domain(self, tournament_results: pd.DataFrame, agent_names: list, domain_names: list, directory: str,
                          domain_set: str = ""):
        data_utility = []
        data_time = []
        data_acceptance_rate = []
        domain_names = ["Domain%s" % i for i in domain_names]

        for i, agent_name in enumerate(agent_names):
            data_utility.append([])
            data_time.append([])
            data_acceptance_rate.append([])
            for j in domain_names:
                rows_utility = tournament_results.loc[(tournament_results["AgentA"] == agent_name) &
                                                      (tournament_results["DomainID"] == j),
                                                      "AgentAUtility"].to_list()

                rows_utility.extend(tournament_results.loc[(tournament_results["AgentB"] == agent_name) &
                                                           (tournament_results["DomainID"] == j),
                                                           "AgentBUtility"].to_list())

                rows_time = tournament_results.loc[(tournament_results["AgentA"] == agent_name) &
                                                   (tournament_results["AgentB"] != agent_name) &
                                                   (tournament_results["DomainID"] == j),
                                                   "Time"].to_list()

                rows_time.extend(tournament_results.loc[(tournament_results["AgentB"] == agent_name) &
                                                        (tournament_results["AgentA"] != agent_name) &
                                                        (tournament_results["DomainID"] == j),
                                                        "Time"].to_list())

                rows_time.extend(tournament_results.loc[(tournament_results["AgentB"] == agent_name) &
                                                        (tournament_results["AgentA"] == agent_name) &
                                                        (tournament_results["DomainID"] == j),
                                                        "Time"].to_list())

                rows_acceptance = tournament_results.loc[(tournament_results["AgentA"] == agent_name) &
                                                         (tournament_results["AgentB"] != agent_name) &
                                                         (tournament_results["DomainID"] == j) &
                                                         (tournament_results["Result"] == "Acceptance"),
                                                         "Time"].to_list()

                rows_acceptance.extend(tournament_results.loc[(tournament_results["AgentB"] == agent_name) &
                                                              (tournament_results["AgentA"] != agent_name) &
                                                              (tournament_results["DomainID"] == j) &
                                                              (tournament_results["Result"] == "Acceptance"),
                                                              "Time"].to_list())

                rows_acceptance.extend(tournament_results.loc[(tournament_results["AgentB"] == agent_name) &
                                                              (tournament_results["AgentA"] == agent_name) &
                                                              (tournament_results["DomainID"] == j) &
                                                              (tournament_results["Result"] == "Acceptance"),
                                                              "Time"].to_list())

                if len(rows_utility) == 0:
                    data_utility[i].append(0.)
                    data_time[i].append(0.)
                    data_acceptance_rate[i].append(0.)
                else:
                    data_utility[i].append(np.mean(rows_utility))
                    data_time[i].append(np.mean(rows_time))
                    data_acceptance_rate[i].append(len(rows_acceptance) / len(rows_time))

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, domain_names, data_utility)

        draw_heatmap(sorted_map, domain_names, sorted_agent_names,
                     os.path.join(directory, f"agent_domain_utility_{domain_set}.png"), "Domains", "Agents")

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, domain_names, data_time)

        draw_heatmap(sorted_map, domain_names, sorted_agent_names,
                     os.path.join(directory, f"agent_domain_time_{domain_set}.png"), "Domains", "Agents")

        sorted_agent_names, sorted_map = sort_y_axis(agent_names, domain_names, data_acceptance_rate)

        draw_heatmap(sorted_map, domain_names, sorted_agent_names,
                     os.path.join(directory, f"agent_domain_acceptance_rate_{domain_set}.png"), "Domains", "Agents",
                     fmt=".0%")


def sort_y_axis(labels_y: List[str], labels_x: List[str], map_values: List[List[float]], descending: bool = True) -> \
        (List[str], List[List[float]]):
    """
        This method sorts the map based on the mean of row.
    :param labels_y: y-axis Labels
    :param labels_x: x-axis Labels
    :param map_values: Map that will be sorted.
    :param descending: Whether the order is descending or not
    :return: Sorted list of y-axis labels and map
    """
    means = np.mean(map_values, axis=1)

    indices = list(range(len(labels_y)))

    indices.sort(key=lambda i: means[i], reverse=descending)

    sorted_labels = []

    for i in indices:
        sorted_labels.append(labels_y[i])

    sorted_map = []

    both_axis = labels_y == labels_x

    for i in indices:
        row = []
        idx = indices if both_axis else list(range(len(labels_x)))
        for j in idx:
            row.append(map_values[i][j])

        sorted_map.append(row)

    return sorted_labels, sorted_map
