from nenv.logger.AbstractLogger import *
import numpy as np


class TournamentWithoutErrorSummaryLogger(AbstractLogger):
    """
        TournamentWithoutErrorSummaryLogger summarize the tournament results for the performance analysis of agents.

        It is a variant of TournamentSummaryLogger, that ignores Error and TimedOut errors.
    """
    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str], estimator_names: List[str]):
        summary = pd.DataFrame(
            columns=["AgentName", "Avg.Utility", "Std.Utility", "Avg.OpponentUtility", "Std.OpponentUtility",
                     "Avg.AcceptanceTime", "Std.AcceptanceTime",
                     "Avg.Round", "Std.Round", "Avg.NashProduct", "Std.NashProduct", "Avg.SocialWelfare",
                     "Std.SocialWelfare", "Avg.NashDistance", "Std.NashDistance", "Avg.KalaiDistance",
                     "Std.KalaiDistance", "Avg.AcceptanceUtility", "Std.AcceptanceUtility", "AcceptanceRate", "Count"])

        tournament_results = tournament_logs.to_data_frame("TournamentResults")

        tournament_without_error_results = tournament_results.loc[(tournament_results["Result"] != "Error") & (tournament_results["Result"] != "TimedOut")]

        for agent_name in agent_names:
            utilities = tournament_without_error_results.loc[(tournament_without_error_results["AgentA"] == agent_name), "AgentAUtility"].to_list()
            utilities.extend(
                tournament_without_error_results.loc[(tournament_without_error_results["AgentB"] == agent_name), "AgentBUtility"].to_list())

            opponent_utilities = tournament_without_error_results.loc[(tournament_without_error_results["AgentA"] == agent_name), "AgentBUtility"]. \
                to_list()
            opponent_utilities.extend(
                tournament_without_error_results.loc[(tournament_without_error_results["AgentB"] == agent_name), "AgentAUtility"].to_list())

            acceptance_utilities = tournament_without_error_results.loc[(tournament_without_error_results["AgentA"] == agent_name) &
                                                                        (tournament_without_error_results[
                                                               "Result"] == "Acceptance"), "AgentAUtility"]. \
                to_list()
            acceptance_utilities.extend(
                tournament_without_error_results.loc[(tournament_without_error_results["AgentB"] == agent_name) &
                                                     (tournament_without_error_results["Result"] == "Acceptance"), "AgentBUtility"].to_list())

            acceptance_times = tournament_without_error_results.loc[
                ((tournament_without_error_results["AgentA"] == agent_name) | (tournament_without_error_results["AgentB"] == agent_name)) & (
                        tournament_without_error_results["Result"] == "Acceptance"), "Time"].to_list()

            rounds = tournament_without_error_results.loc[
                ((tournament_without_error_results["AgentA"] == agent_name) | (tournament_without_error_results["AgentB"] == agent_name)) & (
                        tournament_without_error_results["Result"] == "Acceptance"), "Round"].to_list()

            nash_distances = tournament_without_error_results.loc[(tournament_without_error_results["AgentA"] == agent_name) | (
                    tournament_without_error_results["AgentB"] == agent_name), "NashDistance"].to_list()
            kalai_distances = tournament_without_error_results.loc[(tournament_without_error_results["AgentA"] == agent_name) | (
                    tournament_without_error_results["AgentB"] == agent_name), "KalaiDistance"].to_list()

            nash_products = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "NashProduct"].to_list()
            kalai_sum = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "KalaiSum"].to_list()

            failed_times = tournament_results.loc[
                ((tournament_results["AgentA"] == agent_name) | (tournament_results["AgentB"] == agent_name)) & (
                        tournament_results["Result"] == "Failed"), "Time"].to_list()

            acceptance_count = len(acceptance_times)
            failed_count = len(failed_times)
            total_negotiation = len(nash_distances)

            if total_negotiation == 0:
                summary = summary.append({
                    "AgentName": agent_name,
                    "Avg.Utility": 0,
                    "Std.Utility": 0,
                    "Avg.OpponentUtility": 0,
                    "Std.OpponentUtility": 0,
                    "Avg.AcceptanceTime": 0,
                    "Std.AcceptanceTime": 0,
                    "Avg.Round": 0,
                    "Std.Round": 0,
                    "Avg.NashProduct": 0.,
                    "Std.NashProduct": 0.,
                    "Avg.SocialWelfare": 0.,
                    "Std.SocialWelfare": 0.,
                    "Avg.NashDistance": 0,
                    "Std.NashDistance": 0,
                    "Avg.KalaiDistance": 0,
                    "Std.KalaiDistance": 0,
                    "Avg.AcceptanceUtility": 0,
                    "Std.AcceptanceUtility": 0,
                    "AcceptanceRate": 0,
                    "Count": 0,
                    "Acceptance": 0,
                    "Failed": 0
                }, ignore_index=True)

                continue

            summary = summary.append({
                "AgentName": agent_name,
                "Avg.Utility": np.mean(utilities),
                "Std.Utility": np.std(utilities),
                "Avg.OpponentUtility": np.mean(opponent_utilities),
                "Std.OpponentUtility": np.std(opponent_utilities),
                "Avg.AcceptanceTime": np.mean(acceptance_times),
                "Std.AcceptanceTime": np.std(acceptance_times),
                "Avg.Round": np.mean(rounds),
                "Std.Round": np.std(rounds),
                "Avg.NashProduct": np.mean(nash_products),
                "Std.NashProduct": np.std(nash_products),
                "Avg.SocialWelfare": np.mean(kalai_sum),
                "Std.SocialWelfare": np.std(kalai_sum),
                "Avg.NashDistance": np.mean(nash_distances),
                "Std.NashDistance": np.std(nash_distances),
                "Avg.KalaiDistance": np.mean(kalai_distances),
                "Std.KalaiDistance": np.std(kalai_distances),
                "Avg.AcceptanceUtility": np.mean(acceptance_utilities),
                "Std.AcceptanceUtility": np.std(acceptance_utilities),
                "AcceptanceRate": acceptance_count / total_negotiation,
                "Count": total_negotiation,
                "Acceptance": acceptance_count,
                "Failed": failed_count
            }, ignore_index=True)

        summary.sort_values(by="Avg.Utility", inplace=True, ascending=False)

        summary.to_csv(self.get_path("summary_without_error.csv"), sep=";")
