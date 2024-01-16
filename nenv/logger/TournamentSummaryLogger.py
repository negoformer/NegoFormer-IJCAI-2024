import pandas as pd

from nenv.logger.AbstractLogger import *
import numpy as np


class TournamentSummaryLogger(AbstractLogger):
    """
        TournamentSummaryLogger summarize the tournament results for the performance analysis of agents.
    """

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str],
                          estimator_names: List[str]):
        summary = pd.DataFrame(
            columns=["AgentName", "Avg.Utility", "Std.Utility", "Avg.OpponentUtility", "Std.OpponentUtility",
                     "Avg.AcceptanceTime", "Std.AcceptanceTime",
                     "Avg.Round", "Std.Round", "Avg.NashProduct", "Std.NashProduct", "Avg.SocialWelfare",
                     "Std.SocialWelfare", "Avg.NashDistance", "Std.NashDistance", "Avg.KalaiDistance",
                     "Std.KalaiDistance", "AcceptanceRate", "Count", "Acceptance", "Failed", "Error", "TimedOut"])

        summary_acceptance = pd.DataFrame(columns=["AgentName",
                                                   "Avg.Utility", "Std.Utility", "Avg.OpponentUtility",
                                                   "Std.OpponentUtility",
                                                   "Avg.Round", "Std.Round", "Avg.NashProduct", "Std.NashProduct",
                                                   "Avg.SocialWelfare",
                                                   "Std.SocialWelfare", "Avg.NashDistance", "Std.NashDistance",
                                                   "Avg.KalaiDistance",
                                                   "Std.KalaiDistance"])

        tournament_results = tournament_logs.to_data_frame("TournamentResults")

        for agent_name in agent_names:
            utilities = tournament_results.loc[(tournament_results["AgentA"] == agent_name), "AgentAUtility"].to_list()
            utilities.extend(
                tournament_results.loc[(tournament_results["AgentB"] == agent_name), "AgentBUtility"].to_list())

            opponent_utilities = tournament_results.loc[(tournament_results["AgentA"] == agent_name), "AgentBUtility"]. \
                to_list()
            opponent_utilities.extend(
                tournament_results.loc[(tournament_results["AgentB"] == agent_name), "AgentAUtility"].to_list())

            acceptance_times = tournament_results.loc[
                ((tournament_results["AgentA"] == agent_name) | (tournament_results["AgentB"] == agent_name)) & (
                        tournament_results["Result"] == "Acceptance"), "Time"].to_list()

            rounds = tournament_results.loc[
                ((tournament_results["AgentA"] == agent_name) | (tournament_results["AgentB"] == agent_name))
                , "Round"].to_list()

            nash_distances = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "NashDistance"].to_list()
            kalai_distances = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "KalaiDistance"].to_list()

            nash_products = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "NashProduct"].to_list()
            kalai_sum = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "KalaiSum"].to_list()

            failed_times = tournament_results.loc[
                ((tournament_results["AgentA"] == agent_name) | (tournament_results["AgentB"] == agent_name)) & (
                        tournament_results["Result"] == "Failed"), "Time"].to_list()

            error_times = tournament_results.loc[
                ((tournament_results["AgentA"] == agent_name) | (tournament_results["AgentB"] == agent_name)) & (
                        tournament_results["Result"] == "Error"), "Time"].to_list()

            timed_out_times = tournament_results.loc[
                ((tournament_results["AgentA"] == agent_name) | (tournament_results["AgentB"] == agent_name)) & (
                        tournament_results["Result"] == "TimedOut"), "Time"].to_list()

            self_error_times = tournament_results.loc[
                (((tournament_results["AgentA"] == agent_name) & (tournament_results["Who"] == "A"))
                 | ((tournament_results["AgentB"] == agent_name) & (tournament_results["Who"] == "B")))
                & (tournament_results["Result"] == 'Error'),
                "Time"].to_list()

            self_timed_out_times = tournament_results.loc[
                (((tournament_results["AgentA"] == agent_name) & (tournament_results["Who"] == "A"))
                 | ((tournament_results["AgentB"] == agent_name) & (tournament_results["Who"] == "B")))
                & (tournament_results["Result"] == 'TimedOut'),
                "Time"].to_list()

            acceptance_count = len(acceptance_times)
            failed_count = len(failed_times)
            error_count = len(error_times)
            timed_out_count = len(timed_out_times)
            self_error_count = len(self_error_times)
            self_timed_out_count = len(self_timed_out_times)
            total_negotiation = acceptance_count + failed_count + error_count + timed_out_count

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
                "AcceptanceRate": acceptance_count / total_negotiation,
                "Count": total_negotiation,
                "Acceptance": acceptance_count,
                "Failed": failed_count,
                "Error": error_count,
                "TimedOut": timed_out_count,
                "SelfError": self_error_count,
                "SelfTimedOut": self_timed_out_count
            }, ignore_index=True)

            utilities = tournament_results.loc[(tournament_results["AgentA"] == agent_name) &
                                               (tournament_results[
                                                    "Result"] == "Acceptance"), "AgentAUtility"]. \
                to_list()
            utilities.extend(
                tournament_results.loc[(tournament_results["AgentB"] == agent_name) &
                                       (tournament_results["Result"] == "Acceptance"), "AgentBUtility"].to_list())

            if len(utilities) == 0:
                summary_acceptance = summary_acceptance.append({
                    "AgentName": agent_name,
                    "Avg.Utility": 0.,
                    "Std.Utility": 0.,
                    "Avg.OpponentUtility": 0.,
                    "Std.OpponentUtility": 0.,
                    "Avg.Round": 0.,
                    "Std.Round": 0.,
                    "Avg.NashProduct": 0.,
                    "Std.NashProduct": 0.,
                    "Avg.SocialWelfare": 0.,
                    "Std.SocialWelfare": 0.,
                    "Avg.NashDistance": None,
                    "Std.NashDistance": None,
                    "Avg.KalaiDistance": None,
                    "Std.KalaiDistance": None,
                }, ignore_index=True)

                continue

            opponent_utilities = tournament_results.loc[(tournament_results["AgentA"] == agent_name) &
                                                        (tournament_results["Result"] == "Acceptance"),
            "AgentBUtility"].to_list()

            opponent_utilities.extend(
                tournament_results.loc[
                    (tournament_results["AgentB"] == agent_name) & (tournament_results["Result"] == "Acceptance")
                    , "AgentAUtility"].to_list())

            rounds = tournament_results.loc[
                ((tournament_results["AgentA"] == agent_name) | (tournament_results["AgentB"] == agent_name)) &
                (tournament_results["Result"] == "Acceptance")
                , "Round"].to_list()

            nash_distances = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "NashDistance"].to_list()

            kalai_distances = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "KalaiDistance"].to_list()

            nash_products = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "NashProduct"].to_list()

            kalai_sums = tournament_results.loc[(tournament_results["AgentA"] == agent_name) | (
                    tournament_results["AgentB"] == agent_name), "KalaiSum"].to_list()

            summary_acceptance = summary_acceptance.append({
                "AgentName": agent_name,
                "Avg.Utility": np.mean(utilities),
                "Std.Utility": np.std(utilities),
                "Avg.OpponentUtility": np.mean(opponent_utilities),
                "Std.OpponentUtility": np.std(opponent_utilities),
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
            }, ignore_index=True)

        summary.sort_values(by="Avg.Utility", inplace=True, ascending=False)
        summary_acceptance.sort_values(by="Avg.Utility", inplace=True, ascending=False)

        with pd.ExcelWriter(self.get_path("summary.xlsx")) as f:
            summary.to_excel(f, sheet_name="Summary", index=False)
            summary_acceptance.to_excel(f, sheet_name="Summary Acceptance", index=False)

