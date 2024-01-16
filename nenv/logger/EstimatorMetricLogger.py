from nenv.logger.AbstractLogger import *
from nenv.Agent import AbstractAgent
from tournament_graphs import draw_line
from typing import List, Tuple, Dict
import numpy as np


class EstimatorMetricLogger(AbstractLogger):
    """
        EstimatorMetricLogger logs the performance analysis of each Estimator round by round. RMSE, Spearman and
        Kendal-Tau metrics which are commonly used for the evaluation of an Opponent Model are applied.

        At the end of tournament, it generates overall results containing these metric results. It also draws the
        necessary plots.

        Note: This logger increases the computational time due to the expensive calculation of the metrics. If you have
        strict time for the tournament run, you can look EstimatorOnlyFinalMetricLogger which is a cheaper version of
        this logger.
    """

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        return self.get_metrics(session.agentA, session.agentB)

    def on_accept(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        return self.get_metrics(session.agentA, session.agentB)

    def on_fail(self, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        return self.get_metrics(session.agentA, session.agentB)

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str], estimator_names: List[str]):
        if len(estimator_names) == 0:
            return

        if not os.path.exists(self.get_path("opponent model/")):
            os.mkdir(self.get_path("opponent model/"))

        self.extract_estimator_summary(tournament_logs, estimator_names)
        rmse, kendall, spearman = self.get_estimator_results(tournament_logs, estimator_names)

        self.to_excel(rmse, kendall, spearman)
        self.draw(rmse, kendall, spearman)

    def get_metrics(self, agent_a: AbstractAgent, agent_b: AbstractAgent) -> LogRow:
        row = {}

        for estimator_id in range(len(agent_a.estimators)):
            rmseA, spearmanA, kendallA = agent_a.estimators[estimator_id].calculate_error(agent_b.preference)
            rmseB, spearmanB, kendallB = agent_b.estimators[estimator_id].calculate_error(agent_a.preference)

            log = {
                "RMSE_A": rmseA,
                "RMSE_B": rmseB,
                "SpearmanA": spearmanA,
                "SpearmanB": spearmanB,
                "KendallTauA": kendallA,
                "KendallTauB": kendallB,
                "RMSE": (rmseA + rmseB) / 2.,
                "Spearman": (spearmanA + spearmanB) / 2.,
                "KendallTau": (kendallA + kendallB) / 2.
            }

            row[agent_a.estimators[estimator_id].name] = log

        return row

    def extract_estimator_summary(self, tournament_logs: ExcelLog, estimator_names: List[str]):
        summary = pd.DataFrame(
            columns=["EstimatorName", "Avg.RMSE", "Std.RMSE", "Avg.Spearman", "Std.Spearman", "Avg.KendallTau",
                     "Std.KendallTau"]
        )

        for i in range(len(estimator_names)):
            results = tournament_logs.to_data_frame(estimator_names[i])

            RMSE, spearman, kendall = [], [], []

            RMSE.extend(results["RMSE_A"].to_list())
            RMSE.extend(results["RMSE_B"].to_list())

            spearman.extend(results["SpearmanA"].to_list())
            spearman.extend(results["SpearmanB"].to_list())

            kendall.extend(results["KendallTauA"].to_list())
            kendall.extend(results["KendallTauB"].to_list())

            summary = summary.append({
                "EstimatorName": estimator_names[i],
                "Avg.RMSE": np.mean(RMSE),
                "Std.RMSE": np.std(RMSE),
                "Avg.Spearman": np.mean(spearman),
                "Std.Spearman": np.std(spearman),
                "Avg.KendallTau": np.mean(kendall),
                "Std.KendallTau": np.std(kendall)
            }, ignore_index=True)

        summary.sort_values(by="Avg.RMSE", inplace=True, ascending=True)

        summary.to_excel(self.get_path("opponent model/estimator_summary.xlsx"))

    def get_estimator_results(self, tournament_logs: ExcelLog, estimator_names: list) -> Tuple[Dict[str, List[List[float]]], Dict[str, List[List[float]]], Dict[str, List[List[float]]]]:
        tournament_results = tournament_logs.to_data_frame()

        max_round = max(tournament_results["TournamentResults"]["Round"].to_list())

        rmse = {name: [[] for _ in range(max_round + 1)] for name in estimator_names}
        spearman = {name: [[] for _ in range(max_round + 1)] for name in estimator_names}
        kendall = {name: [[] for _ in range(max_round + 1)] for name in estimator_names}

        for _, row in tournament_results["TournamentResults"].to_dict('index').items():
            agent_a = row["AgentA"]
            agent_b = row["AgentB"]
            domain_name = "Domain%d" % int(row["DomainID"])

            session_path = self.get_path(f"sessions/{agent_a}_{agent_b}_{domain_name}.xlsx")

            for i in range(len(estimator_names)):
                session_log = ExcelLog(file_path=session_path)

                for row_index, estimator_row in enumerate(session_log.log_rows[estimator_names[i]]):
                    if session_log.log_rows["Session"][row_index]["Action"] == "Accept":
                        break

                    _round = session_log.log_rows["Session"][row_index]["Round"]

                    rmse[estimator_names[0]][_round].append(estimator_row["RMSE_A"])
                    spearman[estimator_names[0]][_round].append(estimator_row["SpearmanA"])
                    kendall[estimator_names[0]][_round].append(estimator_row["KendallTauA"])
                    rmse[estimator_names[0]][_round].append(estimator_row["RMSE_B"])
                    spearman[estimator_names[0]][_round].append(estimator_row["SpearmanB"])
                    kendall[estimator_names[0]][_round].append(estimator_row["KendallTauB"])

        return rmse, spearman, kendall

    def to_excel(self, rmse: dict, spearman: dict, kendall: dict):
        rows = []

        rmse_mean, rmse_std = self.get_mean_std(rmse)
        spearman_mean, spearman_std = self.get_mean_std(spearman)
        kendall_mean, kendall_std = self.get_mean_std(kendall)

        for j in range(len(rmse[list(rmse.keys())[0]])):
            row = {"Round": j, "Counts": len(rmse[list(rmse.keys())[0]][j])}

            for estimator_name in rmse:
                row["%s RMSE" % estimator_name] = rmse_mean[estimator_name][j]
                row["%s RMSE Stdev" % estimator_name] = rmse_std[estimator_name][j]
                row["%s Spearman" % estimator_name] = spearman_mean[estimator_name][j]
                row["%s Spearman Stdev" % estimator_name] = spearman_std[estimator_name][j]
                row["%s KendallTau" % estimator_name] = kendall_mean[estimator_name][j]
                row["%s KendallTau Stdev" % estimator_name] = kendall_std[estimator_name][j]

            rows.append(row)

        pd.DataFrame(rows).to_excel(self.get_path("opponent model/estimator_metrics.xlsx"))

    def draw(self, rmse: dict, spearman: dict, kendall: dict):
        rmse_mean, rmse_std = self.get_mean_std(rmse)
        spearman_mean, spearman_std = self.get_mean_std(spearman)
        kendall_mean, kendall_std = self.get_mean_std(kendall)

        draw_line(rmse_mean, self.get_path("opponent model/estimator_rmse.png"), "Rounds", "RMSE")
        draw_line(spearman_mean, self.get_path("opponent model/estimator_spearman.png"), "Rounds", "Spearman")
        draw_line(kendall_mean, self.get_path("opponent model/estimator_kendall_tau.png"), "Rounds", "KendallTau")

        median_round = self.get_median_round(rmse)

        for estimator_name in rmse:
            rmse[estimator_name] = rmse[estimator_name][:median_round]
            spearman[estimator_name] = spearman[estimator_name][:median_round]
            kendall[estimator_name] = kendall[estimator_name][:median_round]

        draw_line(rmse_mean, self.get_path("opponent model/estimator_rmse_median.png"), "Rounds", "RMSE")
        draw_line(spearman_mean, self.get_path("opponent model/estimator_spearman_median.png"), "Rounds", "Spearman")
        draw_line(kendall_mean, self.get_path("opponent model/estimator_kendall_tau_median.png"), "Rounds", "KendallTau")

    def get_median_round(self, results: dict) -> int:
        counts = []

        for estimator_name, rounds in results.items():
            for i, results in enumerate(rounds):
                for j in range(len(results)):
                    counts.append(i)

            break

        return round(float(np.median(counts)))

    def get_mean_std(self, results: dict) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        means, std = {}, {}

        for estimator_name, rounds in results.items():
            means[estimator_name] = []
            std[estimator_name] = []

            for result in rounds:
                means[estimator_name].append(float(np.mean(result)))
                std[estimator_name].append(float(np.std(result)))

        return means, std
