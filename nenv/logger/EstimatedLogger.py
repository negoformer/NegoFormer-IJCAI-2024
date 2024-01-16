import os
from typing import Dict

import pandas as pd

from nenv.logger.AbstractLogger import *
from nenv.utils.Move import *
from tournament_graphs import draw_heatmap
import numpy as np


class EstimatedLogger(AbstractLogger):
    """
        EstimatedLogger logs some information extracted via Estimators, as listed below:
        - Estimated Opponent Utility.
        - Estimated Move.
        - Estimated Nash Product
        - Estimated Social Welfare

        It iterates over all Estimators of all agents to extract the necessary log.

        At the end of the tournament, this logger measures the performance of the Estimators on the move prediction.
        Move prediction task can be considered as a move classification task. Thus, some metrics (e.g., Accuracy, F1)
        which are commonly used for classification tasks are applied for the evaluation. Additionally, this logger also
        creates a Confusion Matrix for each estimator.
    """

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        row = {}

        for estimator_id in range(len(session.agentA.estimators)):
            agentA_utility = session.agentA.preference.get_utility(offer)
            agentB_utility = session.agentB.preference.get_utility(offer)

            estimated_utility_B = session.agentB.estimators[estimator_id].preference.get_utility(offer)
            estimated_utility_A = session.agentA.estimators[estimator_id].preference.get_utility(offer)

            log = {
                "EstimatedOpponentUtilityA": estimated_utility_B,
                "EstimatedOpponentUtilityB": estimated_utility_A,
                "EstimatedMoveA": self._get_move(session, agent, estimate_b=True, estimator_id=estimator_id),
                "EstimatedMoveB": self._get_move(session, agent, estimate_a=True, estimator_id=estimator_id),
                "EstimatedNashProductA": agentA_utility * estimated_utility_B,
                "EstimatedNashProductB": agentB_utility * estimated_utility_A,
                "EstimatedSocialWelfareA": agentA_utility + estimated_utility_B,
                "EstimatedSocialWelfareB": agentB_utility + estimated_utility_A,
            }

            row[session.agentA.estimators[estimator_id].name] = log

        return row

    def on_accept(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        row = {}

        for estimator_id in range(len(session.agentA.estimators)):
            estimated_preference_A = session.agentA.estimators[estimator_id].preference

            log = {
                "EstimatedNashProductA": session.agentA.preference.get_utility(
                    offer) * estimated_preference_A.get_utility(offer),
                "EstimatedSocialWelfareA": session.agentA.preference.get_utility(
                    offer) + estimated_preference_A.get_utility(offer)
            }

            estimated_preference_B = session.agentB.estimators[estimator_id].preference

            log.update({
                "EstimatedNashProductB": session.agentB.preference.get_utility(
                    offer) * estimated_preference_B.get_utility(offer),
                "EstimatedSocialWelfareB": session.agentB.preference.get_utility(
                    offer) + estimated_preference_B.get_utility(offer)
            })

            row[session.agentA.estimators[estimator_id].name] = log

        return row

    def on_fail(self, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        row = {}

        for estimator_id in range(len(session.agentA.estimators)):
            log = {
                "EstimatedNashProductA": 0,
                "EstimatedSocialWelfareA": session.agentA.preference.reservation_value
            }

            log.update({
                "EstimatedNashProductB": 0,
                "EstimatedSocialWelfareB": session.agentB.preference.reservation_value
            })

            row[session.agentA.estimators[estimator_id].name] = log

        return row

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str], estimator_names: List[str]):
        if len(estimator_names) == 0:
            return

        if not os.path.exists(self.get_path("opponent model/")):
            os.mkdir(self.get_path("opponent model/"))

        self.extract_estimator_names(estimator_names)

        moves = self.get_moves()

        accuracy, confusion_matrices = self.get_accuracy_and_confusion_matrix(tournament_logs, estimator_names)

        tp, fp, fn, recall, precision, f1 = self.calculate(confusion_matrices, estimator_names)

        for estimator_id in range(len(estimator_names)):
            confusion_matrix_path = "opponent model/%s_move_confusion_matrix.png" % estimator_names[estimator_id]

            analyze_path = "opponent model/%s_move_analyze.csv" % estimator_names[estimator_id]

            draw_heatmap(confusion_matrices[estimator_id] / (np.sum(confusion_matrices[estimator_id]) + 1e-12), moves,
                         moves, self.get_path(confusion_matrix_path), "Estimated Move", "Real Move")

            self.save_move_analyze(analyze_path, estimator_names[estimator_id], estimator_id, accuracy, tp, fp, fn,
                                   precision, recall, f1)

    def _get_move(self, session: Union[Session, SessionEstimator], agent_no: str, estimate_a: bool = False, estimate_b: bool = False, estimator_id: int = -1) -> str:
        if len(session.action_history) < 3:
            return "-"

        if agent_no == "A":
            offered_utility = session.agentA.estimators[estimator_id].preference.get_utility(
                session.action_history[-1].bid) if estimate_a else session.agentA.preference.get_utility(
                session.action_history[-1].bid)
            prev_offered_utility = session.agentA.estimators[estimator_id].preference.get_utility(
                session.action_history[-3].bid) if estimate_a else session.agentA.preference.get_utility(
                session.action_history[-3].bid)

            opponent_utility = session.agentB.estimators[estimator_id].preference.get_utility(
                session.action_history[-1].bid) if estimate_b else session.agentB.preference.get_utility(
                session.action_history[-1].bid)
            prev_opponent_utility = session.agentB.estimators[estimator_id].preference.get_utility(
                session.action_history[-3].bid) if estimate_b else session.agentB.preference.get_utility(
                session.action_history[-3].bid)
        else:
            offered_utility = session.agentB.estimators[estimator_id].preference.get_utility(
                session.action_history[-1].bid) if estimate_a else session.agentB.preference.get_utility(
                session.action_history[-1].bid)
            prev_offered_utility = session.agentB.estimators[estimator_id].preference.get_utility(
                session.action_history[-3].bid) if estimate_a else session.agentB.preference.get_utility(
                session.action_history[-3].bid)

            opponent_utility = session.agentA.estimators[estimator_id].preference.get_utility(
                session.action_history[-1].bid) if estimate_b else session.agentA.preference.get_utility(
                session.action_history[-1].bid)
            prev_opponent_utility = session.agentA.estimators[estimator_id].preference.get_utility(
                session.action_history[-3].bid) if estimate_b else session.agentA.preference.get_utility(
                session.action_history[-3].bid)

        return get_move(prev_offered_utility, offered_utility, prev_opponent_utility, opponent_utility)

    def extract_estimator_names(self, names: List[str]):
        pd.DataFrame(names).to_excel(self.get_path("opponent model/estimators.xlsx"))

    def get_accuracy_and_confusion_matrix(self, tournament_logs: ExcelLog, estimator_names: List[str]) -> (List[float], List[np.ndarray]):
        moves = self.get_moves()

        confusion_matrices = [np.zeros((len(moves), len(moves)), dtype=np.int32) for _ in
                              range(len(estimator_names))]
        accuracy = [0. for _ in range(len(estimator_names))]

        for row in tournament_logs.log_rows["TournamentResults"]:
            agent_a = row["AgentA"]
            agent_b = row["AgentB"]
            domain_name = "Domain%d" % int(row["DomainID"])

            session_path = self.get_path(f"sessions/{agent_a}_{agent_b}_{domain_name}.xlsx")
            session_log = ExcelLog(file_path=session_path)

            for i in range(len(estimator_names)):
                if estimator_names[i] not in session_log.log_rows:
                    break

                for row_index, session_row in enumerate(session_log.log_rows[estimator_names[i]]):
                    real_move = session_log.log_rows["Session"][row_index]["Move"]

                    if real_move == "-" or real_move is None or str(real_move) == "nan":
                        continue

                    estimated_move = session_row["EstimatedMoveA"]

                    confusion_matrices[i][moves.index(real_move)][moves.index(estimated_move)] += 1

                    if real_move == estimated_move:
                        accuracy[i] += 1

                    estimated_move = session_row["EstimatedMoveB"]

                    confusion_matrices[i][moves.index(real_move)][moves.index(estimated_move)] += 1

                    if real_move == estimated_move:
                        accuracy[i] += 1

        for estimator_id in range(len(estimator_names)):
            accuracy[estimator_id] /= np.sum(confusion_matrices[estimator_id])

        return accuracy, confusion_matrices

    def calculate(self, confusion_matrices: List[np.ndarray], estimator_names: List[str]) -> \
            (List[Dict[str, int]], List[Dict[str, int]], List[Dict[str, int]],
             List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]]):

        moves = self.get_moves()

        tp = [{move: 0 for move in moves} for _ in range(len(estimator_names))]
        fp = [{move: 0 for move in moves} for _ in range(len(estimator_names))]
        fn = [{move: 0 for move in moves} for _ in range(len(estimator_names))]

        for estimator_id in range(len(estimator_names)):
            for i in range(confusion_matrices[estimator_id].shape[0]):
                tp[estimator_id][moves[i]] = confusion_matrices[estimator_id][i][i]

                for j in range(confusion_matrices[estimator_id].shape[1]):
                    if i == j:
                        continue

                    fn[estimator_id][moves[i]] += confusion_matrices[estimator_id][i][j]
                    fp[estimator_id][moves[i]] += confusion_matrices[estimator_id][j][i]

        recall = [{move: tp[i][move] / (tp[i][move] + fn[i][move]) for move in moves} for i in
                  range(len(estimator_names))]
        precision = [{move: tp[i][move] / (tp[i][move] + fp[i][move]) for move in moves} for i in
                     range(len(estimator_names))]
        f1 = [{move: 2 * recall[i][move] * precision[i][move] / (recall[i][move] + precision[i][move]) for move in
               moves} for i in range(len(estimator_names))]

        return tp, fp, fn, recall, precision, f1

    def save_move_analyze(self, analyze_path: str, estimator_name: str, estimator_id: int, accuracy: List[float],
                          tp: List[Dict[str, int]], fp: List[Dict[str, int]], fn: List[Dict[str, int]],
                          precision: List[Dict[str, float]], recall: List[Dict[str, float]], f1: List[Dict[str, float]]):

        moves = self.get_moves()

        with open(self.get_path(analyze_path), "w") as f:
            f.write("Name;%s\n" % estimator_name)
            f.write("Accuracy;%f\n" % accuracy[estimator_id])

            f.write("Move;")

            for move in moves:
                f.write("%s;" % move)

            f.write("\nTP;")

            for move in moves:
                f.write("%d;" % tp[estimator_id][move])

            f.write("\nFP;")

            for move in moves:
                f.write("%d;" % fp[estimator_id][move])

            f.write("\nFN;")

            for move in moves:
                f.write("%d;" % fn[estimator_id][move])

            f.write("\nPrecision;")

            for move in moves:
                f.write("%f;" % precision[estimator_id][move])

            f.write("\nRecall;")

            for move in moves:
                f.write("%f;" % recall[estimator_id][move])

            f.write("\nF1;")

            for move in moves:
                f.write("%f;" % f1[estimator_id][move])

            f.write("\n")

    def get_moves(self) -> List[str]:
        return ["Concession", "Fortunate", "Nice", "Selfish", "Silent", "Unfortunate"]