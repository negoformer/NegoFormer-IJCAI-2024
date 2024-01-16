from typing import Union, List
import numpy as np
import nenv
from nenv import Session, SessionEstimator, Bid
from nenv.utils import LogRow, ExcelLog
from .ParetoWalkerAgent import ParetoWalkerAgent
from tournament_graphs import draw_line
from .utils import extract_pareto_indices


class ParetoLogger(nenv.logger.AbstractLogger):
    """
        This method evaluate the pareto estimation for ParetoWalker agent.
        It assumes that pareto estimation is a classification problem. Therefore, Precision, Recall and F1 Score metrics
        are evaluated. Besides, MSE metric is also considered to analyze the utility calculation error of pareto bids.
    """
    agent_pos: str = ""
    is_estimator_session: bool
    real_pareto: List[nenv.BidPoint]
    average_diff: float

    round_by_round_f1: List[float]
    round_by_round_recall: List[float]
    round_by_round_precession: List[float]
    round_by_round_mse: List[float]
    round_by_round_indices: List[float]
    round_by_round_counts: List[float]

    def __init__(self, log_dir: str):
        super().__init__(log_dir)

        self.round_by_round_f1 = []
        self.round_by_round_recall = []
        self.round_by_round_precession = []
        self.round_by_round_mse = []
        self.round_by_round_indices = []
        self.round_by_round_counts = []

        self.is_estimator_session = True

    def before_session_start(self, session: Union[Session, SessionEstimator]) -> List[str]:
        """
            This method checks whether ParetoWalkerAgent will negotiate in this session.
        """
        if isinstance(session, SessionEstimator):
            self.is_estimator_session = False
        else:
            self.is_estimator_session = False

        if session.agentA.name == "ParetoWalkerAgent":
            self.agent_pos = "A"

            self.real_pareto = self.get_pareto(session.agentA.preference, session.agentB.preference)

            self.average_diff = self.get_average_bid_diff(session.agentA.preference, session.agentB.preference)

            return ["ParetoWalkerAgent_Pareto"]
        elif session.agentB.name == "ParetoWalkerAgent":
            self.agent_pos = "B"

            self.real_pareto = self.get_pareto(session.agentB.preference, session.agentA.preference)

            self.average_diff = self.get_average_bid_diff(session.agentB.preference, session.agentA.preference)

            return ["ParetoWalkerAgent_Pareto"]
        else:
            self.agent_pos = ""

            self.real_pareto = []

            return []

    def on_offer(self, agent: str, offer: Bid, time: float, session: Union[Session, SessionEstimator]) -> LogRow:
        if self.agent_pos == "" or self.is_estimator_session:
            return {}

        if self.agent_pos != agent:
            return {"ParetoWalkerAgent_Pareto": {"Precision": "-", "Recall": "-", "F1": "-", "TP": "-", "FP": "-", "FN": "-", "MSE": "-", "Avg.Diff": self.average_diff, "WalkIndex": "-"}}

        agent: ParetoWalkerAgent = session.agentA if self.agent_pos == "A" else session.agentB

        if agent.pareto_index >= 0:
            last_pareto = agent.pareto

            tp, fp, fn, precision, recall, f1 = self.calculate_tp_fp_fn_precision_recall_f1(last_pareto)

            mse = self.calculate_mse(session, last_pareto)

            return {"ParetoWalkerAgent_Pareto": {"Precision": precision, "Recall": recall, "F1": f1, "TP": tp, "FP": fp, "FN": fn, "MSE": mse, "Avg.Diff": self.average_diff, "WalkIndex": agent.pareto_index}}

        return {"ParetoWalkerAgent_Pareto": {"Precision": "-", "Recall": "-", "F1": "-", "TP": "-", "FP": "-", "FN": "-", "MSE": "-", "Avg.Diff": self.average_diff, "WalkIndex": "-"}}

    def on_session_end(self, final_row: LogRow, session: Union[Session, SessionEstimator]) -> LogRow:
        if self.agent_pos == "":
            return {"ParetoWalkerAgent_Pareto": {"Precision": "-", "Recall": "-", "F1": "-", "TP": "-", "FP": "-", "FN": "-", "MSE": "-", "Avg.Diff": "-", "WalkIndex": "-"}}

        counter = 0

        total_f1, total_recall, total_precision, total_mse, total_indices = [], [], [], [], []
        total_tp, total_fp, total_fn = [], [], []

        for i, row in enumerate(session.session_log.log_rows["ParetoWalkerAgent_Pareto"]):
            if "Precision" not in row or row["Precision"] == "-":
                continue

            total_f1.append(float(row["F1"]))
            total_recall.append(float(row["Recall"]))
            total_precision.append(float(row["Precision"]))
            total_mse.append(float(row["MSE"]))
            total_indices.append(float(row["WalkIndex"]))
            total_tp.append(float(row["TP"]))
            total_fp.append(float(row["FP"]))
            total_fn.append(float(row["FN"]))

            if counter >= len(self.round_by_round_f1):
                self.round_by_round_f1.append(float(row["F1"]))
                self.round_by_round_recall.append(float(row["Recall"]))
                self.round_by_round_precession.append(float(row["Precision"]))
                self.round_by_round_mse.append(float(row["MSE"]))
                self.round_by_round_indices.append(float(row["WalkIndex"]))
                self.round_by_round_counts.append(1)
            else:
                self.round_by_round_f1[counter] += float(row["F1"])
                self.round_by_round_recall[counter] += float(row["Recall"])
                self.round_by_round_precession[counter] += float(row["Precision"])
                self.round_by_round_mse[counter] += float(row["MSE"])
                self.round_by_round_indices[counter] += float(row["WalkIndex"])
                self.round_by_round_counts[counter] += 1

            counter += 1

        if len(total_precision) == 0:
            return {"ParetoWalkerAgent_Pareto": {"Precision": "-", "Recall": "-", "F1": "-", "TP": "-", "FP": "-", "FN": "-", "MSE": "-", "Avg.Diff": "-", "WalkIndex": "-"}}

        precision = np.mean(total_precision)
        recall = np.mean(total_recall)
        f1 = np.mean(total_f1)
        mse = np.mean(total_mse)
        indices = np.mean(total_indices)
        tp = np.mean(total_tp)
        fp = np.mean(total_fp)
        fn = np.mean(total_fn)

        return {"ParetoWalkerAgent_Pareto": {"Precision": precision, "Recall": recall, "F1": f1, "TP": tp, "FP": fp, "FN": fn, "MSE": mse, "WalkIndex": indices, "Avg.Diff": self.average_diff}}

    def on_tournament_end(self, tournament_logs: ExcelLog, agent_names: List[str], domain_names: List[str], estimator_names: List[str]):
        total_f1, total_recall, total_precision, total_mse, total_indices = [], [], [], [], []

        for row in tournament_logs.log_rows["ParetoWalkerAgent_Pareto"]:
            if "Precision" not in row or row["Precision"] == "-":
                continue

            total_f1.append(float(row["F1"]))
            total_recall.append(float(row["Recall"]))
            total_precision.append(float(row["Precision"]))
            total_mse.append(float(row["MSE"]))
            total_indices.append(float(row["WalkIndex"]))

        if len(total_f1) == 0:
            return

        with open(self.get_path("ParetoLogs.csv"), "w") as f:
            f.write("Precision;Recall;F1;MSE;WalkIndices\n")
            f.write(str(float(np.mean(total_precision))) + ";")
            f.write(str(float(np.mean(total_recall))) + ";")
            f.write(str(float(np.mean(total_f1))) + ";")
            f.write(str(float(np.mean(total_mse))) + ";")
            f.write(str(float(np.mean(total_indices))) + ";\n")


        f1 = np.array(self.round_by_round_f1) / np.array(self.round_by_round_counts)
        precision = np.array(self.round_by_round_precession) / np.array(self.round_by_round_counts)
        recall = np.array(self.round_by_round_recall) / np.array(self.round_by_round_counts)

        draw_line({"F1": f1, "Precision": precision, "Recall": recall}, y_axis_name="Value", x_axis_name="Round", save_path=self.get_path("Pareto_Precision_Recall_F1.png"))

        mse = np.array(self.round_by_round_mse) / np.array(self.round_by_round_counts)

        draw_line({"MSE": mse}, y_axis_name="MSE", x_axis_name="Round",
                  save_path=self.get_path("Pareto_MSE.png"))

        walk_indices = np.array(self.round_by_round_indices) / np.array(self.round_by_round_counts)

        draw_line({"WalkIndex": walk_indices}, y_axis_name="WalkIndex", x_axis_name="Round",
                  save_path=self.get_path("Pareto_Walk_Index.png"))

    def calculate_tp_fp_fn_precision_recall_f1(self, pareto: List[nenv.BidPoint]):
        tp = 0
        fp = 0
        fn = 0

        for real in self.real_pareto:
            if real in pareto:
                tp += 1.
            else:
                fn += 1.

        for predicted in pareto:
            if predicted not in self.real_pareto:
                fp += 1.

        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        if recall + precision == 0:
            f1 = 0.
        else:
            f1 = 2 * recall * precision / (precision + recall)

        return tp, fp, fn, precision, recall, f1

    def calculate_mse(self, session: Session, pareto: List[nenv.BidPoint]) -> float:
        sse = 0.

        for bid_point in pareto:
            closest_pareto_point = self.get_closest_real_pareto_point(session, bid_point)

            sse += closest_pareto_point - bid_point

        if len(pareto) == 0:
            return 0.

        mse = sse / len(pareto)

        return mse

    def get_closest_real_pareto_point(self, session: Session, bid_point: nenv.BidPoint) -> nenv.BidPoint:
        closest_one = self.real_pareto[0]

        if self.agent_pos == 'A':
            bid_point = nenv.BidPoint(bid_point.bid, bid_point.utility_a, session.agentB.preference.get_utility(bid_point.bid))
        else:
            bid_point = nenv.BidPoint(bid_point.bid, bid_point.utility_a, session.agentA.preference.get_utility(bid_point.bid))

        for pareto_point in self.real_pareto:
            if pareto_point - bid_point < closest_one - bid_point:
                closest_one = pareto_point

        return closest_one

    def get_pareto(self, preference_a: nenv.Preference, preference_b: nenv.Preference) -> List[nenv.BidPoint]:
        available_bids = preference_a.get_bids_at(0.6, 0., 1.)

        pareto_indices = extract_pareto_indices([(bid.utility, preference_b.get_utility(bid)) for bid in available_bids], 0.6)

        pareto_front = []

        for i in pareto_indices:
            bid = available_bids[i]

            pareto_front.append(nenv.BidPoint(bid, bid.utility, preference_b.get_utility(bid)))

        return pareto_front

    def get_average_bid_diff(self, preference_a: nenv.Preference, preference_b: nenv.Preference) -> float:
        bid_points = [nenv.BidPoint(None, preference_a.get_utility(b), preference_b.get_utility(b)) for b in preference_a.bids]

        total_diff = 0.
        counter = 0.

        for i in range(len(bid_points)):
            for j in range(i + 1, len(bid_points)):
                total_diff += bid_points[i] - bid_points[j]

                counter += 1

        return total_diff / counter
