import pickle as pkl
from typing import List, Dict
import torch
from .TimeEstimator import TimeEstimator
from .Autoformer import Model
from .NegoFormerProcessor import *
from sklearn.linear_model import LinearRegression


class NegoFormer:
    time_estimator: TimeEstimator
    opponent_model: nenv.OpponentModel.AbstractOpponentModel
    model: Model
    bid_point_history: List[OfferPoint]

    INPUT_FEATURE_SIZE: int = 12
    OUTPUT_LENGTH: int = 336
    INPUT_LENGTH: int = 96

    predictions: Dict[float, List[float]]
    current_predictions: Dict[nenv.Bid, Dict[float, float]]

    def __init__(self, time_estimator: TimeEstimator, opponent_model: nenv.OpponentModel.AbstractOpponentModel):
        self.time_estimator = time_estimator
        self.opponent_model = opponent_model

        self.bid_point_history = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device:", self.device)

        self.model = Model(self.INPUT_FEATURE_SIZE, self.OUTPUT_LENGTH).to(self.device)

        self.current_predictions = {}
        self.predictions = {}

    def load_model(self, file_path: str = 'agents//NegoFormerAgent//model.pkl'):
        with open(file_path, "rb") as f:
            self.model.load_state_dict(pkl.load(f))

        self.model.train(False)

    def receive_bid(self, bid: nenv.Bid, t: float):
        self.bid_point_history.append(OfferPoint(1, bid, t))

    def update(self, bid: nenv.Bid, t: float):
        self.bid_point_history.append(OfferPoint(-1, bid, t))

        if bid in self.current_predictions:
            selected_predictions = self.current_predictions[bid]

            for t, pred in selected_predictions.items():
                if t not in self.predictions:
                    self.predictions[t] = []

                self.predictions[t].append(pred)

        self.current_predictions = {}

    def is_ready(self) -> bool:
        return len(self.bid_point_history) >= self.INPUT_LENGTH * 2

    def predict(self, bid: nenv.Bid, t: float) -> float:
        target_times = self.time_estimator.populate(t, self.OUTPUT_LENGTH)
        number_of_prediction = len(target_times)

        X = np.zeros((1, self.INPUT_LENGTH, self.INPUT_FEATURE_SIZE))
        T = np.zeros((1, self.INPUT_LENGTH, 1))
        T_Target = np.zeros((1, self.OUTPUT_LENGTH, 1))

        for i in range(number_of_prediction):
            T_Target[0, i, 0] = target_times[i]

        self.update(bid, t)

        estimated_preference = self.opponent_model.preference

        bid_point_history = self.bid_point_history[-self.INPUT_LENGTH:]

        for i, bid_point in enumerate(bid_point_history):
            T[0, i, 0] = bid_point.t

            if i >= 2:
                x = process(bid_point_history[i - 2], bid_point, t, estimated_preference)
            else:
                x = process(None, bid_point, t, estimated_preference)

            X[0, i, :] = x

        self.bid_point_history.pop(-1)

        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).to(self.device)
            t_tensor = torch.FloatTensor(T).to(self.device)
            t_t_tensor = torch.FloatTensor(T_Target).to(self.device)

            prediction = self.model(x_tensor, x_tensor, t_tensor, t_t_tensor)[0]

        output_length = min(self.OUTPUT_LENGTH, self.time_estimator.get_remaining_round(t))

        prediction = prediction.detach().cpu().numpy()

        y_pred = np.reshape(np.clip(prediction[:output_length, -1], 0., 1.), (output_length, ))

        last_estimated_opp_utility = estimated_preference.get_utility(self.bid_point_history[-1].bid)

        if output_length <= 1:
            slope = 0.
        else:
            slope = self.calculate_slope(T_Target[0, :output_length], y_pred, last_estimated_opp_utility)

        self.current_predictions[bid] = {float(t): pred for t, pred in zip(T_Target[0, :output_length], y_pred)}

        return slope

    def calculate_slope(self, times: np.ndarray, prediction: np.ndarray, last_estimated_opp_utility: float) -> float:
        linear_model = LinearRegression(fit_intercept=False)

        linear_model.fit(times, prediction - last_estimated_opp_utility)

        slope = round(float(linear_model.coef_[0]), 4)

        return slope
