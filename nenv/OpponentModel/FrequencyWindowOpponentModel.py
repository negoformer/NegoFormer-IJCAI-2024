import math
from typing import List

from nenv.Preference import Preference
from nenv.OpponentModel.EstimatedPreference import EstimatedPreference
from nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel
from nenv.Bid import Bid
from scipy.stats import chisquare


class FrequencyWindowOpponentModel(AbstractOpponentModel):
    issues: dict
    offers: List[Bid]
    alpha: float = 10.
    beta: float = 5.
    window_size: int = 48

    @property
    def name(self) -> str:
        return "Frequency Window Opponent Model"

    def __init__(self, reference: Preference):
        super().__init__(reference)
        self.offers = []

        self.issues = {
            issue: IssueEstimator(issue.values) for issue in reference.issues
        }

        for issue in self.issues.keys():
            self.issues[issue].weight = self._pref[issue]

            for value in issue.values:
                self.issues[issue].value_counter[value] = self._pref[issue, value]
                self.issues[issue].value_weights[value] = self._pref[issue, value]

    def update(self, bid: Bid, t: float):
        self.offers.append(bid)

        if t > 0.8:  # Do Not update in the last rounds.
            self.update_weights()
            return

        for issue_name, estimator in self.issues.items():
            estimator.update(bid[issue_name])

        if len(self.offers) < 2:
            self.update_weights()
            return

        if len(self.offers) % self.window_size == 0 and len(self.offers) >= 2 * self.window_size:
            current_window = self.offers[-self.window_size:]
            previous_window = self.offers[-2 * self.window_size:-self.window_size]

            self.update_issues(previous_window, current_window, t)

        self.update_weights()

    def update_issues(self, previous_window, current_window, t):
        not_changed = []
        concession = False

        def frequency(window: list, issue_name: str, issue_obj: IssueEstimator):
            values = []

            for value in issue_obj.value_weights.keys():
                total = 0.

                for bid in window:
                    if bid[issue_name] == value:
                        total += 1.

                values.append((1. + total) / (len(window) + len(issue_obj.value_counter)))

            return values

        for issue_name, issue_obj in self.issues.items():
            fr_current = frequency(current_window, issue_name, issue_obj)
            fr_previous = frequency(previous_window, issue_name, issue_obj)
            p_val = chisquare(fr_previous, fr_current)[1]

            if p_val > 0.05:
                not_changed.append(issue_obj)
            else:
                estimated_current = sum([fr_current[i] * w for i, w in enumerate(issue_obj.value_weights.values())])
                estimated_previous = sum([fr_previous[i] * w for i, w in enumerate(issue_obj.value_weights.values())])

                if estimated_current < estimated_previous:
                    concession = True

        if len(not_changed) != len(self.issues) and concession:
            for issue_obj in not_changed:
                issue_obj.weight += self.alpha * (1. - math.pow(t, self.beta))

        total_issue_weights = sum([issue_obj.weight for issue_obj in self.issues.values()])

        for issue_obj in self.issues.values():
            issue_obj.weight /= total_issue_weights

    def update_weights(self):
        for issue in self.issues.keys():
            self._pref[issue] = self.issues[issue].weight

            for value in issue.values:
                self._pref[issue, value] = self.issues[issue].value_weights[value]


class IssueEstimator:
    weight: float
    value_weights: dict
    value_counter: dict
    gamma: float = 0.25

    def __init__(self, values: list):
        self.value_weights = {value: 1. for value in values}
        self.value_counter = {value: 1. for value in values}

        self.weight = 1.

    def update(self, value: str):
        self.value_counter[value] += 1.

        max_value = max(self.value_counter.values())

        self.value_weights = {value_name: math.pow(self.value_counter[value_name], self.gamma) / math.pow(max_value, self.gamma) for value_name in self.value_counter.keys()}
