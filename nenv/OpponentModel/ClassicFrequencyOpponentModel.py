from typing import List
from nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel
from nenv.Preference import Preference
from nenv.Bid import Bid


class ClassicFrequencyOpponentModel(AbstractOpponentModel):
    issue_counts: dict
    value_counts: dict
    alpha: float
    opponent_bids: List[Bid]

    def __init__(self, reference: Preference):
        super().__init__(reference)

        self.alpha = 0.1
        self.opponent_bids = []
        self.issue_counts = {}
        self.value_counts = {}

        for issue in reference.issues:
            self.issue_counts[issue] = self._pref[issue]

            self.value_counts[issue] = {}

            for value in issue.values:
                self.value_counts[issue][value] = self._pref[issue, value]

        self._pref.normalize()

    @property
    def name(self) -> str:
        return "Classic Frequency Opponent Model"

    def update(self, bid: Bid, t: float):
        self.opponent_bids.append(bid)

        for issue, value in bid:
            self.value_counts[issue][value] += 1.

            if len(self.opponent_bids) >= 2 and self.opponent_bids[-2][issue] == value:
                self.issue_counts[issue] += self.alpha * (1. - t)

        self.update_weights()

    def update_weights(self):
        sum_issues = sum(self.issue_counts.values())

        for issue in self._pref.issues:
            self._pref[issue] = self.issue_counts[issue] / sum_issues

            max_value = max(self.value_counts[issue].values())

            for value in issue.values:
                self._pref[issue, value] = self.value_counts[issue][value] / max_value
