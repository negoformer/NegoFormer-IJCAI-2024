import math
import random
from typing import Union

import nenv
from nenv import Action, Bid


class PonPokoAgent(nenv.AbstractAgent):
    lastReceivedBid: nenv.Bid
    threshold_low: float
    threshold_high: float
    PATTERN_SIZE = 5
    pattern: int

    @property
    def name(self) -> str:
        return "PonPoko"

    def initiate(self, opponent_name: Union[None, str]):
        self.threshold_low = .99
        self.threshold_high = 1.0

        self.pattern = random.choice(list(range(self.PATTERN_SIZE + 1)))

    def receive_offer(self, bid: Bid, t: float):
        self.lastReceivedBid = bid.copy()

    def act(self, t: float) -> Action:
        if self.pattern == 0:
            self.threshold_high = 1 - .1 * t
            self.threshold_low = 1 - .1 * t - .1 * abs(math.sin(t * 40))
        elif self.pattern == 1:
            self.threshold_high = 1.
            self.threshold_low = 1 - .22 * t
        elif self.pattern == 2:
            self.threshold_high = 1. - .1 * t
            self.threshold_low = 1 - .1 * t - .15 * abs(math.sin(t * 20))
        elif self.pattern == 3:
            self.threshold_high = 1. - 0.05 * t
            self.threshold_low = 1. - 0.1 * t

            if t > .99:
                self.threshold_low = 1 - 0.3 * t
        elif self.pattern == 4:
            self.threshold_high = 1. - 0.15 * t * abs(math.sin(t * 20))
            self.threshold_low = 1. - 0.21 * t * abs(math.sin(t * 20))
        else:
            self.threshold_high = 1. - 0.1 * t
            self.threshold_low = 1. - 0.2 * abs(math.sin(t * 40))

        if self.can_accept():
            if self.preference.get_utility(self.lastReceivedBid) > self.threshold_low:
                return self.accept_action

        bid = None
        while bid is None:
            bid = self.selectBidfromList()

            if bid is None:
                self.threshold_low -= 0.0001

        return nenv.Action(bid)

    def selectBidfromList(self):
        bids = []

        for bid in self.preference.bids:
            if self.threshold_low <= bid.utility <= self.threshold_high:
                bids.append(bid)

            if bid.utility < self.threshold_low:
                break

        if len(bids) == 0:
            return None

        return random.choice(bids)
