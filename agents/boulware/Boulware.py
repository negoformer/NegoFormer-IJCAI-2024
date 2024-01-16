from typing import Union

import nenv
from nenv import Action, Bid, Action


class BoulwareAgent(nenv.AbstractAgent):
    """
        Time-Based Agent

        Rustam M. Vahidov, Gregory E. Kersten, and Bo Yu. 2017. Human-Agent Ne-gotiations: The Impact Agents’ Concession
        Schedule and Task Complexity onAgreements. In 50th Hawaii International Conference on System Sciences,
        HICSS2017, Tung Bui (Ed.). ScholarSpace / AIS Electronic Library (AISeL), Hawaii, 1–9
    """
    p0: float   # Initial utility
    p1: float   # Concession ratio
    p2: float   # Final utility

    @property
    def name(self) -> str:
        return "Boulware"

    def initiate(self, opponent_name: Union[None, str]):
        # Set default values
        self.p0 = 1.0
        self.p1 = 0.85
        self.p2 = 0.4

    def receive_offer(self, bid: Bid, t: float):
        # Do nothing when an offer received.

        pass

    def act(self, t: float) -> Action:
        # Calculate target utility to offer
        target_utility = (1 - t) * (1 - t) * self.p0 + 2 * (1 - t) * t * self.p1 + t * t * self.p2

        # Target utility cannot be lower than the reservation value.
        if target_utility < self.preference.reservation_value:
            target_utility = self.preference.reservation_value

        # AC_Next strategy to decide accepting or not
        if self.can_accept() and target_utility <= self.last_received_bids[-1]:
            return self.accept_action

        # Find the closest bid to target utility
        bid = self.preference.get_bid_at(target_utility)

        return Action(bid)
