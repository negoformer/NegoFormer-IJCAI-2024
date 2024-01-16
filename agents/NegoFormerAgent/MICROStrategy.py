import random
from typing import Union
import nenv
from nenv import Action, Bid


class MICROAgent(nenv.AbstractAgent):
    """
        MICRO Agent concedes only when the opponent concedes

        Dave de Jonge. An analysis of the linear bilateral ANAC domains using
        the MiCRO benchmark strategy. In IJCAI 2022, Vienna, Austria, 2022.
    """
    my_last_bid: nenv.Bid   # My last offered bid
    index: int              # Number of unique proposals made by me
    increase: int           # Increase amount
    min_utility: float      # Lower bound

    def __init__(self, preference: nenv.Preference, session_time: int, min_utility: float):
        super().__init__(preference, session_time, [])

        self.min_utility = min_utility

    @property
    def name(self) -> str:
        return "MICRO"

    def initiate(self, opponent_name: Union[None, str]):
        # Set default values
        self.my_last_bid = self.preference.bids[0]
        self.index = 0
        self.increase = 1

    def receive_offer(self, bid: Bid, t: float):
        pass

    def act(self, t: float) -> Action:
        # In first 2 rounds, offer the bid with the highest utility for me
        if len(self.last_received_bids) < 2:
            return nenv.Action(self.my_last_bid)

        # Utility difference of the last two received bids.
        diff = self.last_received_bids[-1].utility - self.last_received_bids[-2].utility

        if diff >= 0:  # If the opponent conceded
            # Increase the number of unique proposals made by me
            self.index += self.increase

            # Check index
            self.index = min(self.index, len(self.preference.bids))

            # Make concession
            self.my_last_bid = self.preference.bids[self.index]

            # Check reservation value and lower bound
            if self.my_last_bid.utility < self.preference.reservation_value or self.my_last_bid < self.min_utility:
                self.my_last_bid = self.preference.get_bid_at(max(self.preference.reservation_value, self.min_utility))

            # AC_Next strategy to decide accepting or not
            if self.preference.bids[self.index] <= self.last_received_bids[-1] >= self.preference.reservation_value:
                return self.accept_action

            return nenv.Offer(self.my_last_bid)
        else:
            # AC_Next strategy to decide accepting or not
            if self.preference.bids[self.index] <= self.last_received_bids[-1] >= self.preference.reservation_value:
                return self.accept_action

            # Select a random bid
            self.my_last_bid = self.preference.bids[self.index] if self.index == 0 else random.choice(self.preference.bids[:self.index + 1])

            # Check reservation value
            if self.my_last_bid.utility < self.preference.reservation_value:
                self.my_last_bid = self.preference.get_bid_at(self.preference.reservation_value)

            return nenv.Offer(self.my_last_bid)
