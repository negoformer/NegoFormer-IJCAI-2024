import math
from typing import Union


class TimeEstimator:
    """
        This component estimates the number of remaining rounds.
    """
    last_receive_time: float
    total_differences: float
    total_counting: int
    number_of_rounds: int

    def __init__(self):
        self.last_receive_time = 0.
        self.total_differences = 0.
        self.total_counting = 0
        self.number_of_rounds = 0

    def update(self, t: float):
        """
            This method updates the parameter when a bid is received.
        :param t: Current negotiation time
        :return: Nothing
        """
        diff = t - self.last_receive_time

        self.last_receive_time = t

        self.number_of_rounds += 1
        self.total_differences += diff * self.number_of_rounds
        self.total_counting += self.number_of_rounds

    def estimated_round_time(self) -> Union[float, None]:
        """
            This method estimates the required time for a round. It employs weighted averaging approach.
        :return: Required normalized negotiation time for a round.
        """
        if self.number_of_rounds < 2:
            return None

        return self.total_differences / self.total_counting

    def get_remaining_round(self, t: float) -> Union[int, None]:
        """
            This method estimates the number of remaining rounds.
        :param t: Current negotiation time
        :return: Number of remaining rounds
        """
        remaining_time = 1. - t

        return math.floor(remaining_time / self.estimated_round_time())
