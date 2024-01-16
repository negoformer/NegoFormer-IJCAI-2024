from typing import Dict, Union
import nenv
from agents.NegoFormerAgent.utils import *


class ParetoEstimator:
    """
        This class estimates the Pareto during negotiation
    """
    opponent_model: nenv.OpponentModel.AbstractOpponentModel
    minimum_utility: float
    preference: nenv.Preference
    available_bids: List[nenv.Bid]
    pareto: Union[None, List[nenv.BidPoint]]
    last_pareto_update: int
    pareto_update_frequency: int

    def __init__(self, preference: nenv.Preference, opponent_model: nenv.OpponentModel.AbstractOpponentModel, minimum_utility: float, update_frequency: int):
        self.preference = preference
        self.opponent_model = opponent_model
        self.minimum_utility = minimum_utility

        self.available_bids = preference.get_bids_at_range(minimum_utility)

        self.pareto = None
        self.last_pareto_update = 0
        self.pareto_update_frequency = update_frequency

    def get_pareto(self, window_size: float) -> List[nenv.BidPoint]:
        """
            This method estimates the pareto front bids.
        :param window_size: Window size
        :return: List of pareto bids
        """
        estimated_preference = self.opponent_model.preference

        if self.pareto is not None and self.last_pareto_update < self.pareto_update_frequency:
            self.last_pareto_update += 1

            self.pareto = [nenv.BidPoint(b.bid, b.utility_a, estimated_preference.get_utility(b.bid)) for b in self.pareto]

            return self.pareto

        pareto_front = []

        pareto_indices = extract_pareto_indices([(bid.utility, estimated_preference.get_utility(bid)) for bid in self.available_bids], self.minimum_utility + window_size * 2)
        for i in pareto_indices:
            bid = self.available_bids[i]

            pareto_front.append(nenv.BidPoint(bid, bid.utility, estimated_preference.get_utility(bid)))

        self.pareto = pareto_front
        self.last_pareto_update = 0

        return pareto_front

    def get_candidate_bids(self, pareto_point: nenv.BidPoint, window_size: float) -> Dict[str, nenv.BidPoint]:
        """
            This method extracts the candidate bids from the pool.
        :param pareto_point: Current pareto point
        :param window_size: Window size
        :return: Dictionary of candidate bids
        """
        candidates = {
            "Pareto": pareto_point
        }

        pool = self.get_pareto_ball(pareto_point, window_size)

        if len(pool) <= 1:
            candidates["Nash"] = pareto_point
            candidates["Kalai"] = pareto_point
            candidates["MaxOpp"] = pareto_point
            candidates["Center"] = pareto_point

            return candidates

        candidates["Nash"] = max(pool, key=lambda b: (b.utility_a * b.utility_b, b.utility_a))
        candidates["Kalai"] = max(pool, key=lambda b: (b.utility_a + b.utility_b, b.utility_a))
        candidates["MaxOpp"] = max(pool, key=lambda b: (b.utility_b, b.utility_a))

        center_bid_point = nenv.BidPoint(None, pareto_point.utility_a - window_size, pareto_point.utility_b)
        candidates["Center"] = min(pool, key=lambda b: b - center_bid_point)

        return candidates

    @staticmethod
    def get_closest_pareto_point_index(point: nenv.BidPoint, pareto: List[nenv.BidPoint]) -> int:
        """
            Find the closest pareto point from a given bid point.
        :param point: Target point
        :param pareto: List of bids on the pareto front.
        :return: Pareto index of the closest pareto bid.
        """
        min_distance = float("inf")
        closest_point = -1

        for i, bid_point in enumerate(pareto):
            if bid_point.bid == point.bid:
                return i

            distance = bid_point - point

            if distance < min_distance:
                min_distance = distance
                closest_point = i

        return closest_point

    def get_pareto_ball(self, pareto_point: nenv.BidPoint, window_size: float, minimum_number_of_bids: int = 5) -> List[nenv.BidPoint]:
        """
            This method generates the pareto ball (i.e., bid pool) from the given pareto point and window size.
        :param pareto_point: Pareto point
        :param window_size: Window size of the pool
        :param minimum_number_of_bids: Minimum number of bids must be in that pool
        :return: Bid pool
        """
        center_utility_agent = pareto_point.utility_a - window_size
        center_utility_opp = pareto_point.utility_b

        # Center Point under the pareto point

        center_point = nenv.BidPoint(None, center_utility_agent, center_utility_opp)

        bids = self.preference.get_bids_at_range(max(self.minimum_utility, center_utility_agent - window_size), center_utility_agent + window_size)

        estimated_preference = self.opponent_model.preference

        pool = []

        for bid in bids:
            bid_point = nenv.BidPoint(bid, bid.utility, estimated_preference.get_utility(bid))

            if bid_point - center_point <= window_size:
                pool.append(bid_point)

        # Minimum number of bids in the pool

        if len(pool) < minimum_number_of_bids:
            bids = self.preference.get_bids_at(center_utility_agent, window_size, 1.0)

            bids.sort(key=lambda b: nenv.BidPoint(b, b.utility, estimated_preference.get_utility(b)) - pareto_point)

            while len(pool) < minimum_number_of_bids and len(bids) > 0:
                bid = bids.pop(0)

                if bid != pareto_point.bid:
                    pool.append(nenv.BidPoint(bid, bid.utility, estimated_preference.get_utility(bid)))

        # Always add Pareto Point
        pool.append(pareto_point)

        return pool
