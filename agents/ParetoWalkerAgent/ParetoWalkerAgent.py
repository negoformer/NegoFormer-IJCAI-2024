from __future__ import annotations

import random
from typing import Union, List, Dict
import nenv
from nenv import Action, Bid
from .MICROStrategy import MICROAgent
from .TimeEstimator import TimeEstimator
from .ParetoEstimator import ParetoEstimator


class ParetoWalkerAgent(nenv.AbstractAgent):
    """
        ParetoWalker Agent employing only ParetoWalker strategy. It does not employ NegoFormer prediction approach.
    """
    opponent_model: nenv.OpponentModel.AbstractOpponentModel
    time_estimator: TimeEstimator
    pareto_estimator: ParetoEstimator
    initial_strategy: MICROAgent
    pareto_index: int
    pareto_bid_point: Union[None | nenv.BidPoint]
    candidates: Dict[str, nenv.BidPoint]
    concession_count: int
    window_size: float
    main_strategy_starting_t: float
    NUMBER_OF_CONCESSION: int = 1
    MINIMUM_UTILITY: float = 0.5
    MINIMUM_OFFER: int = 96
    last_candidate: dict
    pareto: List[nenv.BidPoint]

    @property
    def name(self) -> str:
        return "ParetoWalkerAgent"

    def initiate(self, opponent_name: Union[None, str]):
        self.opponent_model = nenv.OpponentModel.FrequencyWindowOpponentModel(self.preference)

        self.MINIMUM_UTILITY = max(self.preference.reservation_value, self.MINIMUM_UTILITY)

        self.initial_strategy = MICROAgent(self.preference, self.session_time, self.MINIMUM_UTILITY)
        self.initial_strategy.initiate(opponent_name)

        self.time_estimator = TimeEstimator()

        self.pareto_estimator = ParetoEstimator(self.preference, self.opponent_model, self.MINIMUM_UTILITY, self.MINIMUM_OFFER)

        self.pareto_index = -1

        self.pareto_bid_point = None

        self.candidates = {}

        self.concession_count = 0

        self.main_strategy_starting_t = -1

        self.last_candidate = {}
        self.pareto = []

        domain_size = len(self.preference.bids)

        if domain_size < 450:
            self.window_size = 0.050
        elif domain_size < 1500:
            self.window_size = 0.045
        elif domain_size < 4500:
            self.window_size = 0.040
        elif domain_size < 18000:
            self.window_size = 0.035
        elif domain_size < 33000:
            self.window_size = 0.030
        else:
            self.window_size = 0.025

    def receive_offer(self, bid: Bid, t: float):
        self.opponent_model.update(bid, t)

        self.time_estimator.update(t)

        self.initial_strategy.receive_offer(bid, t)

        # Check concession
        if len(self.last_received_bids) >= 2 and self.pareto_index > -1 and t >= 0.95:
            prev_bid = self.last_received_bids[-2]
            estimated_preference = self.opponent_model.preference

            move = nenv.utils.get_move(estimated_preference.get_utility(prev_bid),
                                       estimated_preference.get_utility(bid), prev_bid.utility, bid.utility)

            if move in ["Concession", "Nice", "Fortunate"]:
                self.concession_count += 1

    def act(self, t: float) -> Action:
        if len(self.last_received_bids) >= self.MINIMUM_OFFER:  # In first rounds, apply Initial Strategy
            action = self.initial_strategy.act(t)

            if self.can_accept() and isinstance(action, nenv.Accept):
                return self.accept_action

            bid = action.bid

            return nenv.Offer(bid)

        if t > 0.95:  # Special Ending strategy
            estimated_remaining_round = self.time_estimator.get_remaining_round(t)
            if estimated_remaining_round <= 3:
                max_bid = max(self.last_received_bids, key=lambda b: b.utility)

                if self.can_accept() and self.last_received_bids[-1].utility >= max_bid.utility:
                    return self.accept_action

                return nenv.Offer(max_bid)

        # Update pareto
        pareto = self.pareto_estimator.get_pareto(self.window_size)

        if self.pareto_index == -1:  # If it is first
            self.pareto_index = 0

            self.pareto_bid_point = pareto[0]

            self.main_strategy_starting_t = t

        else:
            self.pareto_bid_point = self.get_pareto_point(t, pareto)

        # Get candidates
        self.candidates = self.pareto_estimator.get_candidate_bids(self.pareto_bid_point, self.window_size)

        # Predict the slopes
        bid_infos = {}

        for candidate_bid_point in self.candidates.values():
            if candidate_bid_point not in bid_infos:
                bid_infos[candidate_bid_point] = random.random()

        candidates = {
            key: (bid_infos[self.candidates[key]], self.candidates[key]) for key in self.candidates
        }

        # Find the min. slope
        bid_point = min(bid_infos.keys(), key=lambda b: (bid_infos[b], -b.utility_a))
        bid = bid_point.bid
        bid.utility = bid_point.utility_a

        # Update Loggers
        self.pareto = pareto
        self.last_candidate = candidates

        # Acceptance conditions
        if self.can_accept() and self.last_received_bids[-1].utility >= bid.utility:
            return self.accept_action

        return nenv.Offer(bid)

    def get_pareto_point(self, t: float, pareto: List[nenv.BidPoint]):
        """
            This method provides an estimated pareto point based on Pareto Walker strategy.
        :param t: Current negotiation time
        :param pareto: List of pareto points
        :return: Selected pareto point based on the strategy
        """
        if t < 0.95:  # Time-Based pareto walking
            # Find nash index on the pareto
            nash_index = 0

            for i, pareto_point in enumerate(pareto):
                if pareto_point.nash_product > pareto[nash_index].nash_product:
                    nash_index = i

            self.pareto_index = min(round((t - self.main_strategy_starting_t) / (0.9 - self.main_strategy_starting_t) * nash_index), nash_index)

            self.pareto_bid_point = pareto[self.pareto_index]

            return self.pareto_bid_point

        # Update last pareto bid
        estimated_preference = self.opponent_model.preference

        self.pareto_bid_point = nenv.BidPoint(self.pareto_bid_point.bid, self.pareto_bid_point.utility_a, estimated_preference.get_utility(self.pareto_bid_point.bid))

        self.pareto_index = self.pareto_estimator.get_closest_pareto_point_index(self.pareto_bid_point, pareto)

        # Update pareto index when concession
        if self.concession_count >= self.NUMBER_OF_CONCESSION:
            self.pareto_index = min(self.pareto_index + 1, len(pareto) - 1)

            self.concession_count = 0

        self.pareto_bid_point = pareto[self.pareto_index]

        return self.pareto_bid_point
