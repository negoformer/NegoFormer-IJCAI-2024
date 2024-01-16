import math
from typing import List, Union
import numpy as np
from nenv.Preference import Preference
from nenv.Bid import Bid


class BidPoint:
    """
        BidPoint class holds the Bid object with the utilit values of each agent.
    """
    __bid: Bid          # Corresponding Bid object
    __utility_a: float  # Utility value of AgentA
    __utility_b: float  # Utility value of AgentB

    def __init__(self, bid: Bid, utility_a: float, utility_b: float):
        """
            Constructor
        :param bid: Corresponding Bid object
        :param utility_a: Utility value of AgentA
        :param utility_b: Utility value of AgentB
        """
        self.__bid = bid
        self.__utility_a = utility_a
        self.__utility_b = utility_b

    @property
    def bid(self) -> Bid:
        """
        :return: Copy of Bid object without utility value
        """
        return self.__bid.copy_without_utility()

    @property
    def utility_a(self) -> float:
        """
        :return: Utility value of AgentA
        """
        return self.__utility_a

    @property
    def utility_b(self) -> float:
        """
        :return: Utility value of AgentB
        """
        return self.__utility_b

    @property
    def nash_product(self) -> float:
        """
        :return: Nash product of this BidPoint
        """
        return self.__utility_a * self.__utility_b

    @property
    def social_welfare(self) -> float:
        """
        :return: Social welfate of this BidPoint
        """
        return self.__utility_a + self.__utility_b

    def distance(self, bid_point) -> float:
        """
            This method calculates the Euclidean distance between the BidPoints
        :param bid_point: Other BidPoint
        :return: Euclidean distance between bid points in the corresponding bid space
        """
        return math.sqrt((self.__utility_a - bid_point.utility_a) ** 2 + (self.__utility_b - bid_point.utility_b) ** 2)

    def __sub__(self, bid_point) -> float:
        """
            "-" operator implementation that returns the distance between the bid points.
            Example: distance = bid_point1 - bid_point2
        :param bid_point: Other BidPoint
        :return: Euclidean distance between bid points in the corresponding bid space
        """
        return self.distance(bid_point)

    def __eq__(self, other):
        """
            "==" operator implementation that compares the offer contents.
        :param other: Bid, BidPoint or offer content
        :return: Whether the offer contents are same, or not
        """
        if isinstance(other, BidPoint):
            return self.__bid.__eq__(other.__bid)
        else:
            return self.__bid.__eq__(other)

    def __str__(self):
        """
        :return: String version of the offer content
        """
        return self.__bid.__str__()

    def __hash__(self):
        """
        :return: The hash value of the offer content
        """
        return self.__bid.__hash__()

    def __repr__(self):
        """
        :return: The representation of the offer content
        """
        return self.__bid.__repr__()

    def __gt__(self, other):
        """
            ">" operator implementation to compare two BidPoints in terms of the nash product
        :param other: Another BidPoint that will be compared
        :return: bid_point > other
        """
        return self.nash_product > other.nash_product

    def __ge__(self, other):
        """
            ">=" operator implementation to compare two BidPoints in terms of the nash product
        :param other: Another BidPoint that will be compared
        :return: bid_point >= other
        """
        return self.nash_product >= other.nash_product

    def __lt__(self, other):
        """
            "<" operator implementation to compare two BidPoints in terms of the nash product
        :param other: Another BidPoint that will be compared
        :return: bid_point < other
        """
        return self.nash_product < other.nash_product

    def __le__(self, other):
        """
            "<=" operator implementation to compare two BidPoints in terms of the nash product
        :param other: Another BidPoint that will be compared
        :return: bid_point <= other
        """
        return self.nash_product <= other.nash_product


class BidSpace:
    """
        Bid space of preferences of the agents.
    """
    prefA: Preference        # Preferences of agentA
    prefB: Preference        # Preferences of agentB
    __bids: List[BidPoint]   # The bid points of the bid space
    __nash_point: BidPoint   # Nash Point of the bid space
    __kalai_point: BidPoint  # Kalai Point of the bid space

    def __init__(self, prefA: Preference, prefB: Preference):
        """
            Constructor
        :param prefA: Preferences of agentA
        :param prefB: Preferences of agentB
        """
        self.prefA = prefA
        self.prefB = prefB
        self.__bids = []
        self.__nash_point = None
        self.__kalai_point = None

    @property
    def bid_points(self):
        """
            The bid points of the bid space. It is initiated when the first call.
        :return: The bid points of the bid space
        """
        if len(self.__bids) > 0:
            return self.__bids

        for bid in self.prefA.bids:
            self.__bids.append(
                BidPoint(bid.copy_without_utility(), self.prefA.get_utility(bid), self.prefB.get_utility(bid)))

            if self.__nash_point is None or self.__kalai_point is None:
                self.__nash_point = BidPoint(bid.copy_without_utility(), self.prefA.get_utility(bid),
                                             self.prefB.get_utility(bid))
                self.__kalai_point = BidPoint(bid.copy_without_utility(), self.prefA.get_utility(bid),
                                              self.prefB.get_utility(bid))

            if self.__nash_point.nash_product < self.__bids[-1].nash_product:
                self.__nash_point = BidPoint(bid.copy_without_utility(), self.prefA.get_utility(bid),
                                             self.prefB.get_utility(bid))
            if self.__kalai_point.social_welfare < self.__bids[-1].social_welfare:
                self.__kalai_point = BidPoint(bid.copy_without_utility(), self.prefA.get_utility(bid),
                                              self.prefB.get_utility(bid))

        return self.__bids

    @property
    def pareto(self) -> List[BidPoint]:
        """
        :return: List of BidPoint on pareto frontier
        """
        bids = self.bid_points

        pareto_indices = [True for _ in range(len(bids))]

        for i in range(len(bids)):
            point_i = np.array([bids[i].utility_a, bids[i].utility_b])
            for j in range(len(bids)):
                point_j = np.array([bids[j].utility_a, bids[j].utility_b])

                if all(point_j >= point_i) and any(point_j > point_i):
                    pareto_indices[i] = False

                    break

        pareto_bids = []

        for index in range(len(bids)):
            if pareto_indices[index]:
                pareto_bids.append(bids[index])

        return pareto_bids

    @property
    def nash_point(self) -> BidPoint:
        """
        :return: Nash point of the bid space as BidPoint
        """
        if self.__nash_point is None:
            _ = self.bid_points

        return self.__nash_point

    @property
    def kalai_point(self) -> BidPoint:
        """
        :return: Kalai point of the bid space as BidPoint
        """
        if self.__kalai_point is None:
            _ = self.bid_points

        return self.__kalai_point

    @property
    def nash_score(self) -> float:
        """
        :return: Nash Product of the Nash point
        """
        return self.nash_point.nash_product

    @property
    def kalai_score(self) -> float:
        """
        :return: Social Welfare of the Kalai point
        """
        return self.kalai_point.social_welfare

    def get_bid_point(self, bid: Bid) -> BidPoint:
        """
            This method converts given bid into BidPoint object
        :param bid: Corresponding bid that will be converted into BidPoint object
        :return:  that is converted from the given bid.
        """
        return BidPoint(bid, self.prefA.get_utility(bid), self.prefB.get_utility(bid))

    def nash_distance(self, target: Union[Bid, BidPoint]) -> float:
        """
            Euclidean distance between the target and Nash point
        :param target: The target as Bid or BidSpace
        :return: Euclidean distance
        """
        if isinstance(target, BidPoint):
            return target - self.nash_point
        if isinstance(target, Bid):
            return self.get_bid_point(target) - self.nash_point

    def kalai_distance(self, target: Union[Bid, BidPoint]) -> float:
        """
            Euclidean distance between the target and Kalai point
        :param target: The target as Bid or BidSpace
        :return: Euclidean distance
        """
        if isinstance(target, BidPoint):
            return target - self.kalai_point
        if isinstance(target, Bid):
            return self.get_bid_point(target) - self.kalai_point

    def __len__(self):
        """
        :return: Number of bids in the bid space
        """
        return len(self.__bids)

    def __iter__(self):
        """
            This method helps you to iterate over all BidPoint in that BidSpace object. Example:

            for bid_point in bid_space:
                ...
        :return: List Iterator
        """
        return self.__bids.__iter__()
