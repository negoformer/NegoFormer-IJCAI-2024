import os
import random
from typing import List, Dict

import numba

from nenv.Issue import Issue
from nenv.Bid import Bid
import json


class Preference:
    """
        Preference class represents the preferences of an agent during a negotiation session. It reads the profile data
        via json file. Thus, you can get the Issue and Issue-Value weights, reservation value, bid space in that domain,
        the utility value of a bid.

        Note: Preference object is mutual.
    """
    profile_json_path: str
    __issues: List[Issue]
    _issue_weights: Dict[Issue, float]
    _value_weights: Dict[Issue, Dict[str, float]]
    _bids: List[Bid]
    __reservation_value: float

    def __init__(self, profile_json_path: str, generate_bids: bool = True):
        """
            Constructor
        :param profile_json_path: Profile json file's path
        :param generate_bids: Automatically generate the bids. Default Value: true
        """
        self.profile_json_path = profile_json_path
        self.__issues = []
        self._issue_weights = {}
        self._value_weights = {}
        self._bids = []

        with open(profile_json_path, "r") as f:
            profile_data = json.load(f)

        self.__reservation_value = profile_data["reservationValue"]

        for issue_name in profile_data["issueWeights"].keys():
            issue = Issue(issue_name, list(profile_data["issues"][issue_name].keys()))

            self.__issues.append(issue)
            self._issue_weights[issue] = profile_data["issueWeights"][issue_name]

            self._value_weights[issue] = {}

            for value_name, value_weight in profile_data["issues"][issue_name].items():
                self._value_weights[issue][value_name] = value_weight

        # Generate bids
        if generate_bids:
            _ = self.bids

    @property
    def bids(self) -> List[Bid]:
        """
            This method provides the list of all possible bids in a domain. It extracts the bids on the first call.
            Also, the bids in the list are assigned the utility value, and they are sorted in descending order.
        :return: Sorted list of all bids in that domain
        """
        if len(self._bids) > 0:
            return self._bids

        bids = [Bid({}, -1)]

        # Generate all bid combinations
        for issue in self.__issues:
            new_bids = []

            for value_name in issue.values:
                for bid in bids:
                    _bid = bid.copy()

                    _bid[issue] = value_name

                    new_bids.append(_bid)

            bids = new_bids

        # Assign the utility of a bid
        for bid in bids:
            bid.utility = self.get_utility(bid)

        # Sort them descending order
        bids = sorted(bids, reverse=True)

        self._bids = bids

        return bids

    def get_utility(self, bid: Bid) -> float:
        """
            This method calculates the utility value of a given bid.
        :param bid: Target bid
        :return: Utility value of the bid
        """
        utility = 0.

        for issue_name, value_name in bid:
            utility += self._issue_weights[issue_name] * self._value_weights[issue_name][value_name]

        return utility

    def get_bid_at(self, target_utility: float) -> Bid:
        """
            This method returns the closest bid to provided target utility. Binary Search approach is applied.
        :param target_utility: Target utility
        :return: The closest bid
        """
        if target_utility >= self._bids[0]:
            return self._bids[0].copy()

        if target_utility <= self._bids[-1]:
            return self._bids[-1].copy()

        low: int = 0
        high: int = len(self._bids) - 1

        while low <= high:
            mid: int = (high + low) // 2

            if target_utility < self._bids[mid]:
                low = mid + 1
            elif target_utility > self._bids[mid]:
                high = mid - 1
            else:
                return self._bids[mid].copy()

        if abs(self._bids[low].utility - target_utility) < abs(self._bids[high].utility - target_utility):
            return self._bids[low].copy()

        return self._bids[high].copy()

    def get_bids_at_range(self, lower_bound: float = 0., upper_bound: float = 1.) -> List[Bid]:
        """
            This method provides a list of bids in the utility range.
        :param lower_bound: The lower bound of the range
        :param upper_bound: The upper bound of the range
        :return: List of bids in that range.
        """
        bids = []

        for bid in self._bids:
            utility = self.get_utility(bid)

            if utility < lower_bound:
                break

            if utility <= upper_bound:
                bids.append(bid.copy())

        return bids

    def get_bids_at(self, target_utility: float, lower_bound: float = 0., upper_bound: float = 0.) -> List[Bid]:
        """
            This method provides a list of bids in the utility window.
        :param target_utility: Center of the utility window
        :param lower_bound: The lower bound of the window = target_utility - lower_bound
        :param upper_bound: The upper bound of the window = target_utility + upper_bound
        :return: List of bids in that window.
        """
        bids = []

        for bid in self._bids:
            utility = self.get_utility(bid)

            if utility < target_utility - lower_bound:
                break

            if utility <= target_utility + upper_bound:
                bids.append(bid.copy())

        return bids

    def get_random_bid(self, lower_bound: float = 0., upper_bound: float = 1.):
        """
            This method randomly selects a bid in a range [max(lower_bound, reservation_value), upper_bound]
        :param lower_bound: Minimum utility value that the bid can obtain. If lower_bound < reservation_value,
        reservation value will be used for the lower bound of the range.
        :param upper_bound: Maximum utility value that the bid can obtain.
        :return: Randomly selected bid in that utility range.
        """
        lower_bound = max(lower_bound, self.__reservation_value)
        upper_bound = min(1., upper_bound)

        if lower_bound > upper_bound:
            return self.get_random_bid(upper_bound, lower_bound)

        target_utility = random.random() * (upper_bound - lower_bound) + lower_bound

        return self.get_bid_at(target_utility)

    def __copy__(self):
        """
        :return: Copy of Preference object
        """
        return Preference(self.profile_json_path)

    def copy(self):
        """
        :return: Copy of Preference object
        """
        return self.__copy__()

    @property
    def issues(self) -> List[Issue]:
        """
        :return: Copy of list of Issue in that domain
        """
        return self.__issues.copy()

    @property
    def reservation_value(self) -> float:
        """
        :return: Provided reservation value
        """
        return self.__reservation_value

    @property
    def issue_weights(self) -> Dict[Issue, float]:
        """
        :return: Copy of dictionary of Issue-Weight pairs.
        """
        return self._issue_weights.copy()

    @property
    def value_weights(self) -> Dict[Issue, Dict[str, float]]:
        """
        :return: Copy of dictionary of Issue-Value - Weight pairs
        """
        return self._value_weights.copy()

    @property
    def max_util_bid(self) -> Bid:
        """
        :return: The maximum utility value in the bid space
        """
        return self.bids[0].copy()

    @property
    def min_util_bid(self) -> Bid:
        """
        :return: The minimum utility value in the bid space
        """
        return self.bids[-1].copy()


def domain_loader(domain_name: str) -> (Preference, Preference):
    """
        This method generates the Preferences for both parties based on the given domain no.
    :param domain_name: The name of the domain
    :return: Preferences of profileA, Preferences of profileB
    """
    domain_path = f"domains/domain{domain_name}/"

    pref1 = Preference(os.path.join(domain_path, "profileA.json"))
    pref2 = Preference(os.path.join(domain_path, "profileB.json"))

    return pref1, pref2
