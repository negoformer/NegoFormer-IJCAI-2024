from typing import Union

import nenv
from agents.Atlas3.etc.negotiatingInfo import negotiatingInfo
from agents.Atlas3.etc.bidSearch import bidSearch
from agents.Atlas3.etc.strategy import strategy
from nenv import Bid


class Atlas3Agent(nenv.AbstractAgent):
    negotiatingInfo: negotiatingInfo
    bidSearch: bidSearch
    strategy: strategy
    rv: float
    offeredBid: nenv.Bid
    supporter_num: int
    CList_index: int

    @property
    def name(self) -> str:
        return "Atlas3"

    def initiate(self, opponent_name: Union[None, str]):
        self.offeredBid = None
        self.supporter_num = 0
        self.CList_index = 0

        self.negotiatingInfo = negotiatingInfo(self.preference)
        self.bidSearch = bidSearch(self.preference, self.negotiatingInfo)
        self.strategy = strategy(self.preference, self.negotiatingInfo)
        self.rv = self.preference.reservation_value

    def act(self, t: float) -> nenv.Action:
        self.negotiatingInfo.updateTimeScale(t)

        CList = self.negotiatingInfo.pb_list

        if t > 1. - self.negotiatingInfo.time_scale * (len(CList) + 1):
            return self.chooseFinalAction(self.offeredBid, CList, t)

        if self.can_accept() and self.strategy.selectAccept(self.offeredBid, t):
            return self.accept_action

        return self.OfferAction(t)

    def chooseFinalAction(self, offeredBid: nenv.Bid, CList: list, t: float) -> nenv.Action:
        offered_bid_util = 0.

        if offeredBid is not None:
            offered_bid_util = self.preference.get_utility(offeredBid)

        if self.CList_index >= len(CList):
            if offered_bid_util >= self.rv:
                return self.accept_action

            return self.OfferAction(t)

        CBid = CList[self.CList_index]
        CBid_util = self.preference.get_utility(CBid)

        if CBid_util > offered_bid_util and CBid_util > self.rv:
            self.CList_index += 1
            self.negotiatingInfo.updateMyBidHistory(CBid)

            return nenv.Action(CBid)
        elif offered_bid_util > self.rv:
            return self.accept_action

        return self.OfferAction(t)

    def OfferAction(self, t: float) -> nenv.Action:
        offerBid = self.bidSearch.getBid(self.preference.get_random_bid(), self.strategy.getThreshold(t))

        self.negotiatingInfo.updateMyBidHistory(offerBid)

        return nenv.Action(offerBid)

    def receive_offer(self, bid: Bid, t: float):
        sender = "OpponentAgent"

        if sender not in self.negotiatingInfo.opponents:
            self.negotiatingInfo.initOpponent(sender)

        self.supporter_num = 1
        self.offeredBid = bid.copy()
        self.negotiatingInfo.updateInfo(sender, bid)
