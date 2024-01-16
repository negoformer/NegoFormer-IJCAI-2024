from typing import Union

import nenv
from agents.Kawaii.NegotiatiInfo import NegotiatingInfo
from agents.Kawaii.BidSearch import BidSearch
from agents.Kawaii.Strategy import Strategy


class Kawaii(nenv.AbstractAgent):
    negotiatingInfo: NegotiatingInfo
    bidSearch: BidSearch
    strategy: Strategy
    offeredBid: nenv.Bid

    def initiate(self, opponent_name: Union[None, str]):
        self.negotiatingInfo = NegotiatingInfo(self.preference)

        self.bidSearch = BidSearch(self.preference, self.negotiatingInfo)
        self.strategy = Strategy(self.preference, self.negotiatingInfo)

        self.offeredBid = None

    @property
    def name(self) -> str:
        return "Kawaii"

    def act(self, t: float) -> nenv.Action:
        if self.can_accept() and self.strategy.selectAccept(self.offeredBid, t):
            return self.accept_action

        return self.OfferAction(t)

    def OfferAction(self, t: float):
        offeredBid = self.bidSearch.getBid(self.preference.get_random_bid(), self.strategy.getThreshold(t))

        self.negotiatingInfo.MyBidHistory.append(offeredBid)

        return nenv.Action(offeredBid)

    def receive_offer(self, bid: nenv.Bid, t: float):
        sender = "OpponentAgent"

        if sender not in self.negotiatingInfo.opponents:
            self.negotiatingInfo.initOpponent(sender)

        self.offeredBid = bid.copy()

        self.negotiatingInfo.opponentsBool[sender] = False

