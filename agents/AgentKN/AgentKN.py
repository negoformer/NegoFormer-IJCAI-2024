from typing import Union

import nenv
from agents.AgentKN.etc.negotiatingInfo import negotiatingInfo
from agents.AgentKN.etc.bidSearch import bidSearch
from agents.AgentKN.etc.negotiationStrategy import strategy
from nenv import Bid


class AgentKN(nenv.AbstractAgent):
    negotiatingInfo: negotiatingInfo
    bidSearch: bidSearch
    negotiatingStrategy: strategy
    mLastReceivedBid: nenv.Bid
    mOfferedBid: nenv.Bid
    nrChosenActions: int
    history: list

    @property
    def name(self) -> str:
        return "AgentKN"

    def initiate(self, opponent_name: Union[None, str]):
        self.mOfferedBid = None
        self.mLastReceivedBid = None
        self.nrChosenActions = 0

        self.negotiatingInfo = negotiatingInfo(self.preference)
        self.bidSearch = bidSearch(self.preference, self.negotiatingInfo)
        self.negotiatingStrategy = strategy(self.preference, self.negotiatingInfo)
        self.history = []
        self.negotiatingInfo.updateOpponentsNum(1)

    def act(self, t: float) -> nenv.Action:
        self.negotiatingInfo.updateTimeScale(t)

        if self.can_accept() and self.negotiatingStrategy.selectAccept(self.mOfferedBid, t):
            return self.accept_action
        else:
            return self.OfferAction(t)

    def OfferAction(self, t: float) -> nenv.Action:
        offerBid = self.bidSearch.getBid(self.preference.get_random_bid(), self.negotiatingStrategy.getThreshold(t))

        self.negotiatingInfo.updateMyBidHistory(offerBid)

        return nenv.Action(offerBid)

    def receive_offer(self, bid: Bid, t: float):
        sender = "OpponentAgent"

        if sender not in self.negotiatingInfo.opponents:
            self.negotiatingInfo.initOpponent(sender)

        self.mOfferedBid = bid.copy()
        self.negotiatingInfo.updateInfo(sender, bid)
        self.negotiatingInfo.updateOfferedValueNum(sender, bid, t)
