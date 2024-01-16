import random
from typing import List, Union
import agents
from agents.Caduceus2015.UtilFunctions import *
import nenv


class Caduceus(nenv.AbstractAgent):
    discountFactor: float
    selfReservationValue: float
    percentageOfOfferingBestBid: float
    random: random.Random
    agents: List[nenv.AbstractAgent]
    scores: List[float]

    def getScore(self, agentIndex: int):
        return self.scores[agentIndex]

    def initiate(self, opponent_name: Union[None, str]):
        self.random = random.Random()
        self.discountFactor = 1.
        self.selfReservationValue = max(0.75, self.preference.reservation_value)
        self.scores = normalize([100, 10, 5, 3, 1])
        self.percentageOfOfferingBestBid = 0.83

        self.agents = [
            agents.ParsAgent(self.preference, self.session_time, []),
            agents.RandomDance(self.preference, self.session_time, []),
            agents.Kawaii(self.preference, self.session_time, []),
            agents.Atlas3Agent(self.preference, self.session_time, []),
            agents.Caduceus2015(self.preference, self.session_time, [])
        ]

        for agent in self.agents:
            agent.initiate(opponent_name)

    @property
    def name(self) -> str:
        return "Caduceus"

    def act(self, t: float) -> nenv.Action:
        if self.isBestOfferTime(t):
            return nenv.Action(self.preference.bids[0])

        bidsFromAgents = []
        possibleActions = []

        for agent in self.agents:
            possibleActions.append(agent.act(t))

        scoreOfAccepts = 0
        scoreOfBids = 0

        agentsWithBids = []

        for i, action in enumerate(possibleActions):
            if isinstance(action, nenv.Accept):
                scoreOfAccepts += self.getScore(i)
            else:
                scoreOfBids += self.getScore(i)
                bidsFromAgents.append(action.bid)
                agentsWithBids.append(i)

        if self.can_accept() and scoreOfAccepts > scoreOfBids:
            return self.accept_action
        elif scoreOfBids > scoreOfAccepts:
            return nenv.Action(self.getRandomizedAction(agentsWithBids, bidsFromAgents))

        return nenv.Action(self.preference.bids[0])

    def getRandomizedAction(self, agentsWithBids: list, bidsFromAgents: list):
        possibilities = [self.getScore(agentWithBid) for agentWithBid in agentsWithBids]

        possibilities = normalize(possibilities)

        randomPick = self.random.random()

        acc = 0.
        for i, possibility in enumerate(possibilities):
            acc += possibility

            if randomPick < acc:
                return bidsFromAgents[i]

        return bidsFromAgents[-1]

    def receive_offer(self, bid: nenv.Bid, t: float):
        for agent in self.agents:
            agent.receive_bid(bid, t)

    def isBestOfferTime(self, t: float):
        return t < self.percentageOfOfferingBestBid
