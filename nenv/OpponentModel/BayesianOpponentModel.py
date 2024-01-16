import math

from nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel
from nenv.OpponentModel.EstimatedPreference import EstimatedPreference, Preference
from nenv.Bid import Bid
from nenv.Issue import Issue


class BayesianOpponentModel(AbstractOpponentModel):
    fWeightHyps: list
    fEvaluatorHyps: list
    fPreviousBidUtility: float
    issues: list
    fExpectedWeight: list
    fBiddingHistory: list

    def __init__(self, reference: Preference):
        super().__init__(reference)

        self.fPreviousBidUtility = 1.
        self.fBiddingHistory = []
        self.issues = reference.issues
        self.fExpectedWeight = [self._pref[issue] for issue in self.issues]

        self.initWeightHyps()

        self.fEvaluatorHyps = []

        for i in range(len(self.issues)):
            lEvalHyps = []

            self.fEvaluatorHyps.append(lEvalHyps)

            issue = self.issues[i]

            lDiscreteEval = {}

            for j in range(len(issue.values)):
                lDiscreteEval[issue.values[j]] = 1000 * j + 1

            lEvalHyps.append({"Prob": 1. / 3, "Desc": "uphill", "DiscreteEval": lDiscreteEval})

            lDiscreteEval = {}

            for j in range(len(issue.values)):
                lDiscreteEval[issue.values[j]] = 1000 * (len(issue.values) - j - 1) + 1

            lEvalHyps.append({"Prob": 1. / 3, "Desc": "downhill", "DiscreteEval": lDiscreteEval})

            if len(issue.values) > 2:
                lTotalTriangularFns = len(issue.values) - 1

                for k in range(1, lTotalTriangularFns):
                    lDiscreteEval = {}

                    for j in range(len(issue.values)):
                        if j < k:
                            lDiscreteEval[issue.values[j]] = 1000 * j / k
                        else:
                            lDiscreteEval[issue.values[j]] = 1000 * (len(issue.values) - j - 1) / (
                                        len(issue.values) - k - 1) + 1

                    lEvalHyps.append({"Prob": 0, "Desc": "triangular%d" % k, "DiscreteEval": lDiscreteEval})

            for eval in lEvalHyps:
                eval["Prob"] = 1. / len(lEvalHyps)

        for i in range(len(self.fExpectedWeight)):
            self.fExpectedWeight[i] = self.getExpectedWeight(i)

    def initWeightHyps(self):
        self.fWeightHyps = []
        lWeightHypsNumber = 11

        for i in range(len(self.issues)):
            lWeightHyps = []

            for j in range(lWeightHypsNumber):
                lHyp = {}
                lHyp["Prob"] = (1. - (j + 1) / lWeightHypsNumber) ** 3
                lHyp["Weight"] = j / (lWeightHypsNumber - 1)
                lWeightHyps.append(lHyp)

            lN = 0

            for j in range(lWeightHypsNumber):
                lN += lWeightHyps[j]["Prob"]

            for j in range(lWeightHypsNumber):
                lWeightHyps[j]["Prob"] /= lN

            self.fWeightHyps.append(lWeightHyps)

    def conditionalDistribution(self, pUtility: float, pPreviousBidUtility: float) -> float:
        lSigma = 0.25
        x = (pPreviousBidUtility - pUtility) / pPreviousBidUtility
        lResult = 1.0 / (lSigma * math.sqrt(2 * math.pi)) * math.exp(-(x * x) / (2. * lSigma * lSigma))

        return lResult

    def getExpectedEvaluationValue(self, pBid: Bid, pIssueNumber: int) -> float:
        lExpectedEval = 0.

        for j in range(len(self.fEvaluatorHyps[pIssueNumber])):
            lExpectedEval = lExpectedEval + self.fEvaluatorHyps[pIssueNumber][j]["Prob"] * \
                            self.get_expected_eval(self.fEvaluatorHyps[pIssueNumber][j]["DiscreteEval"],
                                                   pBid[self.issues[pIssueNumber]])

        return lExpectedEval

    def get_expected_eval(self, discrete_eval: dict, value_name: str) -> float:
        max_value = max(discrete_eval.values())
        if max_value < 0.00001:
            return 0.

        return discrete_eval[value_name] / max_value

    def getExpectedWeight(self, pIssueNumber: int) -> float:
        lExpectedWeight = 0.

        for i in range(len(self.fWeightHyps[pIssueNumber])):
            lExpectedWeight += self.fWeightHyps[pIssueNumber][i]["Prob"] * self.fWeightHyps[pIssueNumber][i]["Weight"]

        return lExpectedWeight

    def getPartialUtility(self, pBid: Bid, pIssueIndex: int) -> float:
        u = 0

        for j in range(len(self.issues)):
            if pIssueIndex == j:
                continue

            w = 0.

            for k in range(len(self.fWeightHyps[j])):
                w += self.fWeightHyps[j][k]["Prob"] * self.fWeightHyps[j][k]["Weight"]

            u += w * self.getExpectedEvaluationValue(pBid, j)

        return u

    def updateWeights(self):
        lBid = self.fBiddingHistory[-1]
        lWeightHyps = []

        for i in range(len(self.fWeightHyps)):
            lTmp = []

            for j in range(len(self.fWeightHyps[i])):
                lHyp = {"Weight": self.fWeightHyps[i][j]["Weight"], "Prob": self.fWeightHyps[i][j]["Prob"]}
                lTmp.append(lHyp)

            lWeightHyps.append(lTmp)

        for j in range(len(self.issues)):
            lN = 0.
            lUtility = 0.

            for i in range(len(self.fWeightHyps[j])):
                lUtility = self.fWeightHyps[j][i]["Weight"] * self.getExpectedEvaluationValue(lBid, j)
                lUtility += self.getPartialUtility(lBid, j)

                lN += self.fWeightHyps[j][i]["Prob"] * self.conditionalDistribution(lUtility, self.fPreviousBidUtility)

            for i in range(len(self.fWeightHyps[j])):
                lUtility = self.fWeightHyps[j][i]["Weight"] * self.getExpectedEvaluationValue(lBid, j)
                lUtility += self.getPartialUtility(lBid, j)

                lWeightHyps[j][i]["Prob"] = self.fWeightHyps[j][i]["Prob"] * self.conditionalDistribution(lUtility,
                                                                                                          self.fPreviousBidUtility) / (lN + 1e-12)

        self.fWeightHyps = lWeightHyps

    def updateEvaluationFns(self):
        lBid = self.fBiddingHistory[-1]

        lEvaluatorHyps = self.fEvaluatorHyps.copy()

        for i in range(len(self.fEvaluatorHyps)):
            lN = 0.

            for j in range(len(self.fEvaluatorHyps[i])):
                lHyp = self.fEvaluatorHyps[i][j]

                lN += lHyp["Prob"] * self.conditionalDistribution(self.getPartialUtility(lBid, i) +
                                                                  self.getExpectedWeight(i) *
                                                                  self.get_expected_eval(lHyp["DiscreteEval"],
                                                                                         self.issues[i].values[j]),
                                                                  self.fPreviousBidUtility)
            for j in range(len(self.fEvaluatorHyps[i])):
                lHyp = self.fEvaluatorHyps[i][j]
                lEvaluatorHyps[i][j]["Prob"] = lHyp["Prob"] * self.conditionalDistribution(
                    self.getPartialUtility(lBid, i) +
                    self.getExpectedWeight(i) *
                    self.get_expected_eval(lHyp["DiscreteEval"], self.issues[i].values[j]),
                    self.fPreviousBidUtility)

                lEvaluatorHyps[i][j]["Prob"] /= (lN + 1e-12)

        self.fEvaluatorHyps = lEvaluatorHyps

    def haveSeenBefore(self, pBid: Bid) -> bool:
        return pBid in self.fBiddingHistory

    def update(self, bid: Bid, t: float):
        if self.haveSeenBefore(bid):
            return

        self.fBiddingHistory.append(bid)

        if len(self.fBiddingHistory) > 1:
            self.updateWeights()
            self.updateEvaluationFns()
        else:
            self.updateEvaluationFns()

        self.fPreviousBidUtility -= 0.003

        for i in range(len(self.fExpectedWeight)):
            self.fExpectedWeight[i] = self.getExpectedWeight(i)

    def getExpectedUtility(self, bid: Bid) -> float:
        u = 0.

        for j in range(len(self.issues)):
            w = self.fExpectedWeight[j]

            u = u + w * self.getExpectedEvaluationValue(bid, j)

        return u

    @property
    def name(self) -> str:
        return "Bayesian Opponent Model"

    def getNormalizedWeight(self, i: Issue, startingNumber: int) -> float:
        sum = 0.

        for issue in self.issues:
            sum += self.getExpectedWeight(self.issues.index(issue) - startingNumber)

        return self.getExpectedWeight(self.issues.index(i) - startingNumber) / sum

    @property
    def preference(self) -> EstimatedPreference:
        for i, issue in enumerate(self.issues):
            self._pref[issue] = self.fExpectedWeight[i]

            for j, value in enumerate(issue.values):
                self._pref[issue, value] = self.get_expected_eval(self.fEvaluatorHyps[i][j]["DiscreteEval"], value)

        self._pref.normalize()

        return self._pref
