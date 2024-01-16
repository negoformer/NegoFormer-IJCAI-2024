from nenv.Preference import Preference
from nenv.Issue import Issue


class EstimatedPreference(Preference):
    """
        Preference object is mutual. Thus, Opponent Models (i.e., Estimators) generate EstimatedPreference object which
        enable to change Issue and Value weights.
    """
    def __init__(self, reference: Preference):
        """
            Constructor
        :param reference: Reference Preference to get domain information.
        """
        super(EstimatedPreference, self).__init__(reference.profile_json_path, generate_bids=False)

        for issue in self._issue_weights.keys():
            self._issue_weights[issue] = 1. - reference.issue_weights[issue]

            for value in issue.values:
                self._value_weights[issue][value] = 1. - reference.value_weights[issue][value]

        self.normalize()

    def __getitem__(self, key) -> float:
        """
            You can reach Issue and Value weight as shown in below:
            - For Issue Weight, you can use Issue object or IssueName (as string):
                estimated_preference[Issue] or estimated_preference[IssueName]
            - For Value Weight: you can use Issue-Value pair where Issue is an Issue object or IssueName as string:
                estimated_preference[Issue, Value] or estimated_preference[IssueName, Value]
        :param key: Issue or Issue-Value pair or IssueName-Value pair
        :return: Weight of Issue or Value
        """
        if isinstance(key, tuple) and len(key) == 2:
            return self._value_weights[key[0]][key[1]]

        return self._issue_weights[key]

    def __setitem__(self, key, weight: float):
        """
            You can reach Issue and Value weight as shown in below:
            - For Issue Weight, you can use Issue object or IssueName (as string):
                estimated_preference[Issue] = 0.5 or estimated_preference[IssueName] = 0.5
            - For Value Weight: you can use Issue-Value pair where Issue is an Issue object or IssueName as string:
                estimated_preference[Issue, Value] = 0.5 or estimated_preference[IssueName, Value] = 0.5
            :param key: Issue or Issue-Value pair or IssueName-Value pair
            :return: Weight of Issue or Value
            """
        if isinstance(key, tuple) and len(key) == 2:
            self._value_weights[key[0]][key[1]] = weight
        else:
            self._issue_weights[key] = weight

    def get_issue_weight(self, issue: Issue) -> float:
        """
        :param issue: Issue object or IssueName as string
        :return: Weight of corresponding Issue
        """
        return self._issue_weights[issue]

    def get_value_weight(self, issue: Issue, value: str) -> float:
        """
        :param issue: Issue object or IssueName as string
        :param value: Value as string
        :return: Weight of corresponding Issue-Value pair
        """
        return self._value_weights[issue][value]

    def set_issue_weight(self, issue: Issue, weight: float):
        """
            Change Issue Weight
        :param issue: Issue object or IssueName as string
        :param weight: New weight that will be assigned
        :return: Nothing
        """
        self._issue_weights[issue] = weight

    def set_value_weight(self, issue: Issue, value: str, weight: float):
        """
            Change Value weight
        :param issue: Issue object or IssueName as string
        :param value: Value as string
        :param weight: New weight that will be assigned
        :return: Nothing
        """
        self._value_weights[issue][value] = weight

    def normalize(self):
        """
            This method normalize the Issue and Value weights.
            - Value weights must be in [0.0-1.0] range
            - Sum of Issue weights must be 1.0
        :return: Nothing
        """
        issue_total = sum(self._issue_weights.values())

        for issue in self.issues:
            if issue_total == 0:
                self._issue_weights[issue] = 1. / len(self.issues)
            else:
                self._issue_weights[issue] /= issue_total

            max_val = max(self._value_weights[issue].values())

            for value in issue.values:
                if max_val == 0:
                    self._value_weights[issue][value] = 1.
                else:
                    self._value_weights[issue][value] /= max_val
