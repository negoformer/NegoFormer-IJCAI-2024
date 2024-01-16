from typing import List


class Issue:
    """
        Issue class holds the issue name and the possible discrete values of the corresponding issue in any domain.
        The objects of this class are mutual.
    """
    __name: str             # Name of the issue
    __values: List[str]     # Possible discrete values of the issue

    def __init__(self, name: str, values: list):
        """
            Constructor
        :param name: Name of the issue as string
        :param values: The name of the values of the issue
        """
        self.__name = name
        self.__values = values

    def __len__(self):
        """
        :return: Number of the values under that issue
        """
        return len(self.__values)

    def __str__(self):
        """
        :return: Name of the issue
        """
        return self.__name

    def __repr__(self):
        """
        :return: Representation of the name
        """
        return self.__name.__repr__()

    def __hash__(self):
        """
            Issue object can be hashed based on its name
        :return: Hash of the issue name
        """
        return self.__name.__hash__()

    def __eq__(self, other):
        """
            "==" operator implementation that check the both issue and issue (or issue name) are the same based on their
            issue name.
        :param other: Issue object or issue name as string
        :return: Whether the issue and given issue are the same based on issue name
        """
        if isinstance(other, Issue):
            return other.__name == self.__name
        elif isinstance(other, str):
            return other == self.__name

        return False

    def __copy__(self):
        """
        :return: Copy of the issue
        """
        return Issue(self.__name, self.__values.copy())

    @property
    def name(self) -> str:
        """
        :return: Issue name as string
        """
        return self.__name

    @property
    def values(self) -> List[str]:
        """
        :return: The list of possible discrete values of the issue
        """
        return self.__values.copy()
