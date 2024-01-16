from typing import Union

import numpy as np
import nenv


class OfferPoint:
    """
        Offer point contains required features in a time-step for NegoFormer model
    """
    who: int
    bid: nenv.Bid
    t: float

    def __init__(self, who: int, bid: nenv.Bid, t: float):
        self.who = who
        self.bid = bid
        self.t = t


def process(prev_offer_point: Union[OfferPoint, None], offer_point: OfferPoint, t: float, estimated_preference: nenv.OpponentModel.EstimatedPreference) -> np.ndarray:
    """
        This method converts Offer Point object into NumPy vector
    :param prev_offer_point: Previous offer point
    :param offer_point: Current offer point
    :param t: Current negotiation time
    :param estimated_preference: Estimated opponent preferences
    :return: NumPy vector for Autoformer model.
    """
    move = ''
    if prev_offer_point is not None:
        if offer_point.who == -1:
            move = nenv.utils.get_move(prev_offer_point.bid.utility, offer_point.bid.utility,
                                       estimated_preference.get_utility(prev_offer_point.bid),
                                       estimated_preference.get_utility(offer_point.bid))
        else:
            move = nenv.utils.get_move(estimated_preference.get_utility(prev_offer_point.bid),
                                       estimated_preference.get_utility(offer_point.bid),
                                       prev_offer_point.bid.utility, offer_point.bid.utility)

    return np.array([
        offer_point.bid.utility,
        t,
        offer_point.who,
        estimated_preference.get_utility(offer_point.bid),
        1. if move == 'Concession' else 0,
        1. if move == 'Selfish' else 0,
        1. if move == 'Fortunate' else 0,
        1. if move == 'Unfortunate' else 0,
        1. if move == 'Silent' else 0,
        1. if move == 'Nice' else 0,
        offer_point.bid.utility * estimated_preference.get_utility(offer_point.bid),
        offer_point.bid.utility + estimated_preference.get_utility(offer_point.bid)
    ])
