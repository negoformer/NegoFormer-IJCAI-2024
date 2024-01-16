from typing import List, Tuple
from numba import prange, njit


@njit(parallel=True)
def extract_pareto_indices(candidates: List[Tuple[float, float]], lower_bound: float) -> List[int]:
    """
        This method extracts the pareto points from a list of bids.
    :param candidates: List of bids as utility tuple (U_A, U_B).
    :param lower_bound: Utility lower bound
    :return: List of indices of pareto bids.
    """
    indices = []

    for i in range(len(candidates)):
        is_pareto = True
        candidate_i = candidates[i]

        for j in prange(len(candidates)):
            if not is_pareto:
                break

            candidate_j = candidates[j]

            if candidate_i[0] > candidate_j[0]:
                break

            if (candidate_j[0] >= candidate_i[0] and candidate_j[1] >= candidate_i[1]) and (
                    candidate_j[0] > candidate_i[0] or candidate_j[1] > candidate_i[1]):
                is_pareto = False

                break

        if is_pareto and candidate_i[0] >= lower_bound:
            indices.append(i)

    return indices


@njit
def populate_time_list(t: float, number_of_sample: int, estimated_round_time: float) -> List[float]:
    """
        Faster implementation to populate time steps.
    :param t: Current negotiation time
    :param number_of_sample: Number of required samples.
    :param estimated_round_time: Estimated round time
    :return: List of time-steps
    """
    time_samples = []

    for i in range(number_of_sample):
        time_samples.append(t)

        t += estimated_round_time

        if t > 1.0:
            break

    return time_samples
