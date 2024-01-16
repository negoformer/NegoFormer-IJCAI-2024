POSITIVE_MOVES = ["Fortunate", "Nice", "Concession"]
NEGATIVE_MOVES = ["Selfish", "Unfortunate", "Silent"]


def get_move(prev_offered_utility: float, offered_utility: float, prev_opponent_utility: float, opponent_utility: float, threshold: float = 0.03) -> str:
    diff_offered = offered_utility - prev_offered_utility
    diff_opponent = opponent_utility - prev_opponent_utility

    if abs(diff_offered) < threshold and abs(diff_opponent) < threshold:
        return "Silent"

    if abs(diff_offered) < threshold and diff_opponent > 0.:
        return "Nice"

    if diff_offered < 0 and diff_opponent >= 0:
        return "Concession"

    if diff_offered <= 0 and diff_opponent < 0:
        return "Unfortunate"

    if diff_offered > 0 and diff_opponent <= 0:
        return "Selfish"

    if diff_offered > 0 and diff_opponent > 0:
        return "Fortunate"

    return ""


def get_move_distribution(moves: list) -> dict:
    dist = {move: 0. for move in POSITIVE_MOVES}
    dist.update({move: 0. for move in NEGATIVE_MOVES})

    for move in moves:
        if move in dist.keys():
            dist[move] += 1

    total = sum(dist.values()) + 1e-8

    return {move: count / total for move, count in dist.items()}


def calculate_behavior_sensitivity(moves: list) -> float:
    count_positives = 0.
    count_negatives = 0.

    for move in moves:
        if move in POSITIVE_MOVES:
            count_positives += 1
        elif move in NEGATIVE_MOVES:
            count_negatives += 1

    return count_positives / (count_negatives + 1e-8)


def calculate_awareness(move_a: list, move_b: list) -> float:
    change_a_counter = 0.
    change_b_counter = 0.

    for i in range(1, min(len(move_a), len(move_b))):
        if (move_a[i] in POSITIVE_MOVES and move_a[i - 1] in NEGATIVE_MOVES) or (move_a[i] in NEGATIVE_MOVES and move_a[i - 1] in POSITIVE_MOVES):
            change_a_counter += 1

            if (move_b[i] in POSITIVE_MOVES and move_b[i - 1] in NEGATIVE_MOVES) or (move_b[i] in NEGATIVE_MOVES and move_b[i - 1] in POSITIVE_MOVES):
                change_b_counter += 1

    return change_b_counter / (change_a_counter + 1e-8)


def calculate_move_correlation(move_a: list, move_b: list) -> float:
    change_a_counter = 0.
    change_b_counter = 0.

    for i in range(1, min(len(move_a), len(move_b))):
        if move_a[i] in POSITIVE_MOVES and move_a[i - 1] in NEGATIVE_MOVES:
            change_a_counter += 1
            if move_b[i] in POSITIVE_MOVES and move_b[i - 1] in NEGATIVE_MOVES:
                change_b_counter += 1
            elif move_b[i] in NEGATIVE_MOVES and move_b[i - 1] in POSITIVE_MOVES:
                change_b_counter -= 1
        elif move_a[i] in NEGATIVE_MOVES and move_a[i - 1] in POSITIVE_MOVES:
            change_a_counter += 1
            if move_b[i] in NEGATIVE_MOVES and move_b[i - 1] in POSITIVE_MOVES:
                change_b_counter += 1
            elif move_b[i] in POSITIVE_MOVES and move_b[i - 1] in NEGATIVE_MOVES:
                change_b_counter -= 1

    return change_b_counter / (change_a_counter + 1e-8)
