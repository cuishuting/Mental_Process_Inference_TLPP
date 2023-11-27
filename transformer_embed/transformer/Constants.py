def predicate_set():
    mental_predicate_set = [1]
    action_predicate_set = [2, 3]
    head_predicate_set = [1, 2, 3]
    total_predicate_set = [1, 2, 3]
    return mental_predicate_set, action_predicate_set, head_predicate_set, total_predicate_set

mental_predicate_set, action_predicate_set, head_predicate_set, total_predicate_set = predicate_set()
PAD = 0

time_horizon = 50
grid_length = 0.1