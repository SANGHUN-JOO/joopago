import go
from pv_mcts import pv_mcts_action
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from pathlib import Path
import numpy as np
from test_random_go_play import random_action

EP_GAME_COUNT = 10

WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)

def first_player_point(ended_state):
    result_v = ended_state.result()
    if result_v == 1:
        return 1   #win
    elif result_v == 0:
        return 0.5 #draw
    else:
        return 0   #lose

def play(next_actions):

    state = go.Position()

    while True:
        if state.is_game_over():
            break
        next_action = next_actions[0] if state.to_play == BLACK else next_actions[1]
        action = next_action(state)

        state = state.play_move(action)

    return first_player_point(state)

def evaluate_algorithm_of(label, next_actions):
    total_point = 0
    for i in range(EP_GAME_COUNT):
        if i % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))
        
        print('\rEvaluate {}/{}'.format(i + 1, EP_GAME_COUNT), end='')
    print('')

    average_point = total_point / EP_GAME_COUNT
    print(label, 'AveragePoint', average_point)

def evaluate_best_player():
    cur_dir = Path(__file__).parent.absolute()
    model = load_model(str(cur_dir) + '\\model\\best.h5')

    next_action0 = pv_mcts_action(model, 0.0)

    next_actions = (next_action0, random_action)
    evaluate_algorithm_of('VS_Random', next_actions)

    K.clear_session()
    del model

if __name__ == '__main__':
    evaluate_best_player()