import go
import numpy as np

state = go.Position()

def random_action(state):
    legal_actions = state.all_legal_moves()
    sum_legal_actions = np.sum(legal_actions)
    possibilities = legal_actions / sum_legal_actions
    num = np.random.choice(len(legal_actions), p = possibilities) # if 9x9 -> 0 ~ 81
    
    if num == len(legal_actions) - 1:
        return None
    
    row = num // go.N
    column = num % go.N

    coord = (row, column)
    
    return coord

if __name__ == '__main__':
    while True:
        if state.is_game_over():
            print(state.result_string())
            break
        
        action = random_action(state)

        state = state.play_move(action)

        print(state.__str__(False))
        print()
