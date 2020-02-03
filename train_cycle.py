from dual_network import dual_network
from self_play import self_play
from train_network import train_network
from evaluate_network import evaluate_network, update_best_player
from evaluate_best_player import evaluate_best_player

dual_network()
count = 0
fail_count = 0
for i in range(30):
    print('Train', i, '======================')
    self_play(fail_count)

    train_network(fail_count)

    #skip = True
    #updated = True
    #if i%10 == 0 and i != 0:
    updated = evaluate_network()
    #    skip = False
    #else:
    #    update_best_player()

    if updated == True:# and skip == False:
        count += 1
        fail_count = 0
    else:
        fail_count += 1

    if count > 4:
        evaluate_best_player()
        count = 0
    
    if fail_count > 3:
        fail_count -= 1