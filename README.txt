Joopago (9x9 go AI with alphazero algorithm)

I've tried to train this AI using my home computer with just a single gpu.
It, however, is not trained to play go yet...

I need your help.

Below things are specs of my pc, circumstances and parametor values of algorithm.

----------------------------------------------------------------------------------
My pc resource details:
CPU : intel core i7-8700 3.20GHz
GPU : NVIDIA GeForce GTX 1080
RAM : 32.0GB
OS : window 10
----------------------------------------------------------------------------------
Program details:
python 3.7.6
tensorflow-gpu 1.14.0
numpy 1.18.1
Keras-Applications   1.0.8
Keras-Preprocessing  1.1.0
----------------------------------------------------------------------------------
Parametor setup value:
go feature : each 4 recent positions(white, black)(8) + color of player(1) = (9)
###### original alphazero : each 8 recent positions(16) + color(1) = (17) ########

input shape : (n, 9, 9, 9) ### original = (n, 19, 19, 17)

dual_network details:
FILTERS = 64 ### original = 256
RESIDUAL_NUM = 9 ### original = 19

mcts details:
EVALUATE_COUNT = 150 # origianl : 1600
C_PUCT = 0.2

self_play details:
GAME_COUNT = 50 # original = 25000

train details:
EPOCHS = 1000 #original 500,000
step_decay(epoch):
        x = 0.08   # orignal 0.02
        if epoch >= 200: x = 0.002   ### original epoch >= 300 : 0.002
        if epoch >= 800: x = 0.0002  ### originalepoch >= 500000 : 0.0002

batch_size=32 ### original = 2048

--------------------------------------------------------------------------------------------

Unfortunately, it took about 30min to finish 1 cycle ( self_play -> train_network -> eveluate_network ) although I reduced a lot of parameter values to make AI simple and light.

My goal is not making this AI as Super human level of go...
I just want to make this AI better than me or random player!

Is it possible to train my go AI using my single computer?

Please give me some advices...

Steps of execution
1) Lunch anaconda3 in window
2) install all the proper things before execute program such as tensorflow or numpy
2) python train_cycle.py
