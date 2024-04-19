import math
from constant import LR, EPOCHS

def step_decay(epoch):
    initial_lrate = LR
    drop = 0.75
    epochs_drop = EPOCHS//10
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate