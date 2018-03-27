from main import *
import numpy as np
import matplotlib.pyplot as plt

trial = 300
t_max = 100
t_his = 1000

w1 = 2*np.pi / 5
w2 = 2*np.pi / 200

trans_p = {"F": {"F": 0.8, "S": 0.2},
           "S": {"F": 0.3, "S": 0.7}}


x, hidden_state = gen_ts_m(w1, w2, trans_p, t_max)

#print(x)
print(hidden_state)
print()

emit_p = trans_emit(w1, w2, t_max)
vit_path, vit_prob = viterbi(x, [1, 0], trans_p, emit_p)

print(vit_prob)
print()
print(vit_path)
print()

print(emit_p)
print()
#print(trans_p)

sto_path = seq_sto(0.5, 101, sym1="S", sym2="F")

# print(len(hidden_state))
# print(len(vit_path))
# print(len(sto_path))

match = 0
matchs = 0
for i in range(len(vit_path)):

    if vit_path[i] == hidden_state[i]:
        match += 1

    if sto_path[i] == hidden_state[i]:
        matchs += 1

print(match)
print(matchs)