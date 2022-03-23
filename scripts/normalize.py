# python normalize.py normalize_log
import sys
import pickle
import numpy as np
from sklearn import preprocessing

f = open(sys.argv[1])
r = [] 
for line in f:
    ts = np.array(eval(line.strip()[64:-1]))
    r.append(ts)
f.close()

r = np.array(r)
m = r.mean(axis=0)
s = r.std(axis=0)
v = r.var(axis=0)
with open('mean_var_file.pkl', 'wb') as f:
    pickle.dump(m, f)
    pickle.dump(v, f)
print(s)
print(v)

#r_scaled = preprocessing.scale(r)
#f = open('normalize_log2')
#for line in f:
#    ts = np.array(eval(line.strip()[64:-1]))
#    print((ts - m) / s)
#    print(ts)
#f.close()
