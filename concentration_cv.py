#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import fsolve
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 14
rcParams['mathtext.fontset'] = 'custom'
rcParams['mathtext.it'] = 'Arial'


def solver(U, totalc, li2s8guess, li2s4guess, li2s6guess, li2sguess):
    kbt = 0.0259
    converged = False
    U01 = 2.237
    U02 = 0.162
    U03 = 2.414
    # for better numerical stability, used in combination with trialfunc
    if U < 1.9: 
        U03prime = U + 0.2
    else:
        U03prime = U 
    U04 = 2.106 - 0.12
    dG1 = (U - U01) * -2
    dG2 = U02
    dG3 = (U - U03) * -2
    dG3prime = (U - U03prime) * -2
    dG4 = (U - U04) * -4
    q1 = np.exp(dG1/kbt)
    q2 = np.exp(dG2/kbt)
    q3 = np.exp(dG3/kbt)
    q3prime = np.exp(dG3prime/kbt)
    q4 = np.exp(dG4/kbt)
    # root[0]:Li2S8, root[1]:Li2S4, root[2]: Li2S6, root[3]: Li2S
    def func(root):
        return [ root[0] * root[1]  - root[2] ** 2 / q2,
                 q1 * root[0] - root[1]  ** 2, 
                 q3 * (totalc - root[0] - root[1] / 2 - root[2] * 3 / 4 - root[3] / 8 ) - root[0],
                 q4 * root[1] - root[3] ** 4,
               ]

    def trialfunc(root):
        return [ root[0] * root[1]  - root[2] ** 2 / q2,
                 q1 * root[0] - root[1]  ** 2,
                 q3prime * (totalc - root[0] - root[1] / 2 - root[2] * 3 / 4 - root[3] / 8 ) - root[0],
                 q4 * root[1] - root[3] ** 4,
               ]
    if U < 2.1:
        root = fsolve(trialfunc, [li2s8guess, li2s4guess, li2s6guess, li2sguess])
        li2s8guess, li2s4guess, li2s6guess, li2sguess = root
    root = fsolve(func, [li2s8guess, li2s4guess, li2s6guess, li2sguess])
    c, x, y, z = root
    return c, x, y, z
    



Urange = [1.8, 2.5]
binsize = 0.001
Ulist = np.linspace(*Urange, num=np.int((Urange[1] - Urange[0]) / binsize) + 1)
totalc = 0.054
li2s8list, li2s4list, li2s6list, li2slist, s3list = [], [], [], [], []
guess = [1.241e-7, 0.01, 0.0013, totalc * 4]

for U in Ulist:
    li2s8, li2s4, li2s6, li2s = solver(U, totalc, *guess)
    li2s8list.append(li2s8)
    li2s4list.append(li2s4)
    li2s6list.append(li2s6)
    li2slist.append(li2s)
    guess = [li2s8, li2s4, li2s6, li2s]

li2s8list = np.array(li2s8list)
li2s4list = np.array(li2s4list)
li2s6list = np.array(li2s6list)
li2slist = np.array(li2slist)
s8list = np.array([totalc] * len(li2s8list))
s8list = s8list - li2s8list - li2s4list / 2 - li2s6list * 6 / 8 - li2slist / 8

# percentage
pli2s8list = li2s8list / (totalc/100)
pli2s4list = li2s4list / (totalc/100) /8 * 4
pli2s6list = li2s6list / (totalc/100) / 8 * 6
pli2slist = li2slist / (totalc/100) / 8
pslist = s8list / (totalc/100)

plt.plot(Ulist, pli2s8list, label='$Li_2S_8$', c='C3')
plt.plot(Ulist, pli2s4list, label='$Li_2S_4$', c='C2')
plt.plot(Ulist, pli2s6list, label='$Li_2S_6$', c='C0')
plt.plot(Ulist, pli2slist, label='$Li_2S$', c='C1')
plt.plot(Ulist, pslist, label='$S_8$', c='C8')
plt.legend()
plt.xlabel('U (V)')
plt.ylabel('converted amount relative to initial $[S_8]$ %')
plt.savefig('li2s-concentration.png', dpi=600, bbox_inches='tight')

dli2s4list1 = np.insert(li2s4list, 0, 0.0)
dli2s4list2 = np.append(li2s4list, li2s4list[-1])
dli2s4list = dli2s4list1 - dli2s4list2
dli2s4list = dli2s4list[1:]
dli2s6list1 = np.insert(li2s6list, 0, 0.0)
dli2s6list2 = np.append(li2s6list, li2s6list[-1])
dli2s6list = dli2s6list1 - dli2s6list2
dli2s6list = dli2s6list[1:]
dli2s8list1 = np.insert(li2s8list, 0, 0.0)
dli2s8list2 = np.append(li2s8list, li2s8list[-1])
dli2s8list = dli2s8list1 - dli2s8list2
dli2s8list = dli2s8list[1:]
dli2slist1 = np.insert(li2slist, 0, 0.0)
dli2slist2 = np.append(li2slist, li2slist[-1])
dli2slist = dli2slist1 - dli2slist2
dli2slist = dli2slist[1:]

chargelist = dli2s4list * 2 + dli2s6list * 2 + dli2s8list * 2 + dli2slist * 2 
chargelist /= binsize

plt.clf()
plt.plot(Ulist, -chargelist)
plt.xlabel('U (V)')
plt.ylabel('current (a.u.)')
plt.savefig('li2s-cv.png', dpi=600, bbox_inches='tight')
