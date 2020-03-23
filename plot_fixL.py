from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy.optimize import least_squares

population = 327.2e6
ratio = 0.6

L = population * ratio

def fun(param, x):
    global L
    k, x0 = param
    return L/(1+np.exp(- k * (x-x0)))

def res(param, x, y):
    return fun(param, x) - y

if __name__ == "__main__":
    df = read_csv('data.csv')
    x0 = np.array([ 1.04432442, 42.53904538])
    print(res(x0, 20, 2000))

    res_robust = least_squares(res, x0, loss='cauchy', gtol=None, ftol=1e-8, xtol=None, max_nfev = 1000000, args=(np.array(df['day']), np.array(df['total'])))

    print(res_robust)

    for day in range(10, 190, 10):
        print(f'{day}: {fun(res_robust.x, day)}')
    Xrange = np.arange(0,35.1,0.1)
    ypred = []
    for x in Xrange:
        ypred.append(fun(res_robust.x, x))

    plt.subplot(211)
    plt.plot(list(df['day']), list(df['total']), '.')
    plt.plot(Xrange, np.array(ypred), '-')

    Xrange = np.arange(0,110.1,0.1)
    ypred = []
    for x in Xrange:
        ypred.append(fun(res_robust.x, x))

    plt.subplot(212)
    plt.plot(list(df['day']), list(df['total']), '.')
    plt.plot(Xrange, np.array(ypred), '-')

    plt.tight_layout(pad=2.0)
    pl.savefig('corona_fixL.png')
