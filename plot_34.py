from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from scipy.optimize import least_squares

def fun(param, x):
    L, k, x0 = param
    return L/(1+np.exp(- k * (x-x0)))

def res(param, x, y):
    #param = L, k, x0
    return fun(param, x) - y

if __name__ == "__main__":
    df = read_csv('data.csv').query('day<35')
    x0_low = np.array([2.00000020e+05, 3.86868186e-01, 4.17216896e+01])
    x0_high = np.array([4.370000020e+06, 2.86868186e-01, 5.17216896e+01])
    inp = [x0_low, x0_high]

    for i in range(2):
        x0 = inp[i]
        res_robust = least_squares(res, x0, loss='cauchy', gtol=None, ftol=1e-9, xtol=None, max_nfev = 1000000, args=(np.array(df['day']), np.array(df['total'])))

        print(res_robust)

        for day in range(10, 190, 10):
            print(f'{day}: {fun(res_robust.x, day)}')
        Xrange = np.arange(0,39.1,0.1)
        ypred = []
        for x in Xrange:
            ypred.append(fun(res_robust.x, x))

        plt.figure(i+1)
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
        pl.savefig(f'corona_init_{i}.png')
