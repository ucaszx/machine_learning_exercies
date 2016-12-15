
import numpy as np
import scipy as sp

# Zhou, Z.-H. Machine Learning[M]. Beijing: Tsinghua University Press: 59-60
def loss(X, y, beta):
    loss = -np.multiply(y, X * beta) + np.log(1 + np.exp(X * beta))
    total = np.sum(loss, axis=0)
    return total

def logistic_newton(input, label):
    X = np.mat(input)
    y = np.mat(label).T
    m,n = np.shape(X)
    X = np.column_stack((X, np.ones(m)))

    max_iter = 10
    beta = np.ones((n+1, 1))
    for k in range(max_iter):
        prob1 = 1.0 - 1.0 / (1.0 + np.exp(X * beta))
        gradient = -np.sum(np.multiply(X, (y - prob1)), axis=0) # excute (n+1) matrix plus m times
        hessian = np.zeros((n+1, n+1))
        for i in range(m):                                        # execute (n+1)*(n+1) matrix plus m times
            num = (prob1[i][0] * (1 - prob1[i][0])).item((0, 0))
            z = np.ones((n+1, n+1)) * num
            hessian = hessian + np.multiply( X[i,:].T * X[i,:], np.mat(z) )
        beta = beta - hessian.I * np.mat(gradient).T
        current_loss = loss(X, y, beta)
        print ("iteration: %d    loss: %f" % (k, current_loss))

    return beta


if __name__=="__main__":

    import pandas as pd
    df = pd.read_csv('watermelon_3a.csv', delimiter='\t', index_col=0)

    X = df[['density', 'ratio_sugar']].values
    y = df['label'].values
    beta = logistic_newton(X, y)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.scatter(df[df['label'] == 0]['density'].values, df[df['label'] == 0]['ratio_sugar'].values, s=30, c='red', marker='s')
    ax.scatter(df[df['label'] == 1]['density'].values, df[df['label'] == 1]['ratio_sugar'].values, s=30, c='green')

    x1 = np.arange(0.0, 1.0, 0.005)
    x2 = ( -beta.item((0, 0)) * x1 -beta.item((2,0)) ) / beta.item((1, 0))
    ax.plot(x1, x2)
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    
    plt.show()

