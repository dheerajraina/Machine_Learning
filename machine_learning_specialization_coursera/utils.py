import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


def load_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    return X, y


def load_coffee_data():
    """ Creates a coffee roasting data set.
        roasting duration: 12-15 minutes is best
        temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5          # 12-15 min is best
    X[:, 0] = X[:, 0] * (285-150) + 150  # 350-500 F (175-260 C) is best
    Y = np.zeros(len(X))

    i = 0
    for t, d in X:
        y = -3/(260-175)*t + 21
        if (t > 175 and t < 260 and d > 12 and d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0
        i += 1

    return (X, Y.reshape(-1, 1))


def plt_roast(X, Y):
    Y = Y.reshape(-1)
    fig, ax = plt.subplots(1, 1,)
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], s=70,
               marker='x', c='red', label="Good Roast")
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], s=100, marker='o',
               facecolors='none', edgecolors='b', linewidth=1,  label="Bad Roast")
    tr = np.linspace(175, 260, 50)
    ax.plot(tr, (-3/85) * tr + 21, color="purple", linewidth=1)
    ax.axhline(y=12, color='purple', linewidth=1)
    ax.axvline(x=175, color='purple', linewidth=1)
    ax.set_title(f"Coffee Roasting", size=16)
    ax.set_xlabel("Temperature \n(Celsius)", size=12)
    ax.set_ylabel("Duration \n(minutes)", size=12)
    ax.legend(loc='upper right')
    plt.show()


def sigmoid(z):
    """
    Compute the sigmoid of z

    Parameters
    ----------
    z : array_like
        A scalar or numpy array of any size.

    Returns
    -------
     g : array_like
         sigmoid(z)
    """
    z = np.clip(z, -500, 500)           # protect against overflow
    g = 1.0/(1.0+np.exp(-z))

    return g


def load_data1():
    X = np.load("data/X.npy")
    y = np.load("data/y.npy")
    return X, y


def gen_data(m, seed=1, scale=0.7):
    """ generate a data set based on a x^2 with added noise """
    c = 0
    x_train = np.linspace(0, 49, m)
    np.random.seed(seed)
    y_ideal = x_train**2 + c
    y_train = y_ideal + scale * y_ideal*(np.random.sample((m,))-0.5)
    x_ideal = x_train  # for redraw when new data included in X
    return x_train, y_train, x_ideal, y_ideal


class lin_model:
    def __init__(self, degree, regularization=False, lambda_=0):
        if regularization:
            self.linear_model = Ridge(alpha=lambda_)
        else:
            self.linear_model = LinearRegression()
        self.poly = PolynomialFeatures(degree, include_bias=False)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        ''' just fits the data. mapping and scaling are not repeated '''
        X_train_mapped = self.poly.fit_transform(X_train.reshape(-1, 1))
        X_train_mapped_scaled = self.scaler.fit_transform(X_train_mapped)
        self.linear_model.fit(X_train_mapped_scaled, y_train)

    def predict(self, X):
        X_mapped = self.poly.transform(X.reshape(-1, 1))
        X_mapped_scaled = self.scaler.transform(X_mapped)
        yhat = self.linear_model.predict(X_mapped_scaled)
        return (yhat)

    def mse(self, y, yhat):
        err = mean_squared_error(y, yhat)/2  # sklean doesn't have div by 2
        return (err)
