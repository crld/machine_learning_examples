import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


num_datasets = 50
noise_variance = 0.5
max_poly = 12
n = 25
ntrain = int(0.9*n)

np.random.seed(2)


def make_poly(x, D):
    n = len(x)
    X = np.empty((N, D+1))
    for d in xrange(D+1):
        X[:,d] = x**d
        if d > 1:
            X[:,d] = (X[:,d] - X[:,d].mean())/X[:,d].std()
    return X


def f(X):
    return np.sin(X)


x_axis = np.linspace(-np.pi, np.pi, 100)
y_axis = f(x_axis)

X = np.linspace(-np.pi, np.pi, N)
np.random.shuffle(X)
f_X= f(X)

Xpoly = make_poly(X, max_poly)
train_scores = np.zeros((num_datasets, max_poly))
test_scores = np.zeros(num_datasets, max_poly)
train_predictions = np.zeros((ntrain, num_datasets, max_poly))
prediction_curves = np.zeros((100, num_datasets, max_poly))


model = LinearRegression()
for k in xrange(num_datasets):
    Y = f_X + np.random.randn(n) * noise_variance

    x_train = Xpoly[:ntrain]
    y_train = Y[:ntrain]

    xtest = Xpoly[ntrain:]
    ytest = Y[ntrain:]

    for d in xrange(max_poly):
        model.fit(x_train[:,:d+2], y_train)
        predictions = model.predict(Xpoly[:,:d+2])

        x_axis_poly = make_poly(x_axis, d+1)
        prediction_axis = model.predict(x_axis_poly)

        prediction_curves[:,k,d] = prediction_axis

        train_predictions = predictions[:ntrain]
        test_predictions = predictions[ntrain:]

        train_score = mse(train_predictions, y_train)
        test_score = mse(test_predictions, ytest)

        train_scores[k,d] = train_score
        test_scores[k,d] = test_score


