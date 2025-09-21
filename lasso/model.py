import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from plot import plot_coefficient_paths, plot_mse_vs_alpha


def soft_threshold(z, r):
    return np.sign(z) * np.maximum(np.abs(z) - r, 0.0)


def mse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.square(np.subtract(y_true, y_pred)).mean()
    

class Lasso: 
    def __init__(self, lr=1e-3, tol=1e-6, alpha=1.0, max_iter=1000, standardize=True):
        self.lr = lr
        self.alpha = alpha 
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.loss_history = []


    def loss(self, X, y, beta):
        n = len(y)
        pred = X @ beta
        resid = y - pred
        return (1 / (2 * n)) * np.sum(resid ** 2) + self.alpha * np.sum(np.abs(beta))
        

    def standardization(self, X, y):
        self.X_mean = X.mean(axis=0)
        self.y_mean = y.mean()
        Xc = X - self.X_mean
        yc = y - self.y_mean

        if self.standardize:
            self.X_std = Xc.std(axis=0, ddof=0)
            self.X_std[self.X_std == 0] = 1.0
        else:
            self.X_std = np.ones(X.shape[1])
        
        Xn = Xc / self.X_std
        return Xn, yc


    def coef(self, beta_std):
        self.coeff = beta_std / self.X_std
        self.intercept = self.y_mean - np.dot(self.X_mean, self.coeff)


    def update(self, X, y, lamb):
        n, p = X.shape
        beta = np.zeros(p)

        for i in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(p):
                X_j = X[:, j]
                resid = y - X @ beta + X_j * beta[j]

                rho_j = np.dot(X_j, resid) / n 

                z_j = rho_j

                beta[j] = soft_threshold(z_j, lamb)

            current_loss = self.loss(X, y, beta)
            self.loss_history.append(current_loss)

            if np.max(np.abs(beta - beta_old)) < self.tol:
                break

        return beta
    

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        X_scaled, y_centered = self.standardization(X, y)

        beta_std = self.update(X_scaled, y_centered, self.alpha)

        self.coeff = beta_std / self.X_std
        self.intercept = self.y_mean - np.dot(self.X_mean, self.coeff)

        return self


    def predict(self, X):
        X = np.array(X)
        return X @ self.coeff + self.intercept


if __name__ == '__main__':
    df = pd.read_csv("data/prostate.csv")
    X = df.drop(columns=["lpsa"])
    y = df["lpsa"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    alphas = np.logspace(-4, 0.5, 50)
    coefs = []
    mses = []

    for alpha in alphas:
        model = Lasso(alpha=alpha, max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        coefs.append(model.coeff)
        mse_val = mean_squared_error(y_test, y_pred)
        mses.append(mse_val)

    coefs = np.array(coefs)
    best_alpha_idx = int(np.argmin(mses))
    best_alpha = alphas[best_alpha_idx]
    best_mse = mses[best_alpha_idx]

    print(f"Best alpha: {best_alpha:.5f}, MSE: {best_mse:.5f}")

    plot_coefficient_paths(alphas, coefs, feature_names=X.columns.tolist())
    plot_mse_vs_alpha(alphas, mses, best_alpha=best_alpha)

