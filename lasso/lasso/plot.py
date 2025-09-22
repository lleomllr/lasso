import matplotlib.pyplot as plt 
import numpy as np 

def plot_coefficient_paths(alphas, coefs, feature_names=None, log_scale=True,
                           title="Trajectoire des coefficients Lasso",
                           save_path=None):
    """
    Trace les trajectoires des coefficients en fonction de alpha.

    Parameters:
        alphas: array-like
        coefs: shape (len(alphas), n_features)
        feature_names: list of str
        log_scale: bool, use log10(alpha) on x-axis
        save_path: optional path to save the figure
    """
    plt.figure(figsize=(10, 6))

    x = np.log10(alphas) if log_scale else alphas

    for i in range(coefs.shape[1]):
        label = feature_names[i] if feature_names else f"Feature {i}"
        plt.plot(x, coefs[:, i], label=label)

    plt.xlabel("log10(alpha)" if log_scale else "alpha")
    plt.ylabel("Coefficient")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_mse_vs_alpha(alphas, mses, best_alpha=None, log_scale=True,
                      title="Erreur quadratique moyenne sur test",
                      save_path=None):
    """
    Trace la MSE en fonction de alpha.

    Parameters:
        alphas: array-like
        mses: list of mean squared errors
        best_alpha: value of best alpha to highlight
        log_scale: whether to use log10(alpha)
        save_path: optional path to save
    """
    plt.figure(figsize=(8, 5))

    x = np.log10(alphas) if log_scale else alphas

    plt.plot(x, mses, marker="o", linestyle="-", label="MSE")
    if best_alpha:
        best_x = np.log10(best_alpha) if log_scale else best_alpha
        plt.axvline(best_x, color="r", linestyle="--", label=f"Best alpha = {best_alpha:.4f}")

    plt.xlabel("log10(alpha)" if log_scale else "alpha")
    plt.ylabel("Mean Squared Error")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
