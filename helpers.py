from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_prec_rec(y: np.ndarray, 
                  y_pred_positive: np.ndarray, 
                  label: str) -> None:
    """
    Function outputs plot of Precision-Recall curve for classificaiton results.
    Args:
        y: (numpy.ndarray) array of true (binary) outputs
        y_pred_positive: (numpy.ndarray) array of predicted (float) outputs
        label: (string) label of the curve
    """
    precision, recall, thresholds = metrics.precision_recall_curve(y, y_pred_positive)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, "b:", linewidth=2, label=label)
    plt.xlabel('Czułość', fontsize=16)
    plt.ylabel('Precyzja', fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.title('Krzywa czułość-precyzja', fontsize=18)
    plt.show()

def plot_roc_curve(y: np.ndarray, 
                   y_pred_positive: np.ndarray, 
                   label: str) -> None:
    """
    Function outputs plot of ROC curve for classificaiton results.
    Args:
        y: (numpy.ndarray) array of true (binary) outputs
        y_pred_positive: (numpy.ndarray) array of predicted (float) outputs
        label: (string) label of the curve
    """
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_positive)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, "b:", linewidth=2, label=label)
    plt.fill_between(fpr, tpr, color='blue', alpha=0.3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Odsetek fałszywie pozytywnych (FPR)', fontsize=16)
    plt.ylabel('Odsetek prawdziwie pozytywnych (TPR)', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.title('Krzywa ROC, AUC={0:.3f}'.format(metrics.roc_auc_score(y, y_pred_positive)), fontsize=18)
    plt.show()

def plot_classification(X: np.ndarray, 
                        y: np.ndarray, 
                        clf: object, 
                        poly: object=None, 
                        title: str=None) -> None:
    """
    Function outputs plot for classificaiton results.
    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) array of input for the given model
        y: (numpy.ndarray) array of outputs matched to X matrix
        clf: (object) object of the fitted classifier
        poly: (object) object of the fitted PolynomialFeatures object
        title: (string) title of the plot
    """
    plt.figure(figsize=(15, 9))

    if poly:
        X1, X2 = np.meshgrid(np.arange(start = X[:, 1].min() - 0.2, stop = X[:, 1].max() + 0.2, step = 0.01),
                         np.arange(start = X[:, 2].min() - 0.2, stop = X[:, 2].max() + 0.2, step = 0.01))
        vals = clf.predict(poly.transform(np.array([X1.ravel(), X2.ravel()]).T))
        plt.contourf(X1, X2, vals.reshape(X1.shape),
                    alpha = 0.3, cmap = ListedColormap(('blue', 'red')))
        plt.plot(X[:, 1][y==0], X[:, 2][y==0], "bo")
        plt.plot(X[:, 1][y==1], X[:, 2][y==1], "ro")

    else:
        X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 0.2, stop = X[:, 0].max() + 0.2, step = 0.01),
                            np.arange(start = X[:, 1].min() - 0.2, stop = X[:, 1].max() + 0.2, step = 0.01))
        plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha = 0.3, cmap = ListedColormap(('blue', 'red')))
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bo")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "ro")
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    plt.xlabel("X_1", fontsize=20)
    plt.ylabel("X_2", fontsize=20)
    plt.title(title, fontsize=22)
    plt.show()


def plot_regression(X: np.ndarray, 
                    y: np.ndarray, 
                    reg: object, 
                    poly: object=None, 
                    title: str=None) -> None:
    """
    Function outputs plot for regression results.
    Args:
        model: (object) fitted model with standard predict(X) public method
        X: (numpy.ndarray) array of input for the given model
        y: (numpy.ndarray) array of outputs matched to X matrix
        reg: (object) object of the fitted regressor
        poly: (object) object of the fitted PolynomialFeatures object
        title: (string) title of the plot
    """

    plt.figure(figsize=(10, 7))
    if poly:
        __x = np.arange(X[:, 1].min(), X[:, 1].max(), step=0.01).reshape(-1, 1)
        __x_poly = poly.transform(__x)
        __y_pred = reg.predict(__x_poly)
        plt.scatter(x=X[:, 1].ravel(), y=y.ravel(), alpha=0.5, s=40, c='blue')

    else:
        __x = np.arange(X.min(), X.max(), step=0.01).reshape(-1, 1)
        __y_pred = reg.predict(__x)
        plt.scatter(x=X.ravel(), y=y.ravel(), alpha=0.5, s=40, c='blue')
    
    plt.plot(__x.ravel(), __y_pred.ravel(), 'r-', linewidth=3)

    plt.xlabel('X_1', size=16)
    plt.ylabel('y', size=16)
    plt.title(title, size=20)
    plt.grid()
    plt.show()
