from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def plot_prec_rec(y_test, y_pred_positive, label):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, y_pred_positive)
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, "b:", linewidth=2, label=label)
    plt.xlabel('Czułość', fontsize=16)
    plt.ylabel('Precyzja', fontsize=16)
    plt.legend(loc="lower left", fontsize=16)
    plt.title('Krzywa czułość-precyzja', fontsize=18)
    plt.show()

def plot_roc_curve(y_test, y_pred_positive, label):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_positive)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, "b:", linewidth=2, label=label)
    plt.fill_between(fpr, tpr, color='blue', alpha=0.3)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('Odsetek fałszywie pozytywnych (FPR)', fontsize=16)
    plt.ylabel('Odsetek prawdziwie pozytywnych (TPR)', fontsize=16)
    plt.legend(loc="lower right", fontsize=16)
    plt.title('Krzywa ROC, AUC={0:.3f}'.format(metrics.roc_auc_score(y_test, y_pred_positive)), fontsize=18)
    plt.show()

def plot_classification(X, y, clf, title):
    plt.figure(figsize=(15, 9))
    X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 0.2, stop = X[:, 0].max() + 0.2, step = 0.01),
                         np.arange(start = X[:, 1].min() - 0.2, stop = X[:, 1].max() + 0.2, step = 0.01))
    plt.contourf(X1, X2, clf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.3, cmap = ListedColormap(('blue', 'red')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bo")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "ro")
    plt.xlabel("X_1", fontsize=20)
    plt.ylabel("X_2", fontsize=20)
    plt.title(title, fontsize=22)
    plt.show()

def plot_classification_poly(X_poly, y, clf_poly, poly, title):
    plt.figure(figsize=(15, 9))
    X1, X2 = np.meshgrid(np.arange(start = X_poly[:, 1].min() - 0.2, stop = X_poly[:, 1].max() + 0.2, step = 0.01),
                         np.arange(start = X_poly[:, 2].min() - 0.2, stop = X_poly[:, 2].max() + 0.2, step = 0.01))
    vals = clf_poly.predict(poly.transform(np.array([X1.ravel(), X2.ravel()]).T))
    plt.contourf(X1, X2, vals.reshape(X1.shape),
                 alpha = 0.3, cmap = ListedColormap(('blue', 'red')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    plt.plot(X_poly[:, 1][y==0], X_poly[:, 2][y==0], "bo")
    plt.plot(X_poly[:, 1][y==1], X_poly[:, 2][y==1], "ro")
    plt.xlabel("X_1", fontsize=20)
    plt.ylabel("X_2", fontsize=20)
    plt.title(title, fontsize=22)
    plt.show()

def plot_regression_poly(X_poly, y_poly, reg, poly, title):
    _x = np.arange(X_poly[:, 1].min(), X_poly[:, 1].max(), step=0.01).reshape(-1, 1)
    _x_poly = poly.transform(_x)
    _y_pred = reg.predict(_x_poly)
    plt.figure(figsize=(10, 7))
    plt.scatter(x=X_poly[:, 1].ravel(), y=y_poly.ravel(), alpha=0.5, s=40, c='blue')
    plt.plot(_x.ravel(), _y_pred.ravel(), 'r-', linewidth=3)

    plt.xlabel('X_1', size=16)
    plt.ylabel('y', size=16)
    plt.title(title, size=20)
    plt.grid()
    plt.show()

def plot_regression(X, y, reg, title):
    _x = np.arange(X.min(), X.max(), step=0.01).reshape(-1, 1)
    _y_pred = reg.predict(_x)
    plt.figure(figsize=(10, 7))
    plt.scatter(x=X.ravel(), y=y.ravel(), alpha=0.5, s=40, c='blue')
    plt.plot(_x.ravel(), _y_pred.ravel(), 'r-', linewidth=3)

    plt.grid()
    plt.xlabel('X_1', size=16)
    plt.ylabel('y', size=16)
    plt.title(title, size=20)
    plt.show()
