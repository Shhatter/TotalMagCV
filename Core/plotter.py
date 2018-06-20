from __future__ import absolute_import

import scikitplot as skplt
import matplotlib.pyplot as plt

# y_true =  [0.4,0.7,0,8]
# y_probas = [0.2,0.5,0,9]
# # predicted probabilities generated by sklearn classifier
# skplt.metrics.plot_roc_curve(y_true, y_probas)
# plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
import scikitplot as skplt
from sklearn.datasets import load_digits
# X, y = load_digits(return_X_y=True)
# random_forest_clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=1)
# predictions = cross_val_predict(random_forest_clf, X, y)
# skplt.metrics.plot_confusion_matrix(y, predictions, normalize=True)
# plt.show()
import matplotlib.pyplot as plt
import scikitplot as skplt

"""
An example showing the plot_roc_curve method
used by a scikit-learn classifier
"""
# import matplotlib.pyplot as plt
# from sklearn.naive_bayes import GaussianNB
# from sklearn.datasets import load_digits as load_data
# import scikitplot as skplt
#
#
# X, y = load_data(return_X_y=True)
# nb = GaussianNB()
# nb.fit(X, y)
# probas = nb.predict_proba(X)
# skplt.metrics.plot_roc(y_true=y, y_probas=probas)
# plt.show()
#
#
#
import matplotlib.pyplot as plt
import numpy as np

FPR = [0.62, 0.67, 0.7, 0.49, 0.5, 0.57, 0.53, 0.51, 0.55, 0.47, 0.24, 0.44, 0.3, 0.42, 0.45, 0.52, 0.63, 0.37, 0.5,
       0.52, 0.53, 0.26, 0.35, 0.59, 0.39, 0.17, 0.19, 0.22, 0.24, 0.41, 0.51, 0.16, 0.19, 0.2, 0.23, 0.26, 0.27, 0.43,
       0.56, 0.15, 0.16, 0.2, 0.21, 0.21, 0.21, 0.22, 0.22, 0.22, 0.23, 0.23, 0.23, 0.24, 0.25, 0.46, 0.12, 0.12, 0.17,
       0.18, 0.19, 0.2, 0.22, 0.22, 0.23, 0.27, 0.3, 0.13, 0.13, 0.14, 0.15, 0.16, 0.16, 0.17, 0.17, 0.17, 0.18, 0.19,
       0.22, 0.22, 0.23, 0.24, 0.26, 0.1, 0.12, 0.14, 0.14, 0.15, 0.15, 0.17, 0.19, 0.2, 0.22, 0.22, 0.23, 0.25, 0.26,
       0.11, 0.13, 0.17, 0.18, 0.2, 0.21]
TPR = [0.78, 0.8, 0.81, 0.84, 0.84, 0.84, 0.86, 0.87, 0.87, 0.88, 0.89, 0.89, 0.9, 0.9, 0.9, 0.9, 0.9, 0.91, 0.91, 0.91,
       0.91, 0.92, 0.92, 0.92, 0.93, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
       0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96, 0.97, 0.97, 0.97, 0.97,
       0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98,
       0.98, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 1, 1,
       1, 1, 1, 1]
skplt.metrics.plot_roc(FPR, TPR)
plt.show()

skplt.metrics.plot_roc()
# # This is the ROC curve
# plt.plot(FPR,TPR)
# plt.show()
#
# # This is the AUC
# auc = np.trapz(FPR,TPR)
