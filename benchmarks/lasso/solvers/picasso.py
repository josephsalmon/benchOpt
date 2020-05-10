import numpy as np
import warnings

from benchopt.base import BaseSolver
from benchopt.util import safe_import


with safe_import() as solver_import:
    import pycasso


class Solver(BaseSolver):
    name = 'picasso'
    sampling_strategy = 'iteration'

    install_cmd = 'pip'
    requirements = ['pycasso']

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd

        n_samples = len(y)

        # warnings.filterwarnings('ignore', category=ConvergenceWarning)
        lmbd_max = np.max(np.abs(self.X.T.dot(y))) / n_samples
        print(self.lmbd / lmbd_max)
        self.lasso = pycasso.Solver(self.X, self.y,
                                    lambdas=[2, self.lmbd / n_samples / lmbd_max],
                                    family="gaussian", penalty="l1",
                                    useintercept=False, prec=1e-14)

    def run(self, n_iter):
        self.lasso.max_ite = n_iter
        self.lasso.train()

    def get_result(self):
        return self.lasso.coef()['beta'][1].flatten()


# example of test:
# import pycasso
# x = np.array([[1,2,3,4,5,0],[3,4,1,7,0,1],[5,6,2,1,4,0]])
# y = np.array([3.1,6.9,11.3])
# n_samples = len(y)
# lambda_max = np.max( np.abs(x.T.dot(y))) / n_samples
# s = pycasso.Solver(x, y, lambdas=(2, 1/2), family="gaussian", penalty="l1", useintercept=False, max_ite=20)
# s.train()

# print(s.coef()['beta'])

# from sklearn.linear_model import Lasso
# clf=Lasso(alpha=1/2 * lambda_max,fit_intercept=False, normalize=False, warm_start=False)
# clf.fit(x,y)
# print(clf.coef_)



# # self.lmbd / n_samples / lmbd_max]
# # self.lmbd/n_samples

# print(s.coef()['beta'])
