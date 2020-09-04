# Optimisation
import cvxopt
import cvxopt.solvers
# Math
import numpy as np
from numpy import linalg
#Scikit
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SVM(object):

    def __init__(self, kernel='linear', C=0, gamma=1, degree=3):

        if C is None:
            C=0
        if gamma is None:
            gamma = 1
        if kernel is None:
            kernel = 'linear'

        C = float(C)
        gamma = float(gamma)
        degree=int(degree)

        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)

    def polynomial_kernel(self, x, y,C=1, d=3):
        # Inputs:
        #   x   : vector of x data.
        #   y   : vector of y data.
        #   c   : is a constant
        #   d   : is the order of the polynomial.
        return (np.dot(x, y) + C) ** d

    def gaussian_kernel(self, x, y, gamma=0.5):
        return np.exp(-gamma*linalg.norm(x - y) ** 2 )

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):

                # Kernel trick.
                if self.kernel == 'linear':
                    K[i, j] = self.linear_kernel(X[i], X[j])
                if self.kernel=='gaussian':
                    K[i, j] = self.gaussian_kernel(X[i], X[j], self.gamma)   # Kernel trick.
                    self.C = None   # Not used in gaussian kernel.
                if self.kernel == 'polynomial':
                    K[i, j] = self.polynomial_kernel(X[i], X[j], self.C, self.degree)


        # Converting into cvxopt format:
        P = cvxopt.matrix(np.outer(y, y) * K, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if self.C is None or self.C==0:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            # Restricting the optimisation with parameter C.
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # Setting options:
        cvxopt.solvers.options['show_progress'] = True
        cvxopt.solvers.options['abstol'] = 1e-10
        cvxopt.solvers.options['reltol'] = 1e-10
        cvxopt.solvers.options['feastol'] = 1e-10

        # Solve QP problem:
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alphas = np.ravel(solution['x'])        # Flatten the matrix into a vector of all the Langrangian multipliers.

        # Support vectors have non zero lagrange multipliers
        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        # Bias (For linear it is the intercept):
        self.b = 0
        for n in range(len(self.alphas)):
            # For all support vectors:
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[n], sv])
        self.b = self.b / len(self.alphas)

        # Weight vector
        if self.kernel == 'linear':
            self.w = np.zeros(n_features)
            for n in range(len(self.alphas)):
                self.w += self.alphas[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

    def project(self, X):
        # Create the decision boundary for the plots. Calculates the hypothesis.
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alphas, self.sv_y, self.sv):
                    # a : Lagrange multipliers, sv : support vectors.
                    # Hypothesis: sign(sum^S a * y * kernel + b)

                    if self.kernel == 'linear':
                        s += a * sv_y * self.linear_kernel(X[i], sv)
                    if self.kernel=='gaussian':
                        s += a * sv_y * self.gaussian_kernel(X[i], sv, self.gamma)   # Kernel trick.
                        self.C = None   # Not used in gaussian kernel.
                    if self.kernel == 'polynomial':
                        s += a * sv_y * self.polynomial_kernel(X[i], sv, self.C, self.degree)

                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        # Hypothesis: sign(sum^S a * y * kernel + b).
        # NOTE: The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0.
        return np.sign(self.project(X))


def formattedOutput(objFit):
    """
    Purpose: To produce readable output summary of the results.
        Input:
            objFit  : The class instance containing the fitted information.
    """
    dash = '=' * 60; #chr(10000)*50
    print(dash)
    print("        SUPPORT VECTOR MACHINE TERMINATION RESULTS")
    print(dash)
    print("               **** In-Sample: ****")
    print("{:>1} support vectors found from {:>1} examples.".format(len(objFit.sv),objFit.number_of_train_examples))
    print("               **** Predictions: ****")
    print("{:>1} out of {:>1} predictions correct.".format(objFit.number_of_test_examples,objFit.correct_predictions))
    print(dash)

def run_custom_svm(kernel='linear', C=0, gamma=0.1, degree=3):
    #--------------------------------------------------------------
    # Purpose: SVM hard margin estimation and plot.
    # Inputs:
    #       run_type    : What margin calculation to perform.
    #                     Accepted Inputs: 'hard','soft','kernel'.
    #--------------------------------------------------------------

    print("Estimating kernel: " + kernel)

    features = datasets.load_breast_cancer().data

    # standardize the features
    features = StandardScaler().fit_transform(features)

    # get the number of features
    num_features = features.shape[1]

    # load the corresponding labels for the features
    labels = datasets.load_breast_cancer().target

    # transform the labels to {-1, +1}
    labels[labels == 0] = -1

    objFit = SVM(kernel=kernel, C=C, gamma=gamma, degree=degree)  # Create model object.

    # split the dataset to 70/30 partition: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    objFit.fit(X_train, y_train)                                            # Fit the training data.
    y_predict = objFit.predict(X_test)                                       # Fit the out-of-sample (predicted data).
    correct_predictions = np.sum(y_predict == y_test)                        # Check how many examples were predicted correctly.
    objFit.correct_predictions=correct_predictions
    objFit.number_of_test_examples = len(X_test)
    objFit.number_of_train_examples = len(X_train)
    objFit.fit_type = 'custom'

    

def run_sklearn_svm(kernel='linear', C='auto', gamma='auto', degree='auto'):
    #------------------------------------------------------------------------------------------------------------------
     # # Run Fit:
    if kernel == 'linear':
        skl = SVC(kernel=kernel,C=C)  # Fit the training data.
    elif kernel == 'poly':
    #     ### NOTE: polynomial kernel has some input 'coef0'. This is interpretted as C by other kernels.
    #     ### sklearn documentation:
    #     ### K(X, Y) = (gamma < X, Y > + coef0) ^ degree
    #     ### source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.polynomial_kernel.html
        skl = SVC(kernel=kernel,coef0=C,  gamma=gamma, degree=degree, tol=0.000001)  # Fit the training data.
    elif kernel == 'rbf':
        skl = SVC(kernel=kernel, gamma=gamma)  # Fit the training data.

    
    features = datasets.load_breast_cancer().data

    # standardize the features
    features = StandardScaler().fit_transform(features)

    # get the number of features
    num_features = features.shape[1]

    # load the corresponding labels for the features
    labels = datasets.load_breast_cancer().target

    # transform the labels to {-1, +1}
    labels[labels == 0] = -1

    #solit the training and testing data: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, stratify=labels)


    train_size = X_train.shape[0]
    test_size = X_test.shape[0]
    skl.fit(X_train, y_train)
    y_predict = skl.predict(X_test)                                       # Fit the out-of-sample (predicted data).
    correct_predictions = np.sum(y_predict == y_test)                        # Check how many examples were predicted correctly.
    skl.correct_predictions=correct_predictions
    print('Support vectors = ', skl.support_vectors_)
    print('Number of support vectors for each class = ', skl.n_support_)
    
    
    # ------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    run_custom_svm(kernel='linear')
    run_custom_svm(kernel='linear',C=100)
    run_custom_svm('polynomial',C=1, degree=3)
    run_custom_svm('gaussian', gamma=0.5)

    run_sklearn_svm(kernel='linear', C=10)          # Hard Margin
    run_sklearn_svm(kernel='linear', C=100)         # Soft Margin
    run_sklearn_svm(kernel='poly', C=1, degree=3,gamma=1)
    run_sklearn_svm(kernel ='rbf', gamma='auto')

    print("Finished")
