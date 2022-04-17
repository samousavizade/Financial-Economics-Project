import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.covariance import MinCovDet, EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS
from scipy.optimize import minimize
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

from returns_estimators import ReturnsEstimators


class RiskEstimators:
    
    def __init__(self):
        pass

    @staticmethod
    def _is_positive_semidefinite(matrix):
        try:
            np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
        if RiskEstimators._is_positive_semidefinite(matrix):
            return matrix

        # Eigendecomposition
        q, V = np.linalg.eigh(matrix)

        if fix_method == "spectral":
            # Remove negative eigenvalues and Reconstruct matrix
            fixed_matrix = V @ np.diag(np.where(q > 0, q, 0)) @ V.T

        elif fix_method == "diag":
            fixed_matrix = matrix - 1.1 * np.min(q) * np.eye(len(matrix))

        else:
            raise NotImplementedError(
                "Method {} not implemented".format(fix_method))

        return_ = pd.DataFrame(fixed_matrix, index=matrix.index, columns=matrix.index) if isinstance(
            matrix, pd.DataFrame) else fixed_matrix

        return return_

    @staticmethod
    def minimum_covariance_determinant(returns,
                                       price_data=False,
                                       assume_centered=False,
                                       support_fraction=None,
                                       random_state=None,
                                       nonpositive_semidefinite_fix_method='spectral'):

        returns = pd.DataFrame(returns) if not isinstance(
            price_data, pd.DataFrame) else returns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns
        
        assets = returns.columns

        returns = returns.dropna().values

        cov_model = MinCovDet(
            random_state=random_state,
            support_fraction=support_fraction,
            assume_centered=assume_centered).\
            fit(returns)

        covariance = pd.DataFrame(
            cov_model.covariance_,
            index=assets,
            columns=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(covariance,
                                                           nonpositive_semidefinite_fix_method)

    @staticmethod
    def empirical_covariance(returns,
                             price_data=False,
                             assume_centered=False,
                             nonpositive_semidefinite_fix_method='spectral'):

        returns = pd.DataFrame(returns) if not isinstance(
            price_data, pd.DataFrame) else returns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns
        
        assets = returns.columns
        
        returns = returns.dropna().values

        cov_model = EmpiricalCovariance(
            assume_centered=assume_centered).\
            fit(returns)

        covariance = pd.DataFrame(
            cov_model.covariance_,
            index=assets,
            columns=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(covariance,
                                                           nonpositive_semidefinite_fix_method)

    @staticmethod
    def shrinked_covariance(returns,
                            price_data=False,
                            shrinkage_type='basic',
                            assume_centered=False,
                            basic_shrinkage=0.1,
                            nonpositive_semidefinite_fix_method='spectral'):

        returns = pd.DataFrame(returns) if not isinstance(
            price_data, pd.DataFrame) else returns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns
        
        assets = returns.columns

        if shrinkage_type == 'basic':
            cov_model = ShrunkCovariance(assume_centered=assume_centered,
                                         shrinkage=basic_shrinkage).\
                fit(returns)

        elif shrinkage_type == 'lw':
            cov_model = LedoitWolf(assume_centered=assume_centered).\
                fit(returns)

        elif shrinkage_type == 'oas':
            cov_model = OAS(assume_centered=assume_centered).\
                fit(returns)

        else:
            raise NotImplementedError(
                "Method {} not implemented".format(shrinkage_type))

        covariance = pd.DataFrame(
            cov_model.covariance_,
            index=assets,
            columns=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(covariance,
                                                           nonpositive_semidefinite_fix_method)

    # threshold_return == benchmark
    # added nonpositive_semidefinite_fix_method input argument
    @staticmethod
    def semi_covariance(returns,
                        price_data=False,
                        threshold_return=0,
                        nonpositive_semidefinite_fix_method='spectral'):

        returns = pd.DataFrame(returns) if not isinstance(
            price_data, pd.DataFrame) else returns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns
        
        assets = returns.columns

        drops = np.fmin(returns - threshold_return, 0)
        T = drops.shape[0]

        covariance = (drops.T @ drops) / T

        covariance = pd.DataFrame(
            covariance,
            index=assets,
            columns=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(covariance,
                                                           nonpositive_semidefinite_fix_method)

    @staticmethod
    def exponential_covariance(returns, price_data=False, window_span=60,
                               nonpositive_semidefinite_fix_method='spectral'):

        returns = pd.DataFrame(returns) if not isinstance(
            price_data, pd.DataFrame) else returns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns

        def _pair_exp_cov(X, Y, span=180):
            covariation = (X - X.mean()) * (Y - Y.mean())
            # Exponentially weight the covariation and take the mean
            return covariation.ewm(span=span).mean().iloc[-1]

        assets = returns.columns
        n = len(assets)

        cov_ = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                cov_[i, j] = cov_[j, i] = _pair_exp_cov(
                    returns.iloc[:, i], returns.iloc[:, j], span=window_span
                )

        cov_ = pd.DataFrame(
            cov_,
            columns=assets,
            index=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(cov_,
                                                           nonpositive_semidefinite_fix_method)

    @staticmethod
    def filter_corr_hierarchical(cor_matrix, method='complete', draw_plot=False):
        cor_matrix = cor_matrix.to_numpy() if isinstance(
            cor_matrix, pd.DataFrame) else cor_matrix
        
        Z = linkage(cor_matrix, method,)

        if draw_plot:
            plt.figure(figsize=(16, 8))
            dn = dendrogram(Z)
            plt.show()
            
        return Z

    def denoise_covariance(self,
                           cov,
                           tn_relation,
                           denoise_method='const_resid_eigen',
                           detone=False,
                           market_component=1,
                           kde_bwidth=0.01,
                           alpha=0):

        corr = self.cov_to_corr(cov)
        std = np.diag(cov) ** 0.5

        e_values, e_vectors = self._get_pca(corr)

        e_max, var = self._find_max_eval(
            np.diag(e_values), tn_relation, kde_bwidth)

        num_facts = e_values.shape[0] - \
            np.diag(e_values)[::-1].searchsorted(e_max)

        if denoise_method == "const_resid_eigen":
            denoised_corr = self._denoised_corr_const_resid_eigen(
                e_values,
                e_vectors,
                num_facts)

        elif denoise_method == "spectral":
            denoised_corr = self._denoised_corr_spectral(
                e_values,
                e_vectors,
                num_facts)

        elif denoise_method == "target_shrink":
            denoised_corr = self._denoised_corr_targ_shrink(
                e_values,
                e_vectors,
                num_facts,
                alpha)

        else:
            raise NotImplementedError(
                "Method {} not implemented".format(denoise_method))

        if detone:
            e_values, e_vectors = self._get_pca(corr)

            e_values_ = e_values[:market_component, :market_component]
            e_vectors_ = e_vectors[:, :market_component]

            corr_ = np.dot(e_vectors_, e_values_).dot(e_vectors_.T)
            detoned_corr = denoised_corr - corr_

            return RiskEstimators.corr_to_cov(detoned_corr, std)

        else:
            return RiskEstimators.corr_to_cov(denoised_corr, std)

    @staticmethod
    def corr_to_cov(corr, std):
        cov = corr * np.outer(std, std)
        return cov

    @staticmethod
    def cov_to_corr(cov):
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)

        # for numerical error
        corr[corr < -1], corr[corr > 1] = -1, +1
        return corr

    @staticmethod
    def is_matrix_invertible(matrix):
        # We should compute the condition number of the matrix to see if it is invertible.
        return np.isfinite(np.linalg.cond(matrix))

    @staticmethod
    def _fit_kde(observations, kde_bwidth=0.01, kde_kernel='gaussian', eval_points=None):
        observations = observations.reshape(-1, 1)

        if len(observations) == 1:
            observations = observations.reshape(1, -1)

        kde = KernelDensity(kernel=kde_kernel,
                            bandwidth=kde_bwidth,).fit(observations)

        if eval_points is None:
            eval_points = np.unique(observations)

        eval_points = eval_points.reshape(-1, 1)

        log_prob = kde.score_samples(eval_points)

        pdf = pd.Series(np.exp(log_prob), index=eval_points.flatten())

        return pdf

    @staticmethod
    def _mp_pdf(var, tn_relation, num_points):
        # tn_relation = T / N
        q = tn_relation

        e_min, e_max = var * (1 - (1. / q) ** .5) ** 2, var * \
            (1 + (1. / q) ** .5) ** 2

        e_values = np.linspace(e_min, e_max, num_points)

        pdf = q / (2 * np.pi * var * e_values) * \
            ((e_max - e_values) * (e_values - e_min)) ** 0.5

        pdf = pd.Series(pdf.ravel(), index=e_values.ravel())

        return pdf

    def _pdf_fit(self, var, eigen_observations, tn_relation, kde_bwidth, num_points=1000):

        # theoretical pdf
        pdf_0 = RiskEstimators._mp_pdf(
            var,
            tn_relation,
            num_points)

        # emprical pdf
        pdf_1 = RiskEstimators._fit_kde(
            eigen_observations,
            kde_bwidth,
            eval_points=pdf_0.index.values)

        sse = ((pdf_1 - pdf_0) ** 2).sum()

        return sse

    def _find_max_eval(self, eigen_observations, tn_relation, kde_bwidth):
        out = minimize(lambda *x: self._pdf_fit(*x),
                       0.5,
                       args=(eigen_observations, tn_relation, kde_bwidth),
                       bounds=((1E-5, 1-1E-5),))

        var = out['x'][0] if out['success'] else 1

        q = tn_relation

        e_max = var * (1 + (1 / q) ** 0.5) ** 2

        return e_max, var

    @staticmethod
    def _get_pca(hermit_matrix):

        e_values, e_vectors = np.linalg.eigh(hermit_matrix)
        indices = e_values.argsort()[::-1]
        e_values, e_vectors = e_values[indices], e_vectors[:, indices]
        e_values = np.diagflat(e_values)

        return e_values, e_vectors

    # function name changed to _denoised_corr_const_resid_eigen from _denoised_corr
    def _denoised_corr_const_resid_eigen(self, eigenvalues, eigenvectors, num_facts):

        e_values_ = np.diag(eigenvalues).copy()
        e_values_[num_facts:] = \
            e_values_[num_facts:].sum() / float(e_values_.shape[0] - num_facts)

        e_values_ = np.diag(e_values_)

        cov_1 = np.dot(eigenvectors, e_values_).dot(eigenvectors.T)
        corr_1 = RiskEstimators.cov_to_corr(cov_1)

        return corr_1

    def _denoised_corr_targ_shrink(self, eigenvalues, eigenvectors, num_facts, alpha=0):

        e_values_L, e_vectors_L = eigenvalues[:num_facts,
                                              :num_facts], eigenvectors[:, :num_facts]
        e_values_R, e_vectors_R = eigenvalues[num_facts:,
                                              num_facts:], eigenvectors[:, num_facts:]

        corr_0 = np.dot(e_vectors_L, e_values_L).dot(e_vectors_L.T)
        corr_1 = np.dot(e_vectors_R, e_values_R).dot(e_vectors_R.T)
        corr_2 = corr_0 + alpha * corr_1 + \
            (1 - alpha) * np.diag(np.diag(corr_1))

        return corr_2

    # this function isnt required. bad input arguments
    def _detoned_corr(self, corr, market_component=1):
        pass

    # kind argument added
    def _denoised_corr_spectral(self, eigenvalues, eigenvectors, num_facts):
        e_values_ = np.diag(eigenvalues).copy()
        e_values_[num_facts:] = 0

        e_values_ = np.diag(e_values_)

        cov_1 = np.dot(eigenvectors, e_values_).dot(eigenvectors.T)
        corr_1 = RiskEstimators.cov_to_corr(cov_1)

        return corr_1
