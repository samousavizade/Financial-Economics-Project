import warnings
import numpy as np
import pandas as pd
from sklearn import covariance
from sklearn.neighbors import KernelDensity
from sklearn.covariance import MinCovDet, EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS
from scipy.optimize import minimize
from scipy.cluster.hierarchy import average, complete, single, dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt

# from mlfinlab.portfolio_optimization.estimators.returns_estimators import ReturnsEstimators
from returns_estimators import ReturnsEstimators


class RiskEstimators:
    """
    This class contains the implementations for different ways to calculate and adjust Covariance matrices.
    The functions related to de-noising and de-toning the Covariance matrix are reproduced with modification
    from Chapter 2 of the the following book:
    Marcos Lopez de Prado “Machine Learning for Asset Managers”, (2020).
    """

    def __init__(self):
        """
        Initialize
        """

        pass

    @staticmethod
    def _is_positive_semidefinite(matrix):
        """
        Helper function to check if a given matrix is positive semidefinite.
        Any method that requires inverting the covariance matrix will struggle
        with a non-positive semidefinite matrix

        :param matrix: (covariance) matrix to test
        :type matrix: np.ndarray, pd.DataFrame
        :return: whether matrix is positive semidefinite
        :rtype: bool
        """
        try:
            # Significantly more efficient than checking eigenvalues (stackoverflow.com/questions/16266720)
            np.linalg.cholesky(matrix + 1e-16 * np.eye(len(matrix)))
            return True
        except np.linalg.LinAlgError:
            return False

    @staticmethod
    def fix_nonpositive_semidefinite(matrix, fix_method="spectral"):
        """
        Check if a covariance matrix is positive semidefinite, and if not, fix it
        with the chosen method.

        The ``spectral`` method sets negative eigenvalues to zero then rebuilds the matrix,
        while the ``diag`` method adds a small positive value to the diagonal.

        :param matrix: raw covariance matrix (may not be PSD)
        :type matrix: pd.DataFrame
        :param fix_method: {"spectral", "diag"}, defaults to "spectral"
        :type fix_method: str, optional
        :raises NotImplementedError: if a method is passed that isn't implemented
        :return: positive semidefinite covariance matrix
        :rtype: pd.DataFrame
        """

        if RiskEstimators._is_positive_semidefinite(matrix):
            return matrix

        warnings.warn(
            "The covariance matrix is non positive semidefinite. Amending eigenvalues."
        )

        # Eigendecomposition
        q, V = np.linalg.eigh(matrix)

        if fix_method == "spectral":
            # Remove negative eigenvalues
            q = np.where(q > 0, q, 0)
            # Reconstruct matrix
            fixed_matrix = V @ np.diag(q) @ V.T
        elif fix_method == "diag":
            min_eig = np.min(q)
            fixed_matrix = matrix - 1.1 * min_eig * np.eye(len(matrix))
        else:
            raise NotImplementedError(
                "Method {} not implemented".format(fix_method))

        if not RiskEstimators._is_positive_semidefinite(fixed_matrix):  # pragma: no cover
            warnings.warn(
                "Could not fix matrix. Please try a different risk model.", UserWarning
            )

        # Rebuild labels if provided
        if isinstance(matrix, pd.DataFrame):
            tickers = matrix.index
            return pd.DataFrame(fixed_matrix, index=tickers, columns=tickers)
        else:
            return fixed_matrix

    # TODO: frequency argument??????
    # TODO: All Cov estimators Function return Dataframe
    @staticmethod
    def minimum_covariance_determinant(returns,
                                       price_data=False, assume_centered=False,
                                       support_fraction=None, random_state=None,
                                       nonpositive_semidefinite_fix_method='spectral'):
        """
        Calculates the Minimum Covariance Determinant for a dataframe of asset prices or returns.

        This function is a wrap of the sklearn's MinCovDet (MCD) class. According to the
        scikit-learn User Guide on Covariance estimation:

        "The idea is to find a given proportion (h) of “good” observations that are not outliers
        and compute their empirical covariance matrix. This empirical covariance matrix is then
        rescaled to compensate for the performed selection of observations".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param assume_centered: (bool) Flag for data with mean significantly equal to zero.
                                       (Read the documentation for MinCovDet class, False by default)
        :param support_fraction: (float) Values between 0 and 1. The proportion of points to be included in the support
                                         of the raw MCD estimate. (Read the documentation for MinCovDet class,
                                         None by default)
        :param random_state: (int) Seed used by the random number generator. (None by default)
        :return: (np.array) Estimated robust covariance matrix.
        """

        if not isinstance(price_data, pd.DataFrame):
            warnings.warn("data is not in a dataframe", RuntimeWarning)
            returns = pd.DataFrame(returns)

        assets = returns.columns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns

        returns = returns.dropna().values

        cov_model = MinCovDet(
            random_state=random_state,
            support_fraction=support_fraction,
            assume_centered=assume_centered).\
            fit(returns)

        covariance = pd.DataFrame(
            cov_model.covariance_, index=assets, columns=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(covariance,
                                                           nonpositive_semidefinite_fix_method)

    @staticmethod
    def empirical_covariance(returns, price_data=False, assume_centered=False,
                             nonpositive_semidefinite_fix_method='spectral'):
        """
        Calculates the Maximum likelihood covariance estimator for a dataframe of asset prices or returns.

        This function is a wrap of the sklearn's EmpiricalCovariance class. According to the
        scikit-learn User Guide on Covariance estimation:

        "The covariance matrix of a data set is known to be well approximated by the classical maximum
        likelihood estimator, provided the number of observations is large enough compared to the number
        of features (the variables describing the observations). More precisely, the Maximum Likelihood
        Estimator of a sample is an unbiased estimator of the corresponding population’s covariance matrix".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param assume_centered: (bool) Flag for data with mean almost, but not exactly zero.
                                       (Read documentation for EmpiricalCovariance class, False by default)
        :return: (np.array) Estimated covariance matrix.
        """

        if not isinstance(price_data, pd.DataFrame):
            warnings.warn("data is not in a dataframe", RuntimeWarning)
            returns = pd.DataFrame(returns)

        assets = returns.columns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns

        returns = returns.dropna().values

        cov_model = EmpiricalCovariance(
            assume_centered=assume_centered).\
            fit(returns)

        covariance = pd.DataFrame(
            cov_model.covariance_, index=assets, columns=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(covariance,
                                                           nonpositive_semidefinite_fix_method)

    @staticmethod
    def shrinked_covariance(returns, price_data=False, shrinkage_type='basic', assume_centered=False,
                            basic_shrinkage=0.1,
                            nonpositive_semidefinite_fix_method='spectral'):
        """
        Calculates the Covariance estimator with shrinkage for a dataframe of asset prices or returns.

        This function allows three types of shrinkage - Basic, Ledoit-Wolf and Oracle Approximating Shrinkage.
        It is a wrap of the sklearn's ShrunkCovariance, LedoitWolf and OAS classes. According to the
        scikit-learn User Guide on Covariance estimation:

        "Sometimes, it even occurs that the empirical covariance matrix cannot be inverted for numerical
        reasons. To avoid such an inversion problem, a transformation of the empirical covariance matrix
        has been introduced: the shrinkage. Mathematically, this shrinkage consists in reducing the ratio
        between the smallest and the largest eigenvalues of the empirical covariance matrix".

        Link to the documentation:
        <https://scikit-learn.org/stable/modules/covariance.html>`_

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param shrinkage_type: (str) Type of shrinkage to use. (``basic`` by default, ``lw``, ``oas``, ``all``)
        :param assume_centered: (bool) Flag for data with mean almost, but not exactly zero.
                                       (Read documentation for chosen shrinkage class, False by default)
        :param basic_shrinkage: (float) Between 0 and 1. Coefficient in the convex combination for basic shrinkage.
                                        (0.1 by default)
        :return: (np.array) Estimated covariance matrix. Tuple of covariance matrices if shrinkage_type = ``all``.
        """

        if not isinstance(price_data, pd.DataFrame):
            warnings.warn("data is not in a dataframe", RuntimeWarning)
            returns = pd.DataFrame(returns)

        assets = returns.columns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns

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
            raise ValueError("Unknown shrinkage type")

        covariance = pd.DataFrame(
            cov_model.covariance_, index=assets, columns=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(covariance,
                                                           nonpositive_semidefinite_fix_method)

    # threshold_return == benchmark
    # added nonpositive_semidefinite_fix_method input argument
    @staticmethod
    def semi_covariance(returns, price_data=False, threshold_return=0,
                        nonpositive_semidefinite_fix_method='spectral'):
        """
        Calculates the Semi-Covariance matrix for a dataframe of asset prices or returns.

        Semi-Covariance matrix is used to calculate the portfolio's downside volatility. Usually, the
        threshold return is zero and the negative volatility is measured. A threshold can be a positive number
        when one assumes a required return rate. If the threshold is above zero, the output is the volatility
        measure for returns below this threshold.

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param threshold_return: (float) Required return for each period in the frequency of the input data.
                                         (If the input data is daily, it's a daily threshold return, 0 by default)
        :param frequency: (int) number of time periods in a year, defaults to 252 (the number
                      of trading days in a year). Ensure that you use the appropriate
                      benchmark, e.g if ``frequency=12`` use the monthly risk-free rate.
        :return: (np.array) Semi-Covariance matrix.
        """

        if not isinstance(price_data, pd.DataFrame):
            warnings.warn("data is not in a dataframe", RuntimeWarning)
            returns = pd.DataFrame(returns)

        assets = returns.columns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns

        drops = np.fmin(returns - threshold_return, 0)
        T = drops.shape[0]

        covariance = (drops.T @ drops) / T

        covariance = pd.DataFrame(covariance, index=assets, columns=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(covariance,
                                                           nonpositive_semidefinite_fix_method)

    @staticmethod
    def exponential_covariance(returns, price_data=False, window_span=60,
                               nonpositive_semidefinite_fix_method='spectral'):
        """
        Calculates the Exponentially-weighted Covariance matrix for a dataframe of asset prices or returns.

        It calculates the series of covariances between elements and then gets the last value of exponentially
        weighted moving average series from covariance series as an element in matrix.

        If a dataframe of prices is given, it is transformed into a dataframe of returns using
        the calculate_returns method from the ReturnsEstimators class.

        :param returns: (pd.DataFrame) Dataframe where each column is a series of returns or prices for an asset.
        :param price_data: (bool) Flag if prices of assets are used and not returns. (False by default)
        :param window_span: (int) Used to specify decay in terms of span for the exponentially-weighted series.
                                  (60 by default)
        :return: (np.array) Exponentially-weighted Covariance matrix.
        """

        if not isinstance(price_data, pd.DataFrame):
            warnings.warn("data is not in a dataframe", RuntimeWarning)
            returns = pd.DataFrame(returns)

        assets = returns.columns

        returns = ReturnsEstimators.calculate_returns(
            returns) if price_data else returns

        n = len(assets)

        def _pair_exp_cov(X, Y, span=180):
            """
            Calculate the exponential covariance between two timeseries of returns.

            :param X: first time series of returns
            :type X: pd.Series
            :param Y: second time series of returns
            :type Y: pd.Series
            :param span: the span of the exponential weighting function, defaults to 180
            :type span: int, optional
            :return: the exponential covariance between X and Y
            :rtype: float
            """
            covariation = (X - X.mean()) * (Y - Y.mean())
            # Exponentially weight the covariation and take the mean
            if span < 10:
                warnings.warn(
                    "it is recommended to use a higher span, e.g 30 days")
            return covariation.ewm(span=span).mean().iloc[-1]

        cov_ = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                cov_[i, j] = cov_[j, i] = _pair_exp_cov(
                    returns.iloc[:, i], returns.iloc[:, j], span=window_span
                )

        cov_ = pd.DataFrame(cov_, columns=assets, index=assets)

        return RiskEstimators.fix_nonpositive_semidefinite(cov_,
                                                           nonpositive_semidefinite_fix_method)

    @staticmethod
    def filter_corr_hierarchical(cor_matrix, method='complete', draw_plot=False):
        """
        Creates a filtered correlation matrix using hierarchical clustering methods from an empirical
        correlation matrix, given that all values are non-negative [0 ~ 1]
        This function allows for three types of hierarchical clustering - complete, single, and average
        linkage clusters. Link to hierarchical clustering methods documentation:
        `<https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html>`_
        It works as follows:
        First, the method creates a hierarchical clustering tree using scipy's hierarchical clustering methods
        from the empirical 2-D correlation matrix.
        Second, it extracts and stores each cluster's filtered value (alpha) and assigns it to it's corresponding leaf.
        Finally, we create a new filtered matrix by assigning each of the correlations to their corresponding
        parent node's alpha value.

        :param cor_matrix: (np.array) Numpy array of an empirical correlation matrix.
        :param method: (str) Hierarchical clustering method to use. (``complete`` by default, ``single``, ``average``)
        :param draw_plot: (bool) Plots the hierarchical cluster tree. (False by default)
        :return: (np.array) The filtered correlation matrix.
        """

        # TODO: This function is not tested yet and maybe bad output

        if isinstance(cor_matrix, pd.DataFrame):
            cor_matrix = cor_matrix.to_numpy()
            warnings.warn("correlation matrix is a dataframe.", RuntimeWarning)

        Z = linkage(cor_matrix, method)

        if draw_plot:
            plt.figure(figsize=(16, 8))
            dn = dendrogram(Z)
            plt.show()

        pass

    # TODO: Covariance estimators Frequency argument??????

    def denoise_covariance(self, cov, tn_relation, denoise_method='const_resid_eigen', detone=False,
                           market_component=1, kde_bwidth=0.01, alpha=0):
        """
        De-noises the covariance matrix or the correlation matrix.

        Two denoising methods are supported:
        1. Constant Residual Eigenvalue Method (``const_resid_eigen``)
        2. Spectral Method (``spectral``)
        3. Targeted Shrinkage Method (``target_shrink``)

        The Constant Residual Eigenvalue Method works as follows:

        First, a correlation is calculated from the covariance matrix (if the input is the covariance matrix).

        Second, eigenvalues and eigenvectors of the correlation matrix are calculated using the linalg.eigh
        function from numpy package.

        Third, a maximum theoretical eigenvalue is found by fitting Marcenko-Pastur (M-P) distribution
        to the empirical distribution of the correlation matrix eigenvalues. The empirical distribution
        is obtained through kernel density estimation using the KernelDensity class from sklearn.
        The fit of the M-P distribution is done by minimizing the Sum of Squared estimate of Errors
        between the theoretical pdf and the kernel. The minimization is done by adjusting the variation
        of the M-P distribution.

        Fourth, the eigenvalues of the correlation matrix are sorted and the eigenvalues lower than
        the maximum theoretical eigenvalue are set to their average value. This is how the eigenvalues
        associated with noise are shrinked. The de-noised covariance matrix is then calculated back
        from new eigenvalues and eigenvectors.

        The Spectral Method works just like the Constant Residual Eigenvalue Method, but instead of replacing
        eigenvalues lower than the maximum theoretical eigenvalue to their average value, they are replaced with
        zero instead.

        The Targeted Shrinkage Method works as follows:

        First, a correlation is calculated from the covariance matrix (if the input is the covariance matrix).

        Second, eigenvalues and eigenvectors of the correlation matrix are calculated using the linalg.eigh
        function from numpy package.

        Third, the correlation matrix composed from eigenvectors and eigenvalues related to noise is
        shrunk using the alpha variable. The shrinkage is done by summing the noise correlation matrix
        multiplied by alpha to the diagonal of the noise correlation matrix multiplied by (1-alpha).

        Fourth, the shrinked noise correlation matrix is summed to the information correlation matrix.

        Correlation matrix can also be detoned by excluding a number of first eigenvectors representing
        the market component.

        These algorithms are reproduced with minor modifications from the following book:
        Marcos Lopez de Prado “Machine Learning for Asset Managers”, (2020).

        :param cov: (np.array) Covariance matrix or correlation matrix.
        :param tn_relation: (float) Relation of sample length T to the number of variables N used to calculate the
                                    covariance matrix.
        :param denoise_method: (str) Denoising methos to use. (``const_resid_eigen`` by default, ``target_shrink``)
        :param detone: (bool) Flag to detone the matrix. (False by default)
        :param market_component: (int) Number of fist eigevectors related to a market component. (1 by default)
        :param kde_bwidth: (float) The bandwidth of the kernel to fit KDE.
        :param alpha: (float) In range (0 to 1) - shrinkage of the noise correlation matrix to use in the
                              Targeted Shrinkage Method. (0 by default)
        :return: (np.array) De-noised covariance matrix or correlation matrix.
        """

        corr = self.cov_to_corr(cov)
        std = np.diag(cov) ** 0.5

        e_values, e_vectors = self._get_pca(corr)

        # TODO: var didnt used???
        e_max, var = self._find_max_eval(
            np.diag(e_values), tn_relation, kde_bwidth)

        num_facts = e_values.shape[0] - \
            np.diag(e_values)[::-1].searchsorted(e_max)

        if denoise_method == "const_resid_eigen":
            # missing parameter
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
        """
        Recovers the covariance matrix from a correlation matrix.

        Requires a vector of standard deviations of variables - square root
        of elements on the main diagonal fo the covariance matrix.

        Formula used: Cov = Corr * OuterProduct(std, std)

        :param corr: (np.array) Correlation matrix.
        :param std: (np.array) Vector of standard deviations.
        :return: (np.array) Covariance matrix.
        """

        cov = corr * np.outer(std, std)
        return cov

    @staticmethod
    def cov_to_corr(cov):
        """
        Derives the correlation matrix from a covariance matrix.

        Formula used: Corr = Cov / OuterProduct(std, std)

        :param cov: (np.array) Covariance matrix.
        :return: (np.array) Covariance matrix.
        """

        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)

        # numerical error
        corr[corr < -1], corr[corr > 1] = -1, +1
        return corr

    @staticmethod
    def is_matrix_invertible(matrix):
        """
        Check if a matrix is invertible or not.
        :param matrix: (Numpy matrix) A matrix whose invertibility we want to check.
        :return: (bool) Boolean value depending on whether the matrix is invertible or not.
        """

        # We should compute the condition number of the matrix to see if it is invertible.
        return np.isfinite(np.linalg.cond(matrix))

    @staticmethod
    def _fit_kde(observations, kde_bwidth=0.01, kde_kernel='gaussian', eval_points=None):
        """
        Fits kernel to a series of observations (in out case eigenvalues), and derives the
        probability density function of observations.

        The function used to fit kernel is KernelDensity from sklearn.neighbors. Fit of the KDE
        can be evaluated on a given set of points, passed as eval_points variable.

        :param observations: (np.array) Array of observations (eigenvalues) eigenvalues to fit kernel to.
        :param kde_bwidth: (float) The bandwidth of the kernel. (0.01 by default)
        :param kde_kernel: (str) Kernel to use [``gaussian`` by default, ``tophat``, ``epanechnikov``, ``exponential``,
                                 ``linear``,``cosine``].
        :param eval_points: (np.array) Array of values on which the fit of the KDE will be evaluated.
                                       If None, the unique values of observations are used. (None by default)
        :return: (pd.Series) Series with estimated pdf values in the eval_points.
        """
        
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
        """
        Derives the pdf of the Marcenko-Pastur distribution.

        Outputs the pdf for num_points between the minimum and maximum expected eigenvalues.
        Requires the variance of the distribution (var) and the relation of T - the number
        of observations of each X variable to N - the number of X variables (T/N).

        :param var: (float) Variance of the M-P distribution.
        :param tn_relation: (float) Relation of sample length T to the number of variables N (T/N).
        :param num_points: (int) Number of points to estimate pdf.
        :return: (pd.Series) Series of M-P pdf values.
        """

        # tn_relation = T / N
        q = tn_relation

        e_min, e_max = var * (1 - (1. / q) ** .5) ** 2, var * (1 + (1. / q) ** .5) ** 2

        e_values = np.linspace(e_min, e_max, num_points)

        pdf = q / (2 * np.pi * var * e_values) * \
            ((e_max - e_values) * (e_values - e_min)) ** 0.5

        pdf = pd.Series(pdf.ravel(), index=e_values.ravel())

        return pdf

    def _pdf_fit(self, var, eigen_observations, tn_relation, kde_bwidth, num_points=1000):
        """
        Calculates the fit (Sum of Squared estimate of Errors) of the empirical pdf
        (kernel density estimation) to the theoretical pdf (Marcenko-Pastur distribution).

        SSE is calculated for num_points, equally spread between minimum and maximum
        expected theoretical eigenvalues.

        :param var: (float) Variance of the M-P distribution. (for the theoretical pdf)
        :param eigen_observations: (np.array) Observed empirical eigenvalues. (for the empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N. (for the theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel. (for the empirical pdf)
        :param num_points: (int) Number of points to estimate pdf. (for the empirical pdf, 1000 by default)
        :return: (float) SSE between empirical pdf and theoretical pdf.
        """
        # 2.4
        # theoretical pdf
        pdf_0 = RiskEstimators._mp_pdf(var, tn_relation, num_points)

        # emprical pdf
        pdf_1 = RiskEstimators._fit_kde(
            eigen_observations, kde_bwidth, eval_points=pdf_0.index.values)

        sse = ((pdf_1 - pdf_0) ** 2).sum()

        return sse

    def _find_max_eval(self, eigen_observations, tn_relation, kde_bwidth):
        """
        Searching for maximum random eigenvalue by fitting Marcenko-Pastur distribution
        to the empirical one - obtained through kernel density estimation. The fit is done by
        minimizing the Sum of Squared estimate of Errors between the theoretical pdf and the
        kernel fit. The minimization is done by adjusting the variation of the M-P distribution.

        :param eigen_observations: (np.array) Observed empirical eigenvalues. (for the empirical pdf)
        :param tn_relation: (float) Relation of sample length T to the number of variables N. (for the theoretical pdf)
        :param kde_bwidth: (float) The bandwidth of the kernel. (for the empirical pdf)
        :return: (float, float) Maximum random eigenvalue, optimal variation of the Marcenko-Pastur distribution.
        """

        # out=minimize(lambda *x:errPDFs(*x),.5,args=(eVal,q,bWidth),
        # bounds=((1E-5,1-1E-5),))
        # if out[’success’]:var=out[’x’][0]
        # else:var=1
        # eMax=var*(1+(1./q)**.5)**2
        # return eMax,var

        out = minimize(lambda *x: self._pdf_fit(*x),
                       0.5,
                       args=(eigen_observations, tn_relation, kde_bwidth),
                       bounds=((1E-5, 1-1E-5),))

        if out['success']:
            var = out['x'][0]

        else:
            var = 1

        q = tn_relation

        e_max = var * (1 + (1 / q) ** 0.5) ** 2

        return e_max, var

    @staticmethod
    def _get_pca(hermit_matrix):
        """
        Calculates eigenvalues and eigenvectors from a Hermitian matrix. In our case, from the correlation matrix.

        Function used to calculate the eigenvalues and eigenvectors is linalg.eigh from numpy package.

        Eigenvalues in the output are placed on the main diagonal of a matrix.

        :param hermit_matrix: (np.array) Hermitian matrix.
        :return: (np.array, np.array) Eigenvalues matrix, eigenvectors array.
        """

        e_values, e_vectors = np.linalg.eigh(hermit_matrix)
        indices = e_values.argsort()[::-1]
        e_values, e_vectors = e_values[indices], e_vectors[:, indices]
        e_values = np.diagflat(e_values)

        return e_values, e_vectors

    # function name changed to _denoised_corr_const_resid_eigen from _denoised_corr
    def _denoised_corr_const_resid_eigen(self, eigenvalues, eigenvectors, num_facts):
        """
        De-noises the correlation matrix using the Constant Residual Eigenvalue method.

        The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
        of the first eigenvalue that is below the maximum theoretical eigenvalue.

        De-noising is done by shrinking the eigenvalues associated with noise (the eigenvalues lower than
        the maximum theoretical eigenvalue are set to a constant eigenvalue, preserving the trace of the
        correlation matrix).

        The result is the de-noised correlation matrix.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
        :param eigenvectors: (float) Eigenvectors array.
        :param num_facts: (float) Threshold for eigenvalues to be fixed.
        :return: (np.array) De-noised correlation matrix.
        """

        e_values_ = np.diag(eigenvalues).copy()
        e_values_[num_facts:] = \
            e_values_[num_facts:].sum() / float(e_values_.shape[0] - num_facts)

        e_values_ = np.diag(e_values_)

        # TODO: use @ instead of .dot method
        cov_1 = np.dot(eigenvectors, e_values_).dot(eigenvectors.T)
        corr_1 = RiskEstimators.cov_to_corr(cov_1)

        return corr_1

    def _denoised_corr_targ_shrink(self, eigenvalues, eigenvectors, num_facts, alpha=0):
        """
        De-noises the correlation matrix using the Targeted Shrinkage method.

        The input is the correlation matrix, the eigenvalues and the eigenvectors of the correlation
        matrix and the number of the first eigenvalue that is below the maximum theoretical eigenvalue
        and the shrinkage coefficient for the eigenvectors and eigenvalues associated with noise.

        Shrinks strictly the random eigenvalues - eigenvalues below the maximum theoretical eigenvalue.

        The result is the de-noised correlation matrix.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
        :param eigenvectors: (float) Eigenvectors array.
        :param num_facts: (float) Threshold for eigenvalues to be fixed.
        :param alpha: (float) In range (0 to 1) - shrinkage among the eigenvectors.
                              and eigenvalues associated with noise. (0 by default)
        :return: (np.array) De-noised correlation matrix.
        """

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
        """
        De-tones the correlation matrix by removing the market component.

        The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
        of the first eigenvalue that is above the maximum theoretical eigenvalue and the number of
        eigenvectors related to a market component.

        :param corr: (np.array) Correlation matrix to detone.
        :param market_component: (int) Number of fist eigevectors related to a market component. (1 by default)
        :return: (np.array) De-toned correlation matrix.
        """
        pass

    # kind argument added
    def _denoised_corr_spectral(self, eigenvalues, eigenvectors, num_facts):
        """
        De-noises the correlation matrix using the Spectral method.
        The input is the eigenvalues and the eigenvectors of the correlation matrix and the number
        of the first eigenvalue that is below the maximum theoretical eigenvalue.
        De-noising is done by shrinking the eigenvalues associated with noise (the eigenvalues lower than
        the maximum theoretical eigenvalue are set to zero, preserving the trace of the
        correlation matrix).
        The result is the de-noised correlation matrix.

        :param eigenvalues: (np.array) Matrix with eigenvalues on the main diagonal.
        :param eigenvectors: (float) Eigenvectors array.
        :param num_facts: (float) Threshold for eigenvalues to be fixed.
        :return: (np.array) De-noised correlation matrix.
        """

        e_values_ = np.diag(eigenvalues).copy()
        e_values_[num_facts:] = 0

        e_values_ = np.diag(e_values_)

        # TODO: use @ instead of .dot method
        cov_1 = np.dot(eigenvectors, e_values_).dot(eigenvectors.T)
        corr_1 = RiskEstimators.cov_to_corr(cov_1)

        return corr_1


class DenoiseMethodNotFound(Exception):
    def __init__(self, message) -> None:
        super().__init__(message)

    pass
