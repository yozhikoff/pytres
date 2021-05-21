import time as time_module

import numpy as np
import scipy.interpolate
import scipy.optimize
from numpy.fft import irfft, rfft
from scipy import stats as stat
from tqdm.notebook import tqdm

from pytres.tres import TRES, AnnotatedTres, get_uniform_grid_on_sphere


class TresSolverNNLS(TRES):
    """
    Class for solving TRES using NNLS.
    """
    def __init__(self, X, time, irf,
                 n_components=2,
                 time_slice=slice(None, None),
                 wavelength_slice=slice(None, None),
                 scattering=True,
                 wavelengths=None,
                 padding_length=2500,
                 t0=None):
        """
        Parameters
        ----------
        X : np.array
            TRES matrix n_time x n_wavelength
        time : np.array
            The time points measurments were made at (size n_time)
        irf : np.array
            Measures IRF (size n_time)
        n_components : int
            Number of components in a luminophore mixture
        time_slice : slice
            Use to crop data
        wavelength_slice : slice
            Use to crop data
        scattering : bool
            Take scatering into account?
        wavelengths : np.array
        padding_length : int
            IRF circular padding length
        t0 : float ot None
            IRF shift - not computed, if provided
        """

        super().__init__(X, time, irf)
        self.time_slice = time_slice
        self.wavelength_slice = wavelength_slice
        self.scattering = scattering
        self.wavelengths = wavelengths
        self.n_components = n_components

        self.t0 = t0

        if len(time) < padding_length * 2:
            padding_length = len(time) // 3
        self.time_padded = self.extend_time(time, padding_length).astype(np.float64)
        self.irf_padded = self.circular_pad(irf, padding_length).astype(np.float64)
        self.tres_annot = None

    @property
    def X_summed(self):
        return self.X[:, self.wavelength_slice].sum(axis=1)

    @staticmethod
    def matrix_nnmf(A, X, W=None):
        """
        Solve argmin_b || AB - X ||_2 for b>=0
        Parameters
        ----------
        A : np.array
        X : np.array
        W : np.array
            weights

        Returns
        -------
        np.array B

        """
        if W is None:
            W = np.ones_like(X)
        W_sqrt = np.sqrt(W)
        B = np.zeros((A.shape[-1], X.shape[-1]))
        for i in range(X.shape[-1]):
            W_ = W_sqrt[:, i]
            B[:, i] = scipy.optimize.nnls(W_[:, None] * A, W_ * X[:, i])[0]
        return B

    @staticmethod
    def matrix_nnmf_without_w(A, X):
        B = np.zeros((A.shape[-1], X.shape[-1]))
        for i in range(X.shape[-1]):
            B[:, i] = scipy.optimize.nnls(A, X[:, i])[0]
        return B

    def matrix_nnmf_poisson(self, A, X, W=None):
        if W is None:
            B = self.matrix_nnmf_without_w(A, X)
            W = A @ B
        return self.matrix_nnmf(A, X, W=1 / np.maximum(W, 0.001))

    @staticmethod
    def conv1d(signal, kernel):
        conved = irfft(rfft(signal) * rfft(kernel), signal.shape[-1])
        return conved

    @staticmethod
    def circular_pad(t, n):
        return np.concatenate([t[-n:], t, t[:n]])

    @staticmethod
    def extend_time(t, n):
        diff = t[1] - t[0]
        start = np.arange(t[0] - n * diff, t[0], diff)
        end = np.arange(t[-1] + diff, t[-1] + (n + 1) * diff, diff)
        return np.concatenate([start, t, end])

    def get_decays(self, tau, t0):
        T = np.exp(-self.time / tau[:, None])
        irf = scipy.interpolate.interp1d(self.time_padded, self.irf_padded)(self.time - t0)
        T_convolved = self.conv1d(T, irf)
        if self.scattering:
            T_convolved = np.vstack([T_convolved, irf])
        return (T_convolved / T_convolved.max(axis=1)[:, None]).T

    def nnls_t0_fast(self, t0, return_spectra=False):
        tau = np.logspace(-4, 6, 128, base=2)
        T = self.get_decays(tau, t0)
        T = np.concatenate([T, np.ones((len(T), 1))], axis=1)
        S = self.matrix_nnmf_poisson(T, self.X_summed[:, None])[:, 0]
        TS = T @ S
        if return_spectra:
            return tau, S
        loss = ((TS - self.X_summed) ** 2) / np.maximum(TS, 0.000001)
        return loss[self.time_slice].sum()

    def nnls(self, tau, t0, return_spectra=False, W=None):
        if (tau < 0).any():
            return 1e10
        T = self.get_decays(tau, t0)
        T = np.concatenate([T, np.ones((len(T), 1))], axis=1)
        S = self.matrix_nnmf_poisson(T, self.X, W)
        TS = T @ S
        if return_spectra:
            return T, S, TS
        loss = ((TS - self.X) ** 2) / np.maximum(TS, 0.000001)
        return loss[self.time_slice, self.wavelength_slice].sum()

    def chi(self, tau, t0, dof):
        return self.nnls(tau, t0) / dof

    def nnls_chi(self, c, v, tau, t0, dof, chi, W=None):
        if c < 0:
            return 1e10 * (-c + 0.01)
        tau = tau + c * v
        return (self.nnls(tau, t0) / dof - chi) ** 2

    def solve(self):
        def print_subset_trace(trace):
            trace = {k: trace[k] for k in ['message', 'nit', 'fun', 'x']}
            for k, v in trace.items():
                print(f'\t{k}: {v}')

        print('Findind IRF shift...')
        time_start = time_module.time()
        if self.t0 is None:
            self.t0 = scipy.optimize.brent(self.nnls_t0_fast, brack=(-0.3, 0.3), tol=1e-6)
        print(f'Elapsed time: {time_module.time() - time_start :.2f}s', f'\tIRF_shift: {self.t0 :.5f}', sep='\n')

        print('\nSolving TRES...')
        time_start = time_module.time()
        tau_initial = np.random.rand(self.n_components) * 5
        # tau_initial = np.array([0.4, 2.5])
        opt_trace = scipy.optimize.minimize(self.nnls, tau_initial, args=(self.t0,), method='Nelder-Mead')
        print(f'Elapsed time: {time_module.time() - time_start :.2f}s')
        print_subset_trace(opt_trace)
        self.tres_annot = self.create_annot(opt_trace.x)
        return self.tres_annot

    def create_annot(self, tau):
        T, S, TS = self.nnls(tau, self.t0, return_spectra=True)
        bg = S[-1]
        S = S[:-1]
        if self.scattering:
            scattering_time = None
            scattering_spectra = S[-1]
            S = S[:-1]
        else:
            scattering_time = None
            scattering_spectra = None
        spectra = S
        lifetimes = tau
        t_start = self.t0
        X_sim = TS
        X = self.X
        time = self.time
        irf = self.irf
        trace = None
        return AnnotatedTres(X, time, spectra, lifetimes, scattering_spectra, scattering_time,
                             t_start, X_sim, trace, bg, irf, wavelengths=self.wavelengths, time_slice=self.time_slice,
                             wavelength_slice=self.wavelength_slice)

    def compute_confidence_indervals(self, random=True):
        chi_squared, chi_sq_ci, _ = self.tres_annot.reduced_chi_squared(self.time_slice, self.wavelength_slice)
        dof = self.tres_annot.degrees_of_freedom(self.time_slice, self.wavelength_slice)
        chi_squared_accepted = chi_squared + chi_sq_ci
        tau = self.tres_annot.lifetimes

        basis_coefs = []
        basis = np.eye(self.n_components)
        for v in basis:
            args = (v, tau, self.t0, dof, chi_squared_accepted)
            c = scipy.optimize.brent(self.nnls_chi, brack=(0, 0.3), tol=1e-6, args=args)
            basis_coefs.append(c)
        basis_coefs = np.array(basis_coefs)
        basis_coefs = np.minimum(basis_coefs, 1)
        print(basis_coefs)

        if not random:
            if self.n_components == 1:
                vectors = np.array([[1], [-1]])

            elif self.n_components == 2:
                grid = np.linspace(0, 2, 150)
                cos = np.cos(grid * np.pi)
                sin = np.sin(grid * np.pi)
                vectors = np.stack([sin, cos]).T

            elif self.n_components == 3:
                vectors = get_uniform_grid_on_sphere(400, r=1).T

            else:
                raise NotImplementedError('Not yet...')

        else:
            print(basis_coefs)
            print(np.power(np.diag(1 / basis_coefs), 1))
            dist = stat.multivariate_normal(mean=np.zeros((self.n_components,)), cov=1)
            vectors = dist.rvs(100 * self.n_components ** 2).T
            vectors = (vectors / np.sqrt((vectors ** 2).sum(axis=0))).T

        vectors = vectors * basis_coefs

        pt = []

        for v in tqdm(vectors):
            args = (v, tau, self.t0, dof, chi_squared_accepted, self.tres_annot.X_sim)
            c = scipy.optimize.brent(self.nnls_chi, brack=(0.05, 0.3), tol=1e-6, args=args)

            pt.append(c * v)

        pt = np.array(pt)
        ci_max = (pt + tau).max(axis=0)
        ci_min = (pt + tau).min(axis=0)

        print('Lifetimes CI:')
        for min_val, max_val in zip(ci_min, ci_max):
            mean = (min_val + max_val)/2
            print(f'{mean :.3f} +/- {max_val - mean :.3f}')

        return pt, vectors