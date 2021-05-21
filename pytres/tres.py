import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stat

from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator


def get_uniform_grid_on_sphere(n, r=1):
    """
    Parameters
    ----------git
    n : int
        Number of points
    r : float
        sphere radius

    Returns
    -------
    np.array
        Random points
    """
    xs = []
    ys = []
    zs = []

    alpha = 4.0 * np.pi * r * r / n
    d = np.sqrt(alpha)
    m_nu = int(np.round(np.pi / d))
    d_nu = np.pi / m_nu
    d_phi = alpha / d_nu
    count = 0
    for m in range(0, m_nu):
        nu = np.pi * (m + 0.5) / m_nu
        m_phi = int(np.round(2 * np.pi * np.sin(nu) / d_phi))
        for n in range(0, m_phi):
            phi = 2 * np.pi * n / m_phi
            xp = r * np.sin(nu) * np.cos(phi)
            yp = r * np.sin(nu) * np.sin(phi)
            zp = r * np.cos(nu)
            xs.append(xp)
            ys.append(yp)
            zs.append(zp)
            count = count + 1
    return np.array([xs, ys, zs])


def IndexFormatter(labels):
    """

    Parameters
    ----------
    labels : iterable

    Returns
    -------
    A formatter that uses labels from a provided iterable

    """
    def fmt_func(x, _):
        rounded = round(x)
        if not 0 <= rounded < len(labels):
            return ''
        elif abs(labels[rounded] - round(labels[rounded])) < 0.01:
            return int(labels[rounded])
        else:
            return labels[rounded]

    return FuncFormatter(fmt_func)


class TRES:
    """
    A base class for working with TRES data.
    """
    def __init__(self, X=None, time=None, irf=None):
        """
        Parameters
        ----------
        X : np.array
            TRES matrix n_time x n_wavelength
        time : np.array
            The time points measurments were made at (size n_time)
        irf : np.array
            Measures IRF (size n_time)
        """
        self.X = X
        self.time = time
        self.irf = irf


class AnnotatedTres(TRES):
    """
    A class for working with solved TRES spectra. Contains visialization methods. Can be constructed from both
    NNLS and VI solvers.
    """
    def __init__(self, X, time, specrta, lifetimes, scattering_spectra, scattering_time, t_start, X_sim, trace, bg,
                 irf=None, wavelengths=None, time_slice=slice(None, None), wavelength_slice=slice(None, None), ):
        """
        Parameters
        ----------
        X : np.array
            TRES matrix n_time x n_wavelength
        time : np.array
            The time points measurments were made at (size n_time)
        lifetimes : np.array
            Solved lifetimes
        scattering_spectra : np.array
            Solved scattering spectra
        scattering_time : float
            DEPRECATED
        t_start : float
            IRF shift
        X_sim : np.array
            Simulated TRES matrix
        trace : dict
            Raw trace from the solver
        bg : np.array
            Background spectra
        irf : np.array
            Measures IRF (size n_time)
        wavelengths : np.array
            Wavelengths measurments were made at
        time_slice : slice
            slice to zoom into time interval of interest
        wavelength_slice : slice
            slice to zoom into wavelengths of interest
        """
        super().__init__(X, time, irf)
        self.spectra = specrta
        self.lifetimes = lifetimes
        self.scattering_spectra = scattering_spectra
        self.scattering_time = scattering_time
        self.t_start = t_start
        self.X_sim = X_sim
        self.trace = trace
        self.bg = bg
        if wavelengths is None:
            wavelengths = np.arange(450, 450 + X.shape[-1] * 15, 10)
        self.wavelengths = wavelengths
        self.time_slice = time_slice
        self.wavelength_slice = wavelength_slice

    def add_second_axis(self, ax, loc='top', type='x_time'):
        """
        Parameters
        Add the second axis to the plot
        ----------
        ax : axes
        loc : str
        type : str

        Returns
        -------
        None
        """
        if type == 'x_time':
            ax.secondary_xaxis(loc, functions=(lambda x: np.searchsorted(self.time, [x])[0], lambda x: x))
        if type == 'y_wave':
            if self.wavelengths is not None:
                ax.secondary_yaxis(loc, functions=(lambda x: x, lambda x: x))
        if type == 'x_wave':
            if self.wavelengths is not None:
                ax.secondary_xaxis(loc, functions=(lambda x: x, lambda x: x))

    @classmethod
    def from_trace(cls, trace, tres, wavelength_slice=slice(None, None),
                   time_slice=slice(None, None), wavelengths=None):
        """
        Create class instance from Pyro trace

        Parameters
        ----------
        trace : trace
        tres : TRES instance

        Returns
        -------
        AnnotatedTres instance
        """
        def extract(trace, key):
            if key in trace:
                return trace[key].cpu().detach().numpy().mean(axis=0)
            else:
                return None

#         spectra = extract(trace, 'S') * extract(trace, 'A')[:, np.newaxis]
        spectra = extract(trace, 'S')
        lifetimes = extract(trace, 'tau')
        scattering_spectra = extract(trace, 'S_sc')
        # if scattering_spectra is not None:
        #     scattering_spectra *= extract(trace, 'A_sc')[:, np.newaxis]
        scattering_time = extract(trace, 'tau_sc')
        t_start = extract(trace, 't0')
        X_sim = extract(trace, 'I')
        if len(X_sim.shape) > 2:
            X_sim = X_sim[0]
        bg = extract(trace, 'xi')
        X = tres.X
        time = tres.time
        irf = tres.irf
        return cls(X, time, spectra, lifetimes, scattering_spectra, scattering_time, t_start, X_sim, trace, bg, irf,
                   wavelength_slice=wavelength_slice, time_slice=time_slice, wavelengths=wavelengths)

    @classmethod
    def from_scipy(cls, tres, trace, fun, args):
        """
        DEPRECATED
        """
        t0, scattering, _ = args
        T, S, TS = fun(trace.x, *args[:-1], return_S=True)
        if scattering:
            scattering_time = 0.005
            scattering_spectra = S[-1]
            S = S[:-1]
        else:
            scattering_time = None
            scattering_spectra = None
        bg = S[-1]
        S = S[:-1]
        spectra = S
        lifetimes = trace.x
        t_start = t0
        X_sim = TS
        X = tres.X
        time = tres.time
        irf = tres.irf
        trace = None
        return cls(X, time, spectra, lifetimes, scattering_spectra, scattering_time, t_start, X_sim, trace, bg, irf)

    def plot_matrix(self, matrix=None, ax=None, title=None, colorbar=True, add_noise=False, vmin=None, vmax=None,
                    log1p=True):
        """
        Plots a given matrix
        Parameters
        ----------
        matrix : np.array
        ax : axes object
        title : str
        colorbar : bool
        add_noise : bool
            Add Poisson noise?
        vmin : float
            Clip min values
        vmax : float
            Clip max values
        log1p : bool
            Do log1p transform?
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        if matrix is None:
            matrix = self.X_sim

        ax.set_title(title)
        ax.set_xlabel('Время, нс')
        ax.set_ylabel('Длина волны, нм')

        ax.xaxis.set_major_formatter(IndexFormatter(self.time))
        ax.yaxis.set_major_formatter(IndexFormatter(self.wavelengths))
        ax.yaxis.set_major_locator(MaxNLocator(8))

        if add_noise:
            matrix = np.random.poisson(matrix)
        if log1p:
            matrix = np.log1p(matrix)
        mappable = ax.imshow(matrix.T, aspect='auto', vmin=vmin, vmax=vmax)

        if colorbar:
            plt.gcf().colorbar(mappable, ax=ax, pad=0.065)

        self.add_second_axis(ax)
        self.add_second_axis(ax, loc='right', type='y_wave')
        self.add_second_axis(ax, loc='top', type='x_wave')

    def plot_matrix_comparison(self, axs=None):
        """
        Plot original and simulated TRES matrices side by sibe
        Parameters
        ----------
        axs : list
            List of 2 matplotlib axes
        """
        if axs is None:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 12))
        else:
            ax1, ax2 = axs

        vmin = min(self.X.min(), self.X_sim.min())
        vmax = min(self.X.max(), self.X_sim.max())

        vmin = np.log1p(vmin)
        vmax = np.log1p(vmax)

        self.plot_matrix(self.X, ax1, 'Экстеримент', vmin=vmin, vmax=vmax)
        self.plot_matrix(self.X_sim, ax2, 'Расчет', vmin=vmin, vmax=vmax)

        plt.tight_layout()

    def plot_residuals_2d(self, ax=None, vmax=None):
        """
        Plot TRES residuals matrix
        Parameters
        ----------
        ax : matplotlib axes
        vmax : float
            Value to clip max
        """
        if ax is None:
            fig, ax = plt.subplots()

        residuals = self.get_residuals(weighted=True, squared=False)
        residuals_slice = residuals[self.time_slice, self.wavelength_slice]
        if vmax is None:
            residuals = np.clip(residuals, np.quantile(residuals_slice, 0.09),
                                np.quantile(residuals_slice, 1 - 0.09))
        print(residuals.shape)
        self.plot_matrix(residuals, ax=ax, title=None, log1p=False, vmax=vmax)
        # TODO FIX
#         xlim = (self.time_slice.start, self.time_slice.stop)
#         ylim = (self.wavelength_slice.stop, self.wavelength_slice.start)
#         ax.set_ylim(*ylim)
#         ax.set_xlim(*xlim)

    def plot_residuals_1d(self, ax=None):
        """
        Plot TRES residuals summed over wavelenghts
        Parameters
        ----------
        ax : matplotlib axes
        vmax : float
            Value to clip max
        """
        if ax is None:
            fig, ax = plt.subplots()

        residuals = self.get_residuals(weighted=True, squared=False).sum(axis=1)
        ax.plot(self.time, residuals)
        self.add_second_axis(ax)
        xlim = (self.time[self.time_slice.start], self.time[self.time_slice.stop])
        print(xlim)
        ylim = (residuals[self.time_slice].min(), residuals[self.time_slice].max())
        # ax.set_ylim(*ylim)
        # ax.set_xlim(*xlim)

    def plot_spectra(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        for t, s in zip(self.lifetimes, self.spectra):
            ax.plot(s, label=f'{t :.3f}')

        if self.scattering_spectra is not None:
            ax.plot(self.scattering_spectra.T, label='sc')

        if self.bg is not None:
            ax.plot(self.bg.T, label='bg')

        ax.xaxis.set_major_formatter(IndexFormatter(self.wavelengths))
        self.add_second_axis(ax, loc='top', type='x_wave')

        ax.legend()

    def plot_amplitude(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        def norm(a):
            return a / a.max()

        for t, s in zip(self.lifetimes, self.spectra):
            ax.plot(norm(s[self.wavelength_slice]), label=f'{s[self.wavelength_slice].max() :.1E}')

        # if self.scattering_spectra is not None:
        #     ax.plot(norm(self.scattering_spectra), label=f'{self.scattering_spectra.max() :.1E}')
        #
        # if self.bg is not None:
        #     ax.plot(norm(self.bg), label=f'{self.bg.max() :.1E}')

        ax.xaxis.set_major_formatter(IndexFormatter(self.wavelengths[self.wavelength_slice]))
        self.add_second_axis(ax, loc='top', type='x_wave')

        ax.legend()

    def plot_fitting_quality(self, axs=None, lim=None):
        """
        Sum TRES matrix over wavelengths and plot zoom-in to the interval with highest intensity. Is useful for
        visual quality inspection.
        Parameters
        ----------
        axs : tuple
            Tuple with 2 axes' objects
        lim : tuple or None
            Tuple with 2 values - X axis limits
        """
        if axs is None:
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10, 5))
        else:
            ax1, ax2 = axs

        poisson_samples = stat.poisson(self.X_sim).rvs((50, *self.X_sim.shape))

        for i in poisson_samples:
            ax1.plot(self.time, i.sum(axis=-1), alpha=0.03, c='C0')
        ax1.plot(self.time, self.X.sum(axis=1), c='C1', linewidth=0.9)

        for i in poisson_samples:
            ax2.plot(self.time, i.sum(axis=-1), alpha=0.03, c='C0')
        ax2.plot(self.time, self.X.sum(axis=1), c='C1', linewidth=0.9)
        x_argmax = self.time[np.argmax(self.X.sum(axis=1))]
        if lim is None:
            ax2.set_xlim(x_argmax - 0.4, x_argmax + 1.0)
        else:
            ax2.set_xlim(*lim)

        self.add_second_axis(ax1)
        self.add_second_axis(ax2)
        return plt.gcf()

    def plot_summary(self, figsize=(14, 6)):
        """
        Plot summary about the analyzis
        Parameters
        ----------
        figsize : tuple
            Tuple with figsize in inches

        """
        print(self.reduced_chi_squared(self.time_slice, self.wavelength_slice))
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows=2, ncols=4, width_ratios=[1, 1, 1.35, 1.35])
        axs = []
        for i in range(4):
            axs.append(fig.add_subplot(gs[0, i]))

        for i in range(4):
            axs.append(fig.add_subplot(gs[1, i]))

        self.plot_fitting_quality((axs[0], axs[1]))
        self.plot_matrix_comparison((axs[2], axs[3]))
        self.plot_spectra(axs[4])
        self.plot_amplitude(axs[5])
        self.plot_residuals_2d(axs[6])
        self.plot_residuals_1d(axs[7])
        plt.tight_layout()

    def plot_repr(self, figsize=(10, 5)):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows=1, ncols=2, width_ratios=[1, 1])
        axs = []
        for i in range(2):
            axs.append(fig.add_subplot(gs[:, i]))

        self.plot_matrix(ax=axs[0])
        self.plot_spectra(axs[1])

    def __repr__(self):
        self.plot_repr()
        plt.show()
        return ''

    def degrees_of_freedom(self, time_slice=slice(None, None), wavelength_slice=slice(None, None)):
        """
        Returns degrees of freedom for NNLS model.
        Parameters
        ----------
        time_slice : slice
        wavelength_slice : slice

        Returns
        -------
        int
        """
        k = len(self.lifetimes)
        j, l_ = self.X[time_slice, wavelength_slice].shape
        return j * l_ - 2 * k - k * l_ - 2 * l_

    def get_residuals(self, weighted=False, squared=True):
        """
        Returns residuals matrix R = TRES - TRES_simulated
        Parameters
        ----------
        weighted : bool
            Perform Poisson weightening

        squared :
            Square the residuals
        """
        data = self.X
        simulated = self.X_sim
        residuals = (data - simulated)
        if squared:
            residuals = residuals ** 2
        if weighted:
            residuals = residuals / np.maximum(simulated, 0.00001)
        return residuals

    def reduced_chi_squared(self, time_slice=slice(None, None), wavelength_slice=slice(None, None)):
        """
        Compute reduced chi square for the solutio
        Parameters
        ----------
        time_slice : slice
        wavelength_slice : slice

        Returns
        -------
        float
        """
        dof = self.degrees_of_freedom(time_slice, wavelength_slice)
        residuals = self.get_residuals(weighted=True)[time_slice, wavelength_slice]
        chi_squared = residuals.sum() / dof
        confidence_interval = np.sqrt(8 / dof)

        fitting_successful = 1 - confidence_interval < chi_squared < 1 + confidence_interval

        return chi_squared, confidence_interval, fitting_successful


