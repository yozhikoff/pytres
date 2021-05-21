import time as time_module
import warnings

import numpy as np
import pyro
import pyro.distributions as dist
import scipy.stats as stat
import torch
import torch.nn
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import Predictive
from pyro.infer import SVI, JitTraceEnum_ELBO
from pyro.infer import autoguide
from pyro.infer.autoguide import AutoDelta, AutoNormal, AutoMultivariateNormal, AutoGuideList
from pyro.optim import ClippedAdam
from pytres import TRES, AnnotatedTres
from pytres.interp1d import Interp1d
from torch.fft import rfft, irfft
from tqdm.auto import tqdm

pyro.enable_validation(True)


def trace_mle(cut_time=slice(None, None), sites_to_map=('gp',), obs='I_obs', prior_weight=50):
    def loss_fn(model, guide, *args, **kwargs):
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = poutine.trace(
            poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

        model_trace.compute_log_prob(lambda site, node: site in sites_to_map)

        loss = 0.0
        for site, node in model_trace.nodes.items():
            if site in sites_to_map:
                loss -= node['log_prob_sum'] * prior_weight
            if site == obs:
                loss -= node['fn'].log_prob(node['value']).sum()

        return loss

    return loss_fn


def make_step(svi, pbar, time_start, args, kwargs, prefix=''):
    loss = svi.step(*args, **kwargs)
    if time_module.time() - time_start > 0.35:
        description = f"{prefix}Loss: {loss:.9E}" if loss > 1e5 else f"Loss: {round(loss, 3)}"
        pbar.set_description(description)
        time_start = time_module.time()
    return loss, time_start


def make_svi(model, guide, args=None, kwargs=None, steps=1000, lr=0.05, cut_time=slice(None, None), max_steps=2000,
             ensure_convergence=False, loss='ELBO'):
    adam_params = {"lr": lr, "betas": (0.90, 0.999), 'weight_decay': 0.005, 'clip_norm': 10}
    optimizer = ClippedAdam(adam_params)

    #     svi = SVI(model, guide, optimizer, loss=trace_mle(cut_time))
    if loss == 'ELBO':
        svi = SVI(model, guide, optimizer, loss=JitTraceEnum_ELBO())
    if loss == 'MLE':
        svi = SVI(model, guide, optimizer, loss=trace_mle())

    pbar = tqdm(range(1, steps + 1))
    time_start = 0

    loss_arr = []

    for i in pbar:
        loss, time_start = make_step(svi, pbar, time_start, args, kwargs)
        loss_arr.append(loss)

    while ensure_convergence:
        std_prev = np.std(loss_arr[-20:-1])
        mean_cur = np.mean(loss_arr[-100:])
        mean_prev = np.mean(loss_arr[-200:-100])
        prob = stat.norm(mean_prev, std_prev).cdf(mean_cur)
        #         print(prob, mean_cur, mean_prev, std_prev)
        if mean_cur < mean_prev and prob < 0.05 and len(loss_arr) < max_steps:
            pbar = tqdm(range(1, 100 + 1), leave=False)
            for j in pbar:
                loss, time_start = make_step(svi, pbar, time_start, args, kwargs, prefix='Extra: ')
                loss_arr.append(loss)
        else:
            break

    return loss


def conv1d(signal, kernel, mode='fft_circular', cut=False, cut_lim=150):
    """
    signal M x N
    kernel N
    """
    kernel_size = int(kernel.shape[-1])

    if mode == 'direct':
        conved = F.conv1d(signal.unsqueeze(1), kernel.flip(0).unsqueeze(0).unsqueeze(0), padding=kernel_size - 1)[:, 0]

    elif mode == 'fft_circular':
        conved = irfft(rfft(signal) * rfft(kernel), signal.shape[-1])

    if cut:
        conved = conved[:, cut_lim: kernel_size + cut_lim]

    return conved


def time_matrix(time, irf, t0, plate, suffix='', scattering=False):
    if suffix != '':
        suffix = f'_{suffix}'

    if scattering:
        lol = pyro.deterministic(f'T_sc', irf / irf.max())
        return lol

    with plate:
        if not scattering:
            tau = pyro.sample(f'tau{suffix}', dist.Gamma(5, 10))[:, np.newaxis]
        else:
            tau = pyro.sample(f'tau{suffix}', dist.Uniform(0.00001, 0.005))[:, np.newaxis]

    T_unbound = pyro.deterministic(f'T_unbound{suffix}', torch.exp(-time / tau))
    T = pyro.deterministic(f'T{suffix}', T_unbound)

    T_convolved = conv1d(T, irf, mode='fft_circular', cut=False)

    if not scattering:
        circular_multiplier = 1 / (1 - torch.exp(-time[-1] / tau))
        circular_multiplier = pyro.deterministic('circular_multiplier', circular_multiplier)
        T_convolved = T_convolved * circular_multiplier
    T_convolved = pyro.deterministic(f'T_convolved{suffix}', T_convolved)

    T_scaled = T_convolved / (T_convolved.max(dim=1)[0][:, np.newaxis])
    T_scaled = pyro.deterministic(f'T_scaled{suffix}', T_scaled)

    return T_scaled


def spectral_matrix(n_points, plate, suffix=''):
    if suffix != '':
        suffix = f'_{suffix}'

    with plate:
        S_unscaled = pyro.sample(f'S_unscaled{suffix}', dist.Exponential(torch.rand(n_points)).to_event(1))
        S = pyro.deterministic(f'S{suffix}', S_unscaled / S_unscaled.max(dim=1)[0][:, np.newaxis], event_dim=1)
    return S


def spectral_matrix_unscaled(n_points, plate, suffix=''):
    if suffix != '':
        suffix = f'_{suffix}'

    with plate:
        S = pyro.sample(f'S{suffix}', dist.Exponential(torch.rand(n_points)).to_event(1))
    return S * 20


def spectral_matrix_gp(n_points, plate, suffix=''):
    if suffix != '':
        suffix = f'_{suffix}'

    with plate:
        l = 150
        pts = torch.arange(n_points, dtype=torch.float64).unsqueeze(-1)
        distance_squared = torch.pow(pts - pts.T, 2).unsqueeze(-1)
        cov = pyro.deterministic('cov', torch.exp(-0.5 * distance_squared / l).T, event_dim=2).contiguous()
        diag_idx = np.diag_indices(n_points, ndim=1)
        cov[:, diag_idx, diag_idx] += torch.rand(1, 1, n_points) / 1000
        gp = pyro.sample('gp',
                         dist.MultivariateNormal(loc=torch.tensor([0.] * n_points), covariance_matrix=cov).to_event(0))
        S = pyro.deterministic('S', F.softplus(gp * 50) * 20, event_dim=1)

    return S


def AutoMixed(model_full, init_loc={}, delta=None):
    guide = AutoGuideList(model_full)

    marginalised_guide_block = poutine.block(model_full, expose_all=True, hide_all=False, hide=['tau'])
    if delta is None:
        guide.append(
            AutoNormal(marginalised_guide_block, init_loc_fn=autoguide.init_to_value(values=init_loc), init_scale=0.05))
    elif delta == 'part' or delta == 'all':
        guide.append(AutoDelta(marginalised_guide_block, init_loc_fn=autoguide.init_to_value(values=init_loc)))

    full_rank_guide_block = poutine.block(model_full, hide_all=True, expose=['tau'])
    if delta is None or delta == 'part':
        guide.append(AutoMultivariateNormal(full_rank_guide_block, init_loc_fn=autoguide.init_to_value(values=init_loc),
                                            init_scale=0.05))
    else:
        guide.append(AutoDelta(full_rank_guide_block, init_loc_fn=autoguide.init_to_value(values=init_loc)))

    return guide


def create_init_trace(tres_annot):
    values = {}
    values['t0'] = tres_annot.t_start
    values['S'] = tres_annot.spectra / 20 + 0.0001
    values['gp'] = tres_annot.spectra / 20 / 50 + 0.0001
    values['S_sc'] = tres_annot.scattering_spectra[None, ...] / 20 + 0.0001
    values['tau'] = tres_annot.lifetimes
    values['xi'] = tres_annot.bg

    return {k: torch.tensor(v) for k, v in values.items()}


class TresSolverVI(TRES):
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
                 t0=0.,
                 initial_trace=None):
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

        assert len(time) == len(irf)
        assert len(time) == len(X)

        self.time_slice = time_slice
        self.wavelength_slice = wavelength_slice
        self.scattering = scattering
        self.wavelengths = wavelengths
        self.n_components = n_components
        self.n_wavelenghs = X.shape[-1]

        self.t0 = t0

        if len(time) < padding_length * 2:
            padding_length = len(time) // 3
        self.time_padded = self.extend_time(time, padding_length).float()
        self.irf_padded = self.circular_pad(irf, padding_length).float()
        self.tres_annot = None

        self.initial_trace = initial_trace

    @staticmethod
    def circular_pad(t, n):
        return torch.tensor(np.concatenate([t[-n:], t, t[:n]]))

    @staticmethod
    def extend_time(t, n):
        diff = t[1] - t[0]
        start = np.arange(t[0] - n * diff, t[0], diff)
        end = np.arange(t[-1] + diff, t[-1] + (n + 1) * diff, diff)
        return torch.tensor(np.concatenate([start, t, end]))

    def model_full(self, data=None, time=None, fix_time=False):
        components_plate = pyro.plate('components', self.n_components)

        if fix_time:
            t0 = pyro.deterministic('t0', self.t0)
        else:
            t0 = pyro.sample('t0', dist.Normal(loc=self.t0, scale=0.01))
        xi = pyro.sample('xi', dist.Exponential(torch.rand(self.n_wavelenghs) / 10).to_event(1))

        irf = Interp1d()(self.time_padded, self.irf_padded, time - t0, None)

        T_scaled = time_matrix(time, irf, t0, components_plate, suffix='')
        S = spectral_matrix_gp(self.n_wavelenghs, components_plate, suffix='')
        ST = pyro.deterministic('ST', S.T @ T_scaled, event_dim=2)

        if self.scattering:
            scattering_plate = pyro.plate('scattering', 1)
            T_sc = time_matrix(time, irf, t0, scattering_plate, suffix='sc', scattering=True)
            S_sc = spectral_matrix_unscaled(self.n_wavelenghs, scattering_plate, suffix='sc') * 20
            ST_sc = pyro.deterministic('ST_sc', S_sc.T @ T_sc, event_dim=2)
            ST = ST + ST_sc

        if data is not None:
            data = data[self.time_slice][self.wavelength_slice]
        pyro.sample('I_obs', dist.Poisson((ST.T + xi)[self.time_slice][self.wavelength_slice]).to_event(2), obs=data)
        pyro.deterministic('I', ST.T + xi, event_dim=2)

    def solve(self):

        pyro.clear_param_store()
        torch.set_default_tensor_type(torch.DoubleTensor)

        if self.initial_trace is not None:
            guide = AutoMixed(self.model_full, init_loc=self.initial_trace)
        else:
            guide = AutoMixed(self.model_full, delta=None)
        device = 'cpu'

        data_tensor, time_tensor, irf_tensor = (torch.DoubleTensor(i).to(device) for i in [self.X, self.time, self.irf])
        args = []
        kwargs = {'data': data_tensor,
                  'time': time_tensor,
                  'fix_time': False}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            make_svi(self.model_full, guide, args, kwargs=kwargs, steps=1000, lr=0.05, ensure_convergence=True)
            make_svi(self.model_full, guide, args, kwargs=kwargs, steps=2500, lr=0.003, ensure_convergence=False)
            make_svi(self.model_full, guide, args, kwargs=kwargs, steps=3500, lr=0.0001, ensure_convergence=True)

        tres_data = TRES(X=kwargs['data'].cpu().numpy(), time=self.time, irf=self.irf)
        trace = Predictive(self.model_full, guide=guide, num_samples=5)(**{**kwargs, **{'data': None}})
        tres_annot = AnnotatedTres.from_trace(trace, tres_data, wavelengths=self.wavelengths,
                                              time_slice=self.time_slice,
                                              wavelength_slice=self.wavelength_slice)

        self.guide = guide

        return tres_annot

    def compute_confidence_indervals(self):

        pt = torch.stack(self.guide[1].quantiles([0.005, 0.995])['tau'], dim=0).detach().numpy()
        ci_max = pt.max(axis=0)
        ci_min = pt.min(axis=0)

        print('Lifetimes CI:')
        for min_val, max_val in zip(ci_min, ci_max):
            mean = (min_val + max_val) / 2
            print(f'{mean :.4f} +/- {(max_val - mean) :.4f}')

        return ci_min, ci_max
