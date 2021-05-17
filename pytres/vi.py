# import sys
# sys.path.append('pytres/')


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time as time_module
import os
from tqdm.auto import tqdm

from pytres.pytres import tres
from pyro import poutine

from pyro.infer import MCMC, NUTS, Predictive
import torch
from torch.nn import functional as F
import torch.nn
import pyro
from pyro import distributions as dist
import torch
import torch.distributions.constraints as constraints
import torch.nn.functional as F
import pyro
from pyro.optim import ClippedAdam
from pyro.infer import SVI, Trace_ELBO, SVGD, RBFSteinKernel, JitTrace_ELBO
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDelta, AutoNormal, AutoMultivariateNormal, AutoGuideList
from pyro.infer import autoguide
import pyro.distributions as dist
from fft_conv import fft_conv
from torch.fft import rfft, irfft
import shutil
from interp1d import Interp1d

import scipy.stats as stat

pyro.enable_validation(True)

import warnings
warnings.filterwarnings('ignore')


def trace_mle(cut_time=slice(None, None), sites_to_map=['gp'], obs='I_obs'):
    def loss_fn(model, guide, *args, **kwargs):
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = poutine.trace(
            poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

        model_trace.compute_log_prob(lambda site, node: site in sites_to_map)

        loss = 0.0
        for site, node in model_trace.nodes.items():
            #             print(site, site in sites_to_map, sites_to_map)
            if site in sites_to_map:
                loss -= node['log_prob_sum'] * 50
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
             ensure_convergence=False):
    adam_params = {"lr": lr, "betas": (0.90, 0.999), 'weight_decay': 0.005, 'clip_norm': 10}
    optimizer = ClippedAdam(adam_params)

    #     svi = SVI(model, guide, optimizer, loss=trace_mle(cut_time))
    svi = SVI(model, guide, optimizer, loss=JitTrace_ELBO())

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

    if mode == 'fft':
        conved = fft_conv(signal.unsqueeze(1), kernel.flip(0).unsqueeze(0).unsqueeze(0), padding=kernel_size - 1)[:, 0]

    elif mode == 'direct':
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


def model_full(data=None, time=None, t0=None,
               n_components=3, scattering=False, fix_time=False,
               n_points=None, time_padded=None, irf_padded=None):
    components_plate = pyro.plate('components', n_components)

    if fix_time:
        t0 = pyro.deterministic('t0', t0)
    else:
        t0 = pyro.sample('t0', dist.Normal(loc=t0, scale=0.01))
    xi = pyro.sample('xi', dist.Exponential(torch.rand(n_points) / 10).to_event(1))

    irf = Interp1d()(time_padded, irf_padded, time - t0, None)

    T_scaled = time_matrix(time, irf, t0, components_plate, suffix='')
    S = spectral_matrix_gp(n_points, components_plate, suffix='')
    ST = pyro.deterministic('ST', S.T @ T_scaled, event_dim=2)

    if scattering:
        scattering_plate = pyro.plate('scattering', 1)
        T_sc = time_matrix(time, irf, t0, scattering_plate, suffix='sc', scattering=True)
        S_sc = spectral_matrix_unscaled(n_points, scattering_plate, suffix='sc') * 20
        ST_sc = pyro.deterministic('ST_sc', S_sc.T @ T_sc, event_dim=2)
        ST = ST + ST_sc

    #     prob = pyro.sample('prob', dist.Beta(1, 1))
    if data is not None:
        data = data
    #     I = pyro.sample('lol', dist.NegativeBinomial(ST.T[:] + xi, probs=prob).to_event(2), obs=data)
    I = pyro.sample('I_obs', dist.Poisson(ST.T + xi).to_event(2), obs=data)
    _ = pyro.deterministic('I', ST.T + xi, event_dim=2)


def AutoMixed(modell_full, init_loc={}, delta=None):
    guide = AutoGuideList(model_full)

    marginalised_guide_block = poutine.block(model_full, expose_all=True, hide_all=False, hide=['tau'])
    if delta is None:
        guide.append(
            AutoNormal(marginalised_guide_block, init_loc_fn=autoguide.init_to_value(values=init_loc), init_scale=0.05))
    elif delta is 'part' or delta is 'all':
        guide.append(AutoDelta(marginalised_guide_block, init_loc_fn=autoguide.init_to_value(values=init_loc)))

    full_rank_guide_block = poutine.block(model_full, hide_all=True, expose=['tau'])
    if delta is None or delta is 'part':
        guide.append(AutoMultivariateNormal(full_rank_guide_block, init_loc_fn=autoguide.init_to_value(values=init_loc),
                                            init_scale=0.05))
    else:
        guide.append(AutoDelta(full_rank_guide_block, init_loc_fn=autoguide.init_to_value(values=init_loc)))

    return guide


class TresSolverVI():
    ...