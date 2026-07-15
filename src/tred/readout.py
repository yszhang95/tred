import torch
import numpy as np
# import logging

from tred.blocking import Block

def nd_readout(block, threshold, adc_hold_delay, adc_down_time, csa_reset_time=1, one_tick=1,
               offset_to_align=0, pixel_axes=(), taxis=-1,
               uncorr_noise=None, thres_noise=None, reset_noise=None, leftover=None, niter=10):
    '''
    locs :: (N, nxpl, nxpl, ..., vdim)
    X :: (N, npxl, npxl, ..., Nt)
    Is X already the summed? No
    Do it once or iteratively?
    Do it Once is faster but complicated
    Do it iteratively is easier but complicated...

    Let us do it once.

    one_tick :: how many points in time for one time tick. Useful for next-to action,
                for instance, threshold-crossing check after CSA reset and ADC down time,
                counting ADC HOLD DELAY after trigger crossing.
    '''
    X = block.data
    locations = block.location
    if threshold.ndim > 0:
        loc_inds = locations.view(-1,block.vdim)[:,:-1].T
        # threshold = threshold[list(loc_inds[i] for i in range(loc_inds.shape[0]))]
        threshold = threshold[loc_inds[0], loc_inds[1]]
    else:
        threshold = threshold.unsqueeze(0).expand(block.nbatches)
    for i in pixel_axes:
        locations = locations.unsqueeze(1)
        threshold = threshold.unsqueeze(1)
    threshold = threshold.unsqueeze(-1)
    olocs = []
    ocharges = []
    if taxis < 0:
        taxis = X.ndim + taxis
    if taxis != X.ndim-1:
        raise NotImplementedError()
    if leftover is not None:
        raise NotImplementedError()
        # FIXME: we need to concate the leftover with the input

    if csa_reset_time > adc_down_time:
        raise ValueError('csa_reset_time > adc_down_time')

    # FIXME: start should be initialized according to leftover
    start = torch.zeros((*tuple(X.shape[i] for i in [0,]+list(pixel_axes)), 1), dtype=torch.int64, device=locations.device)
    trange = torch.arange(X.shape[taxis], device=locations.device).view(*[1 for i in range(X.ndim-1)], -1)
    # info(f'start shape {start .shape}')
    # info(f'trange shape {trange.shape}')

    Nt = X.shape[taxis]
    # logging.debug(f'X shape {X.shape}')
    # info(f'X shape {X.shape}')
    # FIXME: what is an appropriate accumulation function?
    Xacc = X.cumsum(dim=taxis)
    # logging.debug(f'Xacc shape {Xacc.shape}')
    # info(f'Xacc shape {Xacc.shape}')
    if uncorr_noise is not None:
        Xacc += torch.normal(0, torch.full_like(Xacc, fill_value=uncorr_noise, device=Xacc.device))
    # FIXME: reset_noise should be used only if leftover is None
    if (reset_noise is not None) and (leftover is None):
        Xacc += torch.normal(0, torch.full_like(Xacc, fill_value=reset_noise, device=Xacc.device))

    pxl_indices = slice(None, -1, None) # FIXME: hard coded

    iteration = 0
    while True:
        # logging.debug(f'Iteration {iteration}')
        # info(f'Iteration {iteration}')

        if thres_noise:
            thres = threshold + torch.normal(0, torch.full_like(threshold, fill_value=thres_noise, device=threshold.device))
            thres_delay = threshold + torch.normal(0, torch.full_like(threshold, fill_value=thres_noise, device=threshold.device))
        else:
            thres = threshold
            thres_delay = threshold

        mvalid = trange >= start # shape (npxl, npxl, ..., Nt) if taxis = -1
        # logging.debug(f'mvalid shape {mvalid.shape}')
        # info(f'mvalid shape {mvalid.shape}')
        # Xacc = Xacc * mvalid # FIXME: start > trange; we need leftover information
        Xacc[~mvalid] = -1E9

        crossed = torch.zeros_like(Xacc, dtype=torch.int32, device=Xacc.device)
        crossed[...,offset_to_align::one_tick] = (Xacc[...,offset_to_align::one_tick] >= thres) & mvalid[...,offset_to_align::one_tick] # check after start # shape (N, nxpl, ..., Nt) if taxis = -1
        # FIXME:
        cross_t = torch.argmax(crossed.to(torch.int32), dim=taxis, keepdim=True) # shape (N, npxl, .., 1) if taxis = -1

        # logging.debug(f'cross_t shape {cross_t.shape}')
        crossed = torch.gather(Xacc, taxis, cross_t) >= thres # is it really cross at cross_t?
        # crossed shape: (N, npxl, ..., Nt) if taxis = -1
        # logging.debug(f'crossed shape {crossed.shape}')
        # hold_t = cross_t + adc_hold_delay - 1  # samed as cross_t shape, element at adc_hold_delay - 1 is from 0 to adc_hold_delay-1
        hold_t = cross_t + adc_hold_delay
        # logging.debug(f'hold_t shape {hold_t.shape}')
        hold_t_inrange = torch.clamp(hold_t, min=0, max=Nt-1) # same as hold_t shape
        # logging.debug(f'hold_t_inrange shape {hold_t_inrange.shape}')
        Xacc_hold_t = torch.gather(Xacc, taxis, hold_t_inrange) # shape (N, npxl, ..., 1) if taxis = -1
        # logging.debug(f'Xacc_hold_t shape {Xacc_hold_t.shape}')
        delay_crossed = Xacc_hold_t >= thres_delay # shape (N, npxl, ..., 1) if taxis = -1
        # logging.debug(f'delay_crossed shape {delay_crossed.shape}')
        triggered = crossed & delay_crossed & (hold_t < Nt) # shape (N, npxl, ..., 1) if taxis = -1
        # logging.debug(f'triggered shape {triggered.shape}')
        # if iteration % niter == 0 and not mvalid.any():
        if not triggered.any():
            # FIXME: We need deal with leftover on the CSA.
            # FIXME: the leftover should cover at least one
            # FIXME: As the input is current, we need to return current from accumulated charge
            break
        glocs = locations[triggered.squeeze(taxis)]
        pixels = glocs[:,pxl_indices] # 2D array (Ntriggered, vdim-1)
        # print(pixels)
        gtimes = glocs[:,-1] # FIXME
        times = gtimes + cross_t[triggered] # 1D with last dim the
        hold_times = gtimes + hold_t[triggered]
        hits = torch.gather(Xacc, taxis, hold_t_inrange)[triggered] # 1D array
        # start = hold_t + adc_down_time + 1
        start[triggered] = hold_t[triggered] + adc_down_time + one_tick # on discriminator, controlled by adc down time
        start_times = gtimes + start[triggered]
        oloc = torch.cat([pixels, times.unsqueeze(1), hold_times.unsqueeze(1), start_times.unsqueeze(1)], dim=1)
        olocs.append(oloc)
        ocharges.append(hits)
        # if thres_noise is None:
        #     assert torch.all(hits > thres[triggered]).item()
        start[~triggered] = hold_t[~triggered] + one_tick
        start[~crossed] = Nt # crossed not triggered should be at hold_t+1; never crossed needs to be at start.
        # at triggered positions, charges are reset and there is one timestamp missing;
        # everything happens on CSA
        # hold t may be at the last t;
        # FIXME: what happens if the hold_t is the last element?
        Xacc_next_to_hold_t = torch.gather(Xacc, taxis, torch.clamp(hold_t+csa_reset_time, min=0, max=Nt-1))
        # only update the triggered positions
        Xacc[triggered.squeeze(taxis)] -= Xacc_next_to_hold_t[triggered.squeeze(taxis)]
        if reset_noise is not None:
            # FIXME: taxis is assumed to be -1
            Xacc_baseline = torch.normal(0, torch.full(Xacc.shape[:-1], fill_value=reset_noise, device=Xacc.device))
            # print('shape', Xacc_baseline[triggered.squeeze(taxis)].unsqueeze(-1))
            Xacc[triggered.squeeze(taxis)] += Xacc_baseline[triggered.squeeze(taxis)].unsqueeze(-1)

        iteration += 1
    if len(olocs) == 0:
        return torch.zeros((0, len(pixel_axes)+3), dtype=torch.int32, device=locations.device), \
            torch.zeros((0,), dtype=torch.float32, device=X.device)
        raise NotImplementedError("Not sure how to handle empty hit collection")
    return torch.cat(olocs, dim=0), torch.cat(ocharges, dim=0)


def nd_readout_rst(block, threshold, adc_hold_delay, adc_down_time, csa_reset_time=1, one_tick=1,
               offset_to_align=0, pixel_axes=(), taxis=-1,
               uncorr_noise=None, thres_noise=None, reset_noise=None, leftover=None, niter=10):
    '''
    locs :: (N, nxpl, nxpl, ..., vdim)
    X :: (N, npxl, npxl, ..., Nt)
    Is X already the summed? No
    Do it once or iteratively?
    Do it Once is faster but complicated
    Do it iteratively is easier but complicated...

    Let us do it once.

    one_tick :: how many points in time for one time tick. Useful for next-to action,
                for instance, threshold-crossing check after CSA reset and ADC down time,
                counting ADC HOLD DELAY after trigger crossing.
    
    RESET-MODEL VARIANT of nd_readout (copy; original untouched): reset_noise
    is a per-channel constant initial baseline offset + the existing
    per-trigger-reset constant offsets.  Use with external OU output noise
    (uncorr_noise=None here).
    '''
    X = block.data
    locations = block.location
    if threshold.ndim > 0:
        loc_inds = locations.view(-1,block.vdim)[:,:-1].T
        # threshold = threshold[list(loc_inds[i] for i in range(loc_inds.shape[0]))]
        threshold = threshold[loc_inds[0], loc_inds[1]]
    else:
        threshold = threshold.unsqueeze(0).expand(block.nbatches)
    for i in pixel_axes:
        locations = locations.unsqueeze(1)
        threshold = threshold.unsqueeze(1)
    threshold = threshold.unsqueeze(-1)
    olocs = []
    ocharges = []
    if taxis < 0:
        taxis = X.ndim + taxis
    if taxis != X.ndim-1:
        raise NotImplementedError()
    if leftover is not None:
        raise NotImplementedError()
        # FIXME: we need to concate the leftover with the input

    if csa_reset_time > adc_down_time:
        raise ValueError('csa_reset_time > adc_down_time')

    # FIXME: start should be initialized according to leftover
    start = torch.zeros((*tuple(X.shape[i] for i in [0,]+list(pixel_axes)), 1), dtype=torch.int64, device=locations.device)
    trange = torch.arange(X.shape[taxis], device=locations.device).view(*[1 for i in range(X.ndim-1)], -1)
    # info(f'start shape {start .shape}')
    # info(f'trange shape {trange.shape}')

    Nt = X.shape[taxis]
    # logging.debug(f'X shape {X.shape}')
    # info(f'X shape {X.shape}')
    # FIXME: what is an appropriate accumulation function?
    Xacc = X.cumsum(dim=taxis)
    # logging.debug(f'Xacc shape {Xacc.shape}')
    # info(f'Xacc shape {Xacc.shape}')
    if uncorr_noise is not None:
        Xacc += torch.normal(0, torch.full_like(Xacc, fill_value=uncorr_noise, device=Xacc.device))
    # RESET MODEL (KNOWLEDGE 6.30): the initial baseline carries the residual
    # of the LAST reset as a per-channel CONSTANT offset (kTC/charge
    # injection), NOT per-tick white noise — the original per-tick
    # full_like(Xacc) broadcast was the mis-homing bug.  Per-trigger-reset
    # offsets further down (Xacc_baseline block) were already correct.
    if (reset_noise is not None) and (leftover is None):
        Xacc += torch.normal(0, torch.full(Xacc.shape[:-1], fill_value=float(reset_noise),
                                           device=Xacc.device)).unsqueeze(-1)

    pxl_indices = slice(None, -1, None) # FIXME: hard coded

    iteration = 0
    while True:
        # logging.debug(f'Iteration {iteration}')
        # info(f'Iteration {iteration}')

        if thres_noise:
            thres = threshold + torch.normal(0, torch.full_like(threshold, fill_value=thres_noise, device=threshold.device))
            thres_delay = threshold + torch.normal(0, torch.full_like(threshold, fill_value=thres_noise, device=threshold.device))
        else:
            thres = threshold
            thres_delay = threshold

        mvalid = trange >= start # shape (npxl, npxl, ..., Nt) if taxis = -1
        # logging.debug(f'mvalid shape {mvalid.shape}')
        # info(f'mvalid shape {mvalid.shape}')
        # Xacc = Xacc * mvalid # FIXME: start > trange; we need leftover information
        Xacc[~mvalid] = -1E9

        crossed = torch.zeros_like(Xacc, dtype=torch.int32, device=Xacc.device)
        crossed[...,offset_to_align::one_tick] = (Xacc[...,offset_to_align::one_tick] >= thres) & mvalid[...,offset_to_align::one_tick] # check after start # shape (N, nxpl, ..., Nt) if taxis = -1
        # FIXME:
        cross_t = torch.argmax(crossed.to(torch.int32), dim=taxis, keepdim=True) # shape (N, npxl, .., 1) if taxis = -1

        # logging.debug(f'cross_t shape {cross_t.shape}')
        crossed = torch.gather(Xacc, taxis, cross_t) >= thres # is it really cross at cross_t?
        # crossed shape: (N, npxl, ..., Nt) if taxis = -1
        # logging.debug(f'crossed shape {crossed.shape}')
        # hold_t = cross_t + adc_hold_delay - 1  # samed as cross_t shape, element at adc_hold_delay - 1 is from 0 to adc_hold_delay-1
        hold_t = cross_t + adc_hold_delay
        # logging.debug(f'hold_t shape {hold_t.shape}')
        hold_t_inrange = torch.clamp(hold_t, min=0, max=Nt-1) # same as hold_t shape
        # logging.debug(f'hold_t_inrange shape {hold_t_inrange.shape}')
        Xacc_hold_t = torch.gather(Xacc, taxis, hold_t_inrange) # shape (N, npxl, ..., 1) if taxis = -1
        # logging.debug(f'Xacc_hold_t shape {Xacc_hold_t.shape}')
        delay_crossed = Xacc_hold_t >= thres_delay # shape (N, npxl, ..., 1) if taxis = -1
        # logging.debug(f'delay_crossed shape {delay_crossed.shape}')
        triggered = crossed & delay_crossed & (hold_t < Nt) # shape (N, npxl, ..., 1) if taxis = -1
        # logging.debug(f'triggered shape {triggered.shape}')
        # if iteration % niter == 0 and not mvalid.any():
        if not triggered.any():
            # FIXME: We need deal with leftover on the CSA.
            # FIXME: the leftover should cover at least one
            # FIXME: As the input is current, we need to return current from accumulated charge
            break
        glocs = locations[triggered.squeeze(taxis)]
        pixels = glocs[:,pxl_indices] # 2D array (Ntriggered, vdim-1)
        # print(pixels)
        gtimes = glocs[:,-1] # FIXME
        times = gtimes + cross_t[triggered] # 1D with last dim the
        hold_times = gtimes + hold_t[triggered]
        hits = torch.gather(Xacc, taxis, hold_t_inrange)[triggered] # 1D array
        # start = hold_t + adc_down_time + 1
        start[triggered] = hold_t[triggered] + adc_down_time + one_tick # on discriminator, controlled by adc down time
        start_times = gtimes + start[triggered]
        oloc = torch.cat([pixels, times.unsqueeze(1), hold_times.unsqueeze(1), start_times.unsqueeze(1)], dim=1)
        olocs.append(oloc)
        ocharges.append(hits)
        # if thres_noise is None:
        #     assert torch.all(hits > thres[triggered]).item()
        start[~triggered] = hold_t[~triggered] + one_tick
        start[~crossed] = Nt # crossed not triggered should be at hold_t+1; never crossed needs to be at start.
        # at triggered positions, charges are reset and there is one timestamp missing;
        # everything happens on CSA
        # hold t may be at the last t;
        # FIXME: what happens if the hold_t is the last element?
        Xacc_next_to_hold_t = torch.gather(Xacc, taxis, torch.clamp(hold_t+csa_reset_time, min=0, max=Nt-1))
        # only update the triggered positions
        Xacc[triggered.squeeze(taxis)] -= Xacc_next_to_hold_t[triggered.squeeze(taxis)]
        if reset_noise is not None:
            # FIXME: taxis is assumed to be -1
            Xacc_baseline = torch.normal(0, torch.full(Xacc.shape[:-1], fill_value=reset_noise, device=Xacc.device))
            # print('shape', Xacc_baseline[triggered.squeeze(taxis)].unsqueeze(-1))
            Xacc[triggered.squeeze(taxis)] += Xacc_baseline[triggered.squeeze(taxis)].unsqueeze(-1)

        iteration += 1
    if len(olocs) == 0:
        return torch.zeros((0, len(pixel_axes)+3), dtype=torch.int32, device=locations.device), \
            torch.zeros((0,), dtype=torch.float32, device=X.device)
        raise NotImplementedError("Not sure how to handle empty hit collection")
    return torch.cat(olocs, dim=0), torch.cat(ocharges, dim=0)




def nd_readout_prc(block, threshold, adc_hold_delay, adc_down_time, csa_reset_time=1, one_tick=1,
                   offset_to_align=0, pixel_axes=(), taxis=-1,
                   uncorr_noise=None, thres_noise=None, reset_noise=None, leftover=None, niter=10,
                   prc_ticks=1024, prc_sync=False, prc_slot_ticks=16, prc_block_pix=7,
                   prc_perm_seed=20260713):
    '''
    nd_readout + LArPix rolling periodic reset (2x2 Run 1: 1024 x 100 ns =
    102.4 us per channel; DUNE-doc-32080).  Implemented larnd-sim style
    (fee.py): per-channel INDEPENDENT random phase, unconditional reset that
    wipes the accumulated sub-threshold charge; the ~100 ns dead slice maps
    to one fine sample.  Realized as a pure pre-transform on the current
    block — at each reset sample the charge accumulated since the previous
    reset is subtracted — then the untouched nd_readout runs on the result.
    Known approximation: a periodic reset composes with an EARLIER trigger
    reset in the same waveform as an undershoot (channel needs extra charge
    to retrigger afterwards); second-order for 2x2 occupancy.
    prc_ticks is in readout ticks (0.1 us); period in fine samples is
    prc_ticks * one_tick.
    '''
    X = block.data
    P = int(prc_ticks * one_tick)
    phase = None
    if prc_sync:
        phase = sync_prc_phase(block.location, P, one_tick, X.shape[:-1],
                               slot_ticks=prc_slot_ticks, block_pix=prc_block_pix,
                               perm_seed=prc_perm_seed, device=X.device)
    Xp = apply_periodic_reset(X, P, phase=phase)
    from tred.blocking import Block as _Block
    newblock = _Block(location=block.location, data=Xp)
    return nd_readout(newblock, threshold, adc_hold_delay, adc_down_time, csa_reset_time,
                      one_tick, offset_to_align, pixel_axes, taxis,
                      uncorr_noise, thres_noise, reset_noise, leftover, niter)


def sync_prc_phase(location, P, one_tick, phase_shape, slot_ticks=16, block_pix=7,
                   perm_seed=20260713, device=None):
    '''Synchronized rolling PRC phases (user-specified model): every chip is a
    block_pix x block_pix pad block; the in-block pad index p = (iy%7)*7+iz%7
    maps to a reset slot through ONE random-but-frozen permutation of the 64
    slots (the real shift-register->pad routing order is unknown; perm_seed
    freezes the guess).  All chips share the schedule: a single global phase
    per readout call (uniform in [0, P), i.e. free-running clock vs event),
    channel at slot s resets at global_phase + s*slot_ticks*one_tick + k*P in
    GLOBAL event time; each pixel waveform is anchored at its own start
    location[:, -1], so the local phase is the difference mod P.  Per-channel
    marginals are identical to independent phases; only the within-event
    joint structure changes.'''
    gen = torch.Generator(device='cpu').manual_seed(perm_seed)
    perm = torch.randperm(64, generator=gen)
    iy = location[:, 0].to(torch.int64)
    iz = location[:, 1].to(torch.int64)
    tstart = location[:, -1].to(torch.int64)
    pad = (iy % block_pix) * block_pix + iz % block_pix   # 0..48
    slot = perm.to(location.device)[pad]
    gphase = torch.randint(0, P, (1,), device=location.device)
    local = (gphase + slot * slot_ticks * one_tick - tstart) % P
    return local.view(phase_shape).to(device if device is not None else location.device)


def apply_periodic_reset(X, P, phase=None, offset_sigma=None):
    '''Subtract, at every reset sample r = phase + k*P (per channel), the
    charge accumulated since the previous reset, so that cumsum(X')(t) =
    cumsum(X)(t) - cumsum(X)(r_last(t)).  phase: optional tensor of shape
    X.shape[:-1] (defaults to per-channel uniform random in [0, P)).'''
    Nt = X.shape[-1]
    if P <= 0 or P >= Nt:
        return X
    Xacc = X.cumsum(dim=-1)
    if phase is None:
        phase = torch.randint(0, P, X.shape[:-1], device=X.device)
    prev = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
    Xp = X.clone()
    for k in range(Nt // P + 2):
        r = phase + k * P
        valid = r < Nt
        rv = r.clamp(max=Nt - 1)
        cur = torch.gather(Xacc, -1, rv.unsqueeze(-1)).squeeze(-1)
        delta = torch.where(valid, cur - prev, torch.zeros_like(prev))
        if offset_sigma:
            # each periodic reset leaves a fresh constant baseline offset
            # (kTC / charge injection), persisting until the next reset
            delta = delta - torch.where(valid, torch.normal(
                0, torch.full_like(prev, float(offset_sigma))), torch.zeros_like(prev))
        Xp.scatter_add_(-1, rv.unsqueeze(-1), (-delta).unsqueeze(-1))
        prev = torch.where(valid, cur, prev)
    return Xp


def make_ou_noise(shape, Nt, sigma, tau_samples, device, dtype=torch.float32):
    '''Stationary AR(1)/Ornstein-Uhlenbeck noise: per-sample RMS == sigma,
    autocorrelation alpha^k with alpha = exp(-1/tau_samples).  Generated by
    convolving white noise with the exponential kernel sqrt(1-alpha^2)*alpha^j
    (truncated at 1e-4 relative amplitude).'''
    alpha = float(np.exp(-1.0 / max(tau_samples, 1e-6))) if tau_samples > 0 else 0.0
    if alpha <= 0.0:
        return torch.randn(*shape, Nt, device=device, dtype=dtype) * sigma
    L = min(Nt, int(np.ceil(np.log(1e-4) / np.log(alpha))) + 1)
    j = torch.arange(L, device=device, dtype=dtype)
    kern = (alpha ** j) * np.sqrt(1.0 - alpha * alpha)
    B = int(np.prod(shape)) if shape else 1
    eps = torch.randn(B, 1, Nt + L - 1, device=device, dtype=dtype)
    n = torch.nn.functional.conv1d(eps, kern.flip(0).view(1, 1, L))
    return (n.view(*shape, Nt) * sigma)


def nd_readout_ou(block, threshold, adc_hold_delay, adc_down_time, csa_reset_time=1, one_tick=1,
                  offset_to_align=0, pixel_axes=(), taxis=-1,
                  thres_noise=None, leftover=None, niter=10,
                  ou_sigma=1.03, ou_tau_ticks=2.0,
                  prc_ticks=None, prc_sync=False, prc_slot_ticks=16, prc_block_pix=7,
                  prc_perm_seed=20260713):
    '''
    nd_readout with the white per-tick OUTPUT noise replaced by correlated
    OU noise (same per-sample RMS ou_sigma, correlation time ou_tau_ticks in
    0.1-us readout ticks), optionally composed with the rolling periodic
    reset.  Motivation (KNOWLEDGE 6.25-6.27): white noise gives the
    discriminator ~10 independent crossing chances per us -> effective
    threshold below nominal; halving the white RMS fixed the low-T soft
    excess but over-corrected resolution-driven shapes (MIP peak width,
    totN==1).  OU keeps the true RMS and thins only the chance density.
    Realized as a pure pre-transform: adding n(t) to Xacc == adding diff(n)
    to the current; nd_readout is then called with its internal
    uncorr/reset noise DISABLED.  thres_noise passes through unchanged.
    '''
    X = block.data
    if prc_ticks:
        P = int(prc_ticks * one_tick)
        phase = None
        if prc_sync:
            phase = sync_prc_phase(block.location, P, one_tick, X.shape[:-1],
                                   slot_ticks=prc_slot_ticks, block_pix=prc_block_pix,
                                   perm_seed=prc_perm_seed, device=X.device)
        X = apply_periodic_reset(X, P, phase=phase)
    Nt = X.shape[-1]
    n = make_ou_noise(tuple(X.shape[:-1]), Nt, ou_sigma, ou_tau_ticks * one_tick,
                      X.device, X.dtype)
    dn = torch.diff(n, dim=-1, prepend=torch.zeros_like(n[..., :1]))
    from tred.blocking import Block as _Block
    newblock = _Block(location=block.location, data=X + dn)
    return nd_readout(newblock, threshold, adc_hold_delay, adc_down_time, csa_reset_time,
                      one_tick, offset_to_align, pixel_axes, taxis,
                      None, thres_noise, None, leftover, niter)


def nd_readout_full(block, threshold, adc_hold_delay, adc_down_time, csa_reset_time=1, one_tick=1,
                    offset_to_align=0, pixel_axes=(), taxis=-1,
                    thres_noise=None, leftover=None, niter=10,
                    ou_sigma=0.5, ou_tau_ticks=2.0, reset_offset_sigma=0.9,
                    prc_ticks=None, prc_sync=False, prc_slot_ticks=16, prc_block_pix=7,
                    prc_perm_seed=20260713):
    '''
    Full noise/reset fidelity model (KNOWLEDGE 6.30): rolling periodic reset
    with per-reset constant baseline offsets + OU output noise (CSA
    bandwidth) + nd_readout_rst (initial baseline = per-channel constant;
    per-trigger-reset constant offsets native).  thres_noise passes through.
    Deterministic undershoot template deliberately NOT included yet (D=0
    stage; single-variable discipline).
    '''
    X = block.data
    if prc_ticks:
        P = int(prc_ticks * one_tick)
        phase = None
        if prc_sync:
            phase = sync_prc_phase(block.location, P, one_tick, X.shape[:-1],
                                   slot_ticks=prc_slot_ticks, block_pix=prc_block_pix,
                                   perm_seed=prc_perm_seed, device=X.device)
        X = apply_periodic_reset(X, P, phase=phase, offset_sigma=reset_offset_sigma)
    Nt = X.shape[-1]
    n = make_ou_noise(tuple(X.shape[:-1]), Nt, ou_sigma, ou_tau_ticks * one_tick,
                      X.device, X.dtype)
    dn = torch.diff(n, dim=-1, prepend=torch.zeros_like(n[..., :1]))
    from tred.blocking import Block as _Block
    newblock = _Block(location=block.location, data=X + dn)
    return nd_readout_rst(newblock, threshold, adc_hold_delay, adc_down_time, csa_reset_time,
                          one_tick, offset_to_align, pixel_axes, taxis,
                          None, thres_noise, reset_offset_sigma, leftover, niter)
