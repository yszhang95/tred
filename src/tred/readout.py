import torch
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
