import torch
# import logging

from tred.blocking import Block

def nd_readout(block, threshold, adc_hold_delay, adc_down_time, csa_reset_time=1, pixel_axes=(), taxis=-1, leftover=None, niter=10):
    '''
    locs :: (N, nxpl, nxpl, ..., vdim)
    X :: (N, npxl, npxl, ..., Nt)
    Is X already the summed? No
    Do it once or iteratively?
    Do it Once is faster but complicated
    Do it iteratively is easier but complicated...

    Let us do it once.
    '''
    X = block.data
    locations = block.location
    for i in pixel_axes:
        locations = locations.unsqueeze(1)
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
    start = torch.zeros((*tuple(X.shape[i] for i in [0,]+list(pixel_axes)), 1), dtype=torch.int64)
    trange = torch.arange(X.shape[taxis]).view(*[1 for i in range(X.ndim-1)], -1)
    # info(f'start shape {start .shape}')
    # info(f'trange shape {trange.shape}')

    Nt = X.shape[taxis]
    # logging.debug(f'X shape {X.shape}')
    # info(f'X shape {X.shape}')
    # FIXME: what is an appropriate accumulation function?
    Xacc = X.cumsum(dim=taxis)
    # logging.debug(f'Xacc shape {Xacc.shape}')
    # info(f'Xacc shape {Xacc.shape}')

    pxl_indices = slice(None, -1, None) # FIXME: hard coded

    iteration = 0
    while True:
        # logging.debug(f'Iteration {iteration}')
        # info(f'Iteration {iteration}')

        mvalid = trange >= start # shape (npxl, npxl, ..., Nt) if taxis = -1
        # logging.debug(f'mvalid shape {mvalid.shape}')
        # info(f'mvalid shape {mvalid.shape}')
        Xacc = Xacc * mvalid # FIXME: start > trange; we need leftover information

        crossed = (Xacc >= threshold) & mvalid # check after start # shape (N, nxpl, ..., Nt) if taxis = -1
        cross_t = torch.argmax(crossed.to(torch.int32), dim=taxis, keepdim=True) # shape (N, npxl, .., 1) if taxis = -1

        # logging.debug(f'cross_t shape {cross_t.shape}')
        crossed = torch.gather(Xacc, taxis, cross_t) >= threshold # is it really cross at cross_t?
        # crossed shape: (N, npxl, ..., Nt) if taxis = -1
        # logging.debug(f'crossed shape {crossed.shape}')
        # hold_t = cross_t + adc_hold_delay - 1  # samed as cross_t shape, element at adc_hold_delay - 1 is from 0 to adc_hold_delay-1
        hold_t = cross_t + adc_hold_delay
        # logging.debug(f'hold_t shape {hold_t.shape}')
        hold_t_inrange = torch.clamp(hold_t, min=0, max=Nt-1) # same as hold_t shape
        # logging.debug(f'hold_t_inrange shape {hold_t_inrange.shape}')
        Xacc_hold_t = torch.gather(Xacc, taxis, hold_t_inrange) # shape (N, npxl, ..., 1) if taxis = -1
        # logging.debug(f'Xacc_hold_t shape {Xacc_hold_t.shape}')
        delay_crossed = Xacc_hold_t >= threshold # shape (N, npxl, ..., 1) if taxis = -1
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
        start[triggered] = hold_t[triggered] + adc_down_time + 1 # on discriminator, controlled by adc down time
        start_times = gtimes + start[triggered]
        oloc = torch.cat([pixels, times.unsqueeze(1), hold_times.unsqueeze(1), start_times.unsqueeze(1)], dim=1)
        olocs.append(oloc)
        ocharges.append(hits)
        start[~triggered] = hold_t[~triggered]+1
        start[~crossed] = Nt # crossed not triggered should be at hold_t+1; never crossed needs to be at start.
        # at triggered positions, charges are reset and there is one timestamp missing;
        # everything happens on CSA
        # hold t may be at the last t;
        # FIXME: what happens if the hold_t is the last element?
        Xacc_next_to_hold_t = torch.gather(Xacc, taxis, torch.clamp(hold_t+csa_reset_time, min=0, max=Nt-1))
        # only update the triggered positions
        Xacc[triggered.squeeze(taxis)] -= Xacc_next_to_hold_t[triggered.squeeze(taxis)]
        iteration += 1
    if len(olocs) == 0:
        raise NotImplementedError("Not sure how to handle empty hit collection")
    return torch.cat(olocs, dim=0), torch.cat(ocharges, dim=0)
