#!/usr/bin/env python
'''
Select a long muon track (one event, one TPC) from a MiniRun5 EDEPSIM hdf5
and export its segments to a small hdf5 usable by `tred train` (graph_opt.py).

The selection maximizes the drift-distance span of the track: lifetime
sensitivity goes as exp(-t_drift/tau), so a track spanning a large range of
drift distances carries far more lifetime information than a compact cluster.

Segments are evenly subsampled along the track to keep at most --max-segments
rows (memory control: the training loop retains the autograd graph for all
segments of an epoch).

Usage:
  uv run python make_muon_track.py                 # scan, pick best, export
  uv run python make_muon_track.py --scan-only     # just print candidates
  uv run python make_muon_track.py --event 12 --tpc 3   # pick explicitly
'''
import argparse
import h5py
import numpy as np

from tred.io_nd import simple_geo_parser

DEFAULT_INPUT = "/srv/storage1/yousen/storage/2x2run1/MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5"
DEFAULT_MODULE_YAML = "/home/yousen/Documents/NDLAr2x2/tred/tests/playground/2x2_mod2mod_variation.yaml"
DEFAULT_TILE_YAML = "/home/yousen/Documents/NDLAr2x2/tred/tests/playground/multi_tile_layout-2.4.16.yaml"


def tpc_boxes(module_yaml, tile_yaml):
    '''Return (boxes, anode_x): boxes[i] = (3,2) sorted min/max, anode_x[i].'''
    borders = simple_geo_parser(module_yaml, tile_yaml, True).numpy()  # (8,3,2)
    boxes = np.sort(borders, axis=2)
    anode_x = borders[:, 0, 0]  # io_nd convention: anode at borders[:,0,0]
    return boxes, anode_x


def assign_tpc(seg, boxes):
    '''Midpoint-in-box TPC assignment, mirroring tred.io_nd.tpc_label.'''
    mid = np.stack([(seg['x_start'] + seg['x_end']) / 2,
                    (seg['y_start'] + seg['y_end']) / 2,
                    (seg['z_start'] + seg['z_end']) / 2], axis=1)
    tpc = np.full(len(seg), -1, dtype=np.int64)
    for i, box in enumerate(boxes):
        inside = np.ones(len(seg), dtype=bool)
        for ax in range(3):
            inside &= (mid[:, ax] >= box[ax, 0]) & (mid[:, ax] <= box[ax, 1])
        tpc[inside] = i
    return tpc, mid


def scan(seg, boxes, anode_x):
    '''List muon-track candidates per (event, tpc, traj) sorted by drift span.'''
    mu = seg[np.abs(seg['pdg_id']) == 13]
    tpc, mid = assign_tpc(mu, boxes)
    keep = tpc >= 0
    mu, tpc, mid = mu[keep], tpc[keep], mid[keep]
    ddist = np.abs(mid[:, 0] - anode_x[tpc])

    cands = []
    keys = np.stack([mu['event_id'].astype(np.int64), tpc,
                     mu['traj_id'].astype(np.int64)], axis=1)
    uniq = np.unique(keys, axis=0)
    for ev, it, traj in uniq:
        m = (keys == [ev, it, traj]).all(axis=1)
        d = ddist[m]
        cands.append(dict(event=int(ev), tpc=int(it), traj=int(traj),
                          n=int(m.sum()), sum_dx=float(mu['dx'][m].sum()),
                          dmin=float(d.min()), dmax=float(d.max()),
                          span=float(d.max() - d.min())))
    cands.sort(key=lambda c: -c['span'])
    return cands


def export(seg, boxes, anode_x, event, tpc_id, traj, max_segments, out):
    mu = seg[(np.abs(seg['pdg_id']) == 13)
             & (seg['event_id'] == event)
             & (seg['traj_id'] == traj)]
    tpc, mid = assign_tpc(mu, boxes)
    mu, mid = mu[tpc == tpc_id], mid[tpc == tpc_id]
    ddist = np.abs(mid[:, 0] - anode_x[tpc_id])

    order = np.argsort(ddist)
    mu, ddist = mu[order], ddist[order]

    # StepLoader subdivides each row into floor(L/1cm)+1 sub-segments; that is
    # what the training loop pays memory for. Keep the largest evenly-spaced
    # subset of rows whose estimated subdivided count fits the budget.
    def est_subdiv(rows):
        L = np.sqrt((rows['x_end'] - rows['x_start'])**2
                    + (rows['y_end'] - rows['y_start'])**2
                    + (rows['z_end'] - rows['z_start'])**2)
        return int((np.floor(L) + 1).sum())

    for k in range(len(mu), 0, -1):
        pick = np.unique(np.round(np.linspace(0, len(mu) - 1, k)).astype(int))
        if est_subdiv(mu[pick]) <= max_segments:
            break
    mu, ddist = mu[pick], ddist[pick]
    print(f"kept {len(mu)} raw rows -> ~{est_subdiv(mu)} segments after "
          f"StepLoader 1 cm subdivision (budget {max_segments})")

    with h5py.File(out, 'w') as f:
        f.create_dataset('segments', data=mu)
        f.create_dataset('drift_distance_cm', data=ddist)
        f.attrs['event_id'] = event
        f.attrs['tpc_id'] = tpc_id
        f.attrs['traj_id'] = traj
        f.attrs['source'] = DEFAULT_INPUT
    v = 1.59645 * 0.1  # cm/us
    print(f"wrote {out}: {len(mu)} segments, sum dx = {mu['dx'].sum():.2f} cm, "
          f"sum dE = {mu['dE'].sum():.1f} MeV")
    print(f"  drift distance {ddist.min():.2f} - {ddist.max():.2f} cm "
          f"({ddist.min()/v:.0f} - {ddist.max()/v:.0f} us)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', default=DEFAULT_INPUT)
    ap.add_argument('--module-yaml', default=DEFAULT_MODULE_YAML)
    ap.add_argument('--tile-yaml', default=DEFAULT_TILE_YAML)
    ap.add_argument('--event', type=int, default=None)
    ap.add_argument('--tpc', type=int, default=None)
    ap.add_argument('--traj', type=int, default=None)
    ap.add_argument('--max-segments', type=int, default=40)
    ap.add_argument('-o', '--out', default='muon_track.hdf5')
    ap.add_argument('--scan-only', action='store_true')
    args = ap.parse_args()

    seg = h5py.File(args.input)['segments'][:]
    boxes, anode_x = tpc_boxes(args.module_yaml, args.tile_yaml)
    cands = scan(seg, boxes, anode_x)

    print(f"{'event':>6} {'tpc':>4} {'traj':>5} {'nseg':>5} {'dx(cm)':>8} "
          f"{'drift(cm)':>14} {'span(cm)':>9}")
    for c in cands[:15]:
        print(f"{c['event']:>6} {c['tpc']:>4} {c['traj']:>5} {c['n']:>5} "
              f"{c['sum_dx']:>8.2f} {c['dmin']:>6.2f}-{c['dmax']:<7.2f} "
              f"{c['span']:>9.2f}")
    if args.scan_only:
        return

    if args.event is None:
        best = next(c for c in cands if c['n'] >= 10)
    else:
        best = next(c for c in cands if c['event'] == args.event
                    and (args.tpc is None or c['tpc'] == args.tpc)
                    and (args.traj is None or c['traj'] == args.traj))
    print(f"\nselected: event {best['event']} tpc {best['tpc']} traj {best['traj']}")
    export(seg, boxes, anode_x, best['event'], best['tpc'], best['traj'],
           args.max_segments, args.out)


if __name__ == '__main__':
    main()
