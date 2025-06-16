import h5py
import numpy as np

def rotation_matrix_y(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    # standard right-hand-rule rotation about +y
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ])

f = h5py.File('/home/yousen/Public/ndlar_shared/data/tred_2x2_2025010/'
              'MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5','r')
traj = f['trajectories']
segm = f['segments']
events = np.unique(traj['event_id'])

# clock-wise rotation, to negative x; positive x is assumed to be x-direction
angle_bins = np.array([0, -30, -60, -90])

origin     = np.array([-10.0, 5.0, 20.0])

segments_out = { pid: { 'original' : [], 'start' : []}
                 for pid in (13, 2212, 211) }

itrk = 0
for i, eid in enumerate(events):
    emt = (traj['event_id']==eid)
    ems = (segm['event_id']==eid)
    particles = traj[emt]
    tracks    = segm[ems]

    for pid in (13, 2212, 211):
        pars = particles[np.abs(particles['pdg_id'])==pid]
        for ipar, tid in enumerate(pars['file_traj_id']):
            # --- compute 3D displacement & length (in same units as xyz...)
            disp = pars[ipar]['xyz_end'] - pars[ipar]['xyz_start']
            length = np.linalg.norm(disp)

            tks = tracks[tracks['file_traj_id']==tid]

            # keep only segments >5 cm
            mask = length > 5.0
            if not np.any(mask):
                continue

            # compute angle to x-axis, in degrees
            cos_theta = (disp[0]**2 + disp[1]**2)/ length
            # clamp floating‐point noise
            cos_theta = np.clip(cos_theta, -1.0, +1.0)
            theta_deg = np.degrees(np.arccos(cos_theta))

            if np.abs(theta_deg) > 1:
                continue

            else:
                # drift towards positive
                tr = (tks['x_start'] < -3) & (tks['x_start'] > -30)
                tr &= (tks['y_start'] < 60) & (tks['y_start'] > -60)
                tr &= (tks['z_start'] < 30) & (tks['z_start'] > 2)

                tr &= (tks['z_start'] - pars[ipar]['xyz_start'][2]) < 15

                sel_tks = tks[tr]
                if len(sel_tks) > 0:
                    # print("Found")
                    segments_out[pid]['original'].append(sel_tks)
                    xmin = np.min(sel_tks['x_start'])
                    ymin = np.min(sel_tks['y_start'])
                    zmin = np.min(sel_tks['z_start'])
                    segments_out[pid]['start'].append(np.array([xmin, ymin, zmin]))
                    itrk += 1


# now write out per (pid,bin)
segments = { pid : [] for pid in segments_out.keys() }
for pid, data in segments_out.items():
    nel = np.array([len(e) for e in data['original']])
    i = np.argmax(nel)
    ds = data['original'][i]
    displacement = origin - data['start'][i]
    ds['x_start'] += displacement[0]
    ds['y_start'] += displacement[1]
    ds['z_start'] += displacement[2]
    ds['x_end'] += displacement[0]
    ds['y_end'] += displacement[1]
    ds['z_end'] += displacement[2]

    segments[pid].append(ds)
    for iang, ang in enumerate(angle_bins[1:], 1):
        seg = np.copy(ds)
        theta = np.deg2rad(ang)
        R = rotation_matrix_y(theta)
        pts = np.vstack([
            np.stack([seg['x_start'], seg['y_start'], seg['z_start']], axis=1),
            np.stack([seg['x_end'  ], seg['y_end'  ], seg['z_end'  ]], axis=1)
        ])
        # print('pts', pts.shape)
        pts = (pts - origin) @R.T + origin
        print('at angle', ang, 'isochronous track along (0,0,1) rotated to', np.array([[0,0,1]])@R.T)
        N = len(seg)
        seg['x_start'], seg['y_start'], seg['z_start'] = (
            pts[:N, 0], pts[:N, 1], pts[:N, 2]
        )
        seg['x_end'  ], seg['y_end'  ], seg['z_end'  ] = (
            pts[N:, 0], pts[N:, 1], pts[N:, 2]
        )
        segments[pid].append(seg)
for pid, data in segments.items():
    for iang, ang in enumerate(angle_bins[:], 0):
        ds = data[iang]
        outfn = f'segments_pid{pid}_angle{abs(ang):02d}.hdf5'

        with h5py.File(outfn, 'w') as fout:
            fout.create_dataset('segments', data=ds)
            print(f"wrote {outfn}, {len(ds)} records")
