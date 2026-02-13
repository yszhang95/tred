#!/usr/bin/env python3

import numpy as np
import re
import os
import json
import argparse


def export_event_jsons(npz_path, output_dir, prefix='effq'):
    f = np.load(npz_path)

    # Match pattern only for exact keys like 'effq_tpcX_batchY'
    pattern = re.compile(rf'{prefix}_tpc(\d+)_batch(\d+)$')
    matched_keys = [k for k in f.files if pattern.match(k)]

    all_positions = []
    all_charges = []
    all_event_ids = []
    all_tpc_ids = []

    for key in matched_keys:
        match = pattern.match(key)
        tpc = int(match.group(1))
        batch = int(match.group(2))

        data = f[key]
        num_hits = data.shape[0]
        event_id = int(f[f'event_id_tpc{tpc}_batch{batch}'])
        event_ids = np.full(num_hits, event_id, dtype=int)
        tpc_ids = np.full(num_hits, tpc, dtype=int)

        all_positions.append(data[:, :3])
        all_charges.append(data[:, -1])
        all_event_ids.append(event_ids)
        all_tpc_ids.append(tpc_ids)

    # Concatenate all arrays
    positions = np.vstack(all_positions)
    charges = np.hstack(all_charges)
    event_ids = np.hstack(all_event_ids)
    tpc_ids = np.hstack(all_tpc_ids)

    # Sort by event_id
    sort_indices = np.argsort(event_ids)
    positions = positions[sort_indices]
    charges = charges[sort_indices]
    event_ids_sorted = event_ids[sort_indices]
    tpc_ids = tpc_ids[sort_indices]

    # Grouping by event_id
    _, idx_split = np.unique(event_ids_sorted, return_index=True)
    pos_groups = np.split(positions, idx_split[1:])
    charge_groups = np.split(charges, idx_split[1:])
    event_ids_unique = event_ids_sorted[idx_split]
    tpc_groups = np.split(tpc_ids, idx_split[1:])

    # Write each group to JSON
    for eid, pos, chg, tpcs in zip(event_ids_unique, pos_groups, charge_groups, tpc_groups):
        chg *= 1000  # convert ke- to e-
        evt = {
            "runNo": 0,
            "subRunNo": 0,
            "eventNo": int(eid),
            "x": pos[:, 0].tolist(),
            "y": pos[:, 1].tolist(),
            "z": pos[:, 2].tolist(),
            "q": chg.tolist(),
            "nq": [0] * len(chg),
            "cluster_id": tpcs.tolist(),
            "real_cluster_id": tpcs.tolist(),
            "geom": "2x2",
            "type": "wire-cell",
        }

        # Save to subfolder per event
        edir = os.path.join(output_dir, os.path.join('data', str(eid)))
        os.makedirs(edir, exist_ok=True)
        filename = f"{eid}-{prefix}.json"
        with open(os.path.join(edir, filename), "w") as fjson:
            json.dump(evt, fjson, indent=2)

    print(f"Exported {len(event_ids_unique)} event JSONs to '{output_dir}'")


def main():
    parser = argparse.ArgumentParser(description="A script that processes NPZ files to jsons for BEE.")
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='output',
        help='The directory where results will be saved.'
    )
    parser.add_argument(
        '-i', '--input-npz',
        type=str,
        required=True,
        help='The path to the input .npz file.'
    )

    args = parser.parse_args()

    print(f"Output Directory: {args.output_dir}")
    print(f"Input NPZ File: {args.input_npz}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Your main processing logic goes here ---
    export_event_jsons(args.input_npz, args.output_dir, prefix='effq')
    export_event_jsons(args.input_npz, args.output_dir, prefix='hits')


if __name__ == "__main__":
    main()
