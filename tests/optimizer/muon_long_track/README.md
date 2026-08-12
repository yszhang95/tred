# Long muon track lifetime fit

Goal: shrink the noise-induced displacement of the lifetime fit by using a
track with a long drift lever arm instead of the near-anode 0.6 MeV cluster
of the stress test (which has a ~110 µs Fisher floor; see
`../ev2_segments_note.md` and the anchored stress runs).

## Selected track

`make_muon_track.py` scans
`/srv/storage1/yousen/storage/2x2run1/MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5`
for muon trajectories (|pdg|=13) and ranks (event, tpc, traj) by drift-distance
span. Selected: **event 10, TPC 1, traj 0** — span 0.8–29.4 cm of drift
(5–184 µs), 67 MeV over 34 cm.

Memory control: rows are evenly subsampled along drift so the count *after*
StepLoader's 1 cm subdivision stays within `--max-segments` (default 40,
estimated as floor(L/1cm)+1 per row). Exported file: `muon_track.hdf5`
(8 raw rows -> 38 segments). If GPU memory is tight, regenerate with a
smaller budget; on a 24 GB card you can afford more:

    uv run python make_muon_track.py --event 10 --tpc 1 --max-segments 80 -o muon_track.hdf5

## Run

    ./run_muon.sh

Outputs land here: `optimized_muon.log`, `lifetime_fit_results.npz`
(graph_opt.py writes the npz to the working directory; run_muon.sh cd's here
first so nothing in `tests/optimizer/` is overwritten).

Fit settings come from `config_muon_long_track.yaml`: true lifetime 1.0 ms,
two arms starting at 1.2 / 0.8 ms, 1500 epochs each, anchored CPU truth noise
(seed 20260612), no per-epoch prediction noise.

## Machine notes

Paths in the yaml (response, geometry yamls) are absolute — adjust on the
other machine. The input hdf5 path is relative to this directory.
