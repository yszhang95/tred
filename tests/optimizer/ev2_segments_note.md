# ev2 segment analysis (2x2 geometry)

Notes from inspecting `MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5` using the same
geometry/segmentation path as `src/tred/plots/graph_effq.py`.

## How the 2x2 geometry is used (from graph_effq.py)

1. **Parse geometry → TPC borders**
   `simple_geo_parser(module_yaml, tile_yaml, old_geo_config)`
   - `module_yaml = tests/playground/2x2_mod2mod_variation.yaml`
   - `tile_yaml   = tests/playground/multi_tile_layout-2.4.16.yaml`
   - `old_geo_config = True` (graph_effq default)
   - returns an `(8, 3, 2)` tensor: `(x,y,z) × (min,max)` borders → **8 TPCs** (4 modules × 2 anode planes).
2. **Load + subdivide steps** — `StepLoader(h5py.File(input_path), transform=steps_from_ndh5)`
   splits each step into ≈1 cm sub-segments. `labels = int tensor (event_id, vertex_id, pdg_id)`.
3. **Assign to TPCs** — `create_tpc_datasets_from_steps(features, labels, borders, sort_index=0)`
   uses `tpc_label()` (segment midpoint inside a border box) → one `TPCDataset` per TPC;
   `tpcdataset.labels[:,0]` is `event_id`.

### Gotchas
- `event_id` in the file is **1-based** (events 1…N) → **event_id = 0 has 0 segments**.
- `event_id` is stored as `uint32`; the current torch refuses to convert it (see FIXME in
  `steps_from_ndh5`). Pre-cast integer fields to signed before feeding `StepLoader`.
- File has 89,430 raw segments; 29 fall outside all TPC volumes.

## Segment counts per TPC

Subdivided (what the pipeline logs as `N segments`) / raw, for event_id 1 and 2:

| tpc | ev1 (subdiv/raw) | ev2 (subdiv=raw) |
|--:|--:|--:|
| 0 | 550 / 514 | 2 |
| 1 |  77 /  77 | 0 |
| 2 | 789 / 684 | 7 |
| 3 | 385 / 345 | 0 |
| 4 |  31 /  31 | 0 |
| 5 | 350 / 336 | 2 |
| 6 | 142 / 142 | 7 |
| 7 | 127 / 118 | 13 |
| **TOTAL** | **2451 / 2247** | **31** |

(ev2 segments are all ≤1 cm, so subdivision does not change ev2 counts.)

## ev2: total segment length per TPC

Geometric length `sqrt(Δx²+Δy²+Δz²)` equals the file's `dx` field exactly (cm):

| tpc | N | total length (cm) |
|--:|--:|--:|
| 0 | 2 | 0.0197 |
| 2 | 7 | 0.0899 |
| 5 | 2 | 0.1000 |
| 6 | 7 | 0.0926 |
| 7 | 13 | 0.3110 |
| **TOTAL** | **31** | **0.6132** |

TPCs 1, 3, 4 have no ev2 segments; all 31 ev2 segments are inside a TPC.

## ev2: PDG id and parent_id per TPC

PDG key: `11` = e⁻, `22` = γ, `2112` = neutron. `parent_id` looked up from the
`trajectories` table by `(event_id, vertex_id, traj_id)`.

| tpc | N | pdg_id | parent_id |
|--:|--:|--|--|
| 0 | 2 | `11`×2 | `18`×2 |
| 2 | 7 | `11`×6, `22`×1 | `9`×6, `3`×1 |
| 5 | 2 | `2112`×2 | `18`×2 |
| 6 | 7 | `11`×6, `22`×1 | `165`×6, `12`×1 |
| 7 | 13 | `11`×7, `22`×1, `2112`×5 | `156`×7, `22`×5, `155`×1 |

- No `parent_id == -1` → none of these segments are from a primary particle (all secondaries).
- Neutrons (`2112`) are neutral → `dE`/`n_electrons` ≈ 0, produce no ionization charge.

## Saved files

Per-TPC ev2 segments written to repo root as `ev2_tpc{i}.hdf5` for i ∈ {0,2,5,6,7}.
**Photon segments (`pdg_id == 22`) are removed.** The original `segments` structured
dtype is preserved (identical to the source file's `segments` dataset).

Each HDF5 file contains:
- dataset `segments` — raw segment rows (all original fields), pdg 22 dropped
- dataset `seg_length_cm` — per-segment geometric length
- attrs `tpc_id`, `event_id`

Load via `h5py.File("ev2_tpc7.hdf5")["segments"][:]`.

Kept counts / total length after dropping pdg 22:

| tpc | kept N | pdg22 dropped | total length (cm) |
|--:|--:|--:|--:|
| 0 | 2 | 0 | 0.0197 |
| 2 | 6 | 1 | 0.0399 |
| 5 | 2 | 0 | 0.1000 |
| 6 | 6 | 1 | 0.0426 |
| 7 | 12 | 1 | 0.2610 |
| **TOTAL** | **28** | **3** | **0.4632** |

(Built with `uv run`; the env needed `uv add h5py`. Input file lives at
`/srv/storage1/yousen/storage/2x2run1/MiniRun5_1E19_RHC.convert2h5.0000000.EDEPSIM.hdf5`.)
