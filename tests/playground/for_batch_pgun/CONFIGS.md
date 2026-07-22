# pgun threshold-study configs — tag meanings

Each `config_for_pgun_<tag>.yaml` drives one tred fullsim run over the 6
`packet-0050018-2024_07_11_14_*` pgun files (run by
`run_tracks_<tag>.sh`).  Outputs `pgun_mu_3GeV_2mm_<tag>.npz` per packet
(gitignored — archived to HDD).  Downstream pick2/pick3both analysis and the
data/MC comparison live in `2x2_ql/filter_mc_events/output_july11/`; the
physics narrative and every verdict are in
`2x2_ql/threshold_studies/KNOWLEDGE.md` (section refs below).

Common to the 2026-07 campaign unless noted: response
`binaries/response_44_v2a_full.npz` (45×45), lifetime 1.2 ms,
noise uncorr 0.5 / reset 0.9 / thres 0.65 ke, tpc_list [0,1,2,3,6,7],
fluctuate false, threshold table
`2x2_ql/threshold_studies/threshold_july10to11_bin0p2_peakmode_dbin_20260702.hdf5`.

## Obsolete (pre io_group fix, 2026-06; superseded — do NOT use)
| tag | note |
|---|---|
| `20260526`, `20260606`, `old_20260606`, `10pcts_20260606` | early runs before the many_muons io_group remap fix; results raw/wrong. Candidates to drop, not archive. |
| `thres_prompt_hits_2pass_20260629` | first 2-pass prompt-hits run (pre-table-fix baseline). |

## Threshold-table development (2026-07-01/02)
| tag | what changed | KNOWLEDGE |
|---|---|---|
| `thres_prompt_hits_2pass_edgetrim5_20260701` | cluster_highstat edgetrim5 table (foot-biased ~2 ke low) | 6.x history |
| `thres_prompt_hits_2pass_peakmode_20260702` | peakmode estimator, fixed δ=0.2 | 6.x |
| `thres_prompt_hits_2pass_peakmode_dbin_20260702` | peakmode + binned δ (production table); ADC timing hold 1.5 / down 1.2 | 6.5 |

## ADC timing + response footprint (2026-07-04)
| tag | what changed vs dbin_20260702 | KNOWLEDGE |
|---|---|---|
| `..._peakmode_dbin_hd18_20260704` | ADC hold 1.5→1.8, down 1.2→0.9 (larnd 3+15 ticks); **adopted timing** | 6.6 |
| `..._peakmode_dbin_hd18_resp25_20260704` | + response 45×45→125×125 (25×25 pixels) | 6.10/6.14 (prediction falsified) |
| `..._peakmode_dbin_hold15_resp25_20260704` | 25×25 + hold 1.5/1.2 (2×2 factorial corner) | 6.14 |

## Rolling periodic reset (2026-07-13)
| tag | what changed vs hd18 | KNOWLEDGE |
|---|---|---|
| `..._hd18_prc_20260713` | + periodic reset, per-channel independent random phase | 6.17/6.21 |
| `..._hd18_prcsync_20260713` | + periodic reset, synchronized 7×7 rolling (frozen slot perm, one global phase); **baseline from here** | 6.23 |

## Noise-model studies (all on prcsync baseline, 2026-07-13/15)
| tag | what changed | KNOWLEDGE |
|---|---|---|
| `..._prcsync_lowTp05_20260713` | T<4 trigger thresholds +0.5 ke (analysis table unchanged); sensitivity test | 6.25 |
| `..._prcsync_noisehalf_20260713` | waveform noise halved (uncorr 0.25 / reset 0.45) | 6.27 |
| `..._prcsync_ou02_20260713` | correlated (OU) output noise, τ=0.2 µs, RMS unchanged | 6.28 |
| `..._prcsync_ou04_20260714` | correlated output noise, τ=0.4 µs | 6.31 |
| `..._prcsync_qdep_20260714` | noise amplitude scales with accumulated charge σ(t)=0.3+0.73·min(1,S/10ke) | 6.33 |
| `..._prcsync_thres01_20260714` | thres_noise 0.65→0.10 (waveform noise unchanged); refuted, validates 0.65 | 6.29 |
| `..._prcsync_rst_20260714` | reset-model stage 1 (per-reset constant offsets); **falsified on smoke, npz deleted** — config kept for the record | 6.32 |
| `..._prcsync_qdepcorr_20260715` | **combined**: amplitude-vs-charge × correlated (τ=0.4); best data match, adopted noise model | 6.34 |
| `..._prcsync_cfluct_20260715` | combined + issue-#27 per-voxel charge-deposition fluctuation (p=0.05); effect negligible | 6.35 |

## Working configuration stack (as of 2026-07-15)
hd18 timing (1.8/0.9) + synchronized periodic reset + [amplitude-vs-charge ×
correlated noise, τ=0.4] + thres_noise 0.65 + ADC quantization at analysis
(qadc).  Every layer has an independent hardware/data justification; see
KNOWLEDGE 6.34.  Config: `config_for_pgun_..._prcsync_qdepcorr_20260715.yaml`.
