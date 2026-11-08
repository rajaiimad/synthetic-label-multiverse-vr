# redesign_v15: proposed analysis plan v15 (draft, awaiting author approval)

Nothing in this folder modifies existing files. No experiment of plan v15 has been run.

| File | Content |
|---|---|
| `ANALYSIS_PLAN_v15_DRAFT.md` | The complete proposed plan |
| `proxy_variance_components.py`, `proxy_components.json` | Planning inputs: per-generator MPNN − Raw-HGB differences and cohort/seed SDs, read from existing r6 outputs (above-band attack; proxies only) |
| `power_sim_v15.py`, `power_sim_v15.json` | Simulation of the design and of the exact pre-specified analysis over a grid of M (generators), n (test cohorts), S (seeds) |
| `power_sim_v15_proxyrandom.py`, `.json` | Scenario taken from the six prior-drawn proxy generators |
| `power_sim_v15_design.py`, `.json` | Final design comparison at S = 4, with null checks |
| `power_sim_v15_alpha.py`, `.json` | H1 at the final level (95% interval, α 0.025): power, size at the boundaries, coverage |

Rerun: `python proxy_variance_components.py` (needs XRP_GEN=v2), then the `power_sim_*.py` scripts.
