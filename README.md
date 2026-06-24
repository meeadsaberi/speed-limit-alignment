# Speed-Limit Alignment — are posted limits appropriate?

**UNSW Sydney · ADB AI for Safer Roads Innovation Challenge 2026**

The challenge asks: *where are posted speed limits **misaligned** with road function and
vulnerable-road-user (VRU) exposure?* — to support evidence-based **speed-limit review**.
This is **not** about whether drivers speed; it is about whether the *limit itself* is set
at the right level.

## Method (simple, transparent, interpretable by a non-technical official)
For every road link:

1. **Classify the road into a Safe-System context class** — from road function + land use,
   **refined by street-level imagery** where available (a vision-language model reads the
   Mapillary photo). The imagery characteristics are kept in their **own columns next to**
   the provided data (they do not overwrite it), and they fix the data's known weakness
   (the challenge states `LandUse`/`SpeedLimit` are *estimates*).
2. **Assign the Safe-System-appropriate maximum limit** for that context, from a cited
   reference table (WHO / OECD-ITF):

   | Context | Appropriate max (km/h) |
   |---|---|
   | Pedestrians/cyclists mixing with traffic | **30** |
   | Urban / likely VRU presence / intersection | **50** |
   | Rural undivided | **70** |
   | Divided / limited-access / motorway, no VRU | **100–110** |

   *(Hard function gate: motorways and divided no-VRU roads get the high value → never flagged.)*
3. **Recommend a reduction only** — `recommended_limit = min(posted, appropriate_max)`.
   The method **never raises** a limit.
4. **Misalignment = posted − recommended** (km/h the limit exceeds the appropriate level).
5. **Risk flag** prioritises VRU-context links where the limit is materially too high and
   the operating (85th-percentile) speed confirms it is live on the road.
6. **Self-audit against operating speed.** If the 85th-percentile speed is *implausible for the
   assigned class* (a "pedestrian-mix" street where cars run at 90 km/h), the link is flagged
   `needs_review` when the class is unverified (data only), or `imagery_confirmed_conflict` when
   a street photo confirms a real VRU/junction context on a high-speed road. A **targeted imagery
   pass** (`vlm_sample.py target`) verifies the most speed/class-contradictory links.

The score is literally *"how many km/h the limit should come down, and to what"* — readable
by a transport-ministry official, and scalable to any country (uses only globally-available
inputs: functional class, operating speed, land use, street imagery).

References: WHO *Speed Management* (2023), OECD/ITF *Speed and Crash Risk* (2018), World Bank
GRSF; imagery method adapts **V-RoAst** (iRAP attributes from VLMs) and the World Bank GRSF
*Detecting Urban Clues for Road Safety* framework.

## Repository layout
```
model/                 data prep + VLM imagery pipeline
  load.py              harmonise the two networks (TH + MH)  -> data/enriched.gpkg
  enrich.py            land-use fallback + OSM VRU POIs
  vlm_sample.py        VLM road-context characterisation (gpt-5.5, 2 imgs/seg, resumable):
                         `vlm_sample.py`         random 10% per city
                         `vlm_sample.py target`  targeted misalignment-suspect pass
score/limit_alignment.py   THE METHOD -> per-link recommended limit, misalignment, risk flag,
                           current vs proposed classification, verification flags
map/                   interactive map (MapLibre)
  index.html           toggle posted ↔ recommended ↔ risk; current vs proposed classification
  build_alignment_map.py   writes the per-city map layers (map/data/align_*.geojson)
  server.js            tiny static server
findings/Findings_Summary.pdf   the 5-page findings report
run_all.sh             one-command reproduction (load → enrich → score → map layers)
```

## Run
```bash
./run_all.sh        # load → enrich → score → per-city map layers
node map/server.js  # serve the interactive map at http://localhost:8731

# Optional VLM imagery refinement (needs MAPILLARY + OPENAI keys in .env):
python model/vlm_sample.py          # random 10% per city, resumable
python model/vlm_sample.py target   # targeted pass on misalignment-suspect links
python score/limit_alignment.py     # re-score with the imagery refinement
```
Live map: **https://meeadsaberi.github.io/speed-limit-alignment/map/**

## Data note
This is a lean **reproduction** repo. The raw input network and the per-link score **CSV** carry
**TomTom-derived** operating speeds (commercial, challenge-provided) and are **not** committed —
they are provided to ADB through the submission platform. The published per-city map GeoJSON
includes an 85th-percentile operating-speed field (a derived aggregate) so the interactive map
renders; raw probe data is not redistributed. `./run_all.sh` over the challenge dataset
reproduces the score and map layers locally.

## Honesty notes
- VLM imagery refines context where Mapillary coverage exists (Thailand ~77%, Maharashtra ~22%);
  elsewhere a transparent **data-prior** (class + land use) is used, flagged per link.
- The method only recommends **reductions**; it never raises a limit.
- No crash data is used or required — a *proactive* Safe-System appropriateness screen, not a
  crash-prediction model.

## Attribution
Network © OpenStreetMap contributors, Overture Maps Foundation · speeds: TomTom
(challenge-provided) · imagery © Mapillary. Secrets in `.env` are gitignored.
