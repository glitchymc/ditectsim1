# ebike_swap: NYC e-bike battery-swap simulator and station optimizer

This is a modular agent-based simulator of battery swapping for NYC e-bike food delivery. It follows the
*Ebike Battery Swap* deck. It also includes a simulation-based optimizer for station locations and
battery inventory, the deck's first research topic, which NYC DOT is interested in.

```
OD Generator ──► Dispatch Model ◄──► Rider list ◄──► Station list
 (demand.py)      (dispatch.py)       (agents.py)      (agents.py)
                         ▲ discrete-event engine (engine.py) ▲
```

| Deck slide | Module | What is implemented |
|---|---|---|
| OD generator: time-dependent generator | `demand.TimeDependentGenerator` | Hourly piecewise Poisson arrivals with lunch and dinner peaks, scaled by a day-of-week multiplier |
| OD generator: demographic destination sampler | `demand.DemographicDestinationSampler` | Zone weight = (residents or workers, depending on hour) × young-adult share effect × income effect |
| OD generator: restaurant type sampler | `demand.RestaurantTypeSampler` | Cuisine drawn from a day-type prior, then a restaurant of that cuisine near the destination with distance decay |
| Naive dispatch | `dispatch.NaiveRegionalDispatcher` | Regional FIFO queues; the nearest available rider who has enough charge gets the order |
| Naive rider | `agents.Rider` + `engine` | Status: availability, battery, current job. Goes to the restaurant, waits for food, delivers, and swaps below the threshold |
| Naive station | `agents.Station` | First-come-first-served charged packs and a rider queue. Returned packs charge at full power for a time set by their state of charge |
| Battery consumption | `geo.GridRouter` | Energy from route length, elevation gain along the route, and expected stops at traffic lights |
| Expected result / iterative optimization | `optimize.py` | Common-random-number evaluator; the deck's iterative "queued vs. unused" loop; simulated annealing; held-out validation |
| Visualization sketch | `viz.py` | Map of riders colored by battery level, plus per-station battery racks and queue bars, as a PNG or animated GIF |

## Quick start

```bash
pip install numpy pandas matplotlib pytest
python -m pytest -q                                   # 9 tests
python scripts/run_baseline.py --gif                  # one day, uniform layout -> out/baseline/ (example outputs committed under results/baseline/)
python scripts/optimize_stations.py                   # full optimization -> out/optimize/ (results below are committed under results/optimize/)
python scripts/optimize_stations.py --policy least_time   # riders who avoid queues
```

Both scripts accept `--stations K`, `--batteries B` (total budget), `--riders` and `--orders-per-day`.

## Data

The simulator never reads proprietary feeds. Each input comes either from a loader in
`ebike_swap/data/loaders.py` or from the synthetic Manhattan stand-in in `data/synthetic.py`.

| Input | Public source (deck) | Loader / flag |
|---|---|---|
| Restaurant locations and cuisine | DOHMH Restaurant Inspection Results (`43nn-pn8j`) | `load_dohmh_restaurants`, `--restaurants file.csv` |
| Cuisine mix, weekday/weekend skew, prep times | Kaggle *Food Ordering and Delivery App* | `calibrate_from_kaggle`, `--kaggle food_order.csv` |
| Population, income, age by neighbourhood | GeoDa NYC neighbourhood ACS 2008-12 | `load_neighborhoods`, `--neighborhoods file.geojson` |
| Elevation | (any DEM) | `geo.GridTerrain(x0, y0, cell_km, z)` |

Anything you don't supply falls back to the synthetic city, and `city.provenance` records which source each
input came from. The synthetic city is shaped like Manhattan but is not calibrated. **Results from it
show that the method works. They are not findings about NYC.** The default cuisine prior uses the Kaggle
order counts shown on the deck's chart. DOHMH cuisine labels map to the Kaggle categories through
`DOHMH_CUISINE_MAP`. Neighbourhood field names differ between releases, so pass `fields=` /
`young_cols=` to match your file.

The collision and bicycle-counter datasets are not used yet. Two natural hooks: a router that adds
crash risk to its cost (a crash-avoidance path), and calibrating travel speed against the counters.

## Model details and defaults (`config.py`)

* **Battery:** 720 Wh (48 V × 15 Ah). Flat riding uses 15 Wh/km. Climbing costs m·g·Δh / 0.75 with
  m = 120 kg. Traffic lights: 12 per km, half of them a stop, 1 Wh to re-accelerate and 5 s of delay each.
* **Travel:** 18 km/h cruise speed on a street grid rotated 29°, with a 1.05 detour factor.
* **Swap threshold:** with `"auto"`, the threshold is the energy bound for the longest possible job
  (max pickup distance + max OD distance) plus the trip to the farthest point's nearest station. This is
  the deck's assumption. The dispatcher also refuses any job a rider could not finish and still reach a
  station, so no rider is ever stranded. The tests check this.
* **Station:** 0.36 kW per slot (an empty pack charges in 2 h), a 1 minute swap, one charger per slot by
  default.
* **Orders** are cancelled if nobody is assigned within 60 minutes. Shifts are `all_day` or `peaks`.

## Optimization (`optimize.py`)

The decision is to choose **K sites** from a candidate grid (400 m spacing by default) and split a
**battery budget B** across them.

**Objective, in rider-hour equivalents:** swap downtime (detour + queue + swap) + hours of lateness
beyond 45 min + 1 h per undelivered order. The weights are set in `Objective`.

1. **Evaluator.** Order streams and initial rider states are generated once per seed. Every candidate
   layout runs on the same streams (common random numbers), and runs go in parallel across processes.
2. **Starting layouts:**
   * uniform coverage
   * demand k-means: k-means over delivery destinations, with batteries in proportion to cluster size
   * pilot k-means: one run with dense stations records where riders actually decide to swap, and those
     points are clustered
3. **`IterativeRebalancer`, the deck's loop.** It simulates, finds the station with the most queue wait
   and the least-used station, then proposes:
   * moving batteries from surplus or cold stations to the hot one
   * moving the cold station into the far half of the hot station's catchment
   * re-centring every station on the L1-median of its own swap-trigger points

   It keeps the best improving proposal. If nothing improves, it halves the move size and adds random
   moves.
4. **`SimulatedAnnealing`.** It searches the same neighbourhood with a mix of random and guided moves.
   It serves as a check on the guided loop.
5. **Held-out validation.** Every layout is scored again on seeds the optimizer never saw. The script
   reports a paired difference against the uniform layout.

Outputs in `out/optimize/` (the run reported below is committed under `results/optimize/`):
* `holdout_comparison.csv`
* `stations_<layout>.csv`
* `history_*.csv`
* `layouts.png` (marker size = batteries, color = mean queue wait)
* `convergence.png`
* `run.json`

## Results on the synthetic city (method check, not an NYC finding)

`python scripts/optimize_stations.py` with the defaults: K = 15 stations, B = 150 batteries, 400 riders,
about 9.5k orders per day, 6 training seeds and 8 held-out seeds. The table shows held-out cost in
rider-hour equivalents:

| layout | cost ± SE | mean swap downtime | p95 queue wait |
|---|---|---|---|
| simulated annealing | 478 ± 31 | 28.0 min | 99 min |
| demand k-means | 482 ± 18 | 28.3 min | 104 min |
| pilot k-means = iterative | 495 ± 24 | 28.9 min | 99 min |
| uniform coverage | 1569 ± 66 | 89.0 min | 419 min |

* **Placing stations by demand matters a lot.** Every demand-aware layout cuts cost by about 70%
  compared with uniform coverage, and all 8 held-out seeds agree. Uniform coverage overloads Midtown and
  leaves the northern stations idle. This is the queued-vs-unused pattern the deck predicted.
* **Fine-tuning beyond a good k-means start is within the noise.** In a first run that accepted any
  mean improvement on 4 seeds, both local searches beat the pilot on training but lost to it on held-out
  seeds. That is overfitting to the demand realizations. Moves are now accepted only if the paired gain
  exceeds `--z` standard errors (`paired_improvement`). With that rule the deck's guided loop finds no
  significant move, and SA's training gain (475 → 458) mostly disappears on held-out seeds. Getting more
  out of local search needs more seeds per evaluation, or a cheaper surrogate model to screen moves.
* In this scenario, battery supply is the binding constraint, not location: queues stay at about 20
  minutes even in the best layouts. Sweeping `--batteries` next would trace the inventory-versus-wait
  trade-off.

## Extending: the other research topics

Every component is passed into `Simulation(...)` or `Evaluator(...)`, so it can be replaced without
touching the engine:

* **Battery charging optimization:** subclass `Station` (scheduled or price-aware charging, limited
  chargers through `StationParams.chargers_per_station`).
* **Swap-oriented path planning / order assignment:** implement `Dispatcher` (`bind`, `on_order`,
  `on_rider_free`, `on_cancel`). Example: score riders by pickup time plus the expected swap detour.
* **Rider behavior change:** implement `StationPolicy.choose`. `LeastExpectedTime` (riders who avoid
  queues) is included as the first alternative to `NearestStation`.
* **Real street network:** any object with `route(a, b) -> Route(km, minutes, wh)` and a vectorized
  `distance(a, b)` can replace `GridRouter` (for example a wrapper around an OSMnx graph).

## Limitations

* The street grid is a rotated L1 metric, not a real network. Bridges, parks and one-way streets are
  ignored.
* Riders are fully compliant, and fleet size and shifts are exogenous.
* The synthetic demand scale (8,000 orders/day, 400 riders) is a placeholder.
* Swap-station placement constraints (sidewalk space, grid connection, permits) are not modeled beyond the
  candidate grid. Filter `candidate_sites` with real siting data when you have it.
