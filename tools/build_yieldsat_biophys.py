"""Biophysical-variable input `biophys` for YieldSAT (protocols/yieldsat.md, 2.5) [AgriBench].

Each Sentinel-2 observation is converted to five canopy variables with the ESA SNAP Sentinel-2 Toolbox biophysical
processor (Weiss & Baret, S2ToolBox Level-2 ATBD; version 3.0 "S2A" networks; the operational BVNET baseline that
PROSAIL-VAE papers compare against): LAI, FAPAR, FCOVER, canopy chlorophyll content (CCC = LAI x Cab, g/m2) and
canopy water content (CWC = LAI x Cw, kg/m2). Each network is a PROSAIL-trained MLP:
    inputs  B3, B4, B5, B6, B7, B8A, B11, B12 (L2A reflectance, 0-1), cos(view zenith), cos(sun zenith),
            cos(sun azimuth - view azimuth)
    x' = 2 (x - min) / (max - min) - 1;  h = tanh(W1 x' + b1);  y' = W2 h + b2;  y = (y' + 1)(ymax - ymin)/2 + ymin
    then y is clipped to the variable's valid range (ExtremeCases file), as SNAP does.
The coefficients come from senbox-org/s2tbx (GPL-3.0) at a pinned commit and are downloaded to
$project/upstream/snap-biophysical/ (`--fetch`); the implementation is checked against SNAP's own test cases.

Angles [AgriBench approximation]: YieldSAT keeps no acquisition geometry. Sun zenith and azimuth are computed for
each pixel's location and acquisition date at 10:30 local solar time (Sentinel-2's descending-node time). View
zenith is set to 5 degrees (S2 views within about 12 degrees of nadir) and the relative azimuth to 90 degrees
(cos = 0), because which side of the swath a field lies on is unknown. Near-nadir views make the networks only
weakly sensitive to these two.

The five variables are then placed on the same days-since-sowing grid as `s2-aligned` (linear interpolation
between valid observations, NaN outside the observed range): 5 variables x ~17 steps.

    python -m tools.build_yieldsat_biophys --fetch                 (once, login node: downloads ~10 small files)
    python -m tools.build_yieldsat_biophys --subset GER-W          (CPU job)
"""
import argparse
import datetime as dt
import json
import time
import urllib.request

import numpy as np

from tools import yieldsat as ys
from tools.build_yieldsat_s2_aligned import GRID_START, MAX_END, STEP, interpolate, sowing_dates
from tools.embed_yieldsat_galileo import slot_months

COMMIT = "33f36f168cf640ca8bcef0a2f7c7aeee906af687"
RAW = f"https://raw.githubusercontent.com/senbox-org/s2tbx/{COMMIT}/s2tbx-biophysical/src/main/resources/auxdata/3_0/S2A"
AUX = ys.PROJECT / "upstream" / "snap-biophysical" / COMMIT[:12]
VARIABLES = {"LAI": "lai", "FAPAR": "fapar", "FCOVER": "fcover", "LAI_Cab": "ccc", "LAI_Cw": "cwc"}
FILES = ["Normalisation", "Denormalisation", "Weights_Layer1_Neurons", "Weights_Layer1_Bias",
         "Weights_Layer2_Neurons", "Weights_Layer2_Bias", "ExtremeCases", "TestCases"]
BANDS = ["B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12"]
VIEW_ZENITH_DEG = 5.0
COS_RELATIVE_AZIMUTH = 0.0
LOCAL_SOLAR_HOUR = 10.5


def fetch():
    for var in VARIABLES:
        (AUX / var).mkdir(parents=True, exist_ok=True)
        for f in FILES:
            target = AUX / var / f"{var}_{f}"
            if not target.exists():
                urllib.request.urlretrieve(f"{RAW}/{var}/{var}_{f}", target)
    print(f"coefficients in {AUX}")


def load(var):
    read = lambda f: np.loadtxt(AUX / var / f"{var}_{f}", delimiter=",", ndmin=2)
    return {"norm": read("Normalisation"), "denorm": read("Denormalisation")[0], "w1": read("Weights_Layer1_Neurons"),
            "b1": read("Weights_Layer1_Bias")[0], "w2": read("Weights_Layer2_Neurons")[0],
            "b2": read("Weights_Layer2_Bias")[0, 0], "extreme": read("ExtremeCases")[0]}


def network(net, x, clip=True):
    """x (n, 11) -> (n,) SNAP biophysical variable."""
    lo, hi = net["norm"][:, 0], net["norm"][:, 1]
    xn = 2 * (x - lo) / (hi - lo) - 1
    y = np.tanh(xn @ net["w1"].T + net["b1"]) @ net["w2"] + net["b2"]
    y = 0.5 * (y + 1) * (net["denorm"][1] - net["denorm"][0]) + net["denorm"][0]
    if clip:
        _, ymin, ymax = net["extreme"]
        y = np.clip(y, ymin, ymax)
    return y


def check_networks(nets, n=2000):
    """Compare with SNAP's own test cases: 11 inputs first, the (clipped) expected output in the last column;
    rows with no expected value (outside SNAP's definition domain) are skipped."""
    worst = {}
    for var, net in nets.items():
        cases = np.loadtxt(AUX / var / f"{var}_TestCases", delimiter=",", max_rows=n)
        expected = cases[:, -1]
        ok = np.isfinite(expected)
        worst[var] = float(np.max(np.abs(network(net, cases[ok, :11]) - expected[ok])))
        if worst[var] > 1e-3 * max(1.0, np.abs(expected[ok]).max()):
            raise RuntimeError(f"{var}: implementation differs from SNAP test cases by {worst[var]}")
    return worst


def sun_zenith_cos(lat, date):
    """cos(solar zenith) at LOCAL_SOLAR_HOUR local solar time (Cooper declination; no equation of time needed)."""
    doy = date.timetuple().tm_yday
    decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    hour_angle = np.radians(15 * (LOCAL_SOLAR_HOUR - 12))
    phi = np.radians(lat)
    return np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(hour_angle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", choices=ys.SUBSETS)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch()
        return
    out_dir = ys.FEATURES / "biophys"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{args.subset}.npy"
    if target.exists():
        raise SystemExit(f"{target} exists; caches are never overwritten")
    t0 = time.time()
    nets = {var: load(var) for var in VARIABLES}
    test_error = check_networks(nets)
    names = json.loads((ys.FEATURES / "v1" / f"{args.subset}.json").read_text())["feature_names"]
    cols = np.array([[names.index(f"{b}_t{s:02d}") for b in BANDS] for s in range(ys.N_STEPS)])  # (24, 8)
    meta = np.load(ys.FEATURES / "v1" / f"{args.subset}-meta.npz")
    fields, fidx = [str(f) for f in meta["fields"]], meta["field_idx"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    cache = np.load(ys.FEATURES / "v1" / f"{args.subset}.npy", mmap_mode="r")
    latlon = np.load(ys.FEATURES / "latlon" / f"{args.subset}.npy", mmap_mode="r")
    _, _, dates, _ = slot_months(args.subset, fields)
    sow = sowing_dates()

    rel = np.full((len(fields), ys.N_STEPS), np.nan)
    for i, f in enumerate(fields):
        rel[i] = [(d - sow[f]).days if d is not None else np.nan for d in dates[f]]
    last = np.nanmax(rel, axis=1)
    end = int(min(MAX_END, STEP * np.ceil(np.nanpercentile(last, 95) / STEP)))
    grid = np.arange(GRID_START, end + 1, STEP)
    feature_names = [f"{v}_d{g:+04d}" for g in grid for v in VARIABLES.values()]

    tmp = out_dir / f".{args.subset}.npy.tmp"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(len(fidx), len(feature_names)))
    cos_view = np.cos(np.radians(VIEW_ZENITH_DEG))
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        f = fields[fidx[a]]
        slots = np.flatnonzero(np.isfinite(rel[fidx[a]]))
        refl = np.asarray(cache[a:b][:, cols[slots]], np.float64) / 10000.0  # (pixels, slots, 8)
        lat = float(np.mean(latlon[a:b, 0]))
        per_obs = np.full((b - a, len(slots), len(VARIABLES)), np.nan, np.float32)
        for k, s in enumerate(slots):
            x = np.column_stack([refl[:, k], np.full(b - a, cos_view),
                                 np.full(b - a, max(sun_zenith_cos(lat, dates[f][s]), 0.05)),
                                 np.full(b - a, COS_RELATIVE_AZIMUTH)])
            ok = np.isfinite(x).all(1)
            for v, net in enumerate(nets.values()):
                per_obs[ok, k, v] = network(net, x[ok])
        out[a:b] = interpolate(per_obs, rel[fidx[a]][slots], grid).reshape(b - a, -1)
    out.flush()
    del out
    tmp.rename(target)

    arr = np.load(target, mmap_mode="r")
    lai = arr[:, 0::len(VARIABLES)]
    info = {"subset": args.subset, "kind": "dense", "rows": int(len(fidx)), "feature_names": feature_names,
            "variables": VARIABLES, "grid_days_since_sowing": grid.tolist(),
            "median_lai_per_step": [round(float(x), 2) for x in np.nanmedian(lai[::50], axis=0)],
            "snap_test_case_max_abs_error": test_error,
            "geometry": {"view_zenith_deg": VIEW_ZENITH_DEG, "cos_relative_azimuth": COS_RELATIVE_AZIMUTH,
                         "sun": f"computed at {LOCAL_SOLAR_HOUR} h local solar time"},
            "source": f"senbox-org/s2tbx@{COMMIT} auxdata 3_0/S2A (GPL-3.0)",
            "aligned_with": f"v1/{args.subset}-meta.npz",
            "v1_meta_sha256": ys.sha256(ys.FEATURES / "v1" / f"{args.subset}-meta.npz"),
            "seconds": round(time.time() - t0, 1)}
    (out_dir / f"{args.subset}.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: info[k] for k in ("subset", "rows", "grid_days_since_sowing", "median_lai_per_step",
                                           "snap_test_case_max_abs_error", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
