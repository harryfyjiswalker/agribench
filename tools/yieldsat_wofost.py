"""Shared WOFOST (PCSE) set-up for the YieldSAT crop-model inputs `wofost` and `wofost-da` [AgriBench].

Runs in the layered PCSE environment (tools/setup_pcse_env.sh: PCSE 6.0.13, WOFOST 7.2 crop parameters from
ajwdewit/WOFOST_crop_parameters, branch wofost72, pinned commit). Per field:

* Weather: the field's daily ERA5 file through `daily_weather` (tools/build_yieldsat_weather_pheno.py), read from
  60 days before sowing up to the subset-wide cap (same leakage guard as `weather-pheno`: the record ends exactly 60
  days after harvest). Radiation by Hargreaves (0.16 sqrt(Tmax - Tmin) Ra), vapour pressure e°(Tmin) (dew point =
  Tmin, FAO-56), wind 2 m/s; E0/ES0/ET0 by PCSE's own Penman / Penman-Monteith from these. Only Germany and
  Uruguay have radiation, humidity and wind columns, so every country uses the same temperature-based estimates.
* Soil: from the SoilGrids layers in the v1 cache (the `s2-adm` static bands), field mean over pixels,
  thickness-weighted over 0-100 cm: sand, clay, organic carbon and coarse fragments -> wilting point, field
  capacity, saturation and saturated conductivity by the Saxton & Rawls (2006) pedotransfer functions (organic
  matter = 1.724 x SOC, capped at 8 %), water contents scaled by (1 - coarse fragments). Rooting depth 100 cm,
  CRAIRC 0.06. Initial available water: half of the root zone's capacity 60 days before sowing.
* Crop: one default WOFOST 7.2 variety per crop (Winter_wheat_101, Grain_maize_201, Soybean_901,
  Oilseed_rape_1001; rapeseed has no southern-hemisphere variety and none is needed). These are European
  calibrations, so phenology is replaced by the thermal-time constants of `weather-pheno`: temperature-only
  development (IDSL = 0), DTSMTB linear from the crop's base temperature to 30 °C, TSUM1 = flowering thermal time -
  TSUMEM, TSUM2 = maturity - flowering. Everything else (photosynthesis, partitioning, leaf life span) stays default.
  Absolute yields are therefore not calibrated; the features are meant as relative, year-sensitive indices.
* Season: sowing date from the metadata; the crop ends at maturity or at the subset cap, whichever is first.
"""
import datetime as dt
import json

import numpy as np

from tools import yieldsat as ys
from tools.build_yieldsat_s2_gdd import BASE_CAP
from tools.build_yieldsat_weather_pheno import PRE_DAYS, TT_FLOWER, TT_MATURITY, daily_weather

PARAMS_DIR = ys.PROJECT / "upstream" / "wofost-crop-parameters"
CROP_PARAMS_COMMIT = "fc6eda46ea0081eb0ad31fa92ea49fea60e11075"   # branch wofost72
PCSE_VERSION = "6.0.13"
PCSE_PYTHON = ys.PROJECT / "envs" / "pcse" / "bin" / "python"
VARIETY = {"wheat": ("wheat", "Winter_wheat_101"), "corn": ("maize", "Grain_maize_201"),
           "soybean": ("soybean", "Soybean_901"), "rapeseed": ("rapeseed", "Oilseed_rape_1001")}
DEPTHS = [("0-5", 5), ("5-15", 10), ("15-30", 15), ("30-60", 30), ("60-100", 40)]
ROOT_DEPTH_CM = 100.0
WIND_MS = 2.0
ANGSTROM = (0.25, 0.50)

_CROPDATA = None


def cropdata():
    """One YAMLCropDataProvider per process (parsing the YAML files takes a few seconds)."""
    global _CROPDATA
    if _CROPDATA is None:
        from pcse.input import YAMLCropDataProvider
        _CROPDATA = YAMLCropDataProvider(fpath=str(PARAMS_DIR), force_reload=True)
    return _CROPDATA


# ---------------------------------------------------------------- soil

def saxton_rawls(sand, clay, om):
    """Saxton & Rawls (2006): sand, clay as fractions, organic matter in % -> (theta_wp, theta_fc, theta_sat, ksat
    in cm/day)."""
    S, C, OM = sand, clay, min(om, 8.0)
    t1500t = -0.024 * S + 0.487 * C + 0.006 * OM + 0.005 * S * OM - 0.013 * C * OM + 0.068 * S * C + 0.031
    wp = t1500t + (0.14 * t1500t - 0.02)
    t33t = -0.251 * S + 0.195 * C + 0.011 * OM + 0.006 * S * OM - 0.027 * C * OM + 0.452 * S * C + 0.299
    fc = t33t + (1.283 * t33t ** 2 - 0.374 * t33t - 0.015)
    ts33t = 0.278 * S + 0.034 * C + 0.022 * OM - 0.018 * S * OM - 0.027 * C * OM - 0.584 * S * C + 0.078
    ts33 = ts33t + (0.636 * ts33t - 0.107)
    sat = fc + ts33 - 0.097 * S + 0.043
    B = (np.log(1500) - np.log(33)) / (np.log(fc) - np.log(wp))
    ksat_mm_h = 1930 * max(sat - fc, 1e-3) ** (3 - 1 / B)
    return wp, fc, sat, ksat_mm_h * 2.4


def field_soils(subset):
    """field index -> dict of soil properties (field mean over pixels of the SoilGrids layers in the v1 cache)."""
    names = json.loads((ys.FEATURES / "v1" / f"{subset}.json").read_text())["feature_names"]
    meta = np.load(ys.FEATURES / "v1" / f"{subset}-meta.npz")
    fidx = meta["field_idx"]
    bounds = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1, len(fidx)]
    cache = np.load(ys.FEATURES / "v1" / f"{subset}.npy", mmap_mode="r")
    props = ["sand", "clay", "soc", "cfvo"]
    cols = [names.index(f"{p}_{d}") for p in props for d, _ in DEPTHS] + [names.index("dem")]
    w = np.array([t for _, t in DEPTHS], float)
    w /= w.sum()
    out = {}
    for j in range(len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        step = max(1, (b - a) // 400)                     # up to ~400 pixels per field is plenty for a mean
        v = np.nanmean(np.asarray(cache[a:b:step][:, cols], np.float64), axis=0)
        layer = {p: v[k * len(DEPTHS):(k + 1) * len(DEPTHS)] for k, p in enumerate(props)}
        # SoilGrids units: sand/clay g/kg, SOC dg/kg, coarse fragments cm3/dm3.
        sand, clay = np.nansum(layer["sand"] * w) / 1000, np.nansum(layer["clay"] * w) / 1000
        soc_pct = np.nansum(layer["soc"] * w) / 100
        gravel = np.nansum(layer["cfvo"] * w) / 1000
        if not np.isfinite(sand + clay) or sand + clay == 0:
            sand, clay, soc_pct, gravel = 0.40, 0.20, 1.0, 0.0   # generic loam where SoilGrids is missing
        wp, fc, sat, ksat = saxton_rawls(sand, clay, 1.724 * soc_pct)
        keep = 1 - min(gravel, 0.5)
        out[int(fidx[a])] = {"sand": sand, "clay": clay, "soc_pct": soc_pct, "gravel": gravel,
                             "SMW": wp * keep, "SMFCF": fc * keep, "SM0": sat * keep, "ksat_cm_d": ksat,
                             "elev": float(v[-1]) if np.isfinite(v[-1]) else 0.0}
    return out


def soil_params(s):
    k = float(np.clip(s["ksat_cm_d"], 1.0, 100.0))
    return {"SMW": s["SMW"], "SMFCF": s["SMFCF"], "SM0": max(s["SM0"], s["SMFCF"] + 0.02), "CRAIRC": 0.06,
            "SOPE": k, "KSUB": k, "K0": k, "RDMSOL": ROOT_DEPTH_CM}


# ---------------------------------------------------------------- weather

def weather_provider(w, lat, lon, elev):
    """PCSE weather provider from a daily_weather() frame."""
    from pcse.base import WeatherDataContainer, WeatherDataProvider
    from pcse.util import reference_ET

    class Provider(WeatherDataProvider):
        def __init__(self):
            super().__init__()
            self.description = ["YieldSAT ERA5 (Raw weather file), Hargreaves radiation"]
            for day, r in zip(w.index.date, w.itertuples()):
                irrad = float(r.rs) * 1e6
                vap = 6.108 * np.exp(17.27 * r.tmin / (r.tmin + 237.3))       # hPa, dew point = Tmin
                e0, es0, et0 = reference_ET(day, lat, elev, r.tmin, r.tmax, irrad, vap, WIND_MS, *ANGSTROM)
                wdc = WeatherDataContainer(LAT=lat, LON=lon, ELEV=elev, DAY=day, IRRAD=irrad, TMIN=float(r.tmin),
                                           TMAX=float(r.tmax), VAP=float(vap), RAIN=float(r.prec) / 10.0,
                                           E0=e0 / 10.0, ES0=es0 / 10.0, ET0=et0 / 10.0, WIND=WIND_MS)
                self._store_WeatherDataContainer(wdc, day)
            self.first_date, self.last_date = w.index[0].date(), w.index[-1].date()

    return Provider()


# ---------------------------------------------------------------- set-up and run

def phenology_overrides(crop, params):
    base, cap = BASE_CAP[crop]
    tsumem = float(params["TSUMEM"])
    return {"IDSL": 0, "DTSMTB": [0.0, 0.0, base, 0.0, cap, cap - base, 45.0, cap - base],
            "TSUM1": max(100.0, TT_FLOWER[crop] - tsumem), "TSUM2": TT_MATURITY[crop] - TT_FLOWER[crop]}


def agromanagement(crop, sow, cap_days):
    name, variety = VARIETY[crop]
    start = sow - dt.timedelta(days=PRE_DAYS)
    end = sow + dt.timedelta(days=cap_days)
    return [{start: {"CropCalendar": {"crop_name": name, "variety_name": variety, "crop_start_date": sow,
                                      "crop_start_type": "sowing", "crop_end_date": end, "crop_end_type": "earliest",
                                      "max_duration": cap_days + 1},
                     "TimedEvents": None, "StateEvents": None}}]


class FieldSetup:
    """Everything needed to (re)run WOFOST for one field."""

    def __init__(self, field, country, crop, sow, lat, lon, soil, cap_days):
        self.field, self.crop, self.sow, self.cap = field, crop, sow, cap_days
        w = daily_weather(field, country, lat, sow - dt.timedelta(days=PRE_DAYS), sow + dt.timedelta(days=cap_days))
        self.weather = weather_provider(w, lat, lon, soil["elev"])
        self.soil = soil_params(soil)
        self.wav = 0.5 * (self.soil["SMFCF"] - self.soil["SMW"]) * ROOT_DEPTH_CM
        self.agro = agromanagement(crop, sow, cap_days)

    def parameters(self, overrides=None):
        from pcse.base import ParameterProvider
        from pcse.input import WOFOST72SiteDataProvider
        cd = cropdata()
        name, variety = VARIETY[self.crop]
        cd.set_active_crop(name, variety)
        site = WOFOST72SiteDataProvider(WAV=float(np.clip(self.wav, 0, 100)))
        p = ParameterProvider(cropdata=cd, soildata=dict(self.soil), sitedata=site)
        for k, v in phenology_overrides(self.crop, cd).items():
            p.set_override(k, v, check=False)
        for k, v in (overrides or {}).items():
            p.set_override(k, v, check=False)
        return p

    def model(self, kind="WLP", overrides=None):
        from pcse.models import Wofost72_PP, Wofost72_WLP_CWB
        cls = {"PP": Wofost72_PP, "WLP": Wofost72_WLP_CWB}[kind]
        return cls(self.parameters(overrides), self.weather, self.agro)


def days_since_sowing(d, sow, cap):
    return float((d - sow).days) if isinstance(d, dt.date) else float(cap)


def summarise(model, sow, cap, prefix):
    """Season features from a finished run."""
    s = (model.get_summary_output() or [{}])[-1]
    daily = model.get_output()
    lai = np.array([r["LAI"] if r.get("LAI") is not None else 0.0 for r in daily], float)
    doa = s.get("DOA")
    lai_at_flower = np.nan
    if isinstance(doa, dt.date):
        k = [i for i, r in enumerate(daily) if r["day"] == doa]
        lai_at_flower = float(lai[k[0]]) if k else np.nan
    g = lambda key: float(s[key]) if s.get(key) is not None else np.nan
    return {f"{prefix}_twso": g("TWSO"), f"{prefix}_tagp": g("TAGP"), f"{prefix}_laimax": float(lai.max()),
            f"{prefix}_lai_flower": lai_at_flower, f"{prefix}_ctrat": g("CTRAT"), f"{prefix}_dvs_end": g("DVS"),
            "_doa": days_since_sowing(doa, sow, cap), "_dom": days_since_sowing(s.get("DOM"), sow, cap)}
