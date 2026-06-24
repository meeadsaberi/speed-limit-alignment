#!/usr/bin/env python3
"""
Speed Safety Atlas — load + harmonisation layer (Deliverable 1, foundation).
UNSW Sydney · ADB AI for Safer Roads 2026 · SSSR method

Reads the two regional networks (geometry + attributes from the .geojson, which is
self-contained) into a single canonical GeoDataFrame, harmonising the cross-region
schema differences and applying the data-quality rules discovered by profiling:

  * Thailand SpeedLimit is FLOAT and noisy; 412 rows are 0.0 -> treated as missing.
  * Maharashtra SpeedLimit is TEXT ('20'..'80') -> cast to float.
  * Maharashtra carries provider QA flags Pass / ExcludeFromSpeedSPI -> respected.
  * LandUse is ~75% missing in both regions -> coverage flag drives later fallback.
  * StreetImageLink is 'lon,lat,lon,lat' (section start & end) -> parsed to coords.

Auxiliary boundary tables (helmet SPI) are read from the .gpkg for later validation.

Output: data/harmonised.gpkg (layer 'sections') + a printed coverage report that
cross-checks against the standalone profiling numbers.
"""
from __future__ import annotations
import os, glob, warnings
import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", message=".*Sequential read of iterator.*")

# --- paths -------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET = os.path.join(ROOT, "AI for Safer Roads 2026 - Dataset")
ARCHIVE = os.path.join(DATASET, "Archive")
OUT = os.path.join(HERE, "..", "data")

REGIONS = {
    "thailand": {
        "geojson": "ADB_Innovation_Thailand.geojson",
        "gpkg_pat": "*Thailand*(Feature).gpkg",
        "helmet_layer": "Thailand_Province_Boundaries",
        "name_col": "english_ro",
    },
    "maharashtra": {
        "geojson": "ADB_Innovation_Maharashtra.geojson",
        "gpkg_pat": "*Maharashtra*(Feature).gpkg",
        "helmet_layer": "Boundaries_4helmet",
        "name_col": "names_primary",
    },
}

# canonical field -> source column (per region). None = not present in that region.
FIELD_MAP = {
    "section_id":     {"thailand": "OBJECTID",     "maharashtra": "OBJECTID"},
    "name":           {"thailand": "english_ro",   "maharashtra": "names_primary"},
    "road_class":     {"thailand": "RoadClass",    "maharashtra": "RoadClass"},
    "land_use":       {"thailand": "LandUse",      "maharashtra": "LandUse"},
    "speed_limit":    {"thailand": "SpeedLimit",   "maharashtra": "SpeedLimit"},
    "speed_85":       {"thailand": "F85thPercentileSpeed", "maharashtra": "F85thPercentileSpeed"},
    "speed_median":   {"thailand": "MedianSpeed",  "maharashtra": "MedianSpeed"},
    "pct_over_limit": {"thailand": "PercentOverLimit", "maharashtra": "PercentOverLimit"},
    "traffic_weight": {"thailand": "WeightedSample",   "maharashtra": "WeightedSample"},
    "percentile":     {"thailand": "Percentile",   "maharashtra": "Percentile"},
    "urban_pc":       {"thailand": None,           "maharashtra": "UrbanPC"},
    "exclude_spi":    {"thailand": None,           "maharashtra": "ExcludeFromSpeedSPI"},
    "streetview":     {"thailand": "StreetImageLink", "maharashtra": "StreetImageLink"},
}


def num(x):
    """Robust scalar->float (handles TEXT speed limits, blanks, NaN)."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def parse_streetview(s):
    """'lon,lat,lon,lat' -> [(lon,lat), ...]. Order verified: TH ~103°lon,14°lat."""
    if not s or not isinstance(s, str):
        return []
    p = [num(x) for x in s.split(",")]
    return [(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)
            if p[i] is not None and p[i + 1] is not None]


def load_region(region: str) -> gpd.GeoDataFrame:
    cfg = REGIONS[region]
    f = os.path.join(DATASET, cfg["geojson"])
    gdf = gpd.read_file(f)                      # geometry + all attributes, EPSG:4326
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    out = gpd.GeoDataFrame(geometry=gdf.geometry, crs=gdf.crs)
    for canon, src in FIELD_MAP.items():
        col = src[region]
        out[canon] = gdf[col] if (col and col in gdf.columns) else None

    out["region"] = region

    # --- cleaning rules (from profiling) ---
    # speed limit: cast (handles MH TEXT), and treat TH's 0.0 sentinel as missing
    out["speed_limit"] = out["speed_limit"].map(num)
    out.loc[out["speed_limit"] == 0, "speed_limit"] = np.nan
    for c in ["speed_85", "speed_median", "pct_over_limit", "traffic_weight",
              "percentile", "urban_pc"]:
        out[c] = out[c].map(num)

    # land use -> normalised {URBAN, RURAL, None}
    out["land_use"] = (out["land_use"].astype("string").str.upper()
                       .where(out["land_use"].notna(), None))
    out.loc[~out["land_use"].isin(["URBAN", "RURAL"]), "land_use"] = pd.NA

    # road class normalised lower
    out["road_class"] = out["road_class"].astype("string").str.lower()

    # provider QA: MH flags an explicit exclude set; record it (don't silently drop)
    out["exclude_spi"] = out["exclude_spi"].map(
        lambda v: bool(v) if v is not None and not pd.isna(v) else False)

    # parsed imagery coords + count
    out["sv_coords"] = out["streetview"].map(parse_streetview)
    out["n_sv_pts"] = out["sv_coords"].map(len)

    # --- coverage flags (drive the confidence layer) ---
    out["has_limit"]   = out["speed_limit"].notna()
    out["has_85th"]    = out["speed_85"].notna()
    out["has_landuse"] = out["land_use"].notna()
    out["has_imagery_link"] = out["n_sv_pts"] > 0   # link present (not yet imagery hit)

    # scorable = has a usable posted limit and not provider-excluded
    out["scorable"] = out["has_limit"] & (~out["exclude_spi"])
    return out


def load_helmet(region: str) -> gpd.GeoDataFrame | None:
    """Province/district helmet-SPI boundaries from the .gpkg (validation layer)."""
    cfg = REGIONS[region]
    g = glob.glob(os.path.join(ARCHIVE, cfg["gpkg_pat"]))
    if not g:
        return None
    try:
        return gpd.read_file(g[0], layer=cfg["helmet_layer"])
    except Exception as e:
        print(f"  [warn] could not read helmet layer for {region}: {e}")
        return None


def load_all() -> gpd.GeoDataFrame:
    parts = [load_region(r) for r in REGIONS]
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    # stable global id
    gdf["uid"] = gdf["region"].str[:2] + "-" + gdf["section_id"].astype("Int64").astype(str)
    return gdf


def coverage_report(gdf: gpd.GeoDataFrame) -> None:
    print("\n" + "=" * 64)
    print("HARMONISED COVERAGE REPORT (cross-check vs standalone profiling)")
    print("=" * 64)
    for region, g in gdf.groupby("region"):
        n = len(g)
        print(f"\n----- {region.upper()}  (n={n}) -----")
        for flag, label in [("has_limit", "posted limit (V_post)"),
                            ("has_85th", "85th pct (V_ind)"),
                            ("has_landuse", "land use"),
                            ("has_imagery_link", "imagery link")]:
            c = int(g[flag].sum())
            print(f"  {label:24s} {c:6d} / {n}  = {100*c/n:5.1f}%")
        sc = int(g["scorable"].sum())
        excl = int(g["exclude_spi"].sum())
        print(f"  {'SCORABLE (limit & !excl)':24s} {sc:6d} / {n}  = {100*sc/n:5.1f}%"
              f"   [provider-excluded: {excl}]")
        # credibility signal
        both = g[g["has_limit"] & g["has_85th"]]
        if len(both):
            over = int((both["speed_85"] > both["speed_limit"]).sum())
            print(f"  85th > posted (credibility) {over}/{len(both)} = {100*over/len(both):.1f}%")
        print(f"  land use breakdown: {dict(g['land_use'].value_counts(dropna=False))}")


def main():
    os.makedirs(OUT, exist_ok=True)
    gdf = load_all()
    coverage_report(gdf)

    # persist (drop python-object columns geopackage can't store)
    save = gdf.drop(columns=["sv_coords"]).copy()
    save["streetview"] = save["streetview"].astype("string")
    out_path = os.path.join(OUT, "harmonised.gpkg")
    save.to_file(out_path, layer="sections", driver="GPKG")
    print(f"\nwrote {out_path} (layer 'sections', {len(save)} rows, "
          f"{save['scorable'].sum()} scorable)")

    # also stash helmet validation layers
    for region in REGIONS:
        h = load_helmet(region)
        if h is not None:
            h.to_file(out_path, layer=f"helmet_{region}", driver="GPKG")
            print(f"  + helmet_{region} ({len(h)} boundaries)")


if __name__ == "__main__":
    main()
