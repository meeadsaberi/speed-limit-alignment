#!/usr/bin/env python3
"""
Speed Safety Atlas — enrichment layer (Task 4): LandUse fallback + OSM VRU proxy.
UNSW Sydney · ADB AI for Safer Roads 2026 · SSSR method

Two enrichments that reduce reliance on the sparse provider fields and feed the
tier-2 (OSM-proxy) rung of the confidence gradient where imagery is absent:

  1. LandUse fallback (~75% NA in both regions):
       a. Maharashtra: use UrbanPC where present (>=50% -> URBAN).
       b. Otherwise: impute from the nearest LABELLED section centroid within a
          max distance (network-proximity imputation, handoff 6.9). Records
          land_use_src in {provided, urbanpc, imputed, none} -> lowers confidence.

  2. OSM VRU-exposure proxy (Overpass, no key): counts of pedestrian-generating
     POIs (school, marketplace, hospital, college, place_of_worship, bus_station)
     within ~150 m of each section. Cached to data/osm_pois_<region>.gpkg so
     re-runs don't re-hit Overpass. Adds osm_poi_count + osm_vru_proxy (bool).

Input : data/harmonised.gpkg (layer 'sections')
Output: data/enriched.gpkg (layer 'sections')
"""
from __future__ import annotations
import os, time, json, urllib.request, urllib.parse
import numpy as np
import pandas as pd
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

# metric CRS per region for accurate buffering (UTM)
UTM = {"thailand": 32647, "maharashtra": 32643}

# POI types that generate roadside pedestrian / VRU activity
OSM_POI_QUERY = """
[out:json][timeout:180];
(
  node["amenity"~"^(school|college|university|marketplace|hospital|place_of_worship|bus_station)$"]({bbox});
  way["amenity"~"^(school|college|university|marketplace|hospital|place_of_worship|bus_station)$"]({bbox});
  node["shop"="mall"]({bbox});
  node["highway"="bus_stop"]({bbox});
);
out center;
"""
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
POI_BUFFER_M = 150.0


def fetch_osm_pois(region: str, bbox_lonlat) -> gpd.GeoDataFrame:
    """bbox_lonlat = (minlon,minlat,maxlon,maxlat). Overpass wants S,W,N,E."""
    cache = os.path.join(DATA, f"osm_pois_{region}.gpkg")
    if os.path.exists(cache):
        print(f"  [cache] OSM POIs for {region}: {cache}")
        return gpd.read_file(cache)

    minlon, minlat, maxlon, maxlat = bbox_lonlat
    q = OSM_POI_QUERY.format(bbox=f"{minlat},{minlon},{maxlat},{maxlon}")
    last = None
    for ep in OVERPASS_ENDPOINTS:
        for attempt in range(2):
            try:
                print(f"  [overpass] {region} via {ep} (try {attempt+1}) ...")
                data = urllib.parse.urlencode({"data": q}).encode()
                req = urllib.request.Request(ep, data=data,
                        headers={"User-Agent": "UNSW Sydney-ADB-SSSR/1.0"})
                with urllib.request.urlopen(req, timeout=200) as r:
                    js = json.loads(r.read())
                els = js.get("elements", [])
                pts, kinds = [], []
                from shapely.geometry import Point
                for e in els:
                    if e["type"] == "node":
                        lon, lat = e.get("lon"), e.get("lat")
                    else:
                        c = e.get("center") or {}
                        lon, lat = c.get("lon"), c.get("lat")
                    if lon is None:
                        continue
                    tags = e.get("tags", {})
                    pts.append(Point(lon, lat))
                    kinds.append(tags.get("amenity") or tags.get("shop")
                                 or tags.get("highway") or "poi")
                gdf = gpd.GeoDataFrame({"kind": kinds}, geometry=pts, crs=4326)
                gdf.to_file(cache, driver="GPKG")
                print(f"    -> {len(gdf)} POIs cached")
                return gdf
            except Exception as e:
                last = e
                time.sleep(3 * (attempt + 1))
    print(f"  [warn] Overpass failed for {region}: {last}. Proceeding without OSM POIs.")
    return gpd.GeoDataFrame({"kind": []}, geometry=[], crs=4326)


def landuse_fallback(sub: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fill land_use where NA; record provenance in land_use_src."""
    sub = sub.copy()
    sub["land_use_src"] = np.where(sub["land_use"].notna(), "provided", "none")

    # (a) Maharashtra UrbanPC
    if "urban_pc" in sub.columns:
        m = sub["land_use"].isna() & sub["urban_pc"].notna()
        sub.loc[m, "land_use"] = np.where(sub.loc[m, "urban_pc"] >= 50, "URBAN", "RURAL")
        sub.loc[m, "land_use_src"] = "urbanpc"

    # (b) nearest labelled centroid within max distance (metric CRS)
    need = sub["land_use"].isna()
    if need.any() and (~need).any():
        utm = UTM[sub["region"].iloc[0]]
        cent = sub.geometry.representative_point()
        lab = gpd.GeoDataFrame(
            {"land_use_lab": sub.loc[~need, "land_use"].values},
            geometry=cent[~need].values, crs=sub.crs).to_crs(utm)
        unl = gpd.GeoDataFrame(
            {"idx": sub.index[need]},
            geometry=cent[need].values, crs=sub.crs).to_crs(utm)
        joined = gpd.sjoin_nearest(unl, lab, how="left",
                                   max_distance=5000, distance_col="d")
        joined = joined[~joined.index.duplicated(keep="first")]
        fill = joined.set_index("idx")["land_use_lab"]
        sub.loc[fill.index, "land_use"] = fill.values
        sub.loc[fill.dropna().index, "land_use_src"] = "imputed"
    return sub


def osm_proxy(sub: gpd.GeoDataFrame, pois: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Count POIs within POI_BUFFER_M of each section (metric CRS)."""
    sub = sub.copy()
    sub["osm_poi_count"] = 0
    if len(pois) == 0:
        sub["osm_vru_proxy"] = False
        return sub
    utm = UTM[sub["region"].iloc[0]]
    seg = sub[["uid", "geometry"]].to_crs(utm)
    seg["geometry"] = seg.buffer(POI_BUFFER_M)
    p = pois.to_crs(utm)
    j = gpd.sjoin(p, seg, how="inner", predicate="within")
    counts = j.groupby("uid").size()
    sub = sub.merge(counts.rename("poi_n"), left_on="uid", right_index=True, how="left")
    sub["osm_poi_count"] = sub["poi_n"].fillna(0).astype(int)
    sub.drop(columns=["poi_n"], inplace=True, errors="ignore")
    sub["osm_vru_proxy"] = sub["osm_poi_count"] > 0
    return sub


def main():
    src = os.path.join(DATA, "harmonised.gpkg")
    g = gpd.read_file(src, layer="sections")
    print(f"loaded {len(g)} sections")

    out = []
    for region, sub in g.groupby("region"):
        print(f"\n===== enrich {region} (n={len(sub)}) =====")
        sub = landuse_fallback(sub)
        bbox = tuple(sub.total_bounds)  # minlon,minlat,maxlon,maxlat
        pois = fetch_osm_pois(region, bbox)
        sub = osm_proxy(sub, pois)

        print("  land_use_src:", dict(sub["land_use_src"].value_counts()))
        lu = sub["land_use"].notna().mean()
        print(f"  land_use coverage after fallback: {100*lu:.1f}%")
        print(f"  sections with OSM VRU POIs <=150m: "
              f"{int(sub['osm_vru_proxy'].sum())} ({100*sub['osm_vru_proxy'].mean():.1f}%)")
        out.append(sub)

    res = gpd.GeoDataFrame(pd.concat(out, ignore_index=True), crs=g.crs)
    dst = os.path.join(DATA, "enriched.gpkg")
    save = res.copy()
    save["streetview"] = save["streetview"].astype("string")
    save.to_file(dst, layer="sections", driver="GPKG")
    print(f"\nwrote {dst} (layer 'sections', {len(save)} rows)")


if __name__ == "__main__":
    main()
