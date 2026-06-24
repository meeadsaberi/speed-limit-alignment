#!/usr/bin/env python3
"""
Speed-Limit Alignment — the simple, transparent method (Deliverable: Speed Safety Score).
UNSW Sydney · ADB AI for Safer Roads 2026

Question (challenge): is the POSTED limit appropriate for the road's function + VRU context?
NOT about speeding. Method, fully interpretable by a non-technical official:

  1. Classify each link into a Safe-System CONTEXT class (speed-differentiated). VLM imagery
     characterisation is kept in its OWN columns NEXT TO the data (not overriding); where a
     VLM read exists it sets the context, else a transparent data-prior (road class + land use).
  2. Each context -> a PROPOSED limit from the Safe-System reference table (WHO/OECD-ITF):
       ped_cyclist_mix 30 | urban_vru 50 | intersection 50 | rural_undivided 70 |
       divided_limited_access_no_vru 100.    (Hard function gate: motorways -> 100, never flagged.)
  3. MISALIGNMENT = posted - proposed (km/h the limit exceeds the appropriate level).
  4. Cross-check with the 85th-percentile operating speed (is the over-limit live on the road?)
     and the VLM-read sign (does the posted limit match what's signed?).
  5. RISK FLAG prioritising VRU-context segments where the limit is too high.

Outputs: score/limit_alignment.csv  and  score/limit_alignment.geojson  (all attrs in columns).
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
SIMPLIFY = 0.0004   # ~40 m for web/GIS

PROPOSED = {"ped_cyclist_mix": 30, "urban_vru": 50, "intersection": 50,
            "rural_undivided": 70, "divided_limited_access_no_vru": 100}
VRU_CONTEXTS = {"ped_cyclist_mix", "urban_vru", "intersection"}
MARGIN = 10.0       # only flag a material misalignment
# Sanity ceiling: 85th-percentile operating speed above which the assigned class is
# PHYSICALLY IMPLAUSIBLE (a real ped-mix / urban-VRU street cannot sustain these speeds).
# If exceeded, the automated classification (usually the land-use prior) is suspect ->
# flag the link for human verification instead of asserting it as a confident finding.
CLASS_SPEED_CEILING = {"ped_cyclist_mix": 60, "urban_vru": 80,
                       "intersection": 80, "rural_undivided": 100}


def data_prior_context(road_class, land_use):
    rc = (road_class or "").lower(); lu = (land_use or "").upper()
    if "motorway" in rc:
        return "divided_limited_access_no_vru"
    if lu == "URBAN":
        return "urban_vru"
    if lu == "RURAL":
        return "rural_undivided"
    return "rural_undivided"          # unknown land use -> neutral default (70)


def main():
    g = gpd.read_file(os.path.join(DATA, "enriched.gpkg"), layer="sections")
    # merge VLM characterisation (kept side-by-side, vlm_ prefix)
    vp = os.path.join(DATA, "vlm_context.parquet")
    if os.path.exists(vp):
        v = pd.read_parquet(vp)
        v = v[v["n_images"] > 0].rename(columns={
            "context_vlm": "vlm_context", "land_use_vlm": "vlm_land_use",
            "carriageway_vlm": "vlm_carriageway", "vru_vlm": "vlm_vru",
            "intersection_vlm": "vlm_intersection", "road_type_vlm": "vlm_road_type",
            "speed_sign_vlm": "vlm_speed_sign", "n_images": "vlm_n_images",
            "reasoning_vlm": "vlm_reasoning"})
        # first saved thumbnail filename per link (for photo evidence in the map popup)
        v["vlm_img"] = [f"{u}__{json.loads(ids)[0]}.jpg" if ids and json.loads(ids) else None
                        for u, ids in zip(v["uid"], v.get("img_ids", pd.Series([None]*len(v))))]
        keep = ["uid", "vlm_context", "vlm_land_use", "vlm_carriageway", "vlm_vru",
                "vlm_intersection", "vlm_road_type", "vlm_speed_sign", "vlm_n_images",
                "vlm_reasoning", "vlm_img"]
        g = g.merge(v[[c for c in keep if c in v.columns]], on="uid", how="left")
    for c in ["vlm_context", "vlm_speed_sign", "vlm_vru"]:
        if c not in g.columns:
            g[c] = None

    # analysis set = links with a posted limit (where misalignment is defined)
    g = g[g["speed_limit"].notna() & (g["speed_limit"] > 0)].copy()

    # CURRENT classification = what the supplied data (road class + land use) implies, always.
    g["context_data_prior"] = [data_prior_context(rc, lu)
                               for rc, lu in zip(g["road_class"], g["land_use"])]
    g["data_prior_limit"] = g["context_data_prior"].map(PROPOSED).astype(float)
    # PROPOSED classification = imagery (VLM) where available, else the data-prior.
    g["context_source"] = np.where(g["vlm_context"].notna(), "imagery_vlm", "data_prior")
    g["context_final"] = [vc if isinstance(vc, str) and vc else dp
                          for vc, dp in zip(g["vlm_context"], g["context_data_prior"])]
    # safe_system_max = the Safe-System-appropriate MAXIMUM (a ceiling, not a target).
    g["safe_system_max"] = g["context_final"].map(PROPOSED).astype(float)
    posted = pd.to_numeric(g["speed_limit"], errors="coerce")
    v85 = pd.to_numeric(g["speed_85"], errors="coerce")
    # recommended_limit: the method only ever RECOMMENDS REDUCTIONS, never increases.
    g["recommended_limit"] = np.minimum(posted, g["safe_system_max"])
    g["misalignment"] = (posted - g["recommended_limit"]).round(0)         # km/h reduction recommended (>=0)
    g["operating_vs_recommended"] = (v85 - g["recommended_limit"]).round(0)  # is it live on the road?
    # sign cross-check (where VLM read a sign)
    sign = pd.to_numeric(g["vlm_speed_sign"], errors="coerce")
    g["sign_vs_posted"] = (sign - posted).round(0)

    # RISK FLAG: prioritise VRU-context links where the limit is materially too high.
    is_vru = g["context_final"].isin(VRU_CONTEXTS)
    too_high = g["misalignment"] >= MARGIN
    live = (v85 > g["recommended_limit"]) | v85.isna()   # operating speed confirms (or unknown)
    g["risk_flag"] = np.select(
        [too_high & is_vru & live,                    # limit too high, VRUs, and live
         too_high & is_vru,                           # too high, VRUs, but operating speed already low
         too_high],                                   # too high but non-VRU context
        ["High", "Medium", "Low"], default="Aligned")
    g["risk_reason"] = np.select(
        [g["risk_flag"] == "High",
         g["risk_flag"] == "Medium",
         g["risk_flag"] == "Low"],
        ["limit exceeds appropriate level on a VRU road; operating speed confirms",
         "limit exceeds appropriate level on a VRU road; operating speed currently low",
         "limit above appropriate level (non-VRU context)"],
        default="posted limit within appropriate band")

    # SANITY CHECK: is the operating speed physically plausible for the assigned class?
    # A genuine VRU street cannot carry traffic at highway speeds. The right response depends
    # on whether we have imagery to VERIFY the class:
    #   - data_prior (no imagery): the class is a weak land-use guess -> flag NEEDS REVIEW.
    #   - imagery_vlm: the VLM has independently confirmed the VRU/intersection context, so a
    #     high operating speed is a GENUINE high-speed/high-conflict mismatch, not an error.
    ceil = g["context_final"].map(CLASS_SPEED_CEILING)
    g["class_speed_ceiling"] = ceil
    tension = is_vru & ceil.notna() & v85.notna() & (v85 >= ceil)   # speed contradicts the class
    is_prior = g["context_source"] == "data_prior"
    needs_review = tension & is_prior                              # unverified class -> review
    img_confirmed = tension & ~is_prior                           # imagery-confirmed -> genuine
    g["needs_review"] = np.where(needs_review, "Yes", "No")
    g["imagery_confirmed_conflict"] = np.where(img_confirmed, "Yes", "No")
    g["review_reason"] = np.select(
        [needs_review, img_confirmed],
        [("operating 85th=" + v85.round(0).astype("Int64").astype(str)
          + " km/h implausible for '" + g["context_final"].astype(str)
          + "' from land-use only; no imagery — verify class on site/imagery"),
         ("imagery confirms '" + g["context_final"].astype(str)
          + "' yet 85th=" + v85.round(0).astype("Int64").astype(str)
          + " km/h: genuine high-speed VRU/junction conflict — review for design or limit")],
        default="")
    # confidence-aware headline: only UNVERIFIED tension is demoted to "(verify)"
    g["risk_flag_reviewed"] = np.where(
        (g["risk_flag"].isin(["High", "Medium"])) & needs_review,
        g["risk_flag"] + " (verify)", g["risk_flag"])

    cols = ["uid", "region", "road_class", "land_use", "land_use_src",
            "speed_limit", "speed_85", "speed_median", "pct_over_limit",
            "vlm_context", "vlm_land_use", "vlm_carriageway", "vlm_vru",
            "vlm_intersection", "vlm_road_type", "vlm_speed_sign", "vlm_n_images",
            "context_data_prior", "data_prior_limit",
            "context_final", "context_source", "safe_system_max", "recommended_limit",
            "misalignment", "operating_vs_recommended", "sign_vs_posted",
            "class_speed_ceiling", "needs_review", "imagery_confirmed_conflict", "review_reason",
            "risk_flag", "risk_flag_reviewed", "risk_reason", "vlm_img"]
    cols = [c for c in cols if c in g.columns]
    out = g[cols + ["geometry"]].copy()
    out["geometry"] = out.geometry.simplify(SIMPLIFY, preserve_topology=True)

    out.drop(columns="geometry").to_csv(os.path.join(HERE, "limit_alignment.csv"), index=False)
    out.to_file(os.path.join(HERE, "limit_alignment.geojson"), driver="GeoJSON", COORDINATE_PRECISION=5)

    # summary
    print(f"wrote limit_alignment.csv / .geojson  ({len(out)} links with a posted limit)")
    print(f"  context evidence: {dict(out['context_source'].value_counts())}")
    for region, s in out.groupby("region"):
        n = len(s); fl = dict(s["risk_flag"].value_counts())
        print(f"\n  {region} (n={n}):  flags {fl}")
        hi = s[s["risk_flag"] == "High"]
        hi_ok = hi[hi["needs_review"] == "No"]; hi_rev = hi[hi["needs_review"] == "Yes"]
        print(f"    High-risk: {len(hi)} ({100*len(hi)/n:.0f}%)  ->  {len(hi_ok)} confirmed, "
              f"{len(hi_rev)} NEED CLASSIFICATION REVIEW (operating speed implausible for class)")
        print(f"    median misalignment where >0: {s.loc[s.misalignment>0,'misalignment'].median():.0f} km/h")
        nr = (s["needs_review"] == "Yes").sum()
        ic = (s["imagery_confirmed_conflict"] == "Yes").sum()
        print(f"    needs review (unverified class, no imagery): {nr} ({100*nr/n:.0f}%)")
        print(f"    imagery-CONFIRMED high-speed VRU/junction conflict (genuine): {ic}")
    # sign cross-check coverage
    sc = out["sign_vs_posted"].notna().sum()
    if sc:
        disc = (out["sign_vs_posted"].abs() >= 10).sum()
        print(f"\n  VLM read a limit sign on {sc} links; {disc} disagree with posted by >=10 km/h")


if __name__ == "__main__":
    main()
