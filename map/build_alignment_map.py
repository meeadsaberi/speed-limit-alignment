#!/usr/bin/env python3
"""Prep the interactive Speed-Limit-Alignment map: split per city + copy VLM evidence photos.
UNSW Sydney · ADB SSSR."""
import os, json, shutil
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "score", "limit_alignment.geojson")
OUT = os.path.join(HERE, "data"); IMG_OUT = os.path.join(HERE, "img")
VLM_IMG = os.path.join(ROOT, "data", "vlm_images")


def main():
    os.makedirs(OUT, exist_ok=True); os.makedirs(IMG_OUT, exist_ok=True)
    g = gpd.read_file(SRC)
    copied = 0
    for region, sub in g.groupby("region"):
        dst = os.path.join(OUT, f"align_{region}.geojson")
        sub.to_file(dst, driver="GeoJSON", COORDINATE_PRECISION=5)
        print(f"  {region}: {len(sub)} links -> {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
        for fn in sub["vlm_img"].dropna().unique():
            s = os.path.join(VLM_IMG, fn)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(IMG_OUT, fn)); copied += 1
    print(f"copied {copied} evidence photos -> map/img/")


if __name__ == "__main__":
    main()
