#!/usr/bin/env python3
"""
V-RoAst characterisation on a RANDOM, DENSIFIABLE sample of the network.
UNSW Sydney · ADB AI for Safer Roads 2026

- Draws a random 10% of links per city into a recorded sample frame (data/sample_frame.parquet,
  with batch number + location), so we can sequentially add another 10% if coverage is sparse.
- Per sampled segment: fetch up to 2 FLAT Mapillary images along it, classify each with
  gpt-5.5, and AGGREGATE to the MOST-VULNERABLE Safe-System context (Safe-System-conservative).
- Resumable + checkpointed (data/vlm_context.parquet): re-running continues; `densify` adds a batch.

Usage:
  python model/vlm_sample.py            # draw batch 1 if none, then process pending
  python model/vlm_sample.py densify    # draw the next 10% batch, then process
"""
from __future__ import annotations
import os, sys, io, json, base64, math, time, threading, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import geopandas as gpd

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
FRAME = os.path.join(DATA, "sample_frame.parquet")
OUTP = os.path.join(DATA, "vlm_context.parquet")
IMG_DIR = os.path.join(DATA, "vlm_images")   # saved thumbnails: {uid}__{image_id}.jpg
MODEL = os.environ.get("VLM_MODEL", "gpt-5.5")
FRAC = float(os.environ.get("VLM_FRAC", "0.10"))
N_IMAGES = int(os.environ.get("VLM_NIMAGES", "2"))
REGION_FILTER = os.environ.get("VLM_REGION", "")   # "" = all regions; else only this one
WORKERS = 6
SEED = 42

env = {}
for line in open(os.path.join(ROOT, ".env")):
    s = line.strip()
    if s and not s.startswith("#") and "=" in s:
        k, v = s.split("=", 1); env[k] = v
MLY, OAI = env["MAPILLARY_ACCESS_TOKEN"], env["OPENAI_API_KEY"]

PROPOSED_LIMIT = {"ped_cyclist_mix": 30, "urban_vru": 50, "intersection": 50,
                  "rural_undivided": 70, "divided_limited_access_no_vru": 100}

PROMPT = """You are a road-safety assessor applying Safe System speed-limit principles.
Classify this street-level photo of a road in {country} for setting a speed limit. JSON only:
{{"land_use":"urban_builtup|suburban|rural_open","carriageway":"divided|undivided|unclear",
"vru_presence":"high|moderate|none","intersection_visible":true/false,
"road_type":"motorway_expressway|major_arterial|local_road","speed_limit_sign_kmh":null or number,
"safe_system_context":"ped_cyclist_mix|urban_vru|intersection|rural_undivided|divided_limited_access_no_vru",
"reasoning":"one short sentence"}}
ped_cyclist_mix=people walk/cycle mixing with traffic (30); urban_vru=built-up, VRUs likely (50);
intersection=at-grade junction (50); rural_undivided=open road no median (70);
divided_limited_access_no_vru=median/barrier separated, no VRU access e.g. motorway (100-110)."""

_lock = threading.Lock()

# Global Mapillary rate gate. The burst limit is tripped by sustained concurrent requests
# (batch 1 only worked because it ran slowly over hours). Serialise to a safe steady rate.
_mly_gate = threading.Lock()
_mly_next = [0.0]
MLY_MIN_INTERVAL = float(os.environ.get("VLM_MLY_INTERVAL", "0.34"))   # ~3 req/s across all workers


def _mly_throttle():
    with _mly_gate:
        now = time.time()
        wait = _mly_next[0] - now
        if wait > 0:
            time.sleep(wait)
        _mly_next[0] = max(now, _mly_next[0]) + MLY_MIN_INTERVAL


def bbox(lon, lat, d=0.0015):
    dl = d / max(0.1, math.cos(math.radians(lat))); return f"{lon-dl},{lat-d},{lon+dl},{lat+d}"


def mly_images(lon, lat, tries=4):
    """Mapillary images near a point, WITH backoff on rate-limit/5xx so a transient 429
    cannot silently zero out a link (a None return distinguishes 'failed' from 'empty')."""
    q = urllib.parse.urlencode({"access_token": MLY, "fields": "id,is_pano,thumb_1024_url",
                                "bbox": bbox(lon, lat), "limit": 5})
    for a in range(tries):
        try:
            _mly_throttle()
            return json.loads(urllib.request.urlopen(
                f"https://graph.mapillary.com/images?{q}", timeout=20).read()).get("data", [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (a + 1)); continue
            return []                       # genuine 4xx (e.g. nothing here) -> empty
        except Exception:
            time.sleep(1.5 * (a + 1))
    return None                             # exhausted retries -> signal failure, not 'empty'


def flat_images_along(geom, k=N_IMAGES):
    """Up to k distinct flat images sampled along the segment."""
    cs = list(geom.coords) if geom.geom_type == "LineString" else \
         [c for p in geom.geoms for c in p.coords]
    if len(cs) < 2:
        return []
    idx = np.linspace(0, len(cs) - 1, 4).round().astype(int)
    seen, out = set(), []
    for i in idx:
        lon, lat = cs[i]
        ims = mly_images(lon, lat)
        if not ims:                         # None (failed) or [] (none here)
            continue
        flats = [im for im in ims if not im.get("is_pano") and im.get("thumb_1024_url")] or \
                [im for im in ims if im.get("thumb_1024_url")]
        for im in flats:
            if im["id"] not in seen:
                seen.add(im["id"]); out.append(im); break
        if len(out) >= k:
            break
    return out[:k]


def classify(b64, country, tries=4):
    body = json.dumps({"model": MODEL, "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(country=country)},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]}).encode()
    for a in range(tries):
        try:
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {OAI}", "Content-Type": "application/json"})
            j = json.loads(urllib.request.urlopen(req, timeout=120).read())
            return json.loads(j["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(2 * (a + 1)); continue
            raise
        except Exception:
            time.sleep(2 * (a + 1))
    return None


def preflight_openai():
    """Ping OpenAI once before processing so an out-of-quota / bad-key account aborts
    immediately instead of silently fetching imagery it can never classify."""
    body = json.dumps({"model": MODEL, "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Reply JSON {\"ok\":true}"}]}]}).encode()
    try:
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
            headers={"Authorization": f"Bearer {OAI}", "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=60).read()
        return True, "ok"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:
        return False, str(e)[:200]


def process_segment(row):
    country = "Thailand" if row.region == "thailand" else "India"
    imgs = flat_images_along(row.geometry)
    rec = {"uid": row.uid, "region": row.region, "n_images": 0, "img_ids": "[]"}
    if not imgs:
        return rec
    os.makedirs(IMG_DIR, exist_ok=True)
    per = []
    for im in imgs:
        try:
            raw = urllib.request.urlopen(im["thumb_1024_url"], timeout=20).read()
            # save the image alongside its VLM output (auditability)
            try:
                with open(os.path.join(IMG_DIR, f"{row.uid}__{im['id']}.jpg"), "wb") as f:
                    f.write(raw)
            except Exception:
                pass
            res = classify(base64.b64encode(raw).decode(), country)
            if res:
                per.append(res)
        except Exception:
            pass
    if not per:
        return rec
    # aggregate: MOST VULNERABLE (lowest proposed limit) across images
    def lim(r): return PROPOSED_LIMIT.get(r.get("safe_system_context"), 100)
    worst = min(per, key=lim)
    rec.update({
        "n_images": len(per), "img_ids": json.dumps([i["id"] for i in imgs[:len(per)]]),
        "context_vlm": worst.get("safe_system_context"),
        "proposed_limit_vlm": lim(worst),
        "land_use_vlm": worst.get("land_use"),
        "carriageway_vlm": "undivided" if any(p.get("carriageway") == "undivided" for p in per) else worst.get("carriageway"),
        "vru_vlm": ("high" if any(p.get("vru_presence") == "high" for p in per)
                    else "moderate" if any(p.get("vru_presence") == "moderate" for p in per) else "none"),
        "intersection_vlm": any(bool(p.get("intersection_visible")) for p in per),
        "road_type_vlm": worst.get("road_type"),
        "speed_sign_vlm": next((p.get("speed_limit_sign_kmh") for p in per if p.get("speed_limit_sign_kmh")), None),
        "reasoning_vlm": worst.get("reasoning"),
        "contexts_all": json.dumps([p.get("safe_system_context") for p in per]),
    })
    return rec


def draw_batch(g):
    if os.path.exists(FRAME):
        frame = pd.read_parquet(FRAME)
    else:
        frame = pd.DataFrame(columns=["uid", "region", "batch", "lon", "lat"])
    nb = int(frame["batch"].max()) + 1 if len(frame) else 1
    done = set(frame["uid"])
    rng = np.random.RandomState(SEED + nb)
    new = []
    for region, sub in g.groupby("region"):
        if REGION_FILTER and region != REGION_FILTER:   # targeted densify
            continue
        pool = sub[~sub["uid"].isin(done)]
        n = int(round(FRAC * len(sub)))
        pick = pool.sample(min(n, len(pool)), random_state=rng)
        for _, r in pick.iterrows():
            c = r.geometry.representative_point()
            new.append({"uid": r["uid"], "region": region, "batch": nb,
                        "lon": round(c.x, 6), "lat": round(c.y, 6)})
    frame = pd.concat([frame, pd.DataFrame(new)], ignore_index=True)
    frame.to_parquet(FRAME, index=False)
    print(f"drew batch {nb}: {len(new)} links "
          f"(TH {sum(x['region']=='thailand' for x in new)}, MH {sum(x['region']=='maharashtra' for x in new)})")
    return frame


def misclassification_candidates():
    """data_prior links where the assigned class contradicts the 85th-percentile operating
    speed -> highest prior probability of being misclassified. Returns df with a 'mscore'
    (strength of the contradiction, km/h) so we can spend budget on the worst first."""
    csv = os.path.join(ROOT, "score", "limit_alignment.csv")
    d = pd.read_csv(csv)
    d["s85"] = pd.to_numeric(d["speed_85"], errors="coerce")
    d["cap"] = pd.to_numeric(d["safe_system_max"], errors="coerce")
    dp = d[(d["context_source"] == "data_prior") & d["s85"].notna()].copy()
    vru = dp["context_final"].isin(["urban_vru", "intersection"])
    rural = dp["context_final"] == "rural_undivided"
    too_fast = vru & (dp["s85"] >= dp["cap"] + 20)          # 'urban' but highway speeds
    rural_fast = rural & (dp["s85"] >= 95)                  # 'rural' but divided-road speeds
    too_slow = rural & (dp["s85"] <= 45)                    # 'rural' but crawl -> maybe urban VRU
    cand = dp[too_fast | rural_fast | too_slow].copy()
    cand["mscore"] = np.where(cand["s85"] >= cand["cap"], cand["s85"] - cand["cap"],
                              cand["cap"] - cand["s85"])    # distance from class expectation
    return cand[["uid", "region", "mscore"]]


def draw_target(g):
    """Draw a targeted batch: the most class/speed-misaligned data_prior links (worst mscore
    first), up to FRAC of each region's network, excluding anything already sampled."""
    if os.path.exists(FRAME):
        frame = pd.read_parquet(FRAME)
    else:
        frame = pd.DataFrame(columns=["uid", "region", "batch", "lon", "lat"])
    nb = int(frame["batch"].max()) + 1 if len(frame) else 1
    done = set(frame["uid"])
    cand = misclassification_candidates()
    cand = cand[~cand["uid"].isin(done)]
    geo = g.set_index("uid")
    # Total links to draw: VLM_TARGET_N (explicit, for budget control) else 10% of the scored
    # analysis set. Split across regions in proportion to available suspects.
    target_total = int(os.environ.get("VLM_TARGET_N", "0"))
    if not target_total:
        target_total = int(round(FRAC * len(misclassification_candidates())))  # ~10% of suspect set
    avail = cand.groupby("region").size()
    tot_avail = max(1, avail.sum())
    new = []
    for region, sub in g.groupby("region"):
        if REGION_FILTER and region != REGION_FILTER:
            continue
        share = avail.get(region, 0) / tot_avail
        n = min(int(avail.get(region, 0)), int(round(target_total * share)))
        pick = (cand[cand["region"] == region].sort_values("mscore", ascending=False).head(n))
        for uid in pick["uid"]:
            c = geo.loc[uid, "geometry"].representative_point()
            new.append({"uid": uid, "region": region, "batch": nb,
                        "lon": round(c.x, 6), "lat": round(c.y, 6)})
    frame = pd.concat([frame, pd.DataFrame(new)], ignore_index=True)
    frame.to_parquet(FRAME, index=False)
    print(f"drew TARGETED batch {nb}: {len(new)} misalignment-suspect links "
          f"(TH {sum(x['region']=='thailand' for x in new)}, MH {sum(x['region']=='maharashtra' for x in new)})")
    return frame


def main():
    ok, msg = preflight_openai()
    if not ok:
        print(f"ABORT: OpenAI pre-flight failed -> {msg}\n"
              f"  (no imagery fetched; fix billing/key in .env and re-run)")
        sys.exit(1)
    print(f"OpenAI pre-flight OK (model={MODEL})")

    g = gpd.read_file(os.path.join(DATA, "enriched.gpkg"), layer="sections")[["uid", "region", "geometry"]]
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "target":                       # targeted: misclassification-suspect links
        prior = set(pd.read_parquet(OUTP)["uid"]) if os.path.exists(OUTP) else set()
        if os.path.exists(FRAME):
            fr0 = pd.read_parquet(FRAME)
            latest = int(fr0["batch"].max())
            pending = fr0[(fr0["batch"] == latest) & (~fr0["uid"].isin(prior))]
        else:
            pending = []
        if len(pending) > 0:                   # resume the already-drawn target batch (no re-draw)
            frame = pd.read_parquet(FRAME)
            print(f"resuming pending target batch {latest}: {len(pending)} links left to process")
        else:
            frame = draw_target(g)             # last batch finished -> draw the next target batch
    elif mode == "densify" or not os.path.exists(FRAME):
        frame = draw_batch(g)
    else:
        frame = pd.read_parquet(FRAME)

    done = pd.read_parquet(OUTP)["uid"].tolist() if os.path.exists(OUTP) else []
    todo = frame[~frame["uid"].isin(done)]
    if mode == "target":      # ONLY the just-drawn targeted batch — never the older backlog
        todo = todo[todo.batch == int(frame["batch"].max())]
    if REGION_FILTER:                                   # only process the targeted region
        todo = todo[todo.region == REGION_FILTER]
    todo = todo.merge(g[["uid", "geometry"]], on="uid")
    todo = gpd.GeoDataFrame(todo, geometry="geometry")
    print(f"sample frame: {len(frame)} links | already done: {len(done)} | to process now: {len(todo)} "
          f"(model={MODEL}, {N_IMAGES} imgs/seg, {WORKERS} workers)")

    rows, since = [], 0
    if os.path.exists(OUTP):
        rows = pd.read_parquet(OUTP).to_dict("records")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_segment, r): r.uid for r in todo.itertuples()}
        for i, fut in enumerate(as_completed(futs)):
            try:
                rec = fut.result()
            except Exception:
                rec = {"uid": futs[fut], "n_images": 0}
            with _lock:
                rows.append(rec); since += 1
                if since % 25 == 0:
                    pd.DataFrame(rows).to_parquet(OUTP, index=False)
                    withimg = sum(1 for x in rows if x.get("n_images", 0) > 0)
                    print(f"  {len(rows)} done, {withimg} with imagery, "
                          f"{(time.time()-t0)/60:.1f} min elapsed")
    pd.DataFrame(rows).to_parquet(OUTP, index=False)
    df = pd.DataFrame(rows)
    img = df[df["n_images"] > 0]
    print(f"\nDONE. {len(df)} processed, {len(img)} with imagery "
          f"({100*len(img)/max(1,len(df)):.0f}% coverage).")
    if "context_vlm" in df.columns:
        print("  context distribution:", dict(img["context_vlm"].value_counts()))
    for region in ["thailand", "maharashtra"]:
        s = df[df.region == region]; si = s[s["n_images"] > 0]
        print(f"  {region}: {len(s)} sampled, {len(si)} with imagery ({100*len(si)/max(1,len(s)):.0f}%)")


if __name__ == "__main__":
    main()
