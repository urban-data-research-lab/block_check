from __future__ import annotations
#!/usr/bin/env python3
# block_reviewer_austin_orange_filter.py
# - Filters giant/line features to avoid huge message payloads
# - Neighbor borders: blaze orange (solid)
# - Opaque dark-blue numbers
# - Windowed raster read + image downscale to keep PNG small
# - Preserves completion logic & clears A–H for secondary/blank

# import os, io, base64, re
# from pathlib import Path
# from collections import Counter

# import numpy as np
# import pandas as pd
# import geopandas as gpd
# from PIL import Image, ImageDraw, ImageFont

# import rasterio
# from rasterio.windows import from_bounds, Window
# from shapely import affinity
# from shapely.affinity import affine_transform as shp_affine
# from shapely.geometry import box

# import streamlit as st
# from rasterio.io import MemoryFile
# from urllib.parse import urlparse


# ------------------ USER PATHS (Austin) ------------------
# RASTER_PATH  = r"C:\Users\Klio\Dropbox\PC (2)\Desktop\Austin\Texas-Austin_fharent_geo.tif"
# VECTOR_PATH  = r"C:\Users\Klio\Dropbox\PC (2)\Desktop\Austin\Austin_fharent_geo_all_adjusted.geojson"
# OCR_CSV_PATH = r"C:\Users\Klio\Dropbox\PC (2)\Desktop\Austin\docai_extraction_results_atl1geotilt2_austnogeotilt_docai_2025-10-27.csv"
# OUTPUT_DIR   = r"C:\Users\Klio\Dropbox\PC (2)\Desktop\Austin"
# WORK_CSV     = os.path.join(OUTPUT_DIR, Path(OCR_CSV_PATH).stem + "__ui_work.csv")
# ---------------------------------------------------------

# # ------------------ USER PATHS (Austin) ------------------
# RASTER_PATH  = r"/Users/houpuli/Downloads/untitled folder 4/Texas-Austin_fharent_geo.tif"
# VECTOR_PATH  = r"/Users/houpuli/Downloads/untitled folder 4/Austin_fharent_geo_all_adjusted.geojson"
# OCR_CSV_PATH = r"/Users/houpuli/Downloads/untitled folder 4/docai_extraction_results_atl1geotilt2_austnogeotilt_docai_2025-10-27.csv"
# OUTPUT_DIR   = r"/Users/houpuli/Downloads/untitled folder 4/Austin"
# WORK_CSV     = os.path.join(OUTPUT_DIR, Path(OCR_CSV_PATH).stem + "__ui_work.csv")
# ---------------------------------------------------------


# # ------------------ USER PATHS UI ------------------
# st.sidebar.header("📁 Input / Output paths")
# raster_path = st.sidebar.text_input(
#     "Raster (.tif) path",
#     value=st.session_state.get("raster_path", "")
# )
# vector_path = st.sidebar.text_input(
#     "Vector (.geojson) path",
#     value=st.session_state.get("vector_path", "")
# )
# ocr_csv_path = st.sidebar.text_input(
#     "OCR CSV path",
#     value=st.session_state.get("ocr_csv_path", "")
# )
# output_dir = st.sidebar.text_input(
#     "Output directory for decisions CSV",
#     value=st.session_state.get("output_dir", "")
# )
# st.session_state["raster_path"] = raster_path
# st.session_state["vector_path"] = vector_path
# st.session_state["ocr_csv_path"] = ocr_csv_path
# st.session_state["output_dir"]   = output_dir
# # simple check：all paths have to be non-null
# if not (raster_path and vector_path and ocr_csv_path and output_dir):
#     st.warning("👈 Please fill in all paths (Raster, Vector, OCR CSV and output directory) in the left column before continuing.")
#     st.stop()
# RASTER_PATH  = raster_path
# VECTOR_PATH  = vector_path
# OCR_CSV_PATH = ocr_csv_path
# OUTPUT_DIR   = output_dir
# WORK_CSV     = os.path.join(OUTPUT_DIR, Path(OCR_CSV_PATH).stem + "__ui_work.csv")
# # ---------------------------------------------------------


# # ------------------ USER UPLOAD UI (Plan B) ------------------
# st.sidebar.header("📁 Upload input files")
# uploaded_raster = st.sidebar.file_uploader("Raster (.tif)", type=["tif", "tiff"])
# uploaded_vector = st.sidebar.file_uploader("Vector (.geojson)", type=["geojson"])
# uploaded_ocr = st.sidebar.file_uploader("OCR CSV", type=["csv"])

# output_subdir = st.sidebar.text_input("Output folder name (relative, optional)", value="outputs")

# if not (uploaded_raster and uploaded_vector and uploaded_ocr):
#     st.warning("👈 Please upload Raster, Vector and OCR CSV in the sidebar before continuing.")
#     st.stop()

# memfile = MemoryFile(uploaded_raster.read())
# src = memfile.open()

# vector_bytes = io.BytesIO(uploaded_vector.read())
# gdf0 = gpd.read_file(vector_bytes)
# ocr_df = pd.read_csv(uploaded_ocr, dtype=str, low_memory=False).fillna("")

# OUTPUT_DIR = os.path.join(os.getcwd(), output_subdir or "outputs")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# OCR_CSV_NAME = uploaded_ocr.name
# WORK_CSV = os.path.join(
#     OUTPUT_DIR, 
#     Path(OCR_CSV_NAME).stem + "__ui_work.csv"
# )
# # ---------------------------------------------------------


import io
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode

import requests          # 记得 requirements.txt 里加一行：requests
import streamlit as st
import rasterio
from rasterio.io import MemoryFile
import geopandas as gpd
import pandas as pd


# ---------------------------------------------------------


# ----------- FILTER / SIZE LIMIT KNOBS -----------
# Any string column containing these will be excluded (case-insensitive)
EXCLUDE_TERMS = ["red_line", "redline", "red crayon", "redcrayon"]

# Drop polygons that are enormous when expressed in raster pixels (area in px^2)
# Start conservative; raise if you accidentally drop legit big blocks.
MAX_POLY_AREA_PX2 = 5_000_000

# If the rendered window exceeds this many pixels, the PNG is downscaled
MAX_WINDOW_PIXELS = 2_800_000   # ~ (1680 x 1670); adjust as you like
# ------------------------------------------------

FIELDS = list("abcdefgh")
CLASS_OPTIONS  = ["primary","secondary","blank","multiple","obscured"]

DEFAULT_PIXEL_BUFFER = 25
BUFFER_STEP = 100

# ---------- PAGE STYLE ----------
st.set_page_config(layout="wide", page_title="Block Review – Austin (filtered)")
st.markdown("""
<style>
  main .block-container { padding-top:.35rem; padding-bottom:.35rem; }
  .vh-capped img { width:100% !important; height:auto !important; max-height:85vh !important;
                   border-radius:8px; display:block; margin:0 auto; }
  .badge { display:inline-block; padding:2px 6px; border-radius:6px; background:#eee; margin-left:6px; font-size:.8rem; }
  .right-pane * { font-size:0.88rem !important; }
  .stTextInput>div>div>input { font-size:0.88rem !important; padding-top:2px; padding-bottom:2px; }
  .stTextArea textarea { font-size:0.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ---------- helpers ----------
def load_font(px):
    for p in ["C:/Windows/Fonts/arial.ttf","C:/Windows/Fonts/calibri.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, px)
            except: pass
    return ImageFont.load_default()

def pick_font(img_w, img_h, min_px=18, max_px=56, frac=0.055):
    px = int(max(min_px, min(max_px, frac * min(img_w, img_h))))
    return load_font(px)

# Colors
BLUE_NUM = (0, 60, 170, 255)      # dark, opaque blue numbers
ORANGE  = (255, 120, 0, 255)      # blaze orange neighbor borders
RED_EDGE = (255, 0, 0, 255)       # focus outline

def scale_to_uint8(arr):
    arr = arr.astype(np.float32, copy=False)
    mn, mx = np.percentile(arr, (2,98))
    if mx <= mn: mx = mn + 1.0
    arr = (arr - mn) * (255.0/(mx-mn))
    return np.clip(arr,0,255).astype(np.uint8)

def to_display_image(bands):
    if bands.ndim == 3:
        if bands.shape[0] >= 3:
            r,g,b = (scale_to_uint8(bands[0]), scale_to_uint8(bands[1]), scale_to_uint8(bands[2]))
            return Image.merge("RGB", [Image.fromarray(r), Image.fromarray(g), Image.fromarray(b)]).convert("RGBA")
        else:
            g = scale_to_uint8(bands[0])
            return Image.fromarray(g, mode="L").convert("RGBA")
    else:
        g = scale_to_uint8(bands)
        return Image.fromarray(g, mode="L").convert("RGBA")

def encode_img(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

def text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        l,t,r,b = draw.textbbox((0,0), text, font=font)
        return (r-l, b-t)
    return draw.textsize(text, font=font)

def draw_centered_text(draw: ImageDraw.ImageDraw, txt: str, center_xy, font, fill_rgba):
    w,h = text_size(draw, txt, font)
    x = int(center_xy[0] - w/2)
    y = int(center_xy[1] - h/2)
    draw.text((x,y), txt, fill=fill_rgba, font=font)

def digits_only(x): return re.sub(r"\D", "", str(x) if x is not None else "")
def majority_len(series):
    lens = [len(s) for s in series if s]
    return Counter(lens).most_common(1)[0][0] if lens else 12
def norm_zfill(s: str, width: int):
    d = digits_only(s)
    return d if len(d) >= width else d.zfill(width)

# ---- numeric cleaners for A..H ----
_sym_re = re.compile(r"[^\d\.\,\-\+]")
_num_re = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

def _normalize_str(s):
    if s is None: return ""
    s = str(s)
    s = s.replace("O","0").replace("o","0").replace("—","-").replace("–","-").replace("−","-").replace("•",".")
    s = _sym_re.sub("", s)
    if s.count(".") > 1:
        i = s.find("."); s = s[:i+1] + s[i+1:].replace(".","")
    if "," in s:
        if "." in s: s = s.replace(",","")
        else: s = s.replace(",",".",1) if s.count(",")==1 else s.replace(",","")
    m = _num_re.search(s)
    return m.group(0) if m else ""

def _to_dec(s):
    try:
        ss = _normalize_str(s)
        return float(ss) if ss!="" else None
    except:
        return None

def fmt_a_dollar_lt70(x):
    v = _to_dec(x)
    if v is None: return ""
    v = 0 if v < 0 else (69.99 if v >= 70 else v)
    return f"{v:.2f}"

def fmt_b_int(x):
    v = _to_dec(x)
    if v is None: return ""
    return str(int(round(v)))

def fmt_percent_0_100(x):
    v = _to_dec(x)
    if v is None: return ""
    if 0 <= v <= 1: v *= 100
    v = max(0, min(100, v))
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s

def clean_fields(vals: dict):
    out = {}
    out["a"] = fmt_a_dollar_lt70(vals.get("a",""))
    out["b"] = fmt_b_int(vals.get("b",""))
    for c in list("cdefgh"):
        out[c] = fmt_percent_0_100(vals.get(c,""))
    return out

def parse_block_list(txt: str):
    if not txt: return []
    raw = re.split(r"[\s,;]+", str(txt).strip())
    return [r for r in (t.strip() for t in raw) if r]

def clamp_image_if_needed(img: Image.Image, max_pixels=MAX_WINDOW_PIXELS) -> Image.Image:
    w,h = img.size
    if w*h <= max_pixels:
        return img
    scale = (max_pixels / float(w*h))**0.5
    new_w = max(1, int(w*scale))
    new_h = max(1, int(h*scale))
    return img.resize((new_w, new_h), Image.BILINEAR)

# ---------- load raster & vector ----------
# src = rasterio.open(RASTER_PATH)
A   = src.transform
A_params = (A.a, A.b, A.d, A.e, A.c, A.f)

# gdf0 = gpd.read_file(VECTOR_PATH)

# choose an ID column
for candidate in ["global_id","id","ID","OBJECTID","fid","fidc","block_id"]:
    if candidate in gdf0.columns:
        id_field = candidate
        break
else:
    id_field = "_fid"
    gdf0 = gdf0.reset_index().rename(columns={"index":"_fid"})

def is_pixel_space(gdf, raster):
    if gdf.crs is None:
        minx, miny, maxx, maxy = gdf.total_bounds
        w, h = raster.width, raster.height
        margin = max(w, h) * 0.05 + 50
        return (minx >= -margin and miny >= -margin and maxx <= w + margin and maxy <= h + margin)
    return False

# Normalize geometries to raster CRS
if is_pixel_space(gdf0, src):
    gdf_px = gdf0.copy()
    gdf_px["geometry"] = gdf_px["geometry"].map(lambda g: affinity.scale(g, yfact=-1, origin=(0,0)) if g is not None else None)
    gdf_map = gdf_px.copy()
    gdf_map["geometry"] = gdf_px["geometry"].map(lambda g: shp_affine(g, A_params) if g is not None else None)
    gdf_map.set_crs(src.crs, inplace=True)
else:
    gdf_map = gdf0.copy()
    if gdf_map.crs is None:
        gdf_map.set_crs(src.crs, inplace=True)
    elif gdf_map.crs != src.crs:
        gdf_map = gdf_map.to_crs(src.crs)

gdf_map = gdf_map[gdf_map.geometry.notnull()].copy()
if hasattr(gdf_map.geometry, "is_empty"):
    gdf_map = gdf_map[~gdf_map.geometry.is_empty].copy()

# ---- drop objects outside raster ----
raster_poly = box(*src.bounds)
gdf_map = gdf_map[gdf_map.geometry.intersects(raster_poly)].copy()

# ---- content-based drop: any string col contains EXCLUDE_TERMS ----
if EXCLUDE_TERMS:
    def bad_row(row):
        lower_terms = [t.lower() for t in EXCLUDE_TERMS]
        for col in row.index:
            v = row[col]
            if isinstance(v, str) and v:
                s = v.lower()
                if any(term in s for term in lower_terms):
                    return True
        return False
    mask_bad = gdf_map.apply(bad_row, axis=1)
    gdf_map = gdf_map.loc[~mask_bad].copy()

# ---- size-based drop (measure area in PIXELS) ----
Tinv_all = ~src.transform
Tinv_params_all = (Tinv_all.a, Tinv_all.b, Tinv_all.d, Tinv_all.e, Tinv_all.c, Tinv_all.f)

def area_in_pixels(geom):
    try:
        gp = shp_affine(geom, Tinv_params_all)
        return abs(gp.area)
    except Exception:
        return np.inf

gdf_map["area_px2"] = gdf_map.geometry.map(area_in_pixels)
gdf_map = gdf_map[gdf_map["area_px2"] <= MAX_POLY_AREA_PX2].copy()

# Finally, normalize IDs
gdf_map = gdf_map.reset_index(drop=True)
gdf_map[id_field] = gdf_map[id_field].astype(str)
target_len = majority_len(gdf_map[id_field].astype(str).map(digits_only)) or 12
gdf_map["id_norm"] = gdf_map[id_field].astype(str).map(lambda s: norm_zfill(s, target_len))
IDNORM_TO_REAL = dict(zip(gdf_map["id_norm"], gdf_map[id_field].astype(str)))

# ---------- OCR / seeds ----------
if "id" not in ocr_df.columns:
    st.error("OCR CSV must have an 'id' column.")
    st.stop()

def seed_for(fid_norm: str):
    d = digits_only(fid_norm)
    rows = ocr_df[ocr_df["id"].astype(str).map(digits_only) == d]
    if rows.empty:
        return {f:"" for f in FIELDS}
    r = rows.iloc[0]
    raw = {f: r.get(f,"") for f in FIELDS}
    return clean_fields(raw)

# ---------- decisions ----------
DECISION_COLS = ["id","class","a","b","c","d","e","f","g","h","comment",
                 "secondary_ids","blank_neighbor_ids","primary_for_secondary","completed"]

if os.path.exists(WORK_CSV):
    decisions = pd.read_csv(WORK_CSV, dtype=str).reindex(columns=DECISION_COLS, fill_value="")
else:
    decisions = pd.DataFrame(columns=DECISION_COLS)

def ensure_row_exists(d_id):
    m = decisions["id"] == str(d_id)
    if not m.any():
        base = {"id": str(d_id), "class":"primary", "comment":"", "secondary_ids":"",
                "blank_neighbor_ids":"", "primary_for_secondary":"", "completed":"false"}
        base.update({k:"" for k in FIELDS})
        decisions.loc[len(decisions)] = base

def get_decision(fid):
    fid = str(fid)
    m = decisions["id"] == fid
    if m.any():
        return decisions.loc[m].iloc[0].copy()
    s = seed_for(norm_zfill(fid, target_len))
    row = {"id":fid, "class":"primary", **s, "comment":"",
           "secondary_ids":"", "blank_neighbor_ids":"",
           "primary_for_secondary":"", "completed":"false"}
    decisions.loc[len(decisions)] = row
    return decisions.loc[decisions["id"]==fid].iloc[0].copy()

def save_decisions():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    decisions.reindex(columns=DECISION_COLS, fill_value="").to_csv(WORK_CSV, index=False)

def compute_completed(df):
    done = set(df.loc[df["completed"].astype(str).str.lower().eq("true"), "id"].astype(str))
    for _, r in df.iterrows():
        for c in ["secondary_ids","blank_neighbor_ids"]:
            v = str(r.get(c,"") or "")
            if v:
                done.update([t.strip() for t in v.split(",") if t.strip()])
    return done

def pixel_buffer_to_map_units(A, pixel_buf):
    vx = float(np.hypot(A.a, A.d))
    vy = float(np.hypot(A.b, A.e))
    px_size = (vx + vy) / 2.0 if (vx>0 and vy>0) else (abs(A.a) if abs(A.a)>0 else abs(A.e))
    return float(pixel_buf) * (px_size if px_size>0 else 1.0)

def resolve_token_to_real_id(token: str, neigh_map: dict):
    token = str(token).strip()
    if token.isdigit() and int(token) in neigh_map:
        return str(neigh_map[int(token)])
    norm = norm_zfill(token, target_len)
    if norm in IDNORM_TO_REAL:
        return str(IDNORM_TO_REAL[norm])
    return token

# ------- renderer (windowed) -------
if "clip_cache" not in st.session_state: st.session_state.clip_cache = {}

def neighbors_in_buffer(feat_map_geom, buffer_px, current_id_norm: str):
    map_buf = pixel_buffer_to_map_units(A, max(buffer_px,0))
    buffered = feat_map_geom.buffer(map_buf, cap_style=1, join_style=2)
    try:
        sidx = gdf_map.sindex
        cand = list(sidx.intersection(buffered.bounds))
        subset = gdf_map.iloc[cand]
    except Exception:
        subset = gdf_map
    subset = subset[subset.geometry.intersects(buffered)].copy()
    subset = subset[subset["id_norm"] != current_id_norm]
    subset = subset.drop_duplicates(subset=["id_norm"]).copy()
    return subset.sort_values(by=["id_norm"])

def render_window_b64(fid: str, feat_map_geom, buffer_px: int):
    key = (fid, int(buffer_px))
    cache = st.session_state.clip_cache
    if key in cache:
        return cache[key]

    map_buf = pixel_buffer_to_map_units(A, max(buffer_px,0))
    buffered = feat_map_geom.buffer(map_buf, cap_style=1, join_style=2)
    bx0, by0, bx1, by1 = buffered.bounds
    rx0, ry0, rx1, ry1 = src.bounds
    bx0 = max(bx0, rx0); by0 = max(by0, ry0); bx1 = min(bx1, rx1); by1 = min(by1, ry1)
    if not (bx1 > bx0 and by1 > by0):
        raise ValueError("Feature+buffer does not overlap the raster extent.")

    win: Window = from_bounds(bx0, by0, bx1, by1, transform=src.transform)
    out = src.read(window=win)
    out_transform = src.window_transform(win)

    img = to_display_image(out)
    # clamp PNG size to avoid Streamlit message-size overflow
    img = clamp_image_if_needed(img, MAX_WINDOW_PIXELS)

    draw = ImageDraw.Draw(img, "RGBA")
    font = pick_font(*img.size)

    Tinv = ~out_transform
    Tinv_params = (Tinv.a, Tinv.b, Tinv.d, Tinv.e, Tinv.c, Tinv.f)

    # outline focus (solid red)
    focus_px = shp_affine(feat_map_geom, Tinv_params)
    f_polys = list(focus_px.geoms) if getattr(focus_px, "geom_type","")=="MultiPolygon" else [focus_px]
    for p in f_polys:
        draw.line(list(p.exterior.coords), fill=RED_EDGE, width=3, joint="curve")

    # neighbors: blaze orange border + dark blue numbers (opaque)
    current_norm = norm_zfill(fid, target_len)
    neigh_df = neighbors_in_buffer(feat_map_geom, buffer_px, current_norm)
    neigh_map = {}
    for j, (_, nb) in enumerate(neigh_df.iterrows(), start=1):
        nb_px = shp_affine(nb.geometry, Tinv_params)
        # skip giant neighbors after pixel transform too (belt/line cleanup)
        try:
            if abs(nb_px.area) > MAX_POLY_AREA_PX2:
                continue
        except Exception:
            pass

        n_polys = list(nb_px.geoms) if getattr(nb_px, "geom_type","")=="MultiPolygon" else [nb_px]
        for p in n_polys:
            draw.line(list(p.exterior.coords), fill=ORANGE, width=2, joint="curve")
        cx, cy = nb_px.representative_point().coords[0]
        draw_centered_text(draw, str(j), (cx,cy), font, BLUE_NUM)
        neigh_map[j] = str(nb[id_field])

    b64 = encode_img(img)
    cache[key] = (b64, Tinv_params, neigh_map)
    st.session_state.clip_cache = cache
    return b64, Tinv_params, neigh_map

# -------------- session/nav --------------
if "idx" not in st.session_state: st.session_state.idx = 0
if "initialized" not in st.session_state: st.session_state.initialized = False
if "buffer_px" not in st.session_state: st.session_state.buffer_px = DEFAULT_PIXEL_BUFFER
if "last_fid" not in st.session_state: st.session_state.last_fid = None

def next_incomplete_index_from(start_idx):
    done = compute_completed(decisions)
    n = len(gdf_map)
    for step in range(1, n+1):
        j = (start_idx + step) % n
        if str(gdf_map.iloc[j][id_field]) not in done:
            return j
    return start_idx

def prev_incomplete_index_from(start_idx):
    done = compute_completed(decisions)
    n = len(gdf_map)
    for step in range(1, n+1):
        j = (start_idx - step) % n
        if str(gdf_map.iloc[j][id_field]) not in done:
            return j
    return start_idx

def first_incomplete_index():
    done = compute_completed(decisions)
    for i in range(len(gdf_map)):
        if str(gdf_map.iloc[i][id_field]) not in done:
            return i
    return 0

if not st.session_state.initialized:
    st.session_state.idx = first_incomplete_index()
    st.session_state.initialized = True

def goto_index(new_idx):
    st.session_state.idx = new_idx % len(gdf_map)
    st.session_state.buffer_px = DEFAULT_PIXEL_BUFFER

# ----------------- UI -----------------
done_count = len(compute_completed(decisions))
st.markdown(f"**Progress:** {done_count} / {len(gdf_map)}")
st.progress(min(1.0, done_count/max(1,len(gdf_map))))

left, right = st.columns([1.65, 1.35], gap="small")

with left:
    st.markdown("On-the-fly Clip <span style='color:#888;'>(red current • blaze-orange neighbors)</span>", unsafe_allow_html=True)
    if len(gdf_map) == 0:
        st.error("No polygons after filtering. Try relaxing filters or thresholds.")
        st.stop()

    while True:
        feat_row = gdf_map.iloc[st.session_state.idx]
        fid      = str(feat_row[id_field])
        if fid in compute_completed(decisions):
            goto_index(next_incomplete_index_from(st.session_state.idx))
            continue
        break

    row = get_decision(fid)

    if st.session_state.last_fid != fid:
        st.session_state.buffer_px = DEFAULT_PIXEL_BUFFER
        st.session_state.last_fid = fid

    try:
        b64, Tinv_params, neigh_map = render_window_b64(fid, feat_row.geometry, st.session_state.buffer_px)
    except ValueError as e:
        st.warning(f"{e} — marking as skipped & advancing.")
        ensure_row_exists(fid)
        decisions.loc[decisions["id"]==fid, "completed"] = "true"
        save_decisions()
        goto_index(next_incomplete_index_from(st.session_state.idx))
        st.experimental_rerun()

    st.markdown(f'<div class="vh-capped"><img src="{b64}"/></div>', unsafe_allow_html=True)

    z1, z2, z3 = st.columns(3)
    if z1.button("− Buffer", use_container_width=True):
        st.session_state.buffer_px = max(0, st.session_state.buffer_px - BUFFER_STEP); st.rerun()
    if z2.button("+ Buffer", use_container_width=True):
        st.session_state.buffer_px = st.session_state.buffer_px + BUFFER_STEP; st.rerun()
    z3.markdown(f'<span class="badge">buffer: {st.session_state.buffer_px}px</span>', unsafe_allow_html=True)

    st.markdown("---")
    n1, n2, n3 = st.columns(3)
    if n1.button("⬅️ Prev"): goto_index(prev_incomplete_index_from(st.session_state.idx)); st.rerun()
    if n2.button("Skip"):
        ensure_row_exists(fid)
        decisions.loc[decisions["id"]==fid, "completed"] = "true"
        save_decisions(); goto_index(next_incomplete_index_from(st.session_state.idx)); st.rerun()
    if n3.button("Next ➡️"): goto_index(next_incomplete_index_from(st.session_state.idx)); st.rerun()

with right:
    st.markdown('<div class="right-pane">', unsafe_allow_html=True)
    st.subheader(f"ID: {fid}")

    default_class = row.get("class","primary")
    sel_class = st.radio(
        "Classification",
        CLASS_OPTIONS,
        index=CLASS_OPTIONS.index(default_class) if default_class in CLASS_OPTIONS else 0,
        horizontal=True,
        key=f"class_radio_{fid}"
    )

    with st.form(f"edit_form_{fid}", clear_on_submit=False):
        comment_val = row.get("comment","")
        a_val,b_val,c_val,d_val,e_val,f_val,g_val,h_val = [row.get(k,"") for k in "abcdefgh"]
        sec_text = row.get("secondary_ids","")
        blank_text = row.get("blank_neighbor_ids","")
        prim_text = row.get("primary_for_secondary","")

        if sel_class == "primary":
            st.markdown("### Fields (A–H)")
            a_val = st.text_input("A ($<70)", a_val)
            b_val = st.text_input("B (int)",  b_val)
            c_val = st.text_input("C (%)",    c_val)
            d_val = st.text_input("D (%)",    d_val)
            e_val = st.text_input("E (%)",    e_val)
            f_val = st.text_input("F (%)",    f_val)
            g_val = st.text_input("G (%)",    g_val)
            h_val = st.text_input("H (%)",    h_val)

            st.markdown("### Linked blocks (optional)")
            sec_text   = st.text_input("Secondary blocks (blue numbers or real IDs; spaces/commas)", value=sec_text)
            blank_text = st.text_input("Blank blocks (blue numbers or real IDs; spaces/commas)",    value=blank_text)
            comment_val = st.text_area("Comment", comment_val, height=76)

        elif sel_class == "secondary":
            st.info("Enter **one** primary (blue number or real ID). After Save, the app jumps to that primary.")
            prim_text   = st.text_input("Primary (exactly one, required)", value=prim_text)
            comment_val = st.text_area("Comment", comment_val, height=76)

        else:  # blank / multiple / obscured
            st.info("A–H hidden for this classification.")
            if sel_class == "blank":
                blank_text = st.text_input("Additional blank blocks (blue numbers or real IDs; optional)", value=blank_text)
            comment_val = st.text_area("Comment", comment_val, height=76)

        submitted = st.form_submit_button("✅ Save & Next (Enter)")

    if submitted:
        ensure_row_exists(fid)
        decisions.loc[decisions["id"]==fid, ["class","comment"]] = [sel_class, comment_val]

        if sel_class == "primary":
            cleaned = clean_fields({"a":a_val,"b":b_val,"c":c_val,"d":d_val,"e":e_val,"f":f_val,"g":g_val,"h":h_val})
            decisions.loc[decisions["id"]==fid, list("abcdefgh")] = [cleaned[k] for k in list("abcdefgh")]

            sec_list   = [resolve_token_to_real_id(t, neigh_map) for t in parse_block_list(sec_text)]
            blank_list = [resolve_token_to_real_id(t, neigh_map) for t in parse_block_list(blank_text)]

            decisions.loc[decisions["id"]==fid, "secondary_ids"]      = ",".join(sorted({*sec_list}))
            decisions.loc[decisions["id"]==fid, "blank_neighbor_ids"] = ",".join(sorted({*blank_list}))
            decisions.loc[decisions["id"]==fid, "completed"] = "true"

            for nid in sec_list:
                ensure_row_exists(nid)
                decisions.loc[decisions["id"]==nid, list("abcdefgh")] = [""]*8
                decisions.loc[decisions["id"]==nid, ["class","primary_for_secondary","completed"]] = ["secondary", fid, "true"]

            for nid in blank_list:
                ensure_row_exists(nid)
                decisions.loc[decisions["id"]==nid, list("abcdefgh")] = [""]*8
                decisions.loc[decisions["id"]==nid, ["class","completed"]] = ["blank","true"]

            save_decisions()
            goto_index(next_incomplete_index_from(st.session_state.idx)); st.rerun()

        elif sel_class == "secondary":
            primary_tokens = parse_block_list(prim_text)
            if len(primary_tokens) != 1:
                st.warning("Enter exactly **one** primary.")
            else:
                primary_id = resolve_token_to_real_id(primary_tokens[0], neigh_map)
                decisions.loc[decisions["id"]==fid, list("abcdefgh")] = [""]*8
                decisions.loc[decisions["id"]==fid, ["class","primary_for_secondary","completed"]] = ["secondary", primary_id, "true"]

                ensure_row_exists(primary_id)
                existing = str(decisions.loc[decisions["id"]==primary_id, "secondary_ids"].values[0] or "")
                sset = set([x for x in existing.split(",") if x.strip()]) if existing else set()
                sset.add(fid)
                decisions.loc[decisions["id"]==primary_id, "secondary_ids"] = ",".join(sorted(sset))

                save_decisions()
                try:
                    i = gdf_map.index[gdf_map[id_field].astype(str)==str(primary_id)][0]
                    goto_index(i)
                except Exception:
                    goto_index(next_incomplete_index_from(st.session_state.idx))
                st.rerun()

        else:  # blank / multiple / obscured
            decisions.loc[decisions["id"]==fid, list("abcdefgh")] = [""]*8
            decisions.loc[decisions["id"]==fid, "completed"] = "true"

            if sel_class == "blank":
                blanks = [resolve_token_to_real_id(t, neigh_map) for t in parse_block_list(blank_text)]
                decisions.loc[decisions["id"]==fid, "blank_neighbor_ids"] = ",".join(sorted({*blanks}))
                for nid in blanks:
                    ensure_row_exists(nid)
                    decisions.loc[decisions["id"]==nid, list("abcdefgh")] = [""]*8
                    decisions.loc[decisions["id"]==nid, ["class","completed"]] = ["blank","true"]

            save_decisions()
            goto_index(next_incomplete_index_from(st.session_state.idx)); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
c1, c2 = st.columns(2)
if c1.button("💾 Save Now"): save_decisions(); st.toast("Saved.")
if os.path.exists(WORK_CSV):
    with open(WORK_CSV,"rb") as fh:
        c2.download_button("⬇️ Download decisions CSV", data=fh.read(),
                           file_name=Path(WORK_CSV).name, mime="text/csv")



