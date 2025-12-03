#!/usr/bin/env python3
# block_reviewer_austin_gcs_private.py
# - Filters giant/line features to avoid huge message payloads
# - Neighbor borders: blaze orange (solid)
# - Opaque dark-blue numbers
# - Windowed raster read + image downscale to keep PNG small
# - Preserves completion logic & clears A–H for secondary/blank
#
# NOTE: This version supports BOTH local paths and PRIVATE GCS paths
# (gs://... or https://storage.googleapis.com/...) via service account.

import os, io, base64, re, json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image, ImageDraw, ImageFont

import rasterio
from rasterio.windows import from_bounds, Window
from shapely import affinity
from shapely.affinity import affine_transform as shp_affine
from shapely.geometry import box

import streamlit as st
from google.cloud import storage

# =====================================================
#                GCS AUTH (SERVICE ACCOUNT)
# =====================================================

_gcs_client = None  # lazy init

# def get_gcs_client():
#     global _gcs_client
#     if _gcs_client is not None:
#         return _gcs_client

#     # 如果 secrets 里有 JSON，就写到 /tmp 并设置 env 变量
#     if "GCP_SERVICE_ACCOUNT_JSON" in st.secrets:
#         sa_path = "/tmp/gcp_service_account.json"
#         with open(sa_path, "w", encoding="utf-8") as f:
#             f.write(st.secrets["GCP_SERVICE_ACCOUNT_JSON"])
#         os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

#     # 无论如何尝试初始化（本地可用 ADC，云端用上面的 key）
#     _gcs_client = storage.Client()
#     return _gcs_client


_gcs_client = None  # lazy init

def get_gcs_client():
    """
    使用 Streamlit Secrets 里的 GCP_SERVICE_ACCOUNT_JSON 来初始化 GCS 客户端。
    如果 secrets 没配好，会在页面上直接报错，而不是抛 DefaultCredentialsError。
    """
    import textwrap

    global _gcs_client
    if _gcs_client is not None:
        return _gcs_client

    # 1) 检查 secrets 里面有没有这个 key
    if "GCP_SERVICE_ACCOUNT_JSON" not in st.secrets:
        msg = textwrap.dedent(
            """
            ❌ 没有在 Streamlit Secrets 中找到 `GCP_SERVICE_ACCOUNT_JSON`。

            请到 **Manage app → Settings → Secrets** 里添加类似：

            GCP_SERVICE_ACCOUNT_JSON = \"\"\"{
              "type": "service_account",
              "project_id": "block-check-480023",
              ...
            }\"\"\"
            """
        )
        st.error(msg)
        raise RuntimeError("Missing GCP_SERVICE_ACCOUNT_JSON in st.secrets")

    # 2) 把 JSON 内容写到 /tmp 文件里
    sa_json = st.secrets["GCP_SERVICE_ACCOUNT_JSON"]
    sa_path = "/tmp/gcp_service_account.json"
    with open(sa_path, "w", encoding="utf-8") as f:
        f.write(sa_json)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

    # 3) 试着创建 GCS client，如果失败就给出更清楚的提示
    try:
        _gcs_client = storage.Client(project="block-check-480023")
    except Exception as e:
        st.error(
            "❌ 创建 GCS Client 失败，请检查 Streamlit Secrets 里的 "
            "`GCP_SERVICE_ACCOUNT_JSON` 内容是否是完整合法的 service account JSON。\n\n"
            f"错误信息（已截断）：{str(e)[:500]}"
        )
        raise

    return _gcs_client


def _parse_gcs_path(path: str):
    """
    解析路径：
    - gs://bucket/path/to/file.ext
    - https://storage.googleapis.com/bucket/path/to/file.ext
    - 其他 → 视为本地路径
    返回：(mode, bucket, blob)
    mode = "gcs" 或 "local"
    """
    s = str(path).strip()
    if s.startswith("gs://"):
        no_scheme = s[len("gs://"):]
        bucket, *rest = no_scheme.split("/", 1)
        blob = rest[0] if rest else ""
        return "gcs", bucket, blob

    if "storage.googleapis.com" in s:
        # https://storage.googleapis.com/bucket/....../file.ext
        parts = s.split("storage.googleapis.com/", 1)[-1]
        bucket, *rest = parts.split("/", 1)
        blob = rest[0] if rest else ""
        return "gcs", bucket, blob

    return "local", None, None

# 在 session_state 里为 GCS 下载做一个缓存，避免每次都重新拉 8GB TIF
if "gcs_local_cache" not in st.session_state:
    st.session_state.gcs_local_cache = {}

def ensure_local_file(path: str, kind: str) -> str:
    """
    如果 path 是 GCS（gs:// 或 https://storage.googleapis.com/...）：
      - 用 service account 从 GCS 下载到 /tmp 下（带 kind 前缀）
      - 返回本地路径
    如果 path 是普通本地路径：
      - 原样返回
    """
    mode, bucket, blob = _parse_gcs_path(path)
    if mode == "local":
        return path

    if not bucket or not blob:
        raise ValueError(f"Invalid GCS path: {path}")

    cache_key = f"{bucket}/{blob}"
    cache = st.session_state.gcs_local_cache
    if cache_key in cache and os.path.exists(cache[cache_key]):
        return cache[cache_key]

    client = get_gcs_client()
    bucket_obj = client.bucket(bucket)
    blob_obj = bucket_obj.blob(blob)

    ext = Path(blob).suffix or ".dat"
    local_path = f"/tmp/{kind}_{Path(blob).stem}{ext}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob_obj.download_to_filename(local_path)

    cache[cache_key] = local_path
    st.session_state.gcs_local_cache = cache
    return local_path

def upload_local_to_gcs(local_path: str, dest_path: str):
    """
    把本地文件上传回 GCS（dest_path 可以是 gs://... 或 https://storage.googleapis.com/...）
    如果 dest_path 看起来是本地路径，则不上传。
    """
    mode, bucket, blob = _parse_gcs_path(dest_path)
    if mode != "gcs":
        # 视为本地路径：简单复制/移动
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        os.replace(local_path, dest_path)
        return

    client = get_gcs_client()
    bucket_obj = client.bucket(bucket)
    blob_obj = bucket_obj.blob(blob)
    blob_obj.upload_from_filename(local_path)

# =====================================================
#                 STREAMLIT BASIC CONFIG
# =====================================================
st.set_page_config(layout="wide", page_title="Block Review – Austin (GCS private)")

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

# =====================================================
#                 SIDEBAR: PATH CONFIG
# =====================================================
st.sidebar.header("📁 Input / Output paths")

raster_path = st.sidebar.text_input(
    "Raster (.tif) path (local or gs:// or https://storage.googleapis.com/...)",
    value=st.session_state.get("raster_path", "")
)
vector_path = st.sidebar.text_input(
    "Vector (.geojson) path (local or gs:// or https://storage.googleapis.com/...)",
    value=st.session_state.get("vector_path", "")
)
ocr_csv_path = st.sidebar.text_input(
    "OCR CSV path (local or gs:// or https://storage.googleapis.com/...)",
    value=st.session_state.get("ocr_csv_path", "")
)
# 可选：本地输出目录（只在你本机运行时有意义；云端重启会丢）
output_dir = st.sidebar.text_input(
    "Optional local output directory (for debugging; e.g. ./outputs)",
    value=st.session_state.get("output_dir", "./outputs")
)
decisions_gcs_url = st.sidebar.text_input(
    "Decisions CSV (gs://... or https://storage.googleapis.com/...) [recommended]",
    value=st.session_state.get("decisions_gcs_url", "")
)

st.session_state["raster_path"] = raster_path
st.session_state["vector_path"] = vector_path
st.session_state["ocr_csv_path"] = ocr_csv_path
st.session_state["output_dir"]   = output_dir
st.session_state["decisions_gcs_url"] = decisions_gcs_url

# simple check：all core paths have to be non-null
if not (raster_path and vector_path and ocr_csv_path):
    st.warning("👈 Please fill in Raster, Vector and OCR CSV paths (local or GCS) in the left sidebar before continuing.")
    st.stop()

RASTER_PATH  = raster_path
VECTOR_PATH  = vector_path
OCR_CSV_PATH = ocr_csv_path
OUTPUT_DIR   = output_dir
# 本地临时 decisions CSV（用来 download_button）
LOCAL_DECISIONS_TMP = "/tmp/decisions_ui_work.csv"

# =====================================================
#           FILTER / SIZE LIMIT KNOBS & CONSTANTS
# =====================================================
EXCLUDE_TERMS = ["red_line", "redline", "red crayon", "redcrayon"]

MAX_POLY_AREA_PX2 = 5_000_000
MAX_WINDOW_PIXELS = 2_800_000   # ~ (1680 x 1670)

FIELDS = list("abcdefgh")
CLASS_OPTIONS  = ["primary","secondary","blank","multiple","obscured"]

DEFAULT_PIXEL_BUFFER = 25
BUFFER_STEP = 100

# =====================================================
#                      HELPERS
# =====================================================
def load_font(px):
    for p in ["C:/Windows/Fonts/arial.ttf","C:/Windows/Fonts/calibri.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf","/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, px)
            except Exception:
                pass
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
    if mx <= mn:
        mx = mn + 1.0
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
    buf = io.BytesIO()
    img.save(buf, format="PNG")
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

def digits_only(x): 
    return re.sub(r"\D", "", str(x) if x is not None else "")

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
    if s is None:
        return ""
    s = str(s)
    s = (
        s.replace("O","0")
         .replace("o","0")
         .replace("—","-")
         .replace("–","-")
         .replace("−","-")
         .replace("•",".")
    )
    s = _sym_re.sub("", s)
    if s.count(".") > 1:
        i = s.find(".")
        s = s[:i+1] + s[i+1:].replace(".","")
    if "," in s:
        if "." in s:
            s = s.replace(",","")
        else:
            s = s.replace(",",".",1) if s.count(",")==1 else s.replace(",","")
    m = _num_re.search(s)
    return m.group(0) if m else ""

def _to_dec(s):
    try:
        ss = _normalize_str(s)
        return float(ss) if ss != "" else None
    except Exception:
        return None

def fmt_a_dollar_lt70(x):
    v = _to_dec(x)
    if v is None:
        return ""
    v = 0 if v < 0 else (69.99 if v >= 70 else v)
    return f"{v:.2f}"

def fmt_b_int(x):
    v = _to_dec(x)
    if v is None:
        return ""
    return str(int(round(v)))

def fmt_percent_0_100(x):
    v = _to_dec(x)
    if v is None:
        return ""
    if 0 <= v <= 1:
        v *= 100
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
    if not txt:
        return []
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

# =====================================================
#              LOAD RASTER & VECTOR (WITH GCS)
# =====================================================
local_raster_path = ensure_local_file(RASTER_PATH, kind="raster")
src = rasterio.open(local_raster_path)
A   = src.transform
A_params = (A.a, A.b, A.d, A.e, A.c, A.f)

local_vector_path = ensure_local_file(VECTOR_PATH, kind="vector")
gdf0 = gpd.read_file(local_vector_path)

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

# =====================================================
#                 OCR / SEEDS (WITH GCS)
# =====================================================
local_ocr_path = ensure_local_file(OCR_CSV_PATH, kind="ocr")
ocr_df = pd.read_csv(local_ocr_path, dtype=str, low_memory=False).fillna("")
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

# =====================================================
#                    DECISIONS
# =====================================================
DECISION_COLS = [
    "id","class","a","b","c","d","e","f","g","h","comment",
    "secondary_ids","blank_neighbor_ids","primary_for_secondary","completed"
]

def load_decisions_initial():
    """
    优先：从 decisions_gcs_url（GCS 私有）下载；
    若没填，则尝试从本地 OUTPUT_DIR 中读；
    若都没有，则返回空 DataFrame。
    """
    dec_cfg = st.session_state.get("decisions_gcs_url", "").strip()
    if dec_cfg:
        try:
            local_dec = ensure_local_file(dec_cfg, kind="decisions")
            if os.path.exists(local_dec):
                df = pd.read_csv(local_dec, dtype=str).reindex(columns=DECISION_COLS, fill_value="")
                # 缓存一份路径给 download_button 用
                st.session_state["decisions_local_path"] = local_dec
                return df
        except Exception as e:
            st.warning(f"Could not load decisions from GCS ({dec_cfg}): {e}")

    # fallback: local WORK_CSV (for local debugging)
    work_csv_local = os.path.join(OUTPUT_DIR, Path(OCR_CSV_PATH).stem + "__ui_work.csv")
    if os.path.exists(work_csv_local):
        df = pd.read_csv(work_csv_local, dtype=str).reindex(columns=DECISION_COLS, fill_value="")
        st.session_state["decisions_local_path"] = work_csv_local
        return df

    st.session_state["decisions_local_path"] = LOCAL_DECISIONS_TMP
    return pd.DataFrame(columns=DECISION_COLS)

decisions = load_decisions_initial()

def ensure_row_exists(d_id):
    m = decisions["id"] == str(d_id)
    if not m.any():
        base = {
            "id": str(d_id),
            "class": "primary",
            "comment": "",
            "secondary_ids": "",
            "blank_neighbor_ids": "",
            "primary_for_secondary": "",
            "completed": "false"
        }
        base.update({k:"" for k in FIELDS})
        decisions.loc[len(decisions)] = base

def get_decision(fid):
    fid = str(fid)
    m = decisions["id"] == fid
    if m.any():
        return decisions.loc[m].iloc[0].copy()
    s = seed_for(norm_zfill(fid, target_len))
    row = {
        "id": fid,
        "class": "primary",
        **s,
        "comment": "",
        "secondary_ids": "",
        "blank_neighbor_ids": "",
        "primary_for_secondary": "",
        "completed": "false"
    }
    decisions.loc[len(decisions)] = row
    return decisions.loc[decisions["id"]==fid].iloc[0].copy()

def save_decisions():
    """
    核心策略：
    1. 总是写一份到 /tmp/decisions_ui_work.csv
    2. 如果用户填了 decisions_gcs_url，则上传到 GCS（private）
    3. 本地调试时，也写到 OUTPUT_DIR 下的 __ui_work.csv
    """
    os.makedirs(os.path.dirname(LOCAL_DECISIONS_TMP), exist_ok=True)
    decisions.reindex(columns=DECISION_COLS, fill_value="").to_csv(LOCAL_DECISIONS_TMP, index=False)
    st.session_state["decisions_local_path"] = LOCAL_DECISIONS_TMP

    dec_dest = st.session_state.get("decisions_gcs_url", "").strip()
    if dec_dest:
        try:
            upload_local_to_gcs(LOCAL_DECISIONS_TMP, dec_dest)
        except Exception as e:
            st.warning(f"Failed to upload decisions to GCS ({dec_dest}): {e}")

    # optional: local copy for debugging
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        work_csv_local = os.path.join(OUTPUT_DIR, Path(OCR_CSV_PATH).stem + "__ui_work.csv")
        decisions.reindex(columns=DECISION_COLS, fill_value="").to_csv(work_csv_local, index=False)
    except Exception:
        pass

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

# =====================================================
#                  RENDERER (WINDOWED)
# =====================================================
if "clip_cache" not in st.session_state:
    st.session_state.clip_cache = {}

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

# =====================================================
#            SESSION / NAVIGATION STATE
# =====================================================
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "buffer_px" not in st.session_state:
    st.session_state.buffer_px = DEFAULT_PIXEL_BUFFER
if "last_fid" not in st.session_state:
    st.session_state.last_fid = None

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

# =====================================================
#                       UI
# =====================================================
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
        st.session_state.buffer_px = max(0, st.session_state.buffer_px - BUFFER_STEP)
        st.rerun()
    if z2.button("+ Buffer", use_container_width=True):
        st.session_state.buffer_px = st.session_state.buffer_px + BUFFER_STEP
        st.rerun()
    z3.markdown(f'<span class="badge">buffer: {st.session_state.buffer_px}px</span>', unsafe_allow_html=True)

    st.markdown("---")
    n1, n2, n3 = st.columns(3)
    if n1.button("⬅️ Prev"):
        goto_index(prev_incomplete_index_from(st.session_state.idx))
        st.rerun()
    if n2.button("Skip"):
        ensure_row_exists(fid)
        decisions.loc[decisions["id"]==fid, "completed"] = "true"
        save_decisions()
        goto_index(next_incomplete_index_from(st.session_state.idx))
        st.rerun()
    if n3.button("Next ➡️"):
        goto_index(next_incomplete_index_from(st.session_state.idx))
        st.rerun()

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
            goto_index(next_incomplete_index_from(st.session_state.idx))
            st.rerun()

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
            goto_index(next_incomplete_index_from(st.session_state.idx))
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
c1, c2 = st.columns(2)
if c1.button("💾 Save Now"):
    save_decisions()
    st.toast("Saved.")

# download_button：永远从本地临时文件导出，不依赖 GCS 公网
local_dec_path = st.session_state.get("decisions_local_path", LOCAL_DECISIONS_TMP)
if os.path.exists(local_dec_path):
    with open(local_dec_path, "rb") as fh:
        c2.download_button(
            "⬇️ Download decisions CSV",
            data=fh.read(),
            file_name=Path(local_dec_path).name,
            mime="text/csv"
        )


