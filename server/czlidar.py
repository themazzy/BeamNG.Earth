"""Czechia's airborne lidar surface and terrain (CUZK, the State Administration of Land Surveying and Cadastre) for the
road heights (roadz.py's "lidar" source there: the bridge decks measured along the map's own lines, as with the US
3DEP point clouds; GitHub #627 for the interchanges it is about).

Two of CUZK's ArcGIS image services, open data under CC BY 4.0 ("(c) CUZK"):
- dmp1g, the digital surface model (DMP 1G): the lidar's first surface with its structures (bridge decks, buildings)
  and vegetation, flown 2009-2013, 0.4 m on built objects;
- dmr5g, the digital terrain model (DMR 5G): the bare ground (bridges taken out), 0.18 m in the open.
Both answer exportImage in any projection: a square of SIZE pixels per Web-Mercator tile of zoom TILE_Z (~390 m at
50 N; ~1.5 m a pixel, the services' own grid is 2 m), float32 GeoTIFF, heights above the Baltic datum (Bpv). The
surface's pixels go to the deck fit as points (class 1); the terrain's give the datum offset onto our DEM and refuse
a deck the surface does not show (a bridge built after the flights: its "deck" read is the ground under it).

Tiles are kept under <cache>/czlidar/<product>/<z>/<x>/<y>.npy (~256 kB each), capped at TILES_MB, the least
recently read dropped first; one of the server's processes fetches a tile (xproc), the others read the kept copy.
"""
import contextlib
import logging
import math
import os
import threading
import urllib.parse
from collections import OrderedDict

import numpy as np

log = logging.getLogger("earthd.czlidar")

ON = os.environ.get("BNE_ROADZ_CZ", "on") != "off"
DATASET = "cuzk:dmp1g"          # (roadz's dataset name for it: the US ones are 3DEP project names)
# Czechia's box (lat0, lon0, lat1, lon1); the services answer nodata outside the country (no points: the inference)
BOUNDS = (48.55, 12.09, 51.06, 18.86)
HOST = "ags.cuzk.gov.cz"
LANE = "roadz"                  # (its own bucket: the orthophoto tiles share the host)
URL = "https://ags.cuzk.gov.cz/arcgis2/rest/services/{}/ImageServer/exportImage"
PRODUCTS = ("dmp1g", "dmr5g")   # surface, terrain
TILE_Z = 16
SIZE = 256                      # pixels a tile side
R = 6378137.0                   # EPSG:3857's sphere
WORLD = 2.0 * math.pi * R
NODATA = -9999.0                # asked of the service; anything at or below NODATA_BELOW is none
NODATA_BELOW = -1000.0
TOP = 2000.0                    # m: higher is not Czech ground (Snezka 1603 m)
TILES_MB = float(os.environ.get("BNE_CZ_TILES_MB", "1024"))
MEM_TILES = 64


def inside(lat, lon):
    return ON and BOUNDS[0] <= lat <= BOUNDS[2] and BOUNDS[1] <= lon <= BOUNDS[3]


def tile_box(x, y, z=TILE_Z):
    """(x0, y0, x1, y1) of Web-Mercator tile (x, y) in EPSG:3857 metres."""
    n = 1 << z
    s = WORLD / n
    x0 = -WORLD / 2 + x * s
    y1 = WORLD / 2 - y * s
    return (x0, y1 - s, x0 + s, y1)


def tiles_for(box, z=TILE_Z):
    """[(x, y)] of the tiles meeting box (x0, y0, x1, y1) in EPSG:3857."""
    n = 1 << z
    s = WORLD / n
    tx0 = int(math.floor((box[0] + WORLD / 2) / s))
    tx1 = int(math.floor((box[2] + WORLD / 2) / s))
    ty0 = int(math.floor((WORLD / 2 - box[3]) / s))
    ty1 = int(math.floor((WORLD / 2 - box[1]) / s))
    return [(tx, ty) for ty in range(max(0, ty0), min(n - 1, ty1) + 1)
            for tx in range(max(0, tx0), min(n - 1, tx1) + 1)]


def tile_url(product, x, y, z=TILE_Z):
    if product not in PRODUCTS:
        raise ValueError(f"CUZK product {product!r}")
    x0, y0, x1, y1 = tile_box(x, y, z)
    q = {"bbox": f"{x0:.3f},{y0:.3f},{x1:.3f},{y1:.3f}", "bboxSR": 3857, "imageSR": 3857, "size": f"{SIZE},{SIZE}",
         "format": "tiff", "pixelType": "F32", "noData": f"{NODATA:g}", "interpolation": "RSP_BilinearInterpolation",
         "f": "image"}
    return URL.format(product) + "?" + urllib.parse.urlencode(q, safe=",")


def decode(body):
    """(SIZE, SIZE) float64 heights of an exportImage TIFF (NaN: none). Raises ValueError for anything else (an
    ArcGIS error page answered with a 200, a wrong size)."""
    import cog
    import rangeread
    if body[:2] not in (b"II", b"MM"):
        raise ValueError("CUZK exportImage: not a TIFF: " + bytes(body[:80]).decode("latin-1", "replace"))
    c = cog.Cog(rangeread.MemoryRange(body))
    if (c.width, c.height) != (SIZE, SIZE) or c.bands != 1:
        raise ValueError(f"CUZK exportImage: {c.width} x {c.height} x {c.bands}, not {SIZE} x {SIZE}")
    a = np.asarray(c.read_window(0, 0, SIZE, SIZE), np.float64)
    bad = ~np.isfinite(a) | (a <= NODATA_BELOW) | (a > TOP)
    if c.nodata is not None:
        bad |= a == c.nodata
    a[bad] = np.nan
    return a


def pixel_centres(x, y, z=TILE_Z):
    """(px (SIZE, SIZE), py (SIZE, SIZE)) EPSG:3857 centres of tile (x, y)'s pixels, row 0 the north."""
    x0, y0, x1, y1 = tile_box(x, y, z)
    d = (x1 - x0) / SIZE
    c = (np.arange(SIZE) + 0.5) * d
    return np.meshgrid(x0 + c, y1 - c)


def sample(grid, x, y, px, py, z=TILE_Z):
    """Heights of tile (x, y)'s grid (SIZE, SIZE) at EPSG:3857 points px, py (nearest pixel; NaN outside)."""
    x0, y0, x1, y1 = tile_box(x, y, z)
    d = (x1 - x0) / SIZE
    col = np.floor((np.asarray(px, np.float64) - x0) / d)
    row = np.floor((y1 - np.asarray(py, np.float64)) / d)
    ok = (col >= 0) & (col < SIZE) & (row >= 0) & (row < SIZE)
    out = np.full(np.shape(col), np.nan)
    out[ok] = grid[row[ok].astype(np.int64), col[ok].astype(np.int64)]
    return out


class Tiles:
    """The tiles kept under root (None: in memory only), fetched through fetcher (netfetch.Fetcher)."""

    def __init__(self, root, fetcher):
        self.root, self.fetcher = root, fetcher
        self._lock = threading.Lock()
        self._locks = {}
        self._mem = OrderedDict()
        self.stats = {"requests": 0, "bytes": 0}

    def path(self, product, x, y):
        return os.path.join(self.root, product, str(TILE_Z), str(x), f"{y}.npy") if self.root else None

    def _kept(self, product, x, y):
        k = (product, x, y)
        with self._lock:
            got = self._mem.get(k)
            if got is not None:
                self._mem.move_to_end(k)
                return got
        p = self.path(product, x, y)
        if p is None:
            return None
        try:
            got = np.load(p, allow_pickle=False).astype(np.float64)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            log.debug("czlidar tile %s: %s", p, e)
            return None
        if got.shape != (SIZE, SIZE):
            return None
        with contextlib.suppress(OSError):
            os.utime(p)
        self._remember(k, got)
        return got

    def _remember(self, k, grid):
        with self._lock:
            self._mem[k] = grid
            while len(self._mem) > MEM_TILES:
                self._mem.popitem(last=False)

    def get(self, product, x, y, kind="demand"):
        """The tile's heights (SIZE, SIZE): kept, or fetched by one of the server's processes."""
        got = self._kept(product, x, y)
        if got is not None:
            return got
        with self._lock:
            lock = self._locks.setdefault((product, x, y), threading.Lock())
        with lock:
            got = self._kept(product, x, y)
            if got is not None:
                return got
            p = self.path(product, x, y)
            mine = False
            if p:
                import xproc
                got, mine = xproc.once(p + ".lock", lambda: self._kept(product, x, y), xproc.STALE_S, kind)
                if got is not None:
                    return got
            try:
                body = self.fetcher.get(tile_url(product, x, y), kind=kind, lane=LANE).body
                with self._lock:
                    self.stats["requests"] += 1
                    self.stats["bytes"] += len(body)
                got = decode(body)
                if p:
                    import staging
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    tmp = f"{p}.{os.getpid()}.{threading.get_ident()}.tmp"
                    with open(tmp, "wb") as fh:
                        np.save(fh, got.astype(np.float32))
                    staging.replace_cached(tmp, p)
            finally:
                if mine:
                    xproc.release(p + ".lock")
            self._remember((product, x, y), got)
        if p:
            self.trim()
        return got

    def trim(self):
        """The kept tiles cut back to TILES_MB, the least recently read first."""
        if not self.root or not os.path.isdir(self.root):
            return
        files = []
        for dp, _dn, fns in os.walk(self.root):
            for fn in fns:
                if fn.endswith(".npy"):
                    with contextlib.suppress(OSError):
                        st = os.stat(os.path.join(dp, fn))
                        files.append((st.st_mtime, st.st_size, os.path.join(dp, fn)))
        total = sum(f[1] for f in files)
        for _t, size, path in sorted(files):
            if total <= TILES_MB * 1e6:
                break
            with contextlib.suppress(OSError):
                os.remove(path)
                total -= size

    def points(self, product, boxes, kind="demand"):
        """(x (n,), y (n,), z (n,)) EPSG:3857 pixel centres and heights of product inside any of boxes (x0, y0, x1, y1
        in EPSG:3857), NaN-free. Raises what a tile's fetch raises (the cell is then tried again later)."""
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        want = sorted({t for b in boxes for t in tiles_for(b)})
        xs, ys, zs = [], [], []
        for tx, ty in want:
            tb = tile_box(tx, ty)
            near = boxes[(boxes[:, 0] <= tb[2]) & (boxes[:, 2] >= tb[0])
                         & (boxes[:, 1] <= tb[3]) & (boxes[:, 3] >= tb[1])]
            grid = self.get(product, tx, ty, kind)
            px, py = pixel_centres(tx, ty)
            m = np.zeros(px.shape, bool)
            for b in near:
                m |= (px >= b[0]) & (px <= b[2]) & (py >= b[1]) & (py <= b[3])
            m &= np.isfinite(grid)
            xs.append(px[m])
            ys.append(py[m])
            zs.append(grid[m])
        if not xs:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)

    def heights(self, product, px, py, kind="demand"):
        """Heights of product at EPSG:3857 points px, py (n,) (nearest pixel; NaN where none)."""
        px, py = np.asarray(px, np.float64), np.asarray(py, np.float64)
        out = np.full(len(px), np.nan)
        if not len(px):
            return out
        n = 1 << TILE_Z
        s = WORLD / n
        tx = np.floor((px + WORLD / 2) / s).astype(np.int64)
        ty = np.floor((WORLD / 2 - py) / s).astype(np.int64)
        for a, b in sorted(set(zip(tx.tolist(), ty.tolist()))):
            m = (tx == a) & (ty == b)
            out[m] = sample(self.get(product, a, b, kind), a, b, px[m], py[m])
        return out


def credited_near(cache_dir, lat, lon, near_m):
    """Is a surface tile kept on this computer within near_m of (lat, lon) (credits.feature_ids)? Disk only."""
    if not (cache_dir and inside(lat, lon)):
        return False
    lat = max(-85.0, min(85.0, float(lat)))
    x = math.radians(float(lon)) * R
    y = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * R
    d = near_m / max(0.05, math.cos(math.radians(lat)))      # (Mercator metres are longer)
    root = os.path.join(cache_dir, "czlidar", "dmp1g", str(TILE_Z))
    tiles = tiles_for((x - d, y - d, x + d, y + d))
    return len(tiles) <= 400 and any(os.path.exists(os.path.join(root, str(tx), f"{ty}.npy")) for tx, ty in tiles)


_AT = {}
_AT_LOCK = threading.Lock()


def at(cache_dir, fetcher):
    """The process's Tiles of the server cache cache_dir (<cache_dir>/czlidar; None: in memory only)."""
    root = os.path.join(cache_dir, "czlidar") if cache_dir else None
    with _AT_LOCK:
        got = _AT.get((root, id(fetcher)))
        if got is None:
            got = _AT[(root, id(fetcher))] = Tiles(root, fetcher)
        return got
