"""Czechia's bridge deck heights from CUZK's lidar surface (czlidar.py, roadz.RoadZ._cz_from): the tile grid and
requests, the exportImage TIFF, the tiles kept on disk, and the decks measured from the surface on our DEM's datum,
with a bridge the 2009-2013 surface does not show left to the inference."""
import io
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import czlidar  # noqa: E402
import fast  # noqa: E402
import geodesy  # noqa: E402
import roadz  # noqa: E402

NATIVE = fast.AVAILABLE and getattr(fast._ef, "VERSION", 0) >= max(roadz.NATIVE, roadz.GAP_NATIVE)
PRAGUE = (50.0531, 14.4466)         # (lat, lon: the Nusle bridge)


def tiff(a, nodata=None):
    """A float32 TIFF of array a, as ArcGIS's exportImage answers (strips, no compression)."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.fromarray(np.asarray(a, np.float32), mode="F")
    img.save(buf, format="TIFF")
    return buf.getvalue()


class Resp:
    def __init__(self, body):
        self.body = body


class FakeFetcher:
    def __init__(self, body):
        self.body, self.urls = body, []

    def get(self, url, kind="demand", headers=None, lane=None):
        self.urls.append((url, lane))
        return Resp(self.body)


class GridTest(unittest.TestCase):
    def test_tiles_round_the_box(self):
        x, y = roadz.merc_xy(PRAGUE[0], PRAGUE[1])
        (tx, ty), = czlidar.tiles_for((float(x), float(y), float(x), float(y)))
        b = czlidar.tile_box(tx, ty)
        self.assertTrue(b[0] <= x < b[2] and b[1] <= y < b[3])
        self.assertAlmostEqual(b[2] - b[0], czlidar.WORLD / (1 << czlidar.TILE_Z))
        self.assertEqual(len(czlidar.tiles_for((b[0] + 1, b[1] + 1, b[2] + 1, b[3] - 1))), 2)

    def test_url(self):
        u = czlidar.tile_url("dmp1g", 35397, 22224)
        self.assertTrue(u.startswith("https://ags.cuzk.gov.cz/arcgis2/rest/services/dmp1g/ImageServer/exportImage?"))
        for part in ("bboxSR=3857", "imageSR=3857", "size=256,256", "pixelType=F32", "format=tiff", "f=image"):
            self.assertIn(part, u)
        with self.assertRaises(ValueError):
            czlidar.tile_url("../x", 1, 1)

    def test_inside(self):
        self.assertTrue(czlidar.inside(*PRAGUE))
        self.assertFalse(czlidar.inside(48.21, 16.37), "Vienna")
        self.assertIn("lidar", roadz.sources_at(*PRAGUE, lidar=True))
        self.assertNotIn("lidar", roadz.sources_at(*PRAGUE, lidar=False))


class DecodeTest(unittest.TestCase):
    def test_heights_and_nodata(self):
        a = np.full((czlidar.SIZE, czlidar.SIZE), 250.5, np.float32)
        a[0, :10] = czlidar.NODATA
        a[5, 5] = -3.4e38
        got = czlidar.decode(tiff(a))
        self.assertEqual(got.shape, (czlidar.SIZE, czlidar.SIZE))
        self.assertTrue(np.isnan(got[0, :10]).all())
        self.assertTrue(np.isnan(got[5, 5]))
        self.assertEqual(float(got[100, 100]), 250.5)

    def test_error_page_refused(self):
        with self.assertRaises(ValueError):
            czlidar.decode(b'{"error":{"code":400,"message":"Unable to complete operation."}}')
        with self.assertRaises(ValueError):
            czlidar.decode(tiff(np.zeros((10, 10))))


class TilesTest(unittest.TestCase):
    def test_kept_on_disk_and_points_in_boxes(self):
        a = np.arange(czlidar.SIZE * czlidar.SIZE, dtype=np.float32).reshape(czlidar.SIZE, czlidar.SIZE) % 500
        with tempfile.TemporaryDirectory() as d:
            f = FakeFetcher(tiff(a))
            t = czlidar.Tiles(os.path.join(d, "czlidar"), f)
            b = czlidar.tile_box(35397, 22224)
            box = (b[0] + 10, b[1] + 10, b[0] + 40, b[1] + 25)
            x, y, z = t.points("dmp1g", [box])
            self.assertEqual(len(f.urls), 1)
            self.assertEqual(f.urls[0][1], czlidar.LANE)
            self.assertTrue(((x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])).all())
            self.assertGreater(len(z), 50)
            np.testing.assert_allclose(t.heights("dmp1g", x, y), z)
            t2 = czlidar.Tiles(os.path.join(d, "czlidar"), f)
            x2, _y2, z2 = t2.points("dmp1g", [box])
            self.assertEqual(len(f.urls), 1, "the second process reads the kept tile")
            np.testing.assert_array_equal(z, z2)
            self.assertTrue(czlidar.credited_near(d, *roadz.merc_ll(x[0], y[0]), 100.0))
            self.assertFalse(czlidar.credited_near(d, 49.2, 16.6, 100.0), "Brno: nothing kept there")


class FakeDem:
    """Our DEM: f(e, n) in frame's metres."""

    def __init__(self, frame, f):
        self.frame, self.f = frame, f

    def sample(self, la, lo, zoom):
        e, n, _ = self.frame.geodetic_to_enu(np.asarray(la, np.float64), np.asarray(lo, np.float64), 0.0)
        return self.f(np.asarray(e, np.float64), np.asarray(n, np.float64))


class FakeMD:
    def __init__(self, dem):
        self.dem = dem


class FakeTiles:
    """czlidar.Tiles over analytic heights: surface(e, n) and terrain(e, n) in the frame's metres (Baltic datum)."""

    def __init__(self, frame, surface, terrain):
        self.frame, self.f = frame, {"dmp1g": surface, "dmr5g": terrain}

    def _en(self, x, y):
        la, lo = roadz.merc_ll(x, y)
        e, n, _ = self.frame.geodetic_to_enu(la, lo, 0.0)
        return np.asarray(e, np.float64), np.asarray(n, np.float64)

    def points(self, product, boxes, kind="demand"):
        boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
        x0, y0 = boxes[:, :2].min(0)
        x1, y1 = boxes[:, 2:].max(0)
        gx, gy = np.meshgrid(np.arange(x0, x1, 2.39), np.arange(y0, y1, 2.39))
        gx, gy = gx.ravel(), gy.ravel()
        m = np.zeros(len(gx), bool)
        for b in boxes:
            m |= (gx >= b[0]) & (gx <= b[2]) & (gy >= b[1]) & (gy <= b[3])
        gx, gy = gx[m], gy[m]
        return gx, gy, self.f[product](*self._en(gx, gy))

    def heights(self, product, x, y, kind="demand"):
        return self.f[product](*self._en(np.asarray(x, np.float64), np.asarray(y, np.float64)))


@unittest.skipUnless(NATIVE, "earthfast roadz kernels")
class DeckTest(unittest.TestCase):
    """A 160 m bridge carrying a level road over a valley 8 m deep, its sides 8 % (Bpv 250 m at the bridge ends); our
    DEM 0.4 m under CUZK's datum."""
    TOP = 250.0
    SHIFT = 0.4

    def valley(self, e, n):
        return self.TOP - 8.0 * np.clip(1.0 - np.abs(e) / 100.0, 0.0, 1.0)

    def measure(self, surface):
        frame = geodesy.EnuFrame(*PRAGUE)
        rz = roadz.RoadZ(None, object(), FakeMD(FakeDem(frame, lambda e, n: self.valley(e, n) - self.SHIFT)))
        rz.cz = FakeTiles(frame, surface, self.valley)
        xy = np.c_[np.linspace(-80, 80, 17), np.zeros(17)]
        la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        ring = np.c_[np.asarray(lo), np.asarray(la)]
        key = roadz.mapcells.cell_of(*PRAGUE)
        return rz._cz_from(key, frame, [(ring, 4.0)], [(xy, 4.0)], 1, [])

    def test_deck_measured_on_our_datum(self):
        got = self.measure(lambda e, n: np.where(np.abs(n) < 4.5, self.TOP, self.valley(e, n)))
        self.assertEqual(len(got), 1)
        self.assertEqual(int(got.level[0]), 1)
        self.assertTrue(got.corrected)
        z = got.z[np.isfinite(got.z)]
        self.assertGreater(len(z), 30)
        np.testing.assert_allclose(z, self.TOP - self.SHIFT, atol=0.3,
                                   err_msg="the deck, not the valley beside it, on our DEM's datum")

    def test_bridge_missing_from_the_surface_keeps_the_inference(self):
        got = self.measure(self.valley)
        self.assertEqual(len(got), 0, "the surface shows only the valley: a bridge built after the flights")

    def test_unbuilt_rule(self):
        s = np.arange(0.0, 101.0, 4.0)
        self.assertTrue(roadz.RoadZ._cz_unbuilt(s, np.full(len(s), 1.3), np.full(len(s), 1.0)))
        self.assertFalse(roadz.RoadZ._cz_unbuilt(s, np.full(len(s), 7.0), np.full(len(s), 1.0)))
        self.assertFalse(roadz.RoadZ._cz_unbuilt(s[:5], np.full(5, 1.0), np.full(5, 1.0)), "too short to tell")


if __name__ == "__main__":
    unittest.main()
