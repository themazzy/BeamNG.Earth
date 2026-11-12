#!/usr/bin/env python3
"""Check CUZK's lidar services from this computer (czlidar.py): fetch the surface (DMP 1G) and terrain (DMR 5G) tiles
along a line and print both every ~10 m, with the surface over the terrain. On a bridge the surface is the deck and
the terrain the valley under it.

  python3 tools/czlidar_probe.py                       the Nusle bridge, Prague (deck ~40 m over the valley)
  python3 tools/czlidar_probe.py LAT LON LAT LON       any line in Czechia

Uses the standard library's HTTPS (the system proxy is honoured); nothing is kept on disk."""
import math
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import czlidar  # noqa: E402

NUSLE = (50.0690, 14.43005, 50.0630, 14.43055)


class Fetcher:
    def get(self, url, kind="demand", headers=None, lane=None):
        req = urllib.request.Request(url, headers={"User-Agent": "BeamNG.Earth czlidar probe"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read()
        print(f"  GET {url[:110]}... {len(body)} bytes", file=sys.stderr)
        return type("R", (), {"body": body})()


def merc(lat, lon):
    return (math.radians(lon) * czlidar.R, math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) * czlidar.R)


def main(argv):
    la0, lo0, la1, lo1 = (float(v) for v in argv[1:5]) if len(argv) >= 5 else NUSLE
    tiles = czlidar.Tiles(None, Fetcher())
    x0, y0 = merc(la0, lo0)
    x1, y1 = merc(la1, lo1)
    real = math.hypot(x1 - x0, y1 - y0) * math.cos(math.radians((la0 + la1) / 2))
    n = max(2, int(real / 10) + 1)
    xs = [x0 + (x1 - x0) * k / (n - 1) for k in range(n)]
    ys = [y0 + (y1 - y0) * k / (n - 1) for k in range(n)]
    dsm = tiles.heights("dmp1g", xs, ys)
    dtm = tiles.heights("dmr5g", xs, ys)
    print(f"{'m':>6} {'surface':>9} {'terrain':>9} {'over':>7}")
    for k in range(n):
        print(f"{real * k / (n - 1):6.0f} {dsm[k]:9.2f} {dtm[k]:9.2f} {dsm[k] - dtm[k]:7.2f}")
    print(f"{tiles.stats['requests']} requests, {tiles.stats['bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv)
