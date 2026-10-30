# p, odds p/(1-p) and 1/(1-p) as functions of p: two panels (linear y, log y), direct-labelled.  Writes an SVG.
import math, sys
out = sys.argv[1]
W, H = 1000, 1000
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
C = {"p": "#2a78d6", "odds": "#eb6834", "inv": "#1baf7a"}
f = {"p": lambda p: p, "odds": lambda p: p / (1 - p), "inv": lambda p: 1 / (1 - p)}
lab = {"p": "probability  p", "odds": "odds  p/(1−p)", "inv": "1/(1−p)  = 1 + odds"}
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
       f'<text x="40" y="38" font-size="20" fill="{INK}" font-weight="600">Probability, odds and 1/(1−p) as functions of p</text>',
       f'<text x="40" y="62" font-size="14" fill="{INK2}">1/(1−p) is the odds shifted up by exactly 1: it starts at 1 instead of 0, and the two converge as p → 1</text>']
def panel(x0, y0, w, h, ylog, pmax, ymin, ymax, yticks, title):
    def X(p): return x0 + w * p / pmax
    def Y(v):
        if ylog: t = (math.log10(v) - math.log10(ymin)) / (math.log10(ymax) - math.log10(ymin))
        else: t = (v - ymin) / (ymax - ymin)
        return y0 + h * (1 - t)
    svg.append(f'<text x="{x0}" y="{y0 - 14}" font-size="15" fill="{INK}" font-weight="600">{title}</text>')
    for v in yticks:
        yy = Y(v)
        svg.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
        s = f"{v:g}"
        svg.append(f'<text x="{x0 - 8}" y="{yy + 4:.1f}" font-size="12" fill="{MUTED}" text-anchor="end">{s}</text>')
    for p in [i / 10 for i in range(0, 11) if i / 10 <= pmax + 1e-9]:
        xx = X(p)
        svg.append(f'<text x="{xx:.1f}" y="{y0 + h + 18}" font-size="12" fill="{MUTED}" text-anchor="middle">{p:g}</text>')
    svg.append(f'<line x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}" stroke="{AXIS}" stroke-width="1"/>')
    svg.append(f'<text x="{x0 + w / 2}" y="{y0 + h + 40}" font-size="13" fill="{INK2}" text-anchor="middle">p</text>')
    svg.append(f'<clipPath id="c{x0}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath>')
    ends = {}
    for k in ("p", "odds", "inv"):
        pts = []
        n = 600
        for i in range(n + 1):
            p = pmax * i / n
            if p >= 1: break
            v = f[k](p)
            if ylog and v <= 0: continue
            pts.append((X(p), max(min(Y(v), y0 + h + 50), y0 - 50)))
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        svg.append(f'<polyline points="{d}" fill="none" stroke="{C[k]}" stroke-width="2" clip-path="url(#c{x0})"/>')
        ends[k] = pts
    return X, Y, ends
# left: linear, p in [0, 0.95], y in [0, 10]
X1, Y1, e1 = panel(80, 120, 860, 330, False, 1.0, 0, 10, [0, 2, 4, 6, 8, 10], "Linear scale")
# labels placed where each curve is readable
svg.append(f'<text x="{X1(0.97):.1f}" y="{Y1(1) - 8:.1f}" font-size="13" fill="{INK}" text-anchor="end">● </text>')
def label(X, Y, p, v, k, dx=8, dy=-6, anchor="start"):
    svg.append(f'<circle cx="{X(p):.1f}" cy="{Y(v):.1f}" r="4" fill="{C[k]}" stroke="{SURF}" stroke-width="2"/>')
    svg.append(f'<text x="{X(p) + dx:.1f}" y="{Y(v) + dy:.1f}" font-size="13" fill="{INK}" text-anchor="{anchor}">{lab[k]}</text>')
svg.pop()   # (drop the placeholder)
label(X1, Y1, 0.6, 0.6, "p", 8, 18)
label(X1, Y1, 0.85, f["odds"](0.85), "odds", 10, 14)
label(X1, Y1, 0.8, f["inv"](0.8), "inv", -10, -8, "end")
# right: log scale, p in [0, 0.99], y in [0.01, 100]
X2, Y2, e2 = panel(80, 570, 860, 330, True, 1.0, 0.01, 1000, [0.01, 0.1, 1, 10, 100, 1000], "Log scale (same curves)")
label(X2, Y2, 0.3, 0.3, "p", 8, 18)
label(X2, Y2, 0.1, f["odds"](0.1), "odds", 8, 16)
label(X2, Y2, 0.5, f["inv"](0.5), "inv", -8, -10, "end")
svg.append(f'<text x="80" y="965" font-size="13" fill="{MUTED}">At p = 0.5: p = 0.5, odds = 1, 1/(1−p) = 2.  At p = 0.99: odds = 99, 1/(1−p) = 100.</text>')
svg.append('</svg>')
open(out, "w").write("\n".join(svg))
