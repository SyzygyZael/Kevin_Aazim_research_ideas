#!/usr/bin/env python3
"""Distance from every figure label to the nearest drawn LINE.

A "line" is a run of >=28 contiguous ink pixels (horizontal or vertical) at
400 dpi, i.e. ~5 pt, which is longer than any glyph feature apart from radical
and fraction bars. Word boxes are masked out so glyph strokes do not register.
Reports the nearest line's colour and position so each hit can be adjudicated.
"""
import re, sys
import numpy as np

try:
    from PIL import Image
except ImportError:                                          # optional extra
    raise SystemExit("This paper-QA tool needs Pillow and poppler-utils:\n"
                     "  pip install pillow\n"
                     "  apt-get install poppler-utils   (for pdftoppm/pdftotext)\n"
                     "It is not needed to run the experiments.")

S = 400 / 72.0
NAMES = {(0x2A,0x3D,0x66):'indigo', (0xD9,0x8A,0x2B):'amber', (0x3F,0x7D,0x53):'green',
         (0xA2,0x3B,0x3B):'red', (0x25,0x30,0x3A):'ink', (0x8A,0x94,0xA0):'edgegray'}

def colname(rgb):
    best, bd = 'other', 1e9
    for c, n in NAMES.items():
        d = sum((int(rgb[i]) - c[i])**2 for i in range(3))
        if d < bd: bd, best = d, n
    return best if bd < 9000 else 'pale/fill'

def runs_mask(ink, minlen=28):
    out = np.zeros_like(ink)
    for axis in (1, 0):
        a = ink if axis == 1 else ink.T
        o = out if axis == 1 else out.T
        for i in range(a.shape[0]):
            row = a[i]
            if not row.any(): continue
            d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
            st, en = np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]
            for s0, e0 in zip(st, en):
                if e0 - s0 >= minlen: o[i, s0:e0] = True
    return out

def audit(png, bboxhtml, page, pad=3, topn=12):
    pg = re.split(r'<page ', open(bboxhtml).read())[page]
    words = [(float(a), float(b), float(c), float(d), e) for a, b, c, d, e in
             re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" '
                        r'yMax="([\d.]+)">([^<]*)</word>', pg)]
    rgb = np.asarray(Image.open(png).convert('RGB'))
    ink = np.asarray(Image.open(png).convert('L')) < 235
    H, W = ink.shape
    wm = np.zeros_like(ink)
    for (x0, y0, x1, y1, t) in words:
        wm[max(0,int(y0*S)-pad):min(H,int(np.ceil(y1*S))+pad),
           max(0,int(x0*S)-pad):min(W,int(np.ceil(x1*S))+pad)] = True
    lines = runs_mask(ink & ~wm)
    ys, xs = np.nonzero(lines)
    capY = min(w[1] for w in words if w[4] == 'Figure')
    res = []
    for (x0, y0, x1, y1, t) in words:
        if y1 >= capY - 1 or not t.strip(): continue
        a, b, c, d = x0*S, y0*S, x1*S, y1*S
        sel = (xs > a-90) & (xs < c+90) & (ys > b-90) & (ys < d+90)
        if not sel.any(): continue
        px, py = xs[sel], ys[sel]
        dx = np.maximum(np.maximum(a-px, 0), px-c)
        dy = np.maximum(np.maximum(b-py, 0), py-d)
        dist = np.hypot(dx, dy)/S
        k = int(dist.argmin())
        res.append((dist[k], t, colname(rgb[py[k], px[k]]), x0, y0))
    res.sort()
    return res[:topn]

if __name__ == '__main__':
    stem = sys.argv[1]
    for p in [int(a) for a in sys.argv[2:]]:
        print(f'\n=== page {p}: label-to-LINE clearance (pt, smallest first) ===')
        for dist, t, col, x, y in audit(f'hi-{p:02d}.png', f'{stem}_bbox.html', p):
            flag = 'CHECK' if dist < 3.0 else '     '
            print(f'  {flag} {dist:5.2f}  {t[:24]:<24} nearest line={col:<9} @({x:.0f},{y:.0f})')
