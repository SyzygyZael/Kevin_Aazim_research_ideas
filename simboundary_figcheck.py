#!/usr/bin/env python3
"""Objective figure audit: word-word overlaps + graphics strokes crossing text.

Line-through-text test: a stroke that crosses a word continues past the word's
bounding box. So for each row inside the box we require (i) the row is mostly
inked and (ii) the pixels just outside the box on BOTH sides are inked too.
Glyph strokes never satisfy (ii), so this does not fire on ordinary type.
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

DPI = 200
SC = DPI / 72.0
WHITE = 235


def words_per_page(bbox_html):
    html = open(bbox_html).read()
    out = []
    for pg in re.split(r'<page ', html)[1:]:
        out.append([(float(a), float(b), float(c), float(d), e) for a, b, c, d, e in
                    re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" '
                               r'xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', pg)])
    return out


def word_overlaps(ws, pad=0.4):
    bad = []
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            a, b = ws[i], ws[j]
            if (a[2] - pad > b[0] and b[2] - pad > a[0] and
                    a[3] - pad > b[1] and b[3] - pad > a[1] and abs(a[1] - b[1]) >= 1.5):
                bad.append((a, b))
    return bad


def line_through_text(img, ws, margin_pt=2.0, ink_frac=0.88):
    ink = (np.asarray(img.convert('L')) < WHITE)
    H, W = ink.shape
    m = max(2, int(round(margin_pt * SC)))
    hits = []
    for (x0, y0, x1, y1, txt) in ws:
        if not txt.strip():
            continue
        px0, py0 = int(x0 * SC), int(y0 * SC)
        px1, py1 = int(np.ceil(x1 * SC)), int(np.ceil(y1 * SC))
        if px1 - px0 < 6 or py1 - py0 < 5:
            continue
        if px0 - m < 0 or px1 + m >= W or py0 - m < 0 or py1 + m >= H:
            continue
        # A crossing stroke is CONTIGUOUS from outside the box, through it, to
        # the other side. Requiring near-full inking across the extended span
        # rejects glyph stems that merely sit between two inked elements.
        wide = ink[py0:py1, px0 - m:px1 + m]
        for r in range(wide.shape[0]):
            if wide[r].mean() >= ink_frac:
                hits.append((txt, 'horizontal', float(x0), float(y0)))
                break
        tall = ink[py0 - m:py1 + m, px0:px1]
        for c in range(tall.shape[1]):
            if tall[:, c].mean() >= ink_frac:
                hits.append((txt, 'vertical', float(x0), float(y0)))
                break
    return hits


def main(pdf_stem, pages):
    allw = words_per_page(f'{pdf_stem}_bbox.html')
    total_ov = total_ln = 0
    for p in pages:
        ws = allw[p - 1]
        img = Image.open(f'{pdf_stem}-{p:02d}.png')
        ov = word_overlaps(ws)
        ln = line_through_text(img, ws)
        total_ov += len(ov)
        total_ln += len(ln)
        print(f'page {p:>2}: words={len(ws):>4}  word-overlaps={len(ov):>2}  line-through-text={len(ln):>2}')
        for a, b in ov[:8]:
            print(f'      OVERLAP  {a[4]!r} x {b[4]!r}  (y {a[1]:.0f} / {b[1]:.0f})')
        for t, kind, x, y in ln[:8]:
            print(f'      STROKE   {kind:<10} through {t!r} at ({x:.0f},{y:.0f})')
    print(f'TOTAL word-overlaps={total_ov}  line-through-text={total_ln}')
    return total_ov + total_ln


if __name__ == '__main__':
    stem = sys.argv[1]
    pages = [int(a) for a in sys.argv[2:]]
    sys.exit(0 if main(stem, pages) == 0 else 1)
