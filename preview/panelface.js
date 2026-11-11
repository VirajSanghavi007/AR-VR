// The Standard module's face, drawn on a 2D canvas: white backsheet between chamfered mono cells,
// each cell a touch different in tint with a faint sheen towards its centre, fine silver fingers,
// 9 thin busbars and a bevelled frame. Shared by tagar.html (tabletop AR) and roof.html (the
// customer roof viewer) so the panels look the same everywhere. No three.js needed: tests run it
// in node against a recording context.
import { cellLayout, chamferCell, cellShade } from "./panel.js?v=9";

// "#rrggbb" to linear-light [r, g, b] 0..1, the way THREE.Color reads a hex string with colour
// management on (three r152+). The face has always been drawn from those linear values, which is
// what gives the cells their deep near-black blue, so keep it.
export function toLinear(hex) {
  const n = parseInt(hex.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const c = v / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
}

// CSS colour of a linear colour times k, clamped at white.
export function shade(lin, k) {
  return `rgb(${lin.map((v) => Math.round(Math.min(1, v * k) * 255))})`;
}

// Canvas size for a look: W px across, H along so the cells keep their real proportions.
export function faceSize(look, W = 1024) {
  return { W, H: Math.round((W * look.d) / look.w) };
}

// Draw the face into context g (W x H px). Returns the number of cells drawn.
export function drawStandardFace(g, look, W, H) {
  if (look.gap) { g.fillStyle = look.gap; g.fillRect(0, 0, W, H); }
  const base = toLinear(look.cell), cells = cellLayout(look);
  cells.forEach((cell, i) => {
    const x = cell.x * W, y = cell.y * H, w = cell.w * W, h = cell.h * H, row = Math.floor(i / look.cols);
    // half-cut: a pair of rows is one wafer, cut across the middle, so only its outer corners are chamfered
    const top = !look.halfCut || row % 2 === 0, bottom = !look.halfCut || row % 2 === 1;
    const pts = chamferCell(x, y, w, h, w * 0.07, top, bottom), k = cellShade(i);
    const sheen = g.createLinearGradient(x, y, x + w, y + h);
    sheen.addColorStop(0, shade(base, k * 0.92)); sheen.addColorStop(0.5, shade(base, k * 1.12));
    sheen.addColorStop(1, shade(base, k * 0.95));
    g.fillStyle = sheen;
    g.beginPath(); pts.forEach(([px, py], j) => (j ? g.lineTo(px, py) : g.moveTo(px, py))); g.closePath(); g.fill();
    g.save(); g.clip();
    g.fillStyle = "rgba(170,182,196,.10)";                                     // fingers: hair-thin, across the busbars
    for (let fy = y + 2; fy < y + h; fy += 3.2) g.fillRect(x, fy, w, 0.6);
    g.fillStyle = look.busbar;
    for (let b = 1; b <= 9; b++) g.fillRect(x + (w * b) / 10 - 0.6, y, 1.2, h);
    g.restore();
  });
  const f = (look.frameMm / 1000 / look.w) * W, fr = toLinear(look.frame);
  const bevel = (x0, y0, x1, y1, light) => {                                  // lit outer edge, shaded inner lip
    const gr = g.createLinearGradient(x0, y0, x1, y1);
    gr.addColorStop(0, shade(fr, light ? 1.12 : 0.8)); gr.addColorStop(0.6, shade(fr, 1)); gr.addColorStop(1, shade(fr, 0.62));
    return gr;
  };
  if (f > 0) {
    g.fillStyle = bevel(0, 0, 0, f, true); g.fillRect(0, 0, W, f);
    g.fillStyle = bevel(0, H, 0, H - f, false); g.fillRect(0, H - f, W, f);
    g.fillStyle = bevel(0, 0, f, 0, true); g.fillRect(0, f, f, H - 2 * f);
    g.fillStyle = bevel(W, 0, W - f, 0, false); g.fillRect(W - f, f, f, H - 2 * f);
  }
  return cells.length;
}
