// Photo of the AR view for tagar.html: the camera image with the rendered panel over it, plus a caption
// line, so a client can keep or send the preview. No three.js or DOM here, so node can test it.

// Lay the rendered scene over the camera image. Both are RGBA, w x h, rows bottom-up as gl.readPixels
// gives them (camFlip false: the camera rows are already top-down). The scene is premultiplied by alpha,
// as three.js normal blending leaves it on a cleared transparent target. Returns top-down RGBA.
export function composite(cam, scene, w, h, camFlip = true) {
  if (cam.length !== w * h * 4 || scene.length !== w * h * 4) throw new Error("camera and scene must both be w x h RGBA");
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const so = (h - 1 - y) * w * 4, co = (camFlip ? h - 1 - y : y) * w * 4, oo = y * w * 4;
    for (let x = 0; x < w * 4; x += 4) {
      const k = 1 - scene[so + x + 3] / 255;
      out[oo + x] = scene[so + x] + cam[co + x] * k;
      out[oo + x + 1] = scene[so + x + 1] + cam[co + x + 1] * k;
      out[oo + x + 2] = scene[so + x + 2] + cam[co + x + 2] * k;
      out[oo + x + 3] = 255;
    }
  }
  return out;
}

// One caption line: panel type, real size, tilt, and its typical power. For n panels: the count and
// their total peak power; with the site's peak sun hours, also the energy per day (as energy() in panel.js).
export function caption(look, tiltDeg, n = 1, sunHours = 0, derate = 0.8) {
  const size = `${look.w.toFixed(2)} × ${look.d.toFixed(2)} m`;
  let text = n > 1
    ? `${n} × ${look.name} panels · ${size} each · ${tiltDeg}° tilt · ${((n * look.watts) / 1000).toFixed(1)} kW peak`
    : `${look.name} panel · ${size} · ${tiltDeg}° tilt · ~${look.watts} W typical`;
  if (sunHours > 0) text += ` · ~${((n * look.watts * sunHours * derate) / 1000).toFixed(1)} kWh/day`;
  return text;
}

// File name from the local time, e.g. solar-preview-20261101-143005.png
export function shotName(d) {
  const p = (n) => String(n).padStart(2, "0");
  return `solar-preview-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.png`;
}
