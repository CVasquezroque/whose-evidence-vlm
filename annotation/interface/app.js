// Evidence Ownership Annotator: a single-file canvas tool with no dependencies.
// Coordinates: polygons are stored in ORIGINAL image pixels; everything on screen is the
// 1024x1024 experimental canvas (data_contracts.md §1). Sham translations are integer canvas px.
// Mask previews here approximate the renderer; the Python renderer's masks are authoritative.
"use strict";

const N = 1024, DIL = 4, GRAY = 128;
const KEEPOUT_PAD = 12, TAG_ZONE = 34;           // keep shams away from boxes and tag areas
const PCOL = { A: "#ffc800", B: "#00c8ff" };
const SWATCH = { black:"#111", white:"#f4f4f4", gray:"#888", red:"#d32f2f", orange:"#f57c00", yellow:"#fbc02d",
  green:"#388e3c", blue:"#1976d2", purple:"#7b1fa2", pink:"#f06292", brown:"#6d4c41" };
const COND = { I11: ["S_A", "S_B"], I01: ["M_A", "S_B"], I10: ["S_A", "M_B"], I00: ["M_A", "M_B"] };

const $ = (id) => document.getElementById(id);
const stage = $("stage"), ctx = stage.getContext("2d");
let CFG = null, queue = [], cand = null, ann = null, img = null, reviewImg = null;
let person = "A", mode = "draw", view = "orig", drawing = [], dragging = null, dirty = false;
let masks = { A: null, B: null };   // {raw, dil, idx:[...], bbox, area, rawArea, rawBBox}
let showOverlay = true;
let samPts = { A: [], B: [] }, samProp = { A: null, B: null }, samBusy = false;   // SAM-HQ assistance (D-016)

// ---------- API ----------
async function api(path, opts) {
  const r = await fetch(path, opts);
  const j = await r.json();
  if (!r.ok) throw new Error((j && (j.error + (j.details ? ": " + j.details.join("; ") : ""))) || r.statusText);
  return j;
}

// ---------- geometry ----------
function tf() { return cand.canvas_transform; }
function toCanvas([x, y]) {
  const t = tf(), sx = t.resized_size[0] / t.orig_size[0], sy = t.resized_size[1] / t.orig_size[1];
  return [x * sx + t.offset[0], y * sy + t.offset[1]];
}
function toOrig([x, y]) {
  const t = tf(), sx = t.resized_size[0] / t.orig_size[0], sy = t.resized_size[1] / t.orig_size[1];
  return [+(((x - t.offset[0]) / sx).toFixed(2)), +(((y - t.offset[1]) / sy).toFixed(2))];
}
function contentBox() { const t = tf(); return [t.offset[0], t.offset[1], t.offset[0] + t.resized_size[0], t.offset[1] + t.resized_size[1]]; }
function eventPt(ev) {
  const r = stage.getBoundingClientRect();
  return [(ev.clientX - r.left) * N / r.width, (ev.clientY - r.top) * N / r.height];
}

// ---------- masks ----------
function rasterize(polys) {
  const oc = new OffscreenCanvas(N, N), c = oc.getContext("2d");
  c.fillStyle = "#fff";
  for (const poly of polys) {
    c.beginPath();
    poly.forEach((v, i) => { const [x, y] = toCanvas(v); i ? c.lineTo(x, y) : c.moveTo(x, y); });
    c.closePath(); c.fill();
  }
  const d = c.getImageData(0, 0, N, N).data, m = new Uint8Array(N * N);
  for (let i = 0; i < N * N; i++) m[i] = d[i * 4 + 3] >= 128 ? 1 : 0;
  return m;
}
function dilate(m, r) {
  const h = new Uint8Array(N * N), out = new Uint8Array(N * N);
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    if (!m[y * N + x]) continue;
    for (let k = Math.max(0, x - r); k <= Math.min(N - 1, x + r); k++) h[y * N + k] = 1;
  }
  for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
    if (!h[y * N + x]) continue;
    for (let k = Math.max(0, y - r); k <= Math.min(N - 1, y + r); k++) out[k * N + x] = 1;
  }
  return out;
}
function stats(m) {
  let area = 0, x0 = N, y0 = N, x1 = -1, y1 = -1; const idx = [];
  for (let i = 0; i < N * N; i++) if (m[i]) { area++; idx.push(i); const x = i % N, y = (i / N) | 0;
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  return { area, idx, bbox: area ? [x0, y0, x1, y1] : null };
}
function recomputeMasks() {
  for (const p of ["A", "B"]) {
    const polys = ann.persons[p].garment_polygons_orig;
    if (!polys.length) { masks[p] = null; continue; }
    const raw = rasterize(polys), dil = dilate(raw, DIL), rs = stats(raw), ds = stats(dil);
    masks[p] = { raw, dil, idx: ds.idx, area: ds.area, bbox: ds.bbox, rawArea: rs.area, rawBBox: rs.bbox };
  }
}
function forbidden(forPerson) {
  // Pixels a sham may not touch: both person boxes (+pad, + tag zones), both garment masks,
  // and the other person's sham.
  const F = new Uint8Array(N * N);
  const mark = (x0, y0, x1, y1) => { x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0));
    x1 = Math.min(N - 1, Math.ceil(x1)); y1 = Math.min(N - 1, Math.ceil(y1));
    for (let y = y0; y <= y1; y++) F.fill(1, y * N + x0, y * N + x1 + 1); };
  for (const p of ["A", "B"]) {
    let [x0, y0, x1, y1] = cand.persons_canvas[p].bbox_xyxy;
    if (masks[p] && masks[p].bbox) { const b = masks[p].bbox;   // marker box encloses the garment mask too (D-017)
      x0 = Math.min(x0, b[0] - 5); y0 = Math.min(y0, b[1] - 5); x1 = Math.max(x1, b[2] + 5); y1 = Math.max(y1, b[3] + 5); }
    mark(x0 - KEEPOUT_PAD, y0 - KEEPOUT_PAD - TAG_ZONE, x1 + KEEPOUT_PAD, y1 + KEEPOUT_PAD + TAG_ZONE);
    if (masks[p]) for (const i of masks[p].idx) F[i] = 1;
  }
  const o = forPerson === "A" ? "B" : "A", t = ann.sham_translations_canvas[o];
  if (masks[o] && t) for (const i of masks[o].idx) { const x = i % N + t[0], y = ((i / N) | 0) + t[1];
    if (x >= 0 && y >= 0 && x < N && y < N) F[y * N + x] = 1; }
  return F;
}
function shamProblems(p) {
  const t = ann.sham_translations_canvas[p], m = masks[p];
  if (!m) return ["no garment polygon yet"];
  if (!t) return ["no sham placed"];
  const [cx0, cy0, cx1, cy1] = contentBox(), F = forbidden(p); let out = 0, hit = 0;
  for (const i of m.idx) { const x = i % N + t[0], y = ((i / N) | 0) + t[1];
    if (x < cx0 || y < cy0 || x >= cx1 || y >= cy1) { out++; continue; }
    if (F[y * N + x]) hit++; }
  const pr = [];
  if (out) pr.push(`${out} px outside the image content (padding or clipping)`);
  if (hit) pr.push(`${hit} px overlap a person box, tag zone, garment mask or the other sham`);
  return pr;
}
function proposeSham(p) {
  const m = masks[p]; if (!m) return alert("Draw the garment polygon first.");
  const F = forbidden(p), S = new Int32Array((N + 1) * (N + 1));  // summed-area table
  for (let y = 0; y < N; y++) { let row = 0; for (let x = 0; x < N; x++) { row += F[y * N + x];
    S[(y + 1) * (N + 1) + x + 1] = S[y * (N + 1) + x + 1] + row; } }
  const rect = (x0, y0, x1, y1) => S[(y1 + 1) * (N + 1) + x1 + 1] - S[y0 * (N + 1) + x1 + 1] - S[(y1 + 1) * (N + 1) + x0] + S[y0 * (N + 1) + x0];
  const [bx0, by0, bx1, by1] = m.bbox, [cx0, cy0, cx1, cy1] = contentBox();
  let best = null;
  for (let ny0 = cy0; ny0 + (by1 - by0) < cy1; ny0 += 4) for (let nx0 = cx0; nx0 + (bx1 - bx0) < cx1; nx0 += 4) {
    if (rect(nx0, ny0, nx0 + bx1 - bx0, ny0 + by1 - by0) !== 0) continue;
    const dx = nx0 - bx0, dy = ny0 - by0, d = dx * dx + dy * dy;
    if (!best || d < best[2]) best = [dx, dy, d];
  }
  if (!best) return alert(`No free rectangular background region fits ${p}'s mask. You may place the sham by hand (mode s), or reject: no_sham_location.`);
  ann.sham_translations_canvas[p] = [best[0], best[1]]; changed();
}

// ---------- drawing ----------
function appliedMask(cond) {
  const A = new Uint8Array(N * N);
  for (const key of COND[cond]) {
    const p = key[2], m = masks[p]; if (!m) continue;
    const t = key[0] === "S" ? ann.sham_translations_canvas[p] : [0, 0]; if (!t) continue;
    for (const i of m.idx) { const x = i % N + t[0], y = ((i / N) | 0) + t[1];
      if (x >= 0 && y >= 0 && x < N && y < N) A[y * N + x] = 1; }
  }
  return A;
}
function draw() {
  ctx.fillStyle = `rgb(${GRAY},${GRAY},${GRAY})`; ctx.fillRect(0, 0, N, N);
  if (isReview()) { if (reviewImg) ctx.drawImage(reviewImg, 0, 0, N, N); return; }
  if (!img) return;
  const t = tf(); ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, t.offset[0], t.offset[1], t.resized_size[0], t.resized_size[1]);
  if (view !== "orig") {
    const A = appliedMask(view), id = ctx.getImageData(0, 0, N, N), d = id.data;
    for (let i = 0; i < N * N; i++) if (A[i]) { d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = GRAY; }
    ctx.putImageData(id, 0, 0);
  }
  if (!showOverlay) return;
  ctx.lineWidth = 2; ctx.font = "bold 18px system-ui";
  for (const p of ["A", "B"]) {
    const [x0, y0, x1, y1] = cand.persons_canvas[p].bbox_xyxy;
    ctx.strokeStyle = PCOL[p]; ctx.setLineDash([6, 4]); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0); ctx.setLineDash([]);
    ctx.fillStyle = PCOL[p]; ctx.fillRect(x0, Math.max(0, y0 - 22), 22, 22); ctx.fillStyle = "#000"; ctx.fillText(p, x0 + 5, Math.max(17, y0 - 5));
    // Show the chosen colour beside each tag so an A/B colour swap is visible at a glance.
    const col = !isReview() && ann.persons[p].color;
    if (col) { const ty = Math.max(0, y0 - 22); ctx.fillStyle = SWATCH[col]; ctx.fillRect(x0 + 24, ty, 22, 22);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1; ctx.strokeRect(x0 + 24, ty, 22, 22); ctx.lineWidth = 2;
      ctx.fillStyle = "#000"; ctx.fillRect(x0 + 48, ty, ctx.measureText(col).width + 8, 22);
      ctx.fillStyle = "#fff"; ctx.fillText(col, x0 + 52, ty + 17); }
    if (view === "orig") {
      for (const poly of ann.persons[p].garment_polygons_orig) {
        ctx.beginPath(); poly.forEach((v, i) => { const [x, y] = toCanvas(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.closePath(); ctx.fillStyle = PCOL[p] + "40"; ctx.fill(); ctx.strokeStyle = PCOL[p]; ctx.stroke();
        if (mode === "edit" && p === person) for (const v of poly) { const [x, y] = toCanvas(v); ctx.fillStyle = "#fff"; ctx.fillRect(x - 3, y - 3, 6, 6); }
      }
      const sh = ann.sham_translations_canvas[p], m = masks[p];
      if (sh && m) { const [bx0, by0, bx1, by1] = m.bbox; ctx.strokeStyle = PCOL[p]; ctx.setLineDash([2, 3]);
        ctx.strokeRect(bx0 + sh[0], by0 + sh[1], bx1 - bx0 + 1, by1 - by0 + 1); ctx.setLineDash([]);
        ctx.fillStyle = PCOL[p]; ctx.fillText("S" + p, bx0 + sh[0] + 3, by0 + sh[1] + 18); }
    }
  }
  if (view === "orig" && (samPts[person].length || samProp[person])) {
    const prop = samProp[person];
    if (prop) for (const poly of prop.polygons_orig) {
      ctx.beginPath(); poly.forEach((v, i) => { const [x, y] = toCanvas(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.closePath(); ctx.fillStyle = "#ff00ff33"; ctx.fill(); ctx.setLineDash([5, 3]); ctx.strokeStyle = "#ff00ff"; ctx.stroke(); ctx.setLineDash([]);
    }
    samPts[person].forEach(([x0, y0, lab]) => { const [x, y] = toCanvas([x0, y0]); ctx.fillStyle = lab ? "#00e676" : "#ff1744";
      ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI); ctx.fill(); ctx.strokeStyle = "#000"; ctx.stroke(); });
  }
  if (drawing.length) {
    ctx.strokeStyle = "#fff"; ctx.beginPath();
    drawing.forEach((v, i) => { const [x, y] = toCanvas(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
    for (const v of drawing) { const [x, y] = toCanvas(v); ctx.fillStyle = PCOL[person]; ctx.fillRect(x - 3, y - 3, 6, 6); }
  }
}

// ---------- panel ----------
function emptyRecord(role) {
  return { schema: "eob.annotation.v1", role, annotator_id: $("annotator").value.trim(),
    scene_id: cand.scene_id, source_image_id: cand.source_image_id, coco_split: cand.coco_split,
    decision: "draft", reject_reasons: [], checks: {},
    persons: { A: { color: null, color_confidence: null, garment_polygons_orig: [], notes: "" },
               B: { color: null, color_confidence: null, garment_polygons_orig: [], notes: "" } },
    sham_translations_canvas: { A: null, B: null }, review: {}, rater_was_mask_author: null, notes: "" };
}
function isReview() { return $("role").value !== "author"; }
function swatches(p, key, onPick, current) {
  const box = document.createElement("div"); box.className = "swatches";
  for (const c of CFG.garment_colors) {
    const s = document.createElement("div"); s.className = "sw" + (current === c ? " sel" : "");
    s.style.background = SWATCH[c]; s.title = c; s.textContent = c.slice(0, 3);
    s.style.color = ["white", "yellow"].includes(c) ? "#000" : "#fff";
    s.onclick = () => onPick(c); box.appendChild(s);
  }
  return box;
}
function radios(name, opts, current, onPick, disabled) {
  const span = document.createElement("span");
  for (const o of opts) { const l = document.createElement("label"); l.style.marginRight = "8px";
    const r = document.createElement("input"); r.type = "radio"; r.name = name; r.checked = current === o; r.disabled = !!disabled;
    r.onchange = () => onPick(o); l.append(r, " " + o); span.appendChild(l); }
  return span;
}
function renderAuthorPanel() {
  const c = cand;
  $("sceneinfo").innerHTML = `<b>${c.scene_id}</b> · rank ${c.order_rank} · ${c.width}×${c.height} · scale ${c.canvas_transform.scale.toFixed(3)}<br>
    A = ${c.persons.A.side} person (ann ${c.persons.A.coco_ann_id}), B = ${c.persons.B.side} (ann ${c.persons.B.coco_ann_id}) · IoU ${c.bbox_iou.toFixed(3)}<br>
    licence: ${c.license.name || "?"}`;
  const P = $("persons"); P.innerHTML = "";
  for (const p of ["A", "B"]) {
    const pr = ann.persons[p], d = document.createElement("div"); d.className = "person " + p;
    d.innerHTML = `<b>Person ${p}</b> — latent garment colour (as seen in the ORIGINAL):`;
    d.appendChild(swatches(p, "color", (col) => { pr.color = col; markDirty(); renderAuthorPanel(); draw(); }, pr.color));
    d.appendChild(radios("conf" + p, CFG.confidence, pr.color_confidence, (v) => { pr.color_confidence = v; markDirty(); }));
    const pl = document.createElement("div"); pl.className = "polylist small";
    pr.garment_polygons_orig.forEach((poly, i) => { const row = document.createElement("div");
      row.textContent = `polygon ${i + 1}: ${poly.length} vertices`; const b = document.createElement("button"); b.textContent = "delete";
      b.onclick = () => { pr.garment_polygons_orig.splice(i, 1); noteManualEdit(p); changed(); }; row.appendChild(b); pl.appendChild(row); });
    if (pr.mask_source) { const src = document.createElement("div"); src.className = "small muted";
      src.textContent = pr.mask_source === "sam_assisted" ? `mask: SAM-assisted${pr.edited_after_sam ? ", edited by hand" : ""}` : "mask: manual"; pl.appendChild(src); }
    if (!pr.garment_polygons_orig.length) pl.innerHTML = '<span class="warn">no polygon yet</span>';
    d.appendChild(pl); P.appendChild(d);
  }
  renderMetrics();
  const C = $("checks"); C.innerHTML = "";
  for (const [k, label] of Object.entries(CFG.author_checks)) {
    const l = document.createElement("label"); l.className = "chk"; const cb = document.createElement("input"); cb.type = "checkbox";
    cb.checked = ann.checks[k] === true; cb.onchange = () => { ann.checks[k] = cb.checked; markDirty(); };
    l.append(cb, " " + label); C.appendChild(l);
  }
  const D = $("decision"); D.innerHTML = "";
  D.appendChild(radios("decision", ["eligible", "reject", "unsure", "draft"], ann.decision, (v) => { ann.decision = v; markDirty(); renderAuthorPanel(); }));
  const R = $("reasons"); R.innerHTML = ann.decision === "reject" || ann.decision === "unsure" ? "<i>Reasons:</i>" : "";
  if (ann.decision === "reject" || ann.decision === "unsure") for (const r of CFG.reject_reasons) {
    const l = document.createElement("label"); l.className = "chk"; const cb = document.createElement("input"); cb.type = "checkbox";
    cb.checked = ann.reject_reasons.includes(r);
    cb.onchange = () => { ann.reject_reasons = cb.checked ? [...new Set([...ann.reject_reasons, r])] : ann.reject_reasons.filter((x) => x !== r); markDirty(); };
    l.append(cb, " " + r); R.appendChild(l);
  }
  $("notes").value = ann.notes || "";
}
function renderMetrics() {
  const el = CFG.eligibility, lines = [];
  let total = 0;
  for (const p of ["A", "B"]) {
    const m = masks[p];
    if (!m) { lines.push(`<span class="warn">${p}: no mask</span>`); continue; }
    const w = m.rawBBox[2] - m.rawBBox[0] + 1, h = m.rawBBox[3] - m.rawBBox[1] + 1;
    const ok = w >= el.min_garment_width_px && h >= el.min_garment_height_px && m.rawArea >= el.min_garment_area_px;
    total += m.area;
    lines.push(`<span class="${ok ? "ok" : "bad"}">${p}: garment ${w}×${h}px, area ${m.rawArea}px (min ${el.min_garment_width_px}×${el.min_garment_height_px}, ${el.min_garment_area_px})</span>; dilated ${m.area}px`);
    const [bx0, by0, bx1, by1] = cand.persons_canvas[p].bbox_xyxy, rb = m.rawBBox;
    const cx = (rb[0] + rb[2]) / 2, cy = (rb[1] + rb[3]) / 2;
    if (cx < bx0 || cx > bx1 || cy < by0 || cy > by1) lines.push(`<span class="bad">${p}: garment mask centre lies outside ${p}'s box: polygon drawn for the wrong person?</span>`);
    const sp = shamProblems(p);
    lines.push(sp.length ? `<span class="bad">sham ${p}: ${sp.join("; ")}</span>` : `<span class="ok">sham ${p}: ok at (${ann.sham_translations_canvas[p]})</span>`);
  }
  const frac = total / (N * N);
  lines.push(`<span class="${frac <= el.max_masked_canvas_fraction ? "ok" : "bad"}">applied area per condition ≈ ${(100 * frac).toFixed(2)}% of the canvas (max ${100 * el.max_masked_canvas_fraction}%)</span>`);
  if (masks.A && masks.B) { let ov = 0; for (const i of masks.A.idx) if (masks.B.dil[i]) ov++; if (ov) lines.push(`<span class="bad">M_A and M_B overlap by ${ov}px</span>`); }
  $("metrics").innerHTML = lines.join("<br>");
}

// ---------- review mode ----------
function reviewStages() {
  const role = $("role").value, s2 = CFG.stage2_image[role], hidden = s2 === "I01" ? "A" : "B", vis = hidden === "A" ? "B" : "A";
  const yn = ["yes", "no", "unsure"];
  return [
    { key: "stage1_I00", image: "I00", title: "Stage 1: both garments masked (I00)",
      qs: [["fabric_visible_A", `Is ANY fabric of person A's upper-body garment still visible?`, yn],
           ["fabric_visible_B", `Is ANY fabric of person B's upper-body garment still visible?`, yn],
           ["designation_clear", "Can you still tell clearly which person is A and which is B?", yn]] },
    { key: "stage2_single", image: s2, title: `Stage 2: only ${hidden} masked (${s2})`,
      qs: [["hidden_fabric_visible", `Is ANY fabric of ${hidden}'s garment still visible?`, yn],
           ["visible_garment_preserved", `Is ${vis}'s garment fully visible and unedited?`, yn],
           ["designation_clear", "Is the A/B designation clear?", yn]] },
    { key: "stage3_colors", image: "clean", title: "Stage 3: original marked image: label colours", colors: true },
    { key: "stage4_sham", image: "I11", title: "Stage 4: both shams (I11)",
      qs: [["sham_preserves_colors", "Are both garments fully visible, with colours unchanged?", yn],
           ["sham_covers_informative_content", "Does a gray sham patch cover a person, text or other informative content?", yn]] },
  ];
}
function currentStageIndex() { const st = reviewStages(); const i = st.findIndex((s) => !(ann.review[s.key] && ann.review[s.key].locked_utc)); return i < 0 ? st.length : i; }
async function loadReviewImage() {
  const st = reviewStages(), i = currentStageIndex(); reviewImg = null;
  if (i >= st.length) { draw(); return; }
  const im = new Image();
  im.onload = () => { reviewImg = im; draw(); };
  im.onerror = () => { reviewImg = null; draw(); $("hint").innerHTML = `<span class="bad">Rendered image ${st[i].image}.png is not available. Run scripts/render_factorial.py for this scene first.</span>`; };
  im.src = `/rendered/${cand.scene_id}/${st[i].image}.png?t=${Date.now()}`;
}
function renderReviewPanel() {
  $("was_author").checked = ann.rater_was_mask_author === true;
  const S = $("stages"); S.innerHTML = ""; const st = reviewStages(), cur = currentStageIndex();
  st.forEach((s, i) => {
    const box = document.createElement("div"); const locked = !!(ann.review[s.key] && ann.review[s.key].locked_utc);
    box.className = "stage" + (locked ? " locked" : ""); box.innerHTML = `<h4>${s.title}${locked ? " ✓ locked" : ""}</h4>`;
    if (i > cur) { box.innerHTML += '<span class="muted small">hidden until the previous stage is locked</span>'; S.appendChild(box); return; }
    const a = ann.review[s.key] = ann.review[s.key] || {};
    if (s.colors) {
      for (const p of ["A", "B"]) { const d = document.createElement("div"); d.className = "person " + p; d.innerHTML = `<b>${p}</b>`;
        if (!locked) d.appendChild(swatches(p, "c", (c) => { a["color_" + p] = c; markDirty(); renderReviewPanel(); }, a["color_" + p]));
        else d.innerHTML += `: ${a["color_" + p]} (${a["confidence_" + p]})`;
        d.appendChild(radios(s.key + p, CFG.confidence, a["confidence_" + p], (v) => { a["confidence_" + p] = v; markDirty(); }, locked));
        S.appendChild(box); box.appendChild(d); }
    } else for (const [k, q, opts] of s.qs) { const d = document.createElement("div"); d.className = "small"; d.textContent = q + " ";
      d.appendChild(radios(s.key + k, opts, a[k], (v) => { a[k] = v; markDirty(); }, locked)); box.appendChild(d); }
    if (!locked) { const b = document.createElement("button"); b.textContent = "Lock this stage"; b.className = "primary";
      b.onclick = async () => {
        const need = s.colors ? ["color_A", "color_B", "confidence_A", "confidence_B"] : s.qs.map((q) => q[0]);
        const miss = need.filter((k) => !a[k]); if (miss.length) return alert("Answer all questions first: " + miss.join(", "));
        a.locked_utc = new Date().toISOString(); a.image = s.image; finalizeReview(); await save(); renderReviewPanel(); loadReviewImage(); };
      box.appendChild(b); }
    S.appendChild(box);
  });
}
function finalizeReview() {
  const r = ann.review, st = reviewStages(), reasons = [];
  if (!st.every((s) => r[s.key] && r[s.key].locked_utc)) { ann.decision = "draft"; return; }
  const s1 = r.stage1_I00, s2 = r.stage2_single, s3 = r.stage3_colors, s4 = r.stage4_sham;
  if (s1.fabric_visible_A !== "no" || s1.fabric_visible_B !== "no" || s2.hidden_fabric_visible !== "no") reasons.push("cannot_mask_cleanly");
  if (s1.designation_clear !== "yes" || s2.designation_clear !== "yes") reasons.push("designation_unclear");
  if (s2.visible_garment_preserved !== "yes" || s4.sham_preserves_colors !== "yes" || s4.sham_covers_informative_content !== "no") reasons.push("no_sham_location");
  if (s3.color_A === s3.color_B) reasons.push("same_color");
  if (s3.confidence_A === "unsure" || s3.confidence_B === "unsure") reasons.push("ambiguous_color");
  ann.persons.A.color = s3.color_A; ann.persons.B.color = s3.color_B;
  ann.persons.A.color_confidence = s3.confidence_A; ann.persons.B.color_confidence = s3.confidence_B;
  ann.reject_reasons = [...new Set(reasons)]; ann.decision = reasons.length ? "reject" : "eligible";
}

// ---------- scene loading & saving ----------
function markDirty() { dirty = true; $("savestate").textContent = "unsaved changes"; $("savestate").className = "warn"; }
function changed() { recomputeMasks(); markDirty(); renderAuthorPanel(); draw(); }
async function save() {
  if (!$("annotator").value.trim()) { alert("Enter your initials first."); return false; }
  ann.annotator_id = $("annotator").value.trim();
  if (!isReview()) ann.notes = $("notes").value;
  try {
    const r = await api(`/api/annotation/${$("role").value}/${cand.scene_id}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(ann) });
    dirty = false; $("savestate").textContent = "saved " + r.updated_utc; $("savestate").className = "ok";
    if (r.completeness) $("completeness").innerHTML = r.completeness.length
      ? `<span class="warn">Not yet a complete eligible item:</span><br>${r.completeness.map((x) => "• " + x).join("<br>")}`
      : '<span class="ok">Complete: this scene can become an item.</span>';
    refreshQueueStatus(); return true;
  } catch (e) { alert("Save failed: " + e.message); return false; }
}
async function loadScene(id) {
  if (dirty && !confirm("Discard unsaved changes?")) { $("scene").value = cand.scene_id; return; }
  cand = await api(`/api/scene/${id}`); const role = $("role").value;
  ann = (await api(`/api/annotation/${role}/${id}`)) || emptyRecord(role);
  ann.sham_translations_canvas = ann.sham_translations_canvas || { A: null, B: null }; ann.review = ann.review || {};
  dirty = false; drawing = []; samPts = { A: [], B: [] }; samProp = { A: null, B: null }; $("samstate").textContent = ""; $("savestate").textContent = ann.updated_utc ? "loaded " + ann.updated_utc : "new"; $("savestate").className = "muted";
  $("scene").value = id; $("completeness").innerHTML = "";
  $("authorpanel").hidden = isReview(); $("reviewpanel").hidden = !isReview(); $("authortools").style.display = isReview() ? "none" : "";
  document.querySelectorAll(".vbtn").forEach((b) => (b.parentElement.style.display = isReview() ? "none" : ""));
  if (isReview()) { renderReviewPanel(); await loadReviewImage(); setHint(); return; }
  img = new Image(); img.onload = () => { recomputeMasks(); renderAuthorPanel(); draw(); setHint(); }; img.src = `/image/${id}`;
}
async function refreshQueueStatus() {
  queue = await api(`/api/queue?split=${$("split").value}`); const sel = $("scene"), cur = sel.value; sel.innerHTML = "";
  for (const q of queue) { const o = document.createElement("option"); const s = q.status[$("role").value];
    o.value = q.scene_id; o.textContent = `#${q.rank} ${q.scene_id.slice(-12)} ${s ? "[" + s + "]" : ""}`; sel.appendChild(o); }
  if (cur) sel.value = cur;
}
function setHint() {
  const h = isReview() ? "Review: answer from what you SEE in the displayed image only. Then lock each stage."
    : { draw: `Drawing for ${person}: click to add vertices · Enter or double-click closes · Backspace removes the last vertex · Esc cancels`,
        edit: `Editing ${person}: drag a vertex · Shift+click deletes a vertex`,
        sham: `Sham for ${person}: drag, or use the arrow keys (Shift = 10px). Use "Propose sham" for an automatic free location, then CHECK it visually.`,
        sam: `SAM for ${person}: click on ${person}'s garment (green = garment); Shift+click or right-click marks NOT garment (red: skin, background, other person). Then Accept (a) and verify.` }[mode];
  $("hint").textContent = h;
}
function setPerson(p) { person = p; document.querySelectorAll(".pbtn").forEach((b) => b.classList.toggle("active", b.dataset.person === p)); setHint(); draw(); }
function setMode(m) { mode = m; drawing = []; document.querySelectorAll(".mbtn").forEach((b) => b.classList.toggle("active", b.dataset.mode === m)); setHint(); draw(); }
function setView(v) { view = v; document.querySelectorAll(".vbtn").forEach((b) => b.classList.toggle("active", b.dataset.view === v)); draw(); }
function closePolygon() {
  if (drawing.length >= 3) { ann.persons[person].garment_polygons_orig.push(drawing); drawing = []; noteManualEdit(person); changed(); }
  else { drawing = []; draw(); }
}
function noteManualEdit(p) {
  const pr = ann.persons[p];
  if (pr.sam_assist) pr.edited_after_sam = true; else pr.mask_source = "manual";
}

// ---------- SAM-HQ assistance: humans click, SAM proposes, humans accept and edit ----------
async function runSam(p) {
  if (!samPts[p].length || !samPts[p].some((q) => q[2] === 1)) { samProp[p] = null; draw(); return; }
  samBusy = true; $("samstate").textContent = "SAM running… (first call loads the model, ~20 s)";
  try {
    samProp[p] = await api(`/api/sam/${cand.scene_id}`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ points_orig: samPts[p].map((q) => [q[0], q[1]]), labels: samPts[p].map((q) => q[2]) }) });
    $("samstate").textContent = `proposal for ${p}: ${samProp[p].polygons_orig.length} polygon(s), score ${samProp[p].score.toFixed(3)}. Press "Accept SAM" (a) or add points.`;
  } catch (e) { samProp[p] = null; $("samstate").textContent = "SAM error: " + e.message + " (manual polygons still work)"; }
  samBusy = false; draw();
}
function acceptSam(p) {
  const prop = samProp[p]; if (!prop) return alert("No SAM proposal for " + p + ". Click on the garment first (mode m).");
  if (!prop.polygons_orig.length) return alert("SAM proposal is empty. Add positive points or draw manually.");
  const pr = ann.persons[p];
  if (pr.garment_polygons_orig.length && !confirm(`Replace ${p}'s ${pr.garment_polygons_orig.length} existing polygon(s) with the SAM proposal?`)) return;
  pr.garment_polygons_orig = JSON.parse(JSON.stringify(prop.polygons_orig));
  const { polygons_orig, ...prov } = prop;
  pr.mask_source = "sam_assisted"; pr.edited_after_sam = false;
  pr.sam_assist = { ...prov, accepted_utc: new Date().toISOString() };
  samPts[p] = []; samProp[p] = null; $("samstate").textContent = `SAM mask accepted for ${p}. Check sleeves and fragments in the I01/I00 previews; edit with (e) or add polygons with (d).`;
  changed();
}

// ---------- events ----------
stage.addEventListener("mousedown", (ev) => {
  if (isReview() || !cand) return; const pt = eventPt(ev);
  if (mode === "draw") { if (ev.detail >= 2) return; drawing.push(toOrig(pt)); draw(); }
  else if (mode === "edit") {
    let best = null;
    ann.persons[person].garment_polygons_orig.forEach((poly, pi) => poly.forEach((v, vi) => {
      const [x, y] = toCanvas(v), d = Math.hypot(x - pt[0], y - pt[1]); if (d < 10 && (!best || d < best.d)) best = { pi, vi, d }; }));
    if (!best) return;
    if (ev.shiftKey) { const poly = ann.persons[person].garment_polygons_orig[best.pi];
      if (poly.length > 3) { poly.splice(best.vi, 1); noteManualEdit(person); } else alert("A polygon needs 3 vertices; delete the polygon instead."); changed(); }
    else dragging = best;
  } else if (mode === "sham") { dragging = { sham: true, start: pt, t0: (ann.sham_translations_canvas[person] || [0, 0]).slice() }; }
  else if (mode === "sam") {
    if (samBusy) return;
    const o = toOrig(pt); samPts[person].push([o[0], o[1], ev.shiftKey || ev.button === 2 ? 0 : 1]); runSam(person);
  }
});
stage.addEventListener("contextmenu", (ev) => { if (mode === "sam") ev.preventDefault(); });
stage.addEventListener("mousemove", (ev) => {
  if (!dragging) return; const pt = eventPt(ev);
  if (dragging.sham) { ann.sham_translations_canvas[person] = [Math.round(dragging.t0[0] + pt[0] - dragging.start[0]), Math.round(dragging.t0[1] + pt[1] - dragging.start[1])]; draw(); }
  else { ann.persons[person].garment_polygons_orig[dragging.pi][dragging.vi] = toOrig(pt); draw(); }
});
window.addEventListener("mouseup", () => { if (dragging) { if (!dragging.sham) noteManualEdit(person); dragging = null; changed(); } });
stage.addEventListener("dblclick", () => { if (mode === "draw") closePolygon(); });
window.addEventListener("keydown", (ev) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if ((ev.ctrlKey || ev.metaKey) && ev.key === "s") { ev.preventDefault(); save(); return; }
  if (isReview()) return;
  const k = ev.key;
  if (k === "1") setPerson("A"); else if (k === "2") setPerson("B");
  else if (k === "d") setMode("draw"); else if (k === "e") setMode("edit"); else if (k === "s") setMode("sham");
  else if (k === "m" && CFG.sam_available) setMode("sam"); else if (k === "a" && CFG.sam_available) acceptSam(person);
  else if (k === "o") { showOverlay = !showOverlay; $("showoverlay").checked = showOverlay; draw(); }
  else if (k === "n") $("next").click(); else if (k === "p") $("prev").click();
  else if (k === "Enter") closePolygon();
  else if (k === "Escape") { drawing = []; draw(); }
  else if (k === "Backspace" && mode === "draw") { ev.preventDefault(); drawing.pop(); draw(); }
  else if (mode === "sham" && k.startsWith("Arrow")) {
    ev.preventDefault(); const t = ann.sham_translations_canvas[person] || [0, 0], s = ev.shiftKey ? 10 : 1;
    ann.sham_translations_canvas[person] = [t[0] + (k === "ArrowRight" ? s : k === "ArrowLeft" ? -s : 0), t[1] + (k === "ArrowDown" ? s : k === "ArrowUp" ? -s : 0)];
    changed();
  }
});
window.addEventListener("beforeunload", (ev) => { if (dirty) { ev.preventDefault(); ev.returnValue = ""; } });

// ---------- boot ----------
(async function boot() {
  CFG = await api("/api/config");
  for (const s of CFG.splits) $("split").add(new Option(s, s));
  for (const r of CFG.roles) $("role").add(new Option(r, r));
  try { $("annotator").value = localStorage.getItem("eob_annotator") || ""; } catch (e) {}
  $("annotator").onchange = () => { try { localStorage.setItem("eob_annotator", $("annotator").value.trim()); } catch (e) {} };
  document.querySelectorAll(".pbtn").forEach((b) => (b.onclick = () => setPerson(b.dataset.person)));
  document.querySelectorAll(".mbtn").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  document.querySelectorAll(".vbtn").forEach((b) => (b.onclick = () => setView(b.dataset.view)));
  $("showoverlay").onchange = () => { showOverlay = $("showoverlay").checked; draw(); };
  $("propose").onclick = () => proposeSham(person);
  $("samtools").hidden = !CFG.sam_available;
  $("samaccept").onclick = () => acceptSam(person);
  $("samclear").onclick = () => { samPts[person] = []; samProp[person] = null; $("samstate").textContent = ""; draw(); };
  $("save").onclick = save;
  $("was_author").onchange = () => { ann.rater_was_mask_author = $("was_author").checked; markDirty(); };
  $("notes").oninput = markDirty;
  $("split").onchange = async () => { await refreshQueueStatus(); if (queue.length) loadScene(queue[0].scene_id); };
  $("role").onchange = async () => { await refreshQueueStatus(); loadScene($("scene").value); };
  $("scene").onchange = () => loadScene($("scene").value);
  const step = (d) => { const i = queue.findIndex((q) => q.scene_id === cand.scene_id) + d; if (i >= 0 && i < queue.length) loadScene(queue[i].scene_id); };
  $("next").onclick = () => step(1); $("prev").onclick = () => step(-1);
  setPerson("A"); setMode("draw"); setView("orig");
  await refreshQueueStatus(); if (queue.length) loadScene(queue[0].scene_id);
})();
