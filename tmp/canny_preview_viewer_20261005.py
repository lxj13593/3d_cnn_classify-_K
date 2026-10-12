"""Build a self-contained, read-only HTML viewer for the Canny sample preview.

Usage:
    python canny_preview_viewer_20261005.py --output <preview-result-directory>

The input directory must contain preview_index.json and the referenced NPZ files.
Only the HTML artifact is written; input arrays and dataset files are not modified.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Head40 · 20 个样本边缘处理对照</title>
  <style>
    :root { color-scheme: light; font-family: "Microsoft YaHei", sans-serif; color: #20252b; background: #f5f6f8; }
    * { box-sizing: border-box; }
    body { margin: 0 auto; padding: 24px; max-width: 1600px; }
    h1 { margin: 0 0 10px; font-size: 25px; }
    p { line-height: 1.65; margin: 8px 0; }
    .note { color: #4b5563; font-size: 14px; }
    .card { background: white; border: 1px solid #dce0e6; border-radius: 10px; padding: 18px; margin-top: 16px; }
    .controls { display: flex; align-items: end; flex-wrap: wrap; gap: 14px; }
    label { display: flex; flex-direction: column; gap: 7px; font-size: 14px; }
    select, button { font: inherit; padding: 8px 10px; border: 1px solid #b9c1cc; border-radius: 5px; background: white; }
    select { max-width: 100%; }
    #sample-label { flex: 1; min-width: min(440px, 100%); }
    #sample { width: 100%; }
    button { cursor: pointer; }
    button:disabled { cursor: default; opacity: .45; }
    .slice-controls { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; margin-top: 18px; }
    input[type=range] { flex: 1; min-width: 180px; max-width: 850px; }
    #slice-description { min-width: 210px; font-variant-numeric: tabular-nums; }
    .panels { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin-top: 15px; }
    .panel { min-width: 0; }
    .panel h2 { margin: 0 0 9px; font-size: 16px; text-align: center; }
    .canvas-box { background: #111; min-height: 320px; padding: 10px; display: flex; justify-content: center; align-items: center; border-radius: 5px; }
    canvas { display: block; max-width: 100%; height: auto; image-rendering: pixelated; image-rendering: crisp-edges; }
    #info, #config { line-height: 1.8; font-size: 14px; overflow-wrap: anywhere; }
    #path { color: #647080; font-size: 12px; overflow-wrap: anywhere; }
    #orientation { color: #4b5563; font-size: 14px; }
    #error { display: none; color: #a01212; white-space: pre-wrap; }
    @media (max-width: 1050px) { .panels { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 560px) { body { padding: 12px; } .card { padding: 12px; } .panels { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>Head40 · 20 个样本边缘处理对照</h1>
  <p class="note">这是只读预览，不会修改原始数据。标签来自数据清单，不代表重新人工判定；图中没有标注或推测缺陷位置。</p>
  <p class="note">四列显示同一位置的切片。原图和处理后均采用固定 0–1 灰度范围，不逐图拉伸；页面显示 8 位量化，NPZ 保存原始 float32 数组。仅查看未增强的处理效果。</p>
  <section class="card">
    <div class="controls">
      <label>类别<select id="category"><option value="all">全部</option><option value="0">正常</option><option value="1">缺陷</option></select></label>
      <label id="sample-label">样本<select id="sample"></select></label>
      <label>切面<select id="plane"><option value="dw">D–W（固定 H）</option><option value="dh">D–H（固定 W）</option><option value="hw">H–W（固定 D）</option></select></label>
      <label>像素放大<select id="scale"><option value="4">4 倍</option><option value="8" selected>8 倍</option><option value="12">12 倍</option></select></label>
    </div>
    <div class="slice-controls">
      <button id="previous" type="button" aria-label="上一切片">上一切片</button>
      <input id="slice" type="range" min="0" max="0" value="0" step="1" aria-label="切片位置">
      <button id="next" type="button" aria-label="下一切片">下一切片</button>
      <span id="slice-description"></span>
    </div>
    <p id="orientation"></p>
    <div id="info"></div>
    <div id="path"></div>
  </section>
  <section class="card">
    <div class="panels">
      <article class="panel"><h2>原始灰度</h2><div class="canvas-box"><canvas id="image"></canvas></div></article>
      <article class="panel"><h2>Canny 二值边缘</h2><div class="canvas-box"><canvas id="edge_mask"></canvas></div></article>
      <article class="panel"><h2>膨胀后的边缘掩膜</h2><div class="canvas-box"><canvas id="dilated_mask"></canvas></div></article>
      <article class="panel"><h2>处理后：原图 × 掩膜</h2><div class="canvas-box"><canvas id="processed"></canvas></div></article>
    </div>
  </section>
  <section class="card">
    <div id="config"></div>
    <p class="note">D–W / D–H：D 高索引在上（统一头端方向）；H–W：H 高索引在上。切片索引从 0 开始，所有面板保持同一方向。边缘与膨胀掩膜中，白色表示保留位置。</p>
    <p class="note">灰度量化后特别弱的信号可能显示为黑色，请保留 NPZ 用于数值核对。本页不依赖网络或外部脚本，复制整个结果文件夹后仍可离线打开。</p>
  </section>
  <pre id="error"></pre>
  <script id="preview-data" type="application/json">__PREVIEW_DATA__</script>
  <script>
    "use strict";
    const payload = JSON.parse(document.getElementById("preview-data").textContent);
    const samples = payload.samples;
    const keys = ["image", "edge_mask", "dilated_mask", "processed"];
    const cache = new Map();
    const ui = Object.fromEntries(["category", "sample", "plane", "scale", "slice", "previous", "next", "slice-description", "orientation", "info", "path", "config", "error"].map(id => [id, document.getElementById(id)]));
    let current = null;

    function classKey(item) {
      const value = String(item.label ?? "").trim().toLowerCase();
      const names = {"0": "0", normal: "0", "正常": "0", "1": "1", defective: "1", defect: "1", "缺陷": "1"};
      return names[value] ?? "unknown";
    }
    function labelText(item) {
      const key = classKey(item);
      return item.label_cn || (key === "0" ? "正常" : key === "1" ? "缺陷" : String(item.label ?? "未记录"));
    }
    function difficultyText(value) {
      const names = {clear: "清晰", fuzzy: "模糊", ambiguous: "模糊"};
      return names[value] || String(value ?? "未记录");
    }
    function percent(value) {
      return Number.isFinite(Number(value)) ? (100 * Number(value)).toFixed(2) + "%" : "未记录";
    }
    function numberText(value) {
      return Number.isFinite(Number(value)) ? Number(value).toPrecision(7) : "未记录";
    }
    function decode(item, key) {
      const cacheKey = String(item.viewer_id) + ":" + key;
      if (!cache.has(cacheKey)) {
        const binary = atob(item.arrays[key]);
        const array = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) array[i] = binary.charCodeAt(i);
        const expected = item.shape_dhw.reduce((a, b) => a * b, 1);
        if (array.length !== expected) throw new Error("数组长度与体积尺寸不一致：" + cacheKey);
        cache.set(cacheKey, array);
      }
      return cache.get(cacheKey);
    }
    function fillSampleList() {
      const previousId = ui.sample.value;
      const selectedCategory = ui.category.value;
      ui.sample.replaceChildren();
      for (const item of samples) {
        if (selectedCategory !== "all" && classKey(item) !== selectedCategory) continue;
        const option = document.createElement("option");
        option.value = String(item.viewer_id);
        option.textContent = "#" + item.index + " · " + labelText(item) + " · " + difficultyText(item.difficulty) + " · 板 " + String(item.board_key ?? "未记录") + " · " + String(item.sequence ?? "") + " · " + item.shape_dhw.join("×");
        ui.sample.append(option);
      }
      if ([...ui.sample.options].some(option => option.value === previousId)) ui.sample.value = previousId;
      selectSample();
    }
    function selectSample() {
      current = samples.find(item => String(item.viewer_id) === ui.sample.value);
      if (!current) throw new Error("当前类别没有可查看的样本。");
      ui.info.textContent = "标签：" + labelText(current) + "；清晰度：" + difficultyText(current.difficulty) + "；板：" + String(current.board_key ?? "未记录") + "；序号：" + String(current.sequence ?? "未记录") + "；尺寸 D×H×W：" + current.shape_dhw.join("×") + "；膨胀掩膜覆盖比例：" + percent(current.mask_ratio);
      ui.path.textContent = "原始文件：" + String(current.raw_path ?? "未记录");
      setSliceBounds();
    }
    function setSliceBounds() {
      const [D, H, W] = current.shape_dhw;
      const lengths = {dw: H, dh: W, hw: D};
      ui.slice.max = String(lengths[ui.plane.value] - 1);
      ui.slice.value = String(Math.floor(lengths[ui.plane.value] / 2));
      render();
    }
    function render() {
      const [D, H, W] = current.shape_dhw;
      const plane = ui.plane.value;
      const fixed = Number(ui.slice.value);
      const width = plane === "dh" ? H : W;
      const height = plane === "hw" ? H : D;
      const axis = {dw: "H", dh: "W", hw: "D"}[plane];
      const scale = Number(ui.scale.value);
      ui["slice-description"].textContent = "固定 " + axis + " = " + fixed + "（0–" + ui.slice.max + "）";
      ui.orientation.textContent = plane === "dw" ? "横轴 W；纵轴 D；D 高索引在上。" : plane === "dh" ? "横轴 H；纵轴 D；D 高索引在上。" : "横轴 W；纵轴 H；H 高索引在上。";
      ui.previous.disabled = fixed === 0;
      ui.next.disabled = fixed === Number(ui.slice.max);
      for (const key of keys) {
        const values = decode(current, key);
        const canvas = document.getElementById(key);
        canvas.width = width;
        canvas.height = height;
        canvas.style.width = String(width * scale) + "px";
        const context = canvas.getContext("2d");
        context.imageSmoothingEnabled = false;
        const pixels = context.createImageData(width, height);
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            let d, h, w;
            if (plane === "dw") { d = D - 1 - y; h = fixed; w = x; }
            else if (plane === "dh") { d = D - 1 - y; h = x; w = fixed; }
            else { d = fixed; h = H - 1 - y; w = x; }
            const value = values[(d * H + h) * W + w];
            const offset = (y * width + x) * 4;
            pixels.data[offset] = value;
            pixels.data[offset + 1] = value;
            pixels.data[offset + 2] = value;
            pixels.data[offset + 3] = 255;
          }
        }
        context.putImageData(pixels, 0, 0);
      }
    }
    function guarded(action) {
      try { action(); ui.error.style.display = "none"; }
      catch (error) { ui.error.textContent = "查看器错误：" + error.message; ui.error.style.display = "block"; }
    }
    ui.category.addEventListener("change", () => guarded(fillSampleList));
    ui.sample.addEventListener("change", () => guarded(selectSample));
    ui.plane.addEventListener("change", () => guarded(setSliceBounds));
    ui.scale.addEventListener("change", () => guarded(render));
    ui.slice.addEventListener("input", () => guarded(render));
    ui.previous.addEventListener("click", () => guarded(() => { ui.slice.value = String(Math.max(0, Number(ui.slice.value) - 1)); render(); }));
    ui.next.addEventListener("click", () => guarded(() => { ui.slice.value = String(Math.min(Number(ui.slice.max), Number(ui.slice.value) + 1)); render(); }));
    const counts = samples.reduce((result, item) => { const name = labelText(item); result[name] = (result[name] || 0) + 1; return result; }, {});
    ui.category.options[0].textContent = "全部（" + samples.length + "）";
    ui.category.options[1].textContent = "正常（" + (counts["正常"] || 0) + "）";
    ui.category.options[2].textContent = "缺陷（" + (counts["缺陷"] || 0) + "）";
    const config = payload.config || {};
    ui.config.textContent = "处理配置：训练样本数 " + String(config.training_sample_count ?? "未记录") + "；高阈值 " + numberText(config.high_threshold) + "；低阈值 " + numberText(config.low_threshold) + "。详细参数见同目录 preview_index.json。";
    guarded(fillSampleList);
  </script>
</body>
</html>
"""


def encode_gray(array: np.ndarray) -> str:
    if not np.isfinite(array).all():
        raise ValueError("预览数组含有非有限灰度值。")
    quantized = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return base64.b64encode(quantized.tobytes(order="C")).decode("ascii")


def encode_mask(array: np.ndarray) -> str:
    quantized = np.asarray(array, dtype=np.bool_).astype(np.uint8) * 255
    return base64.b64encode(quantized.tobytes(order="C")).decode("ascii")


def build_viewer(output: Path) -> Path:
    output = output.resolve()
    index_path = output / "preview_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    samples = []
    for viewer_id, source in enumerate(index["samples"]):
        item = dict(source)
        array_path = (output / item["array_file"]).resolve()
        try:
            array_path.relative_to(output)
        except ValueError as error:
            raise ValueError("array_file 必须位于结果文件夹内。") from error
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("image", "edge_mask", "dilated_mask", "processed")}
            shape = tuple(int(size) for size in arrays["image"].shape)
            if len(shape) != 3 or any(size < 1 for size in shape):
                raise ValueError(f"无效的 D×H×W 尺寸：{array_path}")
            if any(tuple(array.shape) != shape for array in arrays.values()):
                raise ValueError(f"同一样本的数组尺寸不一致：{array_path}")
            if item.get("shape_dhw") is not None and tuple(item["shape_dhw"]) != shape:
                raise ValueError(f"清单尺寸与数组尺寸不一致：{array_path}")
            item["shape_dhw"] = list(shape)
            item["viewer_id"] = viewer_id
            item["arrays"] = {
                "image": encode_gray(arrays["image"]),
                "edge_mask": encode_mask(arrays["edge_mask"]),
                "dilated_mask": encode_mask(arrays["dilated_mask"]),
                "processed": encode_gray(arrays["processed"]),
            }
        samples.append(item)
    if not samples:
        raise ValueError("preview_index.json 中没有样本。")
    payload = json.dumps({"config": index.get("config", {}), "samples": samples}, ensure_ascii=False, allow_nan=False)
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(old, new)
    destination = output / "查看20个样本.html"
    destination.write_text(HTML.replace("__PREVIEW_DATA__", payload), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="生成自包含的 20 样本边缘处理离线查看器。")
    parser.add_argument("--output", type=Path, required=True, help="包含 preview_index.json 与 NPZ 的结果文件夹")
    args = parser.parse_args()
    destination = build_viewer(args.output)
    print(destination)


if __name__ == "__main__":
    main()
