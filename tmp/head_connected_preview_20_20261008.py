"""Twenty-sample, non-overwriting candidate preview. No training/source edits.

The three still-unvalidated rules in the proposal are made explicit below.
This is an inspectable approximation, NOT a verified anatomical segmentation.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT / "查找结果/Head40_3D_Canny_20样本预览_20261005"
STRUCT26 = np.ones((3, 3, 3), dtype=bool)
STRUCT6 = ndi.generate_binary_structure(3, 1)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segment(image, flipped):
    smooth = ndi.gaussian_filter(image, sigma=(1., 1., 1.), mode="reflect", truncate=4.)
    d, h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    disk = (yy - (h-1)/2)**2 + (xx - (w-1)/2)**2 <= (.30*min(h, w))**2
    # Stable lower body is used only for localization, never as a defect label.
    scores = smooth[4:11].copy()
    scores[:, ~disk] = -1
    pos = np.unravel_index(scores.argmax(), scores.shape)
    seed = (int(pos[0])+4, int(pos[1]), int(pos[2]))
    marker = np.zeros(image.shape, dtype=bool)
    marker[seed] = True
    flags = []
    if smooth[seed] < .75:
        flags.append("弱主体种子：平滑灰度低于0.75")
    # Connected bright core anchors a generous rectangular search box.
    core = ndi.binary_propagation(marker, structure=STRUCT26, mask=smooth >= .75)
    footprint = np.any(core[4:11], axis=0) & disk
    if not footprint.any():
        footprint[seed[1], seed[2]] = True
    py, px = np.where(footprint)
    y0, y1 = max(0, int(py.min())-5), min(h, int(py.max())+6)
    x0, x1 = max(0, int(px.min())-5), min(w, int(px.max())+6)
    scope = np.zeros(image.shape, dtype=bool)
    scope[:, y0:y1, x0:x1] = True
    candidate = (smooth >= .40) & scope
    body = ndi.binary_propagation(marker, structure=STRUCT26, mask=candidate)
    # All-face flood of true in-volume background with 6 adjacency. This drops
    # sealed cavities, but an open inner bore can still connect to outside.
    background = ~body
    exterior_seeds = np.zeros_like(body)
    exterior_seeds[0] = background[0]
    exterior_seeds[-1] = background[-1]
    exterior_seeds[:, 0] = background[:, 0]
    exterior_seeds[:, -1] = background[:, -1]
    exterior_seeds[:, :, 0] = background[:, :, 0]
    exterior_seeds[:, :, -1] = background[:, :, -1]
    exterior = ndi.binary_propagation(exterior_seeds, structure=STRUCT6, mask=background)
    surface = body & ndi.binary_dilation(exterior, structure=STRUCT6)
    # Do not call a boundary created by the rectangular search limit a surface.
    artificial = np.zeros_like(body)
    artificial[:, y0, x0:x1] = True
    artificial[:, y1-1, x0:x1] = True
    artificial[:, y0:y1, x0] = True
    artificial[:, y0:y1, x1-1] = True
    if np.any(body & artificial):
        flags.append("主体碰搜索框：范围不足或背景误连接")
    if body[-1].any():
        flags.append("主体碰D=39：可能原始截断或上方误连接")
    if y0 == 0 or y1 == h or x0 == 0 or x1 == w:
        flags.append("搜索框碰原始H/W边界")
    surface &= ~artificial
    surface[0] = False
    surface[-1] = False
    # Head reference of the existing crop is 14 after flip / 15 otherwise.
    # This reference is NOT an accurately annotated real surface.
    reference = 14 if flipped else 15
    head_start = reference - 6
    allowed = scope.copy()
    allowed[:head_start] = False
    surface &= allowed
    mask = ndi.binary_dilation(surface, structure=STRUCT26, iterations=1) & allowed
    mask[artificial] = False
    processed = (image * mask).astype(np.float32)
    if not mask.any():
        flags.append("空表面Mask")
    # Explicitly report this known unresolved limitation in every result.
    stats = {
        "seed_dhw": list(seed), "seed_value": float(smooth[seed]),
        "scope_yx_exclusive": [y0, y1, x0, x1], "head_reference_d": reference,
        "head_start_d": head_start, "body_ratio": float(body.mean()),
        "surface_ratio": float(surface.mean()), "mask_ratio": float(mask.mean()),
        "retained_intensity_ratio": float(processed.sum()/max(float(image.sum()), 1e-12)),
        "flags": flags, "status": "需重点复核" if flags else "待人工验收",
        "open_inner_wall_exclusion": "未解决：开放孔腔与外界相连，头部窗口内内壁可能保留",
    }
    arrays = {"image": image, "scope_mask": scope, "body_mask": body,
              "surface_mask": surface, "dilated_mask": mask, "processed": processed}
    return arrays, stats


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({"font.family": FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),
                         "axes.unicode_minus": False})
    return plt


def cut(a, plane, fixed):
    if plane == "dw":
        return a[:, fixed, :]
    if plane == "dh":
        return a[:, :, fixed]
    return a[fixed]


def body_overlay(ax, arrays, record, plane, fixed):
    from matplotlib.patches import Rectangle
    a = cut(arrays["image"], plane, fixed)
    b = cut(arrays["body_mask"], plane, fixed)
    rgba = np.zeros((*b.shape, 4))
    rgba[b] = [.1, .9, .5, .42]
    ax.imshow(a, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
    ax.imshow(rgba, origin="lower", interpolation="nearest")
    y0, y1, x0, x1 = record["scope_yx_exclusive"]
    d, h, w = record["shape_dhw"]
    if plane == "dw" and y0 <= fixed < y1:
        rect = (x0-.5, -.5, x1-x0, d)
    elif plane == "dh" and x0 <= fixed < x1:
        rect = (y0-.5, -.5, y1-y0, d)
    elif plane == "hw":
        rect = (x0-.5, y0-.5, x1-x0, y1-y0)
    else:
        rect = None
    if rect:
        ax.add_patch(Rectangle(rect[:2], *rect[2:], fill=False, edgecolor="#ffd166", linewidth=.8))
    sd, sh, sw = record["seed_dhw"]
    if (plane == "dw" and fixed == sh) or (plane == "dh" and fixed == sw) or (plane == "hw" and fixed == sd):
        ax.plot(sw if plane != "dh" else sh, sh if plane == "hw" else sd, "r+", markersize=7)


def detail(plt, target, r, a):
    d, h, w = a["image"].shape
    sd, sh, sw = r["seed_dhw"]
    rows = [("dw", h//2, "D-W中心"), ("dh", w//2, "D-H中心"),
            ("dw", sh, "D-W经过主体种子"), ("dh", sw, "D-H经过主体种子"),
            ("hw", 19, "H-W D=19"), ("hw", 23, "H-W D=23")]
    fig, axes = plt.subplots(len(rows), 5, figsize=(15, 17))
    keys = ["image", "body_mask", "surface_mask", "dilated_mask", "processed"]
    titles = ["原图", "主体叠加／搜索框", "候选头部表面", "表面带Mask", "原图 × Mask"]
    for j, (plane, fixed, label) in enumerate(rows):
        for k, key in enumerate(keys):
            ax = axes[j, k]
            if key == "body_mask":
                body_overlay(ax, a, r, plane, fixed)
            else:
                ax.imshow(cut(a[key], plane, fixed), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            if j == 0:
                ax.set_title(titles[k], fontsize=10)
            if k == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.tick_params(labelsize=7)
    difficulty = "清晰" if r["difficulty"] == "clear" else "模糊"
    msg = "；".join(r["flags"]) or "未触发数值警告，仍需看图验收"
    fig.suptitle(f"{r['index']:02d} {difficulty}{r['label_cn']} | {r['board_key']} 钻孔{r['sequence']}\n"
                 f"候选预览：主体阈值0.40，表面带1体素，头部范围D≥{r['head_start_d']} | Mask {r['mask_ratio']:.1%}\n{msg}", fontsize=12, y=.985)
    fig.text(.5, .012, "绿色=主体；黄色=搜索框；红十字=种子。D高索引在上。开放孔腔内壁尚不能保证排除；不是已验证结果。", ha="center", fontsize=9)
    fig.subplots_adjust(left=.075, right=.99, bottom=.04, top=.925, hspace=.25, wspace=.20)
    fig.savefig(target, dpi=105, facecolor="white")
    plt.close(fig)


def overview(plt, target, items, label):
    fig, axes = plt.subplots(5, 10, figsize=(27, 15))
    for j, (r, a) in enumerate(items):
        row, offset = j % 5, (j // 5)*5
        fixed = a["image"].shape[1]//2
        for k, key in enumerate(("image", "body_mask", "surface_mask", "dilated_mask", "processed")):
            ax = axes[row, offset+k]
            if key == "body_mask":
                body_overlay(ax, a, r, "dw", fixed)
            else:
                ax.imshow(cut(a[key], "dw", fixed), cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
            ax.axis("off")
            title = ["原图", "主体／框", "表面", "表面Mask", "处理后"][k]
            if k == 0:
                title = f"{r['index']:02d} {r['board_key']}\n钻孔{r['sequence']} {'清晰' if r['difficulty']=='clear' else '模糊'}\n{title}"
            else:
                title = "\n\n" + title
            ax.set_title(title, fontsize=9)
    fig.suptitle(f"{label}10样本：主体连通约束 → 候选头部表面带\n全部D-W中心切片；中心切片未必经过凸起。不是纯外表面真值，疑似泄漏／截断详见单样本图。", fontsize=16, y=.985)
    fig.subplots_adjust(left=.01, right=.995, bottom=.012, top=.91, hspace=.38, wspace=.08)
    fig.savefig(target, dpi=100, facecolor="white")
    plt.close(fig)


def viewer(output, index, items):
    # Reuse the established offline UI, changing only the representation panels.
    from canny_preview_viewer_20261005 import HTML
    html = HTML.replace("Head40 · 20 个样本边缘处理对照", "Head40 · 主体连通约束20样本候选预览")
    html = html.replace("repeat(4, minmax(0, 1fr))", "repeat(5, minmax(0, 1fr))")
    html = html.replace("四列显示", "五列显示")
    html = html.replace('<article class="panel"><h2>Canny 二值边缘</h2><div class="canvas-box"><canvas id="edge_mask"></canvas></div></article>',
        '<article class="panel"><h2>主体叠加／搜索框</h2><div class="canvas-box"><canvas id="body_mask"></canvas></div></article><article class="panel"><h2>候选头部表面</h2><div class="canvas-box"><canvas id="surface_mask"></canvas></div></article>')
    html = html.replace("膨胀后的边缘掩膜", "头部表面带Mask")
    html = html.replace('const keys = ["image", "edge_mask", "dilated_mask", "processed"];',
                        'const keys = ["image", "body_mask", "surface_mask", "dilated_mask", "processed"];')
    html = html.replace('const value = values[(d * H + h) * W + w];', '''const at = (d * H + h) * W + w;
            const value = key === "body_mask" ? decode(current, "image")[at] : values[at];''')
    html = html.replace('pixels.data[offset + 3] = 255;', '''pixels.data[offset + 3] = 255;
            if(key === "body_mask" && values[at]) {
              pixels.data[offset] = Math.round(value*.58+25*.42);
              pixels.data[offset+1] = Math.round(value*.58+230*.42);
              pixels.data[offset+2] = Math.round(value*.58+128*.42);
            }''')
    html = html.replace('context.putImageData(pixels, 0, 0);', '''context.putImageData(pixels, 0, 0);
        if(key === "body_mask") {
          const [y0,y1,x0,x1] = current.scope_yx_exclusive;
          context.strokeStyle="#ffd166";context.lineWidth=.3;
          if(plane==="dw" && fixed>=y0 && fixed<y1) context.strokeRect(x0,height-D,x1-x0,D);
          if(plane==="dh" && fixed>=x0 && fixed<x1) context.strokeRect(y0,height-D,y1-y0,D);
          if(plane==="hw") context.strokeRect(x0,H-y1,x1-x0,y1-y0);
          const [sd,sh,sw]=current.seed_dhw;
          if((plane==="dw" && fixed===sh)||(plane==="dh" && fixed===sw)||(plane==="hw" && fixed===sd)) {
            context.fillStyle="red";context.fillRect(plane==="dh"?sh:sw,plane==="hw"?H-1-sh:D-1-sd,1,1);
          }
        }''')
    html = html.replace('percent(current.mask_ratio);', '''percent(current.mask_ratio) + "；头部窗口 D≥" + current.head_start_d + "；状态：" + current.status + "；警告：" + (current.flags.join("；") || "无数值警告，但尚未验收");''')
    html = html.replace('ui.config.textContent = "处理配置：训练样本数 " + String(config.training_sample_count ?? "未记录") + "；高阈值 " + numberText(config.high_threshold) + "；低阈值 " + numberText(config.low_threshold) + "。详细参数见同目录 preview_index.json。";',
       'ui.config.textContent = "候选配置：sigma=1；主体阈值0.40；种子核心阈值0.75；框余量5体素；26连通；6邻接外部背景；表面带膨胀1次。开放孔腔内壁尚未完全排除，搜索框碰撞或D39接触均需复核。不得直接当成正式训练配置。";')
    html = html.replace("图中没有标注或推测缺陷位置。", "没有标注或推测缺陷位置。绿色是候选主体，黄色是搜索框，红点是主体种子。")
    html = html.replace("边缘与膨胀掩膜中，白色表示保留位置。", "表面与表面带中，白色表示候选保留位置。")
    data = []
    for i, (r, a) in enumerate(items):
        item = dict(r, viewer_id=i)
        item["arrays"] = {}
        for k in ("image", "body_mask", "surface_mask", "dilated_mask", "processed"):
            encoded = np.rint(np.clip(a[k], 0, 1)*255).astype(np.uint8)
            item["arrays"][k] = base64.b64encode(encoded.tobytes()).decode("ascii")
        data.append(item)
    payload = json.dumps({"config": index["config"], "samples": data}, ensure_ascii=False, allow_nan=False)
    for old, new in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"), ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        payload = payload.replace(old, new)
    (output/"查看20个样本.html").write_text(html.replace("__PREVIEW_DATA__", payload), encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.relative_to(ROOT / "查找结果")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite: {out}")
    original_index = json.loads((OLD/"preview_index.json").read_text(encoding="utf8"))
    if len(original_index["samples"]) != 20:
        raise ValueError("Expected original twenty samples")
    tracked = [OLD/"preview_index.json", OLD/"edge_config.json", OLD/"20样本清单.csv"]
    tracked += [OLD/r["array_file"] for r in original_index["samples"]]
    old_hashes = {str(p): sha(p) for p in tracked}
    out.mkdir(parents=True, exist_ok=False)
    (out/"单样本对照图").mkdir()
    (out/"处理后体数据").mkdir()
    config = {
        "representation": "seed_connected_head_surface_band_candidate",
        "version": "preview_20261008_v1", "preview_only": True, "validated_for_training": False,
        "original_preview": str(OLD), "source_index_sha256": old_hashes[str(OLD/"preview_index.json")],
        "input": "existing_npz.image; already_direction_unified_and_normalized; no_second_flip_or_normalization",
        "gaussian_sigma": [1, 1, 1], "gaussian_mode": "reflect", "gaussian_truncate": 4,
        "seed": "argmax_smooth_D4_to10_in_central_disk_radius_0.30_minHW",
        "core_threshold": .75, "body_threshold": .40, "search_margin_voxels": 5,
        "scope": "bbox_of_connected_core_D4_to10_footprint_in_central_disk_plus5",
        "body_connectivity": 26, "external_background_connectivity": 6,
        "head_start": "existing_crop_reference_D14_if_flipped_else15_minus6",
        "dilation": "full3x3x3_once", "canny": False, "augment": False,
        "known_limitations": ["阈值和范围为统一预览候选，未做正式训练参数估计", "开放孔腔内壁无法仅用外部洪泛完全排除",
                              "搜索框可截断凸起；相连背景可能泄漏；真实原始截断无法恢复"],
        "source_data_modified": False,
    }
    records, items = [], []
    plt = plotting()
    for r0 in original_index["samples"]:
        source = OLD/r0["array_file"]
        with np.load(source, allow_pickle=False) as archive:
            image = archive["image"].copy()
        if image.dtype != np.float32 or tuple(r0["shape_dhw"]) != image.shape or not np.isfinite(image).all():
            raise ValueError(f"Invalid original image {source}")
        arrays, stats = segment(image, r0["direction_flipped"])
        if not np.array_equal(arrays["processed"], image*arrays["dilated_mask"]):
            raise AssertionError("Mask multiplication changed")
        r = {k: v for k, v in r0.items() if k not in ("edge_ratio", "mask_ratio", "empty_mask")}
        r.update(stats)
        r["source_npz_sha256"] = old_hashes[str(source)]
        np.savez_compressed(out/r["array_file"], **arrays)
        detail(plt, out/r["png_file"], r, arrays)
        records.append(r)
        items.append((r, arrays))
        print(f"Processed {r['index']:02d}/20; mask={r['mask_ratio']:.1%}; flags={r['flags']}", flush=True)
    for label, cn in (("normal", "正常"), ("defective", "缺陷")):
        overview(plt, out/f"{cn}10样本_总览.png", [x for x in items if x[0]["label"] == label], cn)
    index = {"config": config, "samples": records}
    (out/"preview_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf8")
    (out/"处理参数.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf8")
    with (out/"20样本清单.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for r in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    viewer(out, index, items)
    for p, digest in old_hashes.items():
        if sha(Path(p)) != digest:
            raise AssertionError(f"Original changed: {p}")
    validation = {"sample_count": len(records), "old_index_and_arrays_unchanged": True,
                  "same_twenty_ids_in_same_order": [r["sample_id"] for r in records] == [r["sample_id"] for r in original_index["samples"]],
                  "empty_masks": sum(not a["dilated_mask"].any() for _, a in items),
                  "flagged_samples": [r["index"] for r in records if r["flags"]],
                  "training_run": False}
    (out/"必要检查.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf8")
    lines = ["# 20样本主体连通约束预览", "", "本文件夹是独立新结果，旧Canny结果没有覆盖。先打开 `查看20个样本.html`，可切换20个样本、三方向和所有切片。", "",
             "## 本版实际规则", "", "sigma=1辅助分割；D4～10中心区域自动种子；0.75亮核心定位；核心横向框扩展5体素；0.40主体阈值、26连通；6邻接外部背景接触表面；头部参考层减6以下屏蔽；表面膨胀1体素；最终乘未平滑原图。20个样本使用同一规则，没有按标签或逐样本调参。", "",
             "## 必须知道的限制", "", "这只是候选规则的第一次预览，不是已经验证的纯头部外表面。开放孔腔内壁仍可能保留；模糊背景可能和主体误连接；搜索框及原始40层范围可能截断。图中明确标出搜索框、主体和警告，不把这些情况冒充去噪成功。", "",
             "没有Canny二次筛选；没有改数据、标签、加载器或模型，没有训练。输入取旧NPZ的image，未重复翻转或归一化。", "",
             "## 自动复核提示", ""]
    for r in records:
        lines.append(f"- {r['index']:02d} {r['board_key']} 钻孔{r['sequence']}：{'；'.join(r['flags']) or '无数值警告，仍需看图验收'}。")
    (out/"预览说明.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    print("DONE " + str(out), flush=True)
    print(json.dumps(validation, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
