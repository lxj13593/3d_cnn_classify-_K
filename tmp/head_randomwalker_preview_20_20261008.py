"""Read-only-source 20-volume Random Walker preview, distinct v2 output.

Unvalidated automatic seeds and head window are exposed for visual inspection.
No per-label tuning, hole filling, resizing, source edits or training.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import scipy
import skimage
from scipy import ndimage as ndi
from skimage.segmentation import random_walker

from head_connected_preview_20_20261008 import (
    OLD, ROOT, STRUCT6, STRUCT26, plotting, cut, body_overlay, sha,
)


DISPLAY_VERSION = "v2b"


def segment(image, flipped, marker_version="extended"):
    s = ndi.gaussian_filter(image, sigma=1., mode="reflect", truncate=4.)
    d, h, w = image.shape
    yy, xx = np.ogrid[:h, :w]
    disk = (yy-(h-1)/2)**2+(xx-(w-1)/2)**2 <= (.30*min(h,w))**2
    score = s[4:11].copy()
    score[:, ~disk] = -1
    at = np.unravel_index(score.argmax(), score.shape)
    seed = (int(at[0])+4, int(at[1]), int(at[2]))
    initial = np.zeros_like(image, dtype=bool)
    initial[seed] = True
    bright = ndi.binary_propagation(initial, mask=s >= .75, structure=STRUCT26)
    lower = np.zeros_like(bright)
    lower[4:11] = disk
    foreground = bright & lower
    foreground[seed] = True
    footprint = np.any(foreground, axis=0)
    py, px = np.where(footprint)
    # Same footprint/margin as v1: the main change is competing background seeds.
    y0, y1 = max(0, int(py.min())-5), min(h, int(py.max())+6)
    x0, x1 = max(0, int(px.min())-5), min(w, int(px.max())+6)
    scope = np.zeros_like(bright)
    scope[:, y0:y1, x0:x1] = True
    if marker_version == "extended":
        # Extend reliable body seeds along its bright core, not merely D4..10.
        # A narrow footprint guard prevents far-off bright board material seeds.
        guard2d = ndi.binary_dilation(footprint, structure=np.ones((3,3),dtype=bool), iterations=2)
        guard = np.broadcast_to(guard2d, image.shape)
        fg_cutoff = max(.50, .70*float(s[seed]))
        foreground = ndi.binary_propagation(initial, mask=(s >= fg_cutoff) & guard & scope, structure=STRUCT26)
    else:
        fg_cutoff = .75
    ring = np.zeros_like(bright)
    ring[:, y0, x0:x1] = True
    ring[:, y1-1, x0:x1] = True
    ring[:, y0:y1, x0] = True
    ring[:, y0:y1, x1-1] = True
    markers = np.zeros(image.shape, dtype=np.int8)
    markers[~scope | ring | (s <= .18)] = 2
    markers[foreground] = 1
    # D=39 is NOT forced to background: a real protrusion can reach this face.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probabilities = random_walker(s, markers, beta=130., mode="cg_j", tol=1e-4,
                                      return_full_prob=True, spacing=(1.,1.,1.))
    p = np.asarray(probabilities[0], dtype=np.float32)
    if p.shape != image.shape or not np.isfinite(p).all():
        raise ValueError("Invalid Random Walker probability volume")
    if p.min() < -.001 or p.max() > 1.001:
        raise ValueError("Solver probabilities outside allowed numerical tolerance")
    p = np.clip(p, 0, 1)
    candidate = (p >= .5) & scope
    labels, count = ndi.label(candidate, structure=STRUCT26)
    ids = np.unique(labels[foreground])
    keep = np.zeros(count+1, dtype=bool)
    keep[ids[ids > 0]] = True
    body = keep[labels]
    background = ~body
    faces = np.zeros_like(body)
    faces[0] = background[0]
    faces[-1] = background[-1]
    faces[:,0] = background[:,0]
    faces[:,-1] = background[:,-1]
    faces[:,:,0] = background[:,:,0]
    faces[:,:,-1] = background[:,:,-1]
    exterior = ndi.binary_propagation(faces, mask=background, structure=STRUCT6)
    surface = body & ndi.binary_dilation(exterior, structure=STRUCT6)
    surface[ring] = False
    surface[0] = False
    surface[-1] = False
    reference = 14 if flipped else 15
    head_start = reference-6
    allowed = scope.copy()
    allowed[:head_start] = False
    allowed[ring] = False
    surface &= allowed
    # A wide, gradual gray band avoids a second abrupt thin-mask deletion.
    if surface.any():
        distance = ndi.distance_transform_edt(~surface)
        weight = (np.clip((3.-distance)/2., 0., 1.)*allowed).astype(np.float32)
    else:
        weight = np.zeros_like(image)
    mask = weight > 0
    processed = (image*weight).astype(np.float32)
    flags = []
    if s[seed] < .75:
        flags.append("较弱主体种子：需核对定位")
    near_ring = ndi.binary_dilation(ring, structure=STRUCT26, iterations=2)
    if np.any(body & near_ring & allowed):
        flags.append("主体靠近背景标记框：需复核误分或截断")
    if body[-1].any():
        flags.append("主体碰D=39：可能原始截断或上方误连接")
    if not mask.any():
        flags.append("空表面带")
    uncertain = (p > .2) & (p < .8) & allowed
    near_surface = ndi.binary_dilation(surface, structure=STRUCT26, iterations=2)
    uncertainty = float(uncertain[near_surface & allowed].mean()) if np.any(near_surface & allowed) else 1.
    if uncertainty > .35:
        flags.append("表面附近分割概率不确定区域较多")
    if caught:
        flags.append("求解器有警告，见solver_warnings")
    stats = {
        "seed_dhw": list(seed), "seed_value": float(s[seed]),
        "scope_yx_exclusive": [y0,y1,x0,x1], "head_reference_d": reference,
        "head_start_d": head_start, "body_ratio": float(body.mean()),
        "surface_ratio": float(surface.mean()), "mask_ratio": float(mask.mean()),
        "mean_mask_weight": float(weight.mean()), "surface_uncertainty_ratio": uncertainty,
        "foreground_seed_count": int(np.count_nonzero(markers == 1)),
        "foreground_marker_version": marker_version,
        "foreground_brightness_cutoff": fg_cutoff,
        "background_seed_count": int(np.count_nonzero(markers == 2)),
        "body_component_count_26": int(ndi.label(body, structure=STRUCT26)[1]),
        "retained_intensity_ratio": float(processed.sum()/max(float(image.sum()),1e-12)),
        "flags": flags, "status": "需重点复核" if flags else "待人工验收",
        "solver_warnings": [str(w.message) for w in caught],
        "open_inner_wall_exclusion": "开放孔腔内壁仍可能进入头部表面带，未填洞，未宣称已排除",
    }
    arrays = {"image": image, "seed_markers": markers, "body_probability": p,
              "scope_mask": scope, "body_mask": body, "surface_mask": surface,
              "dilated_mask": mask, "surface_weight": weight, "processed": processed}
    return arrays, stats


def draw(ax, key, arrays, r, plane, fixed):
    if key == "body_mask":
        body_overlay(ax, arrays, r, plane, fixed)
    elif key == "seed_markers":
        raw = cut(arrays["image"], plane, fixed)
        marker = cut(arrays[key], plane, fixed)
        rgb = np.repeat(raw[...,None],3,axis=-1)
        rgb[marker == 1] = [.1,1.,.1]
        rgb[marker == 2] = [.15,.4,1.]
        ax.imshow(rgb, origin="lower", interpolation="nearest")
    else:
        ax.imshow(cut(arrays[key],plane,fixed), cmap="gray", vmin=0, vmax=1,
                  origin="lower", interpolation="nearest")


KEYS = ("image","seed_markers","body_probability","body_mask","surface_weight","processed")
TITLES = ("原图","主体绿／背景蓝标记","主体归属概率","主体叠加／框","渐变表面带","原图 × 表面带")


def detailed(plt,path,r,a):
    d,h,w=a["image"].shape
    sd,sh,sw=r["seed_dhw"]
    rows=[("dw",h//2,"D-W中心"),("dh",w//2,"D-H中心"),
          ("dw",sh,"D-W过主体种子"),("dh",sw,"D-H过主体种子"),
          ("hw",19,"H-W D=19"),("hw",23,"H-W D=23")]
    fig,axes=plt.subplots(6,6,figsize=(18,17))
    for j,(plane,fixed,name) in enumerate(rows):
        for k,key in enumerate(KEYS):
            ax=axes[j,k]
            draw(ax,key,a,r,plane,fixed)
            if j==0: ax.set_title(TITLES[k],fontsize=10)
            if k==0: ax.set_ylabel(name,fontsize=9)
            ax.tick_params(labelsize=7)
    flag="；".join(r["flags"]) or "没有触发数值警告，仍需看图验收"
    diff="清晰" if r["difficulty"]=="clear" else "模糊"
    fig.suptitle(f"{r['index']:02d} {diff}{r['label_cn']} | {r['board_key']} 钻孔{r['sequence']}\n"
                 f"{DISPLAY_VERSION} 双标记Random Walker · 宽渐变表面带 · 无填洞 · 范围D≥{r['head_start_d']}\n{flag}",fontsize=12,y=.985)
    fig.text(.5,.012,"绿=主体标记，蓝=背景标记。概率白=更偏主体；并非真值。原图未移动/变形，孔腔不强行填满。",ha="center",fontsize=9)
    fig.subplots_adjust(left=.055,right=.99,top=.925,bottom=.04,hspace=.25,wspace=.20)
    fig.savefig(path,dpi=105,facecolor="white")
    plt.close(fig)


def overview(plt,path,items,cn):
    fig,axes=plt.subplots(10,6,figsize=(17,27))
    for j,(r,a) in enumerate(items):
        for k,key in enumerate(KEYS):
            ax=axes[j,k]
            draw(ax,key,a,r,"dw",a["image"].shape[1]//2)
            ax.axis("off")
            title=TITLES[k]
            if k==0: title=f"{r['index']:02d} {r['board_key']} 钻孔{r['sequence']}\n{title}"
            ax.set_title(title,fontsize=9)
    fig.suptitle(f"{cn}10样本 · {DISPLAY_VERSION}双标记三维分割\nD-W中心切片未必经过凸起；概率与表面带均为候选，未作形态真值",fontsize=15,y=.993)
    fig.subplots_adjust(left=.025,right=.99,top=.96,bottom=.01,hspace=.28,wspace=.08)
    fig.savefig(path,dpi=95,facecolor="white")
    plt.close(fig)


HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Head40·v2双标记三维分割20样本</title><style>
*{box-sizing:border-box}body{font-family:"Microsoft YaHei",sans-serif;margin:20px;background:#f5f6f8;color:#222}h1{font-size:24px}p{line-height:1.6}.card{padding:15px;background:white;border:1px solid #ddd;border-radius:8px;margin:12px 0}.controls{display:flex;gap:15px;flex-wrap:wrap;align-items:center}select,button{font:inherit;padding:7px}#sample{max-width:650px}#slice{width:min(60vw,650px)}.panels{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}h2{font-size:15px;text-align:center}.box{background:#111;min-height:260px;display:flex;align-items:center;justify-content:center;padding:8px}canvas{max-width:100%;height:auto;image-rendering:pixelated}.note{font-size:13px;color:#555}#info{line-height:1.7;overflow-wrap:anywhere}#error{color:#a11;white-space:pre-wrap}@media(max-width:1200px){.panels{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:650px){.panels{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style><h1>Head40 · v2双标记三维分割 · 同样20个样本</h1>
<p>独立新预览，旧两版保留。主体标记绿色、背景标记蓝色；概率图白色表示更偏主体。原图未改动，孔腔不填满，二维两块不强行拼接。</p>
<section class="card"><div class="controls"><select id="category"><option value="all">全部20</option><option value="normal">正常10</option><option value="defective">缺陷10</option></select><select id="sample"></select><select id="plane"><option value="dw">D-W（固定H）</option><option value="dh">D-H（固定W）</option><option value="hw">H-W（固定D）</option></select><select id="scale"><option value="6">6倍</option><option value="10" selected>10倍</option></select></div><p><button id="prev">上一切片</button> <input id="slice" type="range" min="0" step="1"> <button id="next">下一切片</button> <span id="position"></span></p><div id="info"></div></section>
<section class="card panels">__PANELS__</section>
<section class="card note">同一位置对照；D高索引在上，H-W中H高索引在上；所有灰度固定0–1，不逐图拉伸。6列分别为原图、自动标记、主体概率、主体叠加、渐变表面带、处理后。<p>sigma=1辅助定位；主体标记来自低D稳定亮核心；背景标记为外围框和低灰度区；Random Walker beta=130；主体概率≥0.5；表面向周围保留至不足3体素并渐变衰减。头部窗口沿用参考层减6。这些规则尚未验证，不保证完全去噪或保住全部凸起，开放孔腔内壁仍可能保留。</p>原尺寸仍是40×H×W，原数组为float32；网页仅为8位显示，完整数组见NPZ。不依赖网络。</section><pre id="error"></pre>
<script id="data" type="application/json">__DATA__</script><script>
"use strict";
const payload=JSON.parse(document.getElementById('data').textContent),samples=payload.samples;
const keys=['image','seed_markers','body_probability','body_mask','surface_weight','processed'];
const ui=Object.fromEntries(['category','sample','plane','scale','slice','prev','next','position','info','error'].map(k=>[k,document.getElementById(k)]));
let current;const cache=new Map();
function decode(item,key){const id=item.index+':'+key;if(!cache.has(id)){const b=atob(item.arrays[key]),v=Uint8Array.from(b,c=>c.charCodeAt(0));if(v.length!==item.shape_dhw.reduce((a,b)=>a*b,1))throw Error('数组长度不一致');cache.set(id,v);}return cache.get(id);}
function render(){const[D,H,W]=current.shape_dhw,plane=ui.plane.value,fixed=Number(ui.slice.value),width=plane==='dh'?H:W,height=plane==='hw'?H:D;
ui.position.textContent='固定 '+({dw:'H',dh:'W',hw:'D'}[plane])+'='+fixed+' / '+ui.slice.max;
ui.prev.disabled=fixed===0;ui.next.disabled=fixed===Number(ui.slice.max);
for(const key of keys){const v=decode(current,key),raw=decode(current,'image'),canvas=document.getElementById(key);canvas.width=width;canvas.height=height;canvas.style.width=width*Number(ui.scale.value)+'px';const ctx=canvas.getContext('2d'),pix=ctx.createImageData(width,height);
for(let y=0;y<height;y++)for(let x=0;x<width;x++){let d,h,w;if(plane==='dw'){d=D-1-y;h=fixed;w=x;}else if(plane==='dh'){d=D-1-y;h=x;w=fixed;}else{d=fixed;h=H-1-y;w=x;}const at=(d*H+h)*W+w,o=(y*width+x)*4,z=(key==='seed_markers'||key==='body_mask')?raw[at]:v[at];pix.data[o]=pix.data[o+1]=pix.data[o+2]=z;pix.data[o+3]=255;
if(key==='seed_markers'&&v[at]===1){pix.data[o]=25;pix.data[o+1]=255;pix.data[o+2]=25;}else if(key==='seed_markers'&&v[at]===2){pix.data[o]=38;pix.data[o+1]=102;pix.data[o+2]=255;}else if(key==='body_mask'&&v[at]){pix.data[o]=Math.round(z*.58+25*.42);pix.data[o+1]=Math.round(z*.58+230*.42);pix.data[o+2]=Math.round(z*.58+128*.42);}}
ctx.putImageData(pix,0,0);if(key==='body_mask'){const[y0,y1,x0,x1]=current.scope_yx_exclusive;ctx.strokeStyle='#ffd166';ctx.lineWidth=.3;if(plane==='dw'&&fixed>=y0&&fixed<y1)ctx.strokeRect(x0,0,x1-x0,D);if(plane==='dh'&&fixed>=x0&&fixed<x1)ctx.strokeRect(y0,0,y1-y0,D);if(plane==='hw')ctx.strokeRect(x0,H-y1,x1-x0,y1-y0);}}
}
function bounds(){const[D,H,W]=current.shape_dhw;ui.slice.max=({dw:H,dh:W,hw:D}[ui.plane.value])-1;ui.slice.value=Math.floor(Number(ui.slice.max)/2);render();}
function choose(){current=samples.find(r=>String(r.index)===ui.sample.value);ui.info.textContent=current.label_cn+' / '+(current.difficulty==='clear'?'清晰':'模糊')+' / '+current.board_key+' / 钻孔'+current.sequence+' / '+current.shape_dhw.join('×')+'；表面带覆盖'+(current.mask_ratio*100).toFixed(1)+'%；D≥'+current.head_start_d+'；'+current.status+'；'+(current.flags.join('；')||'未触发数值警告，但尚未验收');bounds();}
function fill(){ui.sample.replaceChildren();for(const r of samples)if(ui.category.value==='all'||r.label===ui.category.value){const o=document.createElement('option');o.value=r.index;o.textContent=String(r.index).padStart(2,'0')+' '+r.label_cn+' '+(r.difficulty==='clear'?'清晰':'模糊')+' '+r.board_key+' 钻孔'+r.sequence;ui.sample.append(o);}choose();}
function safe(fn){try{fn();ui.error.textContent='';}catch(e){ui.error.textContent=e.stack||e.message;}}
ui.category.addEventListener('change',()=>safe(fill));ui.sample.addEventListener('change',()=>safe(choose));ui.plane.addEventListener('change',()=>safe(bounds));ui.scale.addEventListener('change',()=>safe(render));ui.slice.addEventListener('input',()=>safe(render));ui.prev.addEventListener('click',()=>safe(()=>{ui.slice.value=Math.max(0,Number(ui.slice.value)-1);render();}));ui.next.addEventListener('click',()=>safe(()=>{ui.slice.value=Math.min(Number(ui.slice.max),Number(ui.slice.value)+1);render();}));safe(fill);
</script></html>'''


def viewer(out,index,items):
    samples=[]
    for r,a in items:
        row=dict(r, arrays={})
        for key in KEYS:
            v=a[key].astype(np.uint8) if key=='seed_markers' else np.rint(np.clip(a[key],0,1)*255).astype(np.uint8)
            row['arrays'][key]=base64.b64encode(v.tobytes()).decode('ascii')
        samples.append(row)
    payload=json.dumps(dict(config=index['config'],samples=samples),ensure_ascii=False,allow_nan=False)
    for x,y in (("&","\\u0026"),("<","\\u003c"),(">","\\u003e"),("\u2028","\\u2028"),("\u2029","\\u2029")):
        payload=payload.replace(x,y)
    panels=''.join(f'<article><h2>{name}</h2><div class="box"><canvas id="{key}"></canvas></div></article>' for key,name in zip(KEYS,TITLES))
    # Never substitute version strings after embedding base64 volume data.
    html = HTML.replace('v2',DISPLAY_VERSION).replace('__PANELS__',panels)
    html = html.replace('主体标记来自低D稳定亮核心',
        '主体标记来自低D定位后延伸的相连亮核心，阈值max(0.50,0.70×种子灰度)，受核心横向范围扩展2体素约束' if DISPLAY_VERSION == 'v2b' else '主体标记来自低D稳定亮核心')
    (out/'查看20个样本.html').write_text(html.replace('__DATA__',payload),encoding='utf8')


def main():
    global DISPLAY_VERSION
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--marker-version',choices=('lower','extended'),default='extended')
    parser.add_argument('--viewer-only',action='store_true')
    args=parser.parse_args()
    DISPLAY_VERSION = 'v2b' if args.marker_version == 'extended' else 'v2'
    out=args.output.resolve()
    out.relative_to(ROOT/'查找结果')
    if args.viewer_only:
        # Repair only this turn's new viewer, never touch an old preview.
        if out.name != 'Head40_双标记三维分割_20样本预览_20261008_v2b':
            raise ValueError('Viewer repair is restricted to the new v2b output')
        index=json.loads((out/'preview_index.json').read_text(encoding='utf8'))
        items=[]
        for r in index['samples']:
            with np.load(out/r['array_file'],allow_pickle=False) as saved:
                items.append((r,{k:saved[k] for k in saved.files}))
        viewer(out,index,items)
        print('VIEWER REPAIRED '+str(out),flush=True)
        return
    if out.exists(): raise FileExistsError(f'Refusing overwrite: {out}')
    old_index=json.loads((OLD/'preview_index.json').read_text(encoding='utf8'))
    if len(old_index['samples'])!=20: raise ValueError('Expected same 20 samples')
    v1=ROOT/'查找结果/Head40_头部连通约束_20样本预览_20261008_v1'
    # Include all old artifacts, not just arrays, in preservation checks.
    v2=ROOT/'查找结果/Head40_双标记三维分割_20样本预览_20261008_v2'
    old_folders=(OLD,v1,v2) if out != v2 and v2.exists() else (OLD,v1)
    tracked=[p for folder in old_folders for p in folder.rglob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in tracked}
    out.mkdir(parents=True,exist_ok=False)
    for folder in ('单样本对照图','处理后体数据'): (out/folder).mkdir()
    config=dict(representation='random_walker_head_surface_band_candidate',version='preview_20261008_'+DISPLAY_VERSION,
                preview_only=True,validated_for_training=False,input='old_npz.image_no_reflip_no_renormalization',
                gaussian_sigma=[1,1,1],gaussian_mode='reflect',gaussian_truncate=4,
                foreground=('seed_connected_bright_core>=max(0.50,0.70*seedgray)_allD_in_lower_core_footprint_plus2' if args.marker_version == 'extended' else 'D4..10_seed_connected_bright_core>=0.75_in_central_disk_radius0.3_minHW'),
                marker_version=args.marker_version,
                background='outside_search_rectangle_or_rectangle_lateral_faces_or_smoothed_gray<=0.18',
                rectangle_margin_voxels=5,random_walker_beta=130,solver='cg_j',solver_tol=1e-4,
                foreground_probability_cutoff=.5,head_start='crop_reference_D14_if_flipped_else15_minus6',
                surface_band='euclidean_distance_to_surface; weight=clip((3-distance)/2,0,1); within_scope_head_window',
                hole_filling=False,closing=False,canny=False,source_data_modified=False,
                numpy_version=np.__version__,scipy_version=scipy.__version__,skimage_version=skimage.__version__,
                known_limitations=['自动背景标记可能误标凸起','头部窗口和搜索框未验证','开放孔腔内壁可能保留','RW概率不是正确率或标注真值'],
                reference='https://scikit-image.org/docs/0.25.x/api/skimage.segmentation.html#skimage.segmentation.random_walker')
    plt=plotting()
    records=[];items=[]
    for old in old_index['samples']:
        with np.load(OLD/old['array_file'],allow_pickle=False) as a:
            image=a['image'].copy()
        if image.dtype!=np.float32 or image.shape!=tuple(old['shape_dhw']) or not np.isfinite(image).all():
            raise ValueError('Invalid source volume')
        arrays,stats=segment(image,old['direction_flipped'],args.marker_version)
        if not np.array_equal(arrays['processed'],image*arrays['surface_weight']): raise AssertionError('Unexpected intensity transform')
        r={k:v for k,v in old.items() if k not in ('edge_ratio','mask_ratio','empty_mask')}
        r.update(stats);r['source_npz_sha256']=hashes[str(OLD/old['array_file'])]
        np.savez_compressed(out/r['array_file'],**arrays)
        detailed(plt,out/r['png_file'],r,arrays)
        records.append(r);items.append((r,arrays))
        print(f"Processed {r['index']:02d}/20: mask={r['mask_ratio']:.1%}; warnings={r['flags']}",flush=True)
    for label,cn in (('normal','正常'),('defective','缺陷')):
        overview(plt,out/f'{cn}10样本_总览.png',[x for x in items if x[0]['label']==label],cn)
    index=dict(config=config,samples=records)
    (out/'preview_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'处理参数.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'20样本清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader()
        for r in records: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    viewer(out,index,items)
    for p,digest in hashes.items():
        if sha(Path(p))!=digest: raise AssertionError('Old artifact modified: '+p)
    checks=dict(samples=20,same_sample_ids_and_order=[r['sample_id'] for r in records]==[r['sample_id'] for r in old_index['samples']],
                old_versions_unchanged=True,old_folders=[str(p) for p in old_folders],old_artifact_count=len(hashes),
                empty_masks=sum(not a['dilated_mask'].any() for _,a in items),
                flagged_samples=[r['index'] for r in records if r['flags']],training_run=False)
    (out/'必要检查.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    notes=[f'# {DISPLAY_VERSION}：双标记三维分割20样本预览','','同样20个训练样本，旧版全部保留，没有训练或修改数据。打开查看20个样本.html，可检查全部切片。','',
           '## 实际变化','','固定灰度0.40区域生长 → 主体/背景双标记Random Walker；一体素硬边缘带 → 距离不足3体素的渐变灰度带。自动定位、方向、原图归一化和头部窗口不变，未逐样本调参。','',
           '## 看图注意','','原图、标记、概率、主体、渐变Mask、处理后六列。蓝色是算法预设背景，若蓝色落在凸起上，应判断为标记失败；不能把概率白色视为真实主体。','',
           '二维中心切片可能出现同一孔的两侧壁，不为了视觉连起来而填孔。开放孔腔内壁尚不能保证排除；无数值警告不等于保住全部缺陷，框或窗口仍可能截断。','',
           '不能以警告减少、Mask变连续作为模型效果提升依据；仅供人工验收。','',
           '## 每样本提示','']
    if args.marker_version == 'extended':
        notes[6] += ' 修正v2低D标记不足：主体标记沿相连亮核心向头部延伸，阈值max(0.50,0.70×种子灰度)，限制在低D核心横向范围外扩2体素内；不是人工标注的真实主体，仍可能错标。'
    for r in records: notes.append(f"- {r['index']:02d} {r['board_key']} 钻孔{r['sequence']}：{'；'.join(r['flags']) or '无数值警告，仍需验收'}。")
    (out/'预览说明.md').write_text('\n'.join(notes)+'\n',encoding='utf8')
    print('DONE '+str(out),flush=True)
    print(json.dumps(checks,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
