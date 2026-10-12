"""Same 20 samples: H/W rectangular crop only; preserve all 40 depth slices.

The positioning statistic is used ONLY for a generous crop box, never a voxel
segmentation or denoising mask. All grayscale inside the crop is bit-identical.
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import math
from pathlib import Path

import numpy as np

from head_connected_preview_20_20261008 import ROOT, OLD, plotting, cut, sha


def position_box(image):
    _,h,w=image.shape
    yy,xx=np.ogrid[:h,:w]
    disk=(yy-(h-1)/2)**2+(xx-(w-1)/2)**2 <= (.30*min(h,w))**2
    # Raw, unchanged lower-body gray is enough to propose a transverse center.
    # No Gaussian, region growing, connected components or head-surface mask.
    projection=np.mean(image[4:11],axis=0)
    peak=float(projection[disk].max())
    bright=(projection >= .80*peak) & disk
    if not bright.any(): raise ValueError('Cannot locate a proposed crop center')
    by,bx=np.where(bright)
    weights=projection[bright].astype(np.float64)
    cy=float(np.average(by,weights=weights))
    cx=float(np.average(bx,weights=weights))
    # 72% of original width OR lower bright footprint + six voxels each side.
    # Never force a small crop when the observed footprint is relatively broad.
    nh=min(h,max(math.ceil(.72*h),int(by.max()-by.min()+1)+12))
    nw=min(w,max(math.ceil(.72*w),int(bx.max()-bx.min()+1)+12))
    y0=int(np.clip(round(cy-(nh-1)/2),0,h-nh));y1=y0+nh
    x0=int(np.clip(round(cx-(nw-1)/2),0,w-nw));x1=x0+nw
    flags=[]
    if peak < .60: flags.append('下部定位亮区较弱，需核对框是否正确')
    if y0==0 or y1==h or x0==0 or x1==w:
        flags.append('裁剪框碰原图横向边界，需复核偏心或范围不足')
    if nh==h or nw==w:
        flags.append('定位范围较宽：对应方向保留原尺寸，不强行缩小')
    return [y0,y1,x0,x1],dict(center_yx=[cy,cx],localization_peak=peak,
        localization_bright_threshold=.80*peak,localization_slice_d=[4,11],
        flags=flags,status='需核对范围' if flags else '待看图验收，未保证包含全部凸起')


def center_half_box(image):
    """Centered integer crop: ceil(n/2), no intensity-dependent positioning."""
    _, h, w = image.shape
    nh, nw = (h+1)//2, (w+1)//2
    y0, x0 = (h-nh)//2, (w-nw)//2
    y1, x1 = y0+nh, x0+nw
    if (y0+y1-1) != (h-1) or (x0+x1-1) != (w-1):
        raise ValueError('Expected exactly symmetric center crop for these sizes')
    return [y0,y1,x0,x1], dict(center_yx=[(h-1)/2,(w-1)/2],
        rounding='ceil_half_to_preserve_symmetric_odd_width', flags=[],
        status='原图几何中心裁剪，不随亮区移动；需看图确认凸起是否碰框')


def plane_box(ax,shape,box,plane,fixed):
    from matplotlib.patches import Rectangle
    d,h,w=shape;y0,y1,x0,x1=box
    if plane=='dw' and y0 <= fixed < y1: rect=(x0-.5,-.5,x1-x0,d)
    elif plane=='dh' and x0 <= fixed < x1: rect=(y0-.5,-.5,y1-y0,d)
    elif plane=='hw': rect=(x0-.5,y0-.5,x1-x0,y1-y0)
    else: return
    ax.add_patch(Rectangle(rect[:2],*rect[2:],fill=False,edgecolor='#ffd000',linewidth=1.2))


def cropped_cut(crop,box,plane,fixed):
    y0,y1,x0,x1=box
    if plane=='dw': return crop[:,fixed-y0,:]
    if plane=='dh': return crop[:,:,fixed-x0]
    return crop[fixed]


def draw(ax,image,crop,box,plane,fixed,col):
    y0,y1,x0,x1=box
    if col==2:
        value=cropped_cut(crop,box,plane,fixed)
        if plane=='dw': extent=(x0-.5,x1-.5,-.5,image.shape[0]-.5)
        elif plane=='dh': extent=(y0-.5,y1-.5,-.5,image.shape[0]-.5)
        else: extent=(x0-.5,x1-.5,y0-.5,y1-.5)
        ax.imshow(value,cmap='gray',vmin=0,vmax=1,origin='lower',interpolation='nearest',extent=extent)
    else:
        ax.imshow(cut(image,plane,fixed),cmap='gray',vmin=0,vmax=1,origin='lower',interpolation='nearest')
        if col==1: plane_box(ax,image.shape,box,plane,fixed)


def detail(plt,path,r,image,crop):
    y0,y1,x0,x1=r['roi_yx_exclusive']
    cy=(y0+y1-1)//2;cx=(x0+x1-1)//2
    rows=[('dw',cy,f'D-W 固定H={cy}'),('dh',cx,f'D-H 固定W={cx}'),
          ('hw',10,'H-W D=10'),('hw',19,'H-W D=19'),('hw',23,'H-W D=23')]
    fig,axes=plt.subplots(5,3,figsize=(10,14))
    for j,(plane,fixed,name) in enumerate(rows):
        for k in range(3):
            ax=axes[j,k]
            draw(ax,image,crop,r['roi_yx_exclusive'],plane,fixed,k)
            if j==0: ax.set_title(['原图','原图＋裁剪框','仅H/W裁剪后'][k],fontsize=11)
            if k==0: ax.set_ylabel(name,fontsize=9)
            ax.tick_params(labelsize=7)
    diff='清晰' if r['difficulty']=='clear' else '模糊'
    fig.suptitle(f"{r['index']:02d} {diff}{r['label_cn']} | {r['board_key']} 钻孔{r['sequence']}\n"
                 f"{r['shape_dhw']} → {r['crop_shape_dhw']}；D=40不变；框内灰度逐体素相同\n"
                 + (r['crop_description'] if 'crop_description' in r else '原图几何中心，H/W各保留向上取整的一半；不滤波、不填孔、不改灰度' if r.get('crop_mode')=='center-half'
                  else '仅矩形裁剪，不分割、不滤波、不填孔、不改变灰度'),fontsize=12,y=.98)
    fig.text(.5,.015,'同一位置切片；D高索引在上。保留上方全部40层信息，框内背景仍在。黄框仅表示候选裁剪范围，不是表面线。',ha='center',fontsize=8)
    fig.subplots_adjust(left=.13,right=.99,top=.91,bottom=.05,hspace=.25,wspace=.16)
    fig.savefig(path,dpi=110,facecolor='white');plt.close(fig)


def overview(plt,path,items,cn):
    fig,axes=plt.subplots(5,6,figsize=(18,15))
    for j,(r,image,crop) in enumerate(items):
        row,offset=j%5,(j//5)*3
        y0,y1,x0,x1=r['roi_yx_exclusive'];fixed=(y0+y1-1)//2
        for k in range(3):
            ax=axes[row,offset+k];draw(ax,image,crop,r['roi_yx_exclusive'],'dw',fixed,k)
            ax.axis('off')
            if k==0: title=f"{r['index']:02d} {r['board_key']} 钻孔{r['sequence']}\n原图"
            elif k==1: title='\n候选裁剪框'
            else: title=f"\n裁剪后 {crop.shape[1]}×{crop.shape[2]}"
            ax.set_title(title,fontsize=9)
    subtitle=items[0][0].get('crop_description') or ('原图中心向两侧对称裁剪，H/W各保留一半（向上取整）' if items[0][0].get('crop_mode')=='center-half' else '只缩小H/W')
    fig.suptitle(f'{cn}10样本：{subtitle}；D仍40\n同一D-W切片。没有主体Mask，不填孔；框内灰度不变，上方背景不删除。',fontsize=14,y=.98)
    fig.subplots_adjust(left=.015,right=.99,top=.915,bottom=.01,hspace=.25,wspace=.08)
    fig.savefig(path,dpi=105,facecolor='white');plt.close(fig)


HTML=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Head40·20样本仅ROI裁剪</title><style>
*{box-sizing:border-box}body{font-family:"Microsoft YaHei",sans-serif;background:#f5f6f8;margin:22px;color:#222}h1{font-size:24px}p{line-height:1.6}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:16px;margin:14px 0}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}select,button{font:inherit;padding:7px}#sample{max-width:650px}#slice{width:min(65vw,650px)}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}h2{font-size:17px;text-align:center}.box{background:#111;min-height:400px;display:flex;justify-content:center;align-items:center;padding:8px}canvas{max-width:100%;height:auto;image-rendering:pixelated}.note{font-size:13px;color:#555}#error{color:#a11;white-space:pre-wrap}@media(max-width:800px){.panels{grid-template-columns:1fr}}
</style><h1>Head40 · 同样20个样本 · 仅H/W矩形裁剪</h1><p>原图 / 原图＋黄色裁剪框 / 实际裁剪体数据。D方向40层全部保留；框内灰度完全不变。没有去噪、分割、边缘掩膜、填孔、缩放或重新归一化。框内背景仍然存在。</p>
<section class="card"><div class="controls"><select id="category"><option value="all">全部20</option><option value="normal">正常10</option><option value="defective">缺陷10</option></select><select id="sample"></select><select id="plane"><option value="dw">D-W（固定H）</option><option value="dh">D-H（固定W）</option><option value="hw">H-W（固定D）</option></select><select id="scale"><option value="6">6倍</option><option value="10" selected>10倍</option></select></div><p><button id="prev">上一切片</button> <input id="slice" type="range" min="0" step="1"> <button id="next">下一切片</button> <span id="position"></span></p><div id="info"></div></section>
<section class="card panels"><article><h2>原图</h2><div class="box"><canvas id="original"></canvas></div></article><article><h2>原图＋裁剪框</h2><div class="box"><canvas id="box"></canvas></div></article><article><h2>仅H/W裁剪后</h2><div class="box"><canvas id="cropped"></canvas></div></article></section>
<section class="card note">同一原图坐标对照；D高索引在上，H-W中H高索引在上。固定H/W在裁剪范围外时，右侧明确显示“切片位于ROI外”，不是数据全空。页面按同一像素倍率显示，不将裁剪图拉伸到原图尺寸。<p>定位：原图D4～10均值图，在中心区域估计亮区中心，仅用于放置矩形框。框宽至少原H/W的72%，且至少覆盖定位亮区并向两侧各留6体素。不是人工确认的头部边界，需检查偏心、倾斜、凸起是否碰框；不能保证全部样本范围都正确。</p>完整float32体数据保存在NPZ；网页为8位显示，无网络依赖。原始方向与归一化不重复处理；未修改数据集或训练代码，未训练。</section><pre id="error"></pre><script id="data" type="application/json">__DATA__</script><script>
"use strict";const payload=JSON.parse(document.getElementById('data').textContent),samples=payload.samples;
const ui=Object.fromEntries(['category','sample','plane','scale','slice','prev','next','position','info','error'].map(k=>[k,document.getElementById(k)]));let current;const cache=new Map();
function decode(item,key){const id=item.index+':'+key;if(!cache.has(id)){const b=atob(item.arrays[key]),v=Uint8Array.from(b,c=>c.charCodeAt(0)),shape=key==='image'?item.shape_dhw:item.crop_shape_dhw;if(v.length!==shape.reduce((a,b)=>a*b,1))throw Error('数组长度不一致');cache.set(id,v);}return cache.get(id);}
function paint(key){const[D,H,W]=current.shape_dhw,[CD,CH,CW]=current.crop_shape_dhw,[y0,y1,x0,x1]=current.roi_yx_exclusive,plane=ui.plane.value,f=Number(ui.slice.value),isCrop=key==='cropped';const width=plane==='dh'?(isCrop?CH:H):(isCrop?CW:W),height=plane==='hw'?(isCrop?CH:H):D,canvas=document.getElementById(key);canvas.width=width;canvas.height=height;canvas.style.width=width*Number(ui.scale.value)+'px';const ctx=canvas.getContext('2d');const outside=isCrop&&((plane==='dw'&&(f<y0||f>=y1))||(plane==='dh'&&(f<x0||f>=x1)));if(outside){ctx.fillStyle='#eee';ctx.font='3px sans-serif';ctx.fillText('切片位于ROI外',1,10);return;}const v=decode(current,isCrop?'cropped':'image'),pixels=ctx.createImageData(width,height);
for(let y=0;y<height;y++)for(let x=0;x<width;x++){let d,h,w;if(plane==='dw'){d=D-1-y;h=f-(isCrop?y0:0);w=x;}else if(plane==='dh'){d=D-1-y;h=x;w=f-(isCrop?x0:0);}else{d=f;h=height-1-y;w=x;}const at=(d*(isCrop?CH:H)+h)*(isCrop?CW:W)+w,o=(y*width+x)*4,z=v[at];pixels.data[o]=pixels.data[o+1]=pixels.data[o+2]=z;pixels.data[o+3]=255;}ctx.putImageData(pixels,0,0);
if(key==='box'){ctx.strokeStyle='#ffd000';ctx.lineWidth=.4;if(plane==='dw'&&f>=y0&&f<y1)ctx.strokeRect(x0,0,x1-x0,D);if(plane==='dh'&&f>=x0&&f<x1)ctx.strokeRect(y0,0,y1-y0,D);if(plane==='hw')ctx.strokeRect(x0,H-y1,x1-x0,y1-y0);}}
function render(){ui.position.textContent='原图固定 '+({dw:'H',dh:'W',hw:'D'}[ui.plane.value])+'='+ui.slice.value+' / '+ui.slice.max;ui.prev.disabled=Number(ui.slice.value)===0;ui.next.disabled=Number(ui.slice.value)===Number(ui.slice.max);for(const k of ['original','box','cropped'])paint(k);}
function bounds(){const[D,H,W]=current.shape_dhw,[y0,y1,x0,x1]=current.roi_yx_exclusive,p=ui.plane.value;ui.slice.max=({dw:H,dh:W,hw:D}[p])-1;ui.slice.value=p==='dw'?Math.floor((y0+y1-1)/2):p==='dh'?Math.floor((x0+x1-1)/2):19;render();}
function choose(){current=samples.find(r=>String(r.index)===ui.sample.value);ui.info.textContent=current.label_cn+' / '+(current.difficulty==='clear'?'清晰':'模糊')+' / '+current.board_key+' / 钻孔'+current.sequence+'；'+current.shape_dhw.join('×')+' → '+current.crop_shape_dhw.join('×')+'；H范围['+current.roi_yx_exclusive.slice(0,2).join(',')+')，W范围['+current.roi_yx_exclusive.slice(2).join(',')+')；'+(current.flags.join('；')||'范围待看图确认');bounds();}
function fill(){ui.sample.replaceChildren();for(const r of samples)if(ui.category.value==='all'||r.label===ui.category.value){const o=document.createElement('option');o.value=r.index;o.textContent=String(r.index).padStart(2,'0')+' '+r.label_cn+' '+(r.difficulty==='clear'?'清晰':'模糊')+' '+r.board_key+' 钻孔'+r.sequence;ui.sample.append(o);}choose();}
function safe(fn){try{fn();ui.error.textContent='';}catch(e){ui.error.textContent=e.stack||e.message;}}
ui.category.addEventListener('change',()=>safe(fill));ui.sample.addEventListener('change',()=>safe(choose));ui.plane.addEventListener('change',()=>safe(bounds));ui.scale.addEventListener('change',()=>safe(render));ui.slice.addEventListener('input',()=>safe(render));ui.prev.addEventListener('click',()=>safe(()=>{ui.slice.value=Math.max(0,Number(ui.slice.value)-1);render();}));ui.next.addEventListener('click',()=>safe(()=>{ui.slice.value=Math.min(Number(ui.slice.max),Number(ui.slice.value)+1);render();}));safe(fill);
</script></html>'''


def build_viewer(out,index,items):
    rows=[]
    for r,image,crop in items:
        row=dict(r,arrays={})
        for key,a in [('image',image),('cropped',crop)]:
            quant=np.rint(a*255).astype(np.uint8)
            row['arrays'][key]=base64.b64encode(quant.tobytes()).decode('ascii')
        rows.append(row)
    payload=json.dumps(dict(config=index['config'],samples=rows),ensure_ascii=False,allow_nan=False)
    for x,y in (("&","\\u0026"),("<","\\u003c"),(">","\\u003e"),("\u2028","\\u2028"),("\u2029","\\u2029")):
        payload=payload.replace(x,y)
    html=HTML
    if index['config'].get('crop_mode')=='center-55':
        html=html.replace('Head40·20样本仅ROI裁剪','Head40·中心55%裁剪20样本')
        html=html.replace('仅H/W矩形裁剪','原图中心55% H/W裁剪')
        html=html.replace('定位：原图D4～10均值图，在中心区域估计亮区中心，仅用于放置矩形框。框宽至少原H/W的72%，且至少覆盖定位亮区并向两侧各留6体素。不是人工确认的头部边界，需检查偏心、倾斜、凸起是否碰框；不能保证全部样本范围都正确。',
            '裁剪规则：以原始Head40的H/W几何中心为基准，各保留55%，体素数向上取整：29→16、37→21、49→27，D仍40。不在半宽裁剪结果上继续裁，不根据亮区移动框。29→16无法两边去除同样数量体素，两侧分别去除6和7个，中心误差0.5体素。偏心或倾斜凸起仍需看图复核。')
    if index['config'].get('crop_mode')=='center-half':
        html=html.replace('Head40·20样本仅ROI裁剪','Head40·原图中心半宽裁剪20样本')
        html=html.replace('仅H/W矩形裁剪','原图中心半宽H/W裁剪')
        html=html.replace('定位：原图D4～10均值图，在中心区域估计亮区中心，仅用于放置矩形框。框宽至少原H/W的72%，且至少覆盖定位亮区并向两侧各留6体素。不是人工确认的头部边界，需检查偏心、倾斜、凸起是否碰框；不能保证全部样本范围都正确。',
            '裁剪规则：使用原图H/W几何中心，向两侧对称保留原宽的一半，奇数向上取整；29→15、37→19、49→25。不定位亮区、不根据标签移动框，不在上次裁剪图上继续裁。固定中心可能截断偏心或倾斜头部，请检查凸起是否碰框。')
    (out/'查看20个样本.html').write_text(html.replace('__DATA__',payload),encoding='utf8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--crop-mode',choices=('generous','center-half'),default='generous')
    parser.add_argument('--resume-generated',action='store_true',help='Resume only this task\'s unfinished center-half preview')
    args=parser.parse_args()
    out=args.output.resolve();out.relative_to(ROOT/'查找结果')
    if args.resume_generated and (args.crop_mode!='center-half' or out.name!='Head40_原图中心半宽裁剪_20样本预览_20261008_v1'):
        raise ValueError('Resume restricted to this task\'s center-half output')
    if out.exists() and not args.resume_generated: raise FileExistsError('Refusing overwrite: '+str(out))
    index0=json.loads((OLD/'preview_index.json').read_text(encoding='utf8'))
    if len(index0['samples'])!=20: raise ValueError('Expected same original20 samples')
    old_folders=[p for p in (ROOT/'查找结果').iterdir() if p.is_dir() and '20样本预览' in p.name and p.resolve()!=out]
    tracked=[p for folder in old_folders for p in folder.rglob('*') if p.is_file()]
    hashes={str(p):sha(p) for p in tracked}
    out.mkdir(parents=True,exist_ok=args.resume_generated)
    for folder in ('单样本对照图','裁剪后体数据'): (out/folder).mkdir(exist_ok=args.resume_generated)
    plt=plotting();items=[];records=[]
    for original in index0['samples']:
        with np.load(OLD/original['array_file'],allow_pickle=False) as a: image=a['image'].copy()
        if image.shape!=tuple(original['shape_dhw']) or image.dtype!=np.float32 or not np.isfinite(image).all():
            raise ValueError('Unexpected source array')
        box,stats=(center_half_box(image) if args.crop_mode=='center-half' else position_box(image));y0,y1,x0,x1=box
        cropped=image[:,y0:y1,x0:x1].copy()
        if cropped.shape[0]!=40 or not np.array_equal(cropped,image[:,y0:y1,x0:x1]):
            raise AssertionError('Not an exact rectangular crop')
        r={k:v for k,v in original.items() if k not in ('edge_ratio','mask_ratio','empty_mask','array_file')}
        r.update(stats);r.update(crop_mode=args.crop_mode,roi_yx_exclusive=box,crop_shape_dhw=list(cropped.shape),
            kept_volume_ratio=float(cropped.size/image.size),
            array_file='裁剪后体数据/'+Path(original['array_file']).name,
            source_npz_sha256=hashes[str(OLD/original['array_file'])],grayscale_bit_identical=True)
        np.savez_compressed(out/r['array_file'],image=image,cropped=cropped,roi_yx_exclusive=np.array(box,dtype=np.int16))
        detail(plt,out/r['png_file'],r,image,cropped)
        records.append(r);items.append((r,image,cropped))
        print(f"Cropped {r['index']:02d}/20: {list(image.shape)} -> {list(cropped.shape)} box={box}",flush=True)
    for label,cn in [('normal','正常'),('defective','缺陷')]:
        overview(plt,out/f'{cn}10样本_总览.png',[x for x in items if x[0]['label']==label],cn)
    config=dict(version=('roi_center_half_preview_20261008_v1' if args.crop_mode=='center-half' else 'roi_only_preview_20261008_v1'),crop_mode=args.crop_mode,preview_only=True,source_preview=str(OLD),
        input='original_npz.image_direction_unified_normalized_once',depth='all40_slices',
        localization=('original_geometric_center; no_gray_statistic' if args.crop_mode=='center-half' else 'unfiltered_D4..10_mean_in_central_disk_radius0.3_minHW; bright>=0.8*peak; weighted_center'),
        crop_width=('ceil(original_width/2); start=(original_width-crop_width)//2' if args.crop_mode=='center-half' else 'max(ceil(0.72*original_width),lower_bright_extent+12)_clipped_to_original'),
        denoising=False,segmentation=False,edge_mask=False,hole_filling=False,resize=False,
        intensity_transform=False,dataset_modified=False,training_run=False,
        caveat=('固定原图中心半宽，偏心/倾斜凸起可能碰框；框内上方背景不删除' if args.crop_mode=='center-half' else '宽松横向定位框，未人工标定全部凸起；框内上方背景不删除'))
    index=dict(config=config,samples=records)
    (out/'preview_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
    (out/'处理参数.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'20样本清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader()
        for r in records: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    build_viewer(out,index,items)
    for path,digest in hashes.items():
        if sha(Path(path))!=digest: raise AssertionError('Old preview changed: '+path)
    checks=dict(sample_count=20,same_ids_and_order=[r['sample_id'] for r in records]==[r['sample_id'] for r in index0['samples']],
        all_depths40=True,all_crops_bit_identical=True,old_versions_unchanged=True,
        old_artifact_count=len(hashes),training_run=False)
    (out/'必要检查.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    notes=['# 20样本：只做宽松H/W裁剪','','同样10正常+10缺陷，样本顺序不变，旧全部结果不覆盖。','',
        '这版只实际切取矩形体数据：image[:,y0:y1,x0:x1]。D的40层全部保留，原图灰度逐体素一致，没有滤波、填孔、主体Mask、边缘乘法、重采样、方向二次翻转或二次归一化。','',
        '矩形定位从原图D4～10的亮区估计中心，宽度至少原H/W的72%，并给定位亮区各侧6体素余量。统计亮区仅为了确定框位置，不用于删除框内任何体素。','',
        '候选框需要人工看图确认，尤其偏心、倾斜和较大凸起。框内上方背景仍然存在，这不是精准去噪或头部外轮廓提取。','',
        '先打开查看20个样本.html，三列对照所有方向与切片；也有正常、缺陷总览和20张详细图。页面用相同像素倍率显示裁剪前后，图变窄来自真实H/W减小，不是把钻孔压缩。','',
        '仅预览，未改数据集、清单、加载器、模型和训练代码。','']
    if args.crop_mode=='center-half':
        notes[0]='# 20样本：原图中心向两侧对称保留半宽'
        notes[6]='从原始Head40的几何中心裁剪，H/W各保留一半。奇数宽度向上取整：29→15，37→19，49→25，保证两侧去除相同数量体素。不基于灰度定位，不随钻孔偏心移动，不读取上一版裁剪后的体数据。'
        notes[8]='这是用户指定的固定中心半宽范围，不是人工确认的完整头部边界。偏心或倾斜时可能截断头部/凸起，需要看图复核；不得自动移动框冒充固定中心结果。框内上方背景仍保留。'
    for r in records: notes.append(f"- {r['index']:02d} {r['board_key']} 钻孔{r['sequence']}：{r['shape_dhw']} → {r['crop_shape_dhw']}；框{r['roi_yx_exclusive']}。")
    (out/'预览说明.md').write_text('\n'.join(notes)+'\n',encoding='utf8')
    print('DONE '+str(out),flush=True);print(json.dumps(checks,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
