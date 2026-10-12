"""Re-crop the same two batches from cached ORIGINAL images, never half crops."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from head_roi_only_preview_20_20261008 import ROOT, plotting, detail, overview, build_viewer, sha

DEST = ROOT/'查找结果/Head40_原图中心55百分比裁剪_40样本预览_20261008_v1'
SOURCES = [
    ('第一批20个', ROOT/'查找结果/Head40_原图中心半宽裁剪_20样本预览_20261008_v1'),
    ('新增20个', ROOT/'查找结果/Head40_原图中心半宽裁剪_新增20样本预览_20261008_v1'),
]


def main():
    if DEST.exists():
        raise FileExistsError('Refusing overwrite: '+str(DEST))
    indexes = [(name, source, json.loads((source/'preview_index.json').read_text(encoding='utf8')))
               for name, source in SOURCES]
    if any(len(index['samples']) != 20 for _, _, index in indexes):
        raise ValueError('Expected two batches of twenty')
    old_hashes = {str(p):sha(p) for _, source in SOURCES for p in source.rglob('*') if p.is_file()}
    DEST.mkdir(parents=True)
    plt = plotting()
    total = []
    for name, source, original in indexes:
        out = DEST/name
        out.mkdir()
        for child in ('裁剪后体数据', '单样本对照图'):
            (out/child).mkdir()
        records, items = [], []
        for old in original['samples']:
            with np.load(source/old['array_file'], allow_pickle=False) as arrays:
                image = arrays['image'].copy()
            if image.shape != tuple(old['shape_dhw']) or image.dtype != np.float32 or not np.isfinite(image).all():
                raise ValueError('Unexpected original array')
            d,h,w = image.shape
            # Integer arithmetic for exact ceil(0.55*n); no banker rounding.
            nh,nw = (55*h+99)//100, (55*w+99)//100
            y0,x0 = (h-nh)//2, (w-nw)//2
            box = [y0,y0+nh,x0,x0+nw]
            crop = image[:,y0:y0+nh,x0:x0+nw].copy()
            if d != 40 or not np.array_equal(crop, image[:,y0:y0+nh,x0:x0+nw]):
                raise AssertionError('Not an exact crop')
            r = dict(old)
            r.pop('source_npz_sha256', None)
            r.update(crop_mode='center-55', roi_yx_exclusive=box, crop_shape_dhw=list(crop.shape),
                     crop_description='原图中心裁剪，H/W各保留55%（向上取整）',
                     center_yx=[(h-1)/2,(w-1)/2], actual_crop_center_yx=[(box[0]+box[1]-1)/2,(box[2]+box[3]-1)/2],
                     rounding='ceil(55*n/100);start=floor((n-kept)/2)',
                     source_original_npz=str(source/old['array_file']),
                     source_original_npz_sha256=old_hashes[str(source/old['array_file'])],
                     kept_volume_ratio=float(crop.size/image.size), grayscale_bit_identical=True,
                     status='固定原图中心55%裁剪；需看图确认凸起是否碰框',
                     flags=['奇数原宽29保留16：中心误差0.5体素'] if h==29 else [])
            np.savez_compressed(out/r['array_file'],image=image,cropped=crop,roi_yx_exclusive=np.asarray(box,dtype=np.int16))
            detail(plt,out/r['png_file'],r,image,crop)
            items.append((r,image,crop));records.append(r);total.append(r)
            print(f"Rendered {r['index']:02d}/40: {list(image.shape)} -> {list(crop.shape)}",flush=True)
        config = dict(version='center55_preview_20261008_v1', crop_mode='center-55', fraction=.55,
                      crop_width='ceil(0.55*original_width)', positioning='start=floor((original_width-kept)/2)',
                      depth=40, source_preview=str(source), source_array='image, not cropped',
                      denoising=False,segmentation=False,resize=False,intensity_transform=False,
                      dataset_modified=False,training_run=False)
        index = dict(config=config,samples=records)
        for filename,value in [('preview_index.json',index),('处理参数.json',config)]:
            (out/filename).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
        with (out/'20样本清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader()
            for r in records:
                writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
        for label,cn in [('normal','正常'),('defective','缺陷')]:
            overview(plt,out/f'{cn}10样本_总览.png',[item for item in items if item[0]['label']==label],cn)
        build_viewer(out,index,items)
    for path,digest in old_hashes.items():
        if sha(Path(path)) != digest:
            raise AssertionError('Previous preview changed: '+path)
    checks=dict(sample_count=len(total),unique_sample_ids=len({r['sample_id'] for r in total}),
                all_depths40=True,all_crops_bit_identical=True,old_results_unchanged=True,training_run=False)
    (DEST/'必要检查.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    (DEST/'预览说明.md').write_text(
        '# 原图中心55%裁剪：两批共40个样本\n\n'
        '与之前两批使用相同样本，从缓存的原始Head40重新裁剪，不读取半宽 cropped 数组。\n\n'
        'H/W各保留原尺寸的55%，向上取整：29→16、37→21、49→27；深度仍40。'
        '起点 floor((原宽-新宽)/2)。29→16存在不可避免的半体素中心误差，两边去除6和7个体素。\n\n'
        '无去噪、分割、填孔、缩放、二次归一化；框内灰度逐体素不变。原数据集、加载器、训练代码和旧预览均未修改。\n\n'
        '分别打开第一批20个、新增20个目录里的查看20个样本.html；支持所有切片和三个方向。'
        '固定中心裁剪仍需检查偏心或倾斜凸起是否碰框。\n',encoding='utf8')
    print('DONE '+str(DEST),flush=True)
    print(json.dumps(checks,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
