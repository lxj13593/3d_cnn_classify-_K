"""A second distinct set of 20, loaded with the unchanged Head40 loader.

Extract in the existing torch environment, render separately without torch.
No training or edits to data/old previews; same geometric center-half crop.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from head_roi_only_preview_20_20261008 import (
    ROOT, OLD, center_half_box, detail, overview, build_viewer, plotting, sha,
)

PREVIOUS = ROOT/'查找结果/Head40_原图中心半宽裁剪_20样本预览_20261008_v1'
MANIFEST = ROOT/'datasets/resnet18_head40_toml_direction_unified_multiboard_5fold_clear_fuzzy_20260924/fold_0/train.csv'
LABEL_CN = {'normal':'正常','defective':'缺陷'}
sys.path.insert(0,str(ROOT))


def select(dataset, previous):
    old_ids={r['sample_id'] for r in previous['samples']}
    old_hashes={r['raw_sha256'].lower() for r in previous['samples']}
    used_ids=set(old_ids);used_hashes=set(old_hashes)
    board_counts=Counter(r['board'] for r in previous['samples'])
    class_boards=Counter()
    selected={'normal':[],'defective':[]}
    slots=[('clear',29),('fuzzy',29),('clear',29),
           ('clear',37),('fuzzy',37),('clear',37),('fuzzy',37),
           ('clear',49),('fuzzy',49),('clear',49)]
    for difficulty,side in slots:
        for label in ('normal','defective'):
            candidates=[i for i,s in enumerate(dataset.samples)
                if s.label==label and s.difficulty==difficulty and s.shape_dhw==(40,side,side)
                and s.sample_id not in used_ids and s.raw_sha256.lower() not in used_hashes]
            if not candidates:
                raise ValueError(f'No distinct candidate: {label}/{difficulty}/{side}')
            def rank(i):
                s=dataset.samples[i]
                return (board_counts[s.board],class_boards[label,s.board],
                        hashlib.sha256(('20261008_additional20|'+s.sample_id).encode()).hexdigest())
            i=min(candidates,key=rank);s=dataset.samples[i]
            selected[label].append(i);used_ids.add(s.sample_id);used_hashes.add(s.raw_sha256.lower())
            board_counts[s.board]+=1;class_boards[label,s.board]+=1
    return selected['normal']+selected['defective']


def extract(out,dataset_root):
    import torch
    from data_operate.data_load_resnet18_head40_toml_direction_unified_multiboard import (
        MultiBoardRawDataset,get_normalization_params,get_direction_standardization_params,
    )
    from canny_preview_20_20261005 import board_key
    if out.exists(): raise FileExistsError('Refusing to overwrite: '+str(out))
    previous=json.loads((PREVIOUS/'preview_index.json').read_text(encoding='utf8'))
    if len(previous['samples'])!=20: raise ValueError('Expected first20 sample exclusion list')
    torch.set_num_threads(1)
    dataset=MultiBoardRawDataset(MANIFEST,augment=False,dataset_root=dataset_root,verify_files=False)
    if any(s.validation_fold==0 for s in dataset.samples): raise ValueError('Unexpected fold0 validation sample')
    chosen=select(dataset,previous)
    # Verify the twenty chosen sources before creating any new result folder.
    source_hashes={}
    for i in chosen:
        s=dataset.samples[i]
        if s.raw_path.stat().st_size!=int(np.prod(s.shape_dhw)): raise ValueError('RAW byte count mismatch')
        digest=sha(s.raw_path)
        if digest!=s.raw_sha256.lower(): raise ValueError('RAW hash does not match original split: '+str(s.raw_path))
        source_hashes[str(s.raw_path)]=digest
    old_preview_hashes={str(p):sha(p) for p in PREVIOUS.rglob('*') if p.is_file()}
    out.mkdir(parents=True,exist_ok=False)
    for name in ('单样本对照图','裁剪后体数据'): (out/name).mkdir()
    records=[]
    for number,i in enumerate(chosen,start=21):
        s=dataset.samples[i]
        image=dataset[i]['volume'][0].numpy().copy()
        if image.dtype!=np.float32 or image.shape!=s.shape_dhw or not np.isfinite(image).all():
            raise ValueError('Unexpected loaded image')
        box,stats=center_half_box(image);y0,y1,x0,x1=box
        cropped=image[:,y0:y1,x0:x1].copy()
        if not np.array_equal(cropped,image[:,y0:y1,x0:x1]) or cropped.shape[0]!=40:
            raise AssertionError('Unexpected pixel transform')
        prefix=f'{number:02d}_{LABEL_CN[s.label]}_{board_key(s.board)}_钻孔{s.sequence}'
        record=dict(index=number,sample_id=s.sample_id,label=s.label,label_cn=LABEL_CN[s.label],
            difficulty=s.difficulty,board=s.board,board_key=board_key(s.board),sequence=s.sequence,
            shape_dhw=list(s.shape_dhw),crop_shape_dhw=list(cropped.shape),
            raw_path=str(s.raw_path),raw_sha256=s.raw_sha256,direction_flipped=s.direction_flip_required,
            validation_fold=s.validation_fold,crop_mode='center-half',roi_yx_exclusive=box,
            kept_volume_ratio=float(cropped.size/image.size),grayscale_bit_identical=True,
            png_file=f'单样本对照图/{prefix}.png',array_file=f'裁剪后体数据/{prefix}.npz',**stats)
        np.savez_compressed(out/record['array_file'],image=image,cropped=cropped,roi_yx_exclusive=np.array(box,dtype=np.int16))
        records.append(record)
        print(f"Extracted {number}: {s.label}/{s.difficulty} {record['board_key']} {s.sequence} {list(s.shape_dhw)} -> {list(cropped.shape)}",flush=True)
    config=dict(version='center_half_additional20_20261008_v1',crop_mode='center-half',preview_only=True,
        source_manifest=str(MANIFEST),source_manifest_sha256=sha(MANIFEST),dataset_root=str(dataset.dataset_root),
        exclusion_preview=str(PREVIOUS),selection='per_class_3_at29_4_at37_3_at49;6clear4fuzzy;prioritize_board_diversity_then_sha256',
        normalization=get_normalization_params(),direction=get_direction_standardization_params(),augment=False,
        crop_width='ceil(original_width/2);fixed_original_geometric_center',depth=40,
        denoising=False,segmentation=False,edge_mask=False,resize=False,training_run=False,
        caveat='固定中心可能截断偏心/倾斜凸起；中心切片未必经过凸起，需看图确认')
    (out/'preview_index.json').write_text(json.dumps(dict(config=config,samples=records),ensure_ascii=False,indent=2),encoding='utf8')
    (out/'处理参数.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    with (out/'新增20样本清单.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader()
        for r in records: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
    for path,digest in source_hashes.items():
        if sha(Path(path))!=digest: raise AssertionError('Source RAW changed')
    for path,digest in old_preview_hashes.items():
        if sha(Path(path))!=digest: raise AssertionError('Previous preview changed')
    old_ids={r['sample_id'] for r in previous['samples']}
    old_hashes={r['raw_sha256'].lower() for r in previous['samples']}
    checks=dict(sample_count=20,normal_count=10,defective_count=10,
        original_width_counts=dict(Counter(str(r['shape_dhw'][2]) for r in records)),
        per_class_width_counts={label:dict(Counter(str(r['shape_dhw'][2]) for r in records if r['label']==label)) for label in LABEL_CN},
        difficulty_counts=dict(Counter(r['difficulty'] for r in records)),
        distinct_boards=len({r['board'] for r in records}),
        sample_id_overlap_with_previous=len(old_ids & {r['sample_id'] for r in records}),
        raw_hash_overlap_with_previous=len(old_hashes & {r['raw_sha256'].lower() for r in records}),
        new_ids_unique=len({r['sample_id'] for r in records})==20,
        new_content_hashes_unique=len({r['raw_sha256'].lower() for r in records})==20,
        all_depths40=True,all_crops_bit_identical=True,source_raws_unchanged=True,previous_preview_unchanged=True,
        old_preview_hashes=old_preview_hashes,training_run=False)
    (out/'必要检查.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
    notes=['# 新增20个：原图中心半宽裁剪','','编号21～40，与上一批01～20按样本ID和RAW内容哈希排除重复。',
        '', '正常10、缺陷10。每类29尺寸3个、37尺寸4个、49尺寸3个；每类清晰6个、模糊4个。',
        '', '复用原Head40加载器：原方向统一与单样本P1/P99归一化，无增强。然后以原图几何中心对称裁H/W，29→15、37→19、49→25，D40不变。',
        '', '只实际裁矩形，框内灰度逐体素完全相同。没有滤波、分割、填孔、Mask、缩放或二次归一化。原始数据、清单、加载器及旧预览均未修改，未训练。',
        '', '固定中心可能截断偏心头部或较大凸起；框内上方背景保留。请看黄色框以及三方向相邻切片，不以一张中心切片判断凸起丢失。','']
    for r in records: notes.append(f"- {r['index']:02d} {r['label_cn']} {r['board_key']} 钻孔{r['sequence']}：{r['shape_dhw']} → {r['crop_shape_dhw']}。")
    (out/'预览说明.md').write_text('\n'.join(notes)+'\n',encoding='utf8')
    print('EXTRACT DONE',flush=True)
    print(json.dumps({k:v for k,v in checks.items() if k!='old_preview_hashes'},ensure_ascii=False),flush=True)


def render(out):
    index=json.loads((out/'preview_index.json').read_text(encoding='utf8'))
    if len(index['samples'])!=20 or index['config']['version']!='center_half_additional20_20261008_v1':
        raise ValueError('Not the expected new preview')
    plt=plotting();items=[]
    for r in index['samples']:
        target=out/r['png_file']
        if target.exists(): raise FileExistsError('Refusing to overwrite '+str(target))
        with np.load(out/r['array_file'],allow_pickle=False) as a: image,crop=a['image'],a['cropped']
        detail(plt,target,r,image,crop);items.append((r,image,crop))
        print(f"Rendered {r['index']}/40",flush=True)
    for label,cn in [('normal','正常'),('defective','缺陷')]:
        overview(plt,out/f'{cn}10样本_总览.png',[x for x in items if x[0]['label']==label],cn)
    build_viewer(out,index,items)
    checks=json.loads((out/'必要检查.json').read_text(encoding='utf8'))
    for path,digest in checks['old_preview_hashes'].items():
        if sha(Path(path))!=digest: raise AssertionError('Previous preview changed during render')
    print('RENDER DONE '+str(out),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('extract','render'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=Path('H:/多板数据集/multiboard_head40_toml'))
    args=parser.parse_args();out=args.output.resolve();out.relative_to(ROOT/'查找结果')
    if args.phase=='extract': extract(out,args.dataset_root)
    else: render(out)


if __name__=='__main__': main()
