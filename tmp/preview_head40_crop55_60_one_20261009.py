"""One actual sample: original, center55, center60; two orthogonal D views."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from head_connected_preview_20_20261008 import ROOT, plotting, sha

SOURCE = ROOT/'查找结果/Head40_原图中心55百分比裁剪_40样本预览_20261008_v1/新增20个'
OUTPUT = ROOT/'查找结果/Head40_55与60裁剪_单样本双视图_20261009_v1'


def crop(image, percent):
    d,h,w=image.shape
    nh,nw=(percent*h+99)//100,(percent*w+99)//100
    y0,x0=(h-nh)//2,(w-nw)//2
    box=[y0,y0+nh,x0,x0+nw]
    return image[:,y0:y0+nh,x0:x0+nw].copy(),box


def make_picture(plt,path,record,image,crop55,box55,crop60,box60,views):
    from matplotlib.patches import Rectangle
    # Identical pixels-per-voxel in every panel: crops are NOT stretched wider.
    fig_width,fig_height=17.,2.6+3.05*len(views)
    fig=plt.figure(figsize=(fig_width,fig_height))
    scale=.050
    column_centers=[1.7,5.05,8.4,11.75,15.1]
    titles=['原图','原图＋55%框','55%裁剪后','原图＋60%框','60%裁剪后']
    for row,(plane,fixed,name) in enumerate(views):
        bottom=fig_height-2.1-2.-row*3.05
        original=image[:,fixed,:] if plane=='dw' else image[:,:,fixed]
        fixed_axis='H' if plane=='dw' else 'W'
        fig.text(.025,(bottom+2.38)/fig_height,f'{name}（固定{fixed_axis}={fixed}）',fontsize=13,weight='bold')
        for col,center in enumerate(column_centers):
            is_crop=col in (2,4)
            selected,box=(crop55,box55) if col in (1,2) else (crop60,box60)
            y0,y1,x0,x1=box
            lo,hi=(x0,x1) if plane=='dw' else (y0,y1)
            if is_crop:
                values=selected[:,fixed-y0,:] if plane=='dw' else selected[:,:,fixed-x0]
                extent=(lo-.5,hi-.5,-.5,39.5)
            else:
                values=original
                extent=(-.5,original.shape[1]-.5,-.5,39.5)
            panel_width=values.shape[1]*scale
            ax=fig.add_axes([(center-panel_width/2)/fig_width,bottom/fig_height,
                             panel_width/fig_width,40*scale/fig_height])
            ax.imshow(values,cmap='gray',vmin=0,vmax=1,origin='lower',
                      interpolation='nearest',aspect='equal',extent=extent)
            if col in (1,3):
                ax.add_patch(Rectangle((lo-.5,-.5),hi-lo,40,fill=False,
                                       edgecolor='#ffd000',linewidth=1.4))
            ax.set_title(titles[col]+(f'\n40×{values.shape[1]}' if is_crop else ''),fontsize=11,pad=10)
            ax.set_xlabel('W' if plane=='dw' else 'H',fontsize=9)
            ax.set_ylabel('D',fontsize=9)
            ax.tick_params(labelsize=8)
    fig.suptitle(f"清晰缺陷 · {record['board_key']} · 钻孔{record['sequence']}\n"
        '原始40×49×49 → 55%：40×27×27；60%：40×30×30',fontsize=16,y=.97)
    fig.text(.5,.09,'两个视图均取原图几何中心切片；D高索引在上。黄框只表示裁剪范围，不是钻孔边界。',ha='center',fontsize=10)
    fig.text(.5,.055,'所有面板按相同体素倍率显示；仅中心裁H/W，不去噪、不缩放、不改变灰度，D的40层全部保留。',ha='center',fontsize=10)
    fig.savefig(path,dpi=140,facecolor='white');plt.close(fig)


def main():
    if OUTPUT.exists(): raise FileExistsError('Refusing overwrite: '+str(OUTPUT))
    index=json.loads((SOURCE/'preview_index.json').read_text(encoding='utf8'))
    record=next(r for r in index['samples'] if r['index']==38)
    source_file=SOURCE/record['array_file']
    digest=sha(source_file)
    with np.load(source_file,allow_pickle=False) as arrays:
        image=arrays['image'].copy();previous55=arrays['cropped'].copy()
    if image.shape!=(40,49,49) or image.dtype!=np.float32 or not np.isfinite(image).all():
        raise ValueError('Unexpected cached original')
    image55,box55=crop(image,55);image60,box60=crop(image,60)
    assert np.array_equal(image55,previous55)
    assert image55.shape==(40,27,27) and image60.shape==(40,30,30)
    for a,box in [(image55,box55),(image60,box60)]:
        y0,y1,x0,x1=box
        assert np.array_equal(a,image[:,y0:y1,x0:x1])
    OUTPUT.mkdir(parents=True)
    plt=plotting()
    views=[('dw',24,'正面 D-W'),('dh',24,'侧面 D-H')]
    make_picture(plt,OUTPUT/'正面侧面_55与60裁剪对照.png',record,image,image55,box55,image60,box60,views)
    for view,filename in zip(views,['正面_DW_对照.png','侧面_DH_对照.png']):
        make_picture(plt,OUTPUT/filename,record,image,image55,box55,image60,box60,[view])
    np.savez_compressed(OUTPUT/'样本体数据.npz',image=image,crop55=image55,crop60=image60)
    info=dict(sample_id=record['sample_id'],board=record['board'],board_key=record['board_key'],
        sequence=record['sequence'],label='defective',difficulty='clear',original_shape_dhw=list(image.shape),
        source_npz=str(source_file),source_npz_sha256=digest,raw_path=record['raw_path'],raw_sha256=record['raw_sha256'],
        direction_flipped=record['direction_flipped'],normalization='existing original Head40 P1/P99, not repeated',
        front='D-W at original H=24',side='D-H at original W=24',crop55_box_yx_exclusive=box55,
        crop60_box_yx_exclusive=box60,rounding='ceil(percent*n/100);start=floor((n-kept)/2)',
        grayscale_bit_identical=True,previous55_match=True,source_unchanged=sha(source_file)==digest,
        denoising=False,resize=False,full20percent_sampling_started=False)
    if not info['source_unchanged']: raise AssertionError('Source modified')
    (OUTPUT/'样本与处理说明.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(output=str(OUTPUT),original_shape=list(image.shape),
        crop55_shape=list(image55.shape),crop60_shape=list(image60.shape),source_unchanged=True,
        previous55_match=True,full_sampling_started=False),ensure_ascii=False),flush=True)


if __name__=='__main__': main()
