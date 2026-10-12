"""Render the four user-highlighted examples, without changing old artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from head_connected_preview_20_20261008 import ROOT, plotting, cut, body_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.relative_to(ROOT/'查找结果')
    target = out/'重点4样本_原图旧版新版对照.png'
    if target.exists():
        raise FileExistsError(target)
    index = json.loads((out/'preview_index.json').read_text(encoding='utf8'))
    old = ROOT/'查找结果/Head40_头部连通约束_20样本预览_20261008_v1'
    plt = plotting()
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    for j, sample_index in enumerate((4, 9, 10, 20)):
        r = index['samples'][sample_index-1]
        with np.load(out/r['array_file'], allow_pickle=False) as saved:
            a = {k:saved[k] for k in saved.files}
        with np.load(old/r['array_file'], allow_pickle=False) as saved:
            previous = saved['processed']
        fixed = a['image'].shape[1]//2
        values = [a['image'],previous,a['body_mask'],a['surface_weight'],a['processed']]
        for k, value in enumerate(values):
            ax = axes[j,k]
            if k == 2:
                body_overlay(ax,a,r,'dw',fixed)
            else:
                ax.imshow(cut(value,'dw',fixed), cmap='gray', vmin=0, vmax=1,
                          origin='lower', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_title(['原图','旧v1处理后','新v2b主体叠加','新渐变表面带','新v2b处理后'][k],fontsize=11)
            if k == 0:
                ax.set_ylabel(f"{sample_index:02d} {r['label_cn']}\n{r['board_key']}\n钻孔{r['sequence']}",fontsize=10)
    fig.suptitle('重点4样本：同一D-W中心切片对照\n减少上方误连，不强行拼接两侧孔壁；新结果仍非形态真值',fontsize=15,y=.98)
    fig.text(.5,.017,'全部灰度固定0–1。新表面带为渐变权重，所以灰度可能变暗；孔腔未填平，缺陷是否保住还需原图多切片检查。',ha='center',fontsize=9)
    fig.subplots_adjust(left=.10,right=.99,top=.90,bottom=.05,hspace=.16,wspace=.08)
    fig.savefig(target,dpi=110,facecolor='white')
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
