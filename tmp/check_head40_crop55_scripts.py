"""Necessary smoke checks only; no training and no persistent checkpoint."""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    files = [
        'data_operate/head40_center_crop55.py',
        'model/resnet18_3d_head40_stem733_crop55.py',
        'train_val/main_resnet18_head40_stem733_crop55_ce_5fold.py',
        'test/test_resnet18_head40_stem733_crop55_ce_3boards.py',
    ]
    for relative in files:
        compile((ROOT/relative).read_text(encoding='utf8'),relative,'exec')
    print('Syntax: all four files OK',flush=True)

    import numpy as np
    import torch
    from data_operate import head40_center_crop55 as crop
    from data_operate import data_load_resnet18_head40_toml_direction_unified_multiboard as original
    from model.resnet18_3d_head40_stem733_crop55 import resnet18_3d, architecture_metadata
    torch.set_num_threads(1)
    train = importlib.import_module('train_val.main_resnet18_head40_stem733_crop55_ce_5fold')
    # Python ships a package named test; load the project test entry by path.
    spec = importlib.util.spec_from_file_location('crop55_locked_test', ROOT/files[3])
    test = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = test
    spec.loader.exec_module(test)
    config = train.load_split_config(train.DEFAULT_SPLIT_ROOT)
    assert config['sample_count'] == 4364 and config['locked_test_counts']['sample_count'] == 778
    assert train.EXPERIMENT_NAME == test.EXPERIMENT_NAME
    assert train.CHECKPOINT_FILES == test.CHECKPOINT_FILES
    assert train.DEFAULT_MODEL_ROOT == test.DEFAULT_MODEL_ROOT
    assert train.DEFAULT_SPLIT_ROOT == test.DEFAULT_SPLIT_ROOT
    assert len(str(train.DEFAULT_MODEL_ROOT/'fold_0'/train.CHECKPOINT_FILES['best_loss'])) < 230

    train_root = Path('H:/多板数据集/multiboard_head40_toml')
    manifest = train.DEFAULT_SPLIT_ROOT/'fold_0/train.csv'
    dataset = crop.CenterCrop55Dataset(manifest,augment=False,dataset_root=train_root,verify_files=False)
    reference = original.MultiBoardRawDataset(manifest,augment=False,dataset_root=train_root,verify_files=False)
    preview_root = ROOT/'查找结果/Head40_原图中心55百分比裁剪_40样本预览_20261008_v1/新增20个'
    preview = json.loads((preview_root/'preview_index.json').read_text(encoding='utf8'))
    by_id = {s.sample_id:i for i,s in enumerate(dataset.samples)}
    selected = [preview['samples'][i] for i in (0,3,7)]
    model = resnet18_3d().eval()
    parameter_count = sum(p.numel() for p in model.parameters())
    for row in selected:
        i = by_id[row['sample_id']]
        item = dataset[i]
        y0,y1,x0,x1 = crop.crop_box(dataset.samples[i].shape_dhw)
        assert torch.equal(item['volume'],reference[i]['volume'][:,:,y0:y1,x0:x1])
        with np.load(preview_root/row['array_file'],allow_pickle=False) as arrays:
            assert np.array_equal(item['volume'][0].numpy(),arrays['cropped'])
        assert item['input_shape_dhw'] == 'x'.join(str(v) for v in item['volume'].shape[1:])
        with torch.no_grad():
            logits = model(item['volume'].unsqueeze(0))
        assert tuple(logits.shape) == (1,2) and torch.isfinite(logits).all()
        print(f"Real RAW/preview match: {row['shape_dhw']} -> {list(item['volume'].shape[1:])}; output [1,2]",flush=True)

    train_loader,val_loader = crop.get_fold_loaders(train.DEFAULT_SPLIT_ROOT,0,dataset_root=train_root,verify_files=False)
    train.dataset_integrity_check(train_loader,val_loader,0)
    batch = next(iter(train_loader))
    assert tuple(batch['volume'].shape[-3:]) in set(dataset.shapes)
    assert len(set(batch['input_shape_dhw'])) == 1
    test_loader = crop.make_loader(train.DEFAULT_SPLIT_ROOT/'locked_test_manifest.csv',
        dataset_root=Path('H:/多板数据集/test_3boards_head40_toml'),batch_size=4,num_workers=0,
        augment=False,seed=42,verify_files=False,expected_dataset_name=original.TEST_DATASET_NAME)
    assert len(test_loader.dataset) == 778
    flipped = next(i for i,s in enumerate(test_loader.dataset.samples) if s.direction_flip_required)
    test_item = test_loader.dataset[flipped]
    test_original = original.MultiBoardRawDataset(train.DEFAULT_SPLIT_ROOT/'locked_test_manifest.csv',
        augment=False,dataset_root=Path('H:/多板数据集/test_3boards_head40_toml'),verify_files=False,
        expected_dataset_name=original.TEST_DATASET_NAME)
    y0,y1,x0,x1 = crop.crop_box(test_loader.dataset.samples[flipped].shape_dhw)
    assert test_item['direction_flipped'] and torch.equal(test_item['volume'],test_original[flipped]['volume'][:,:,y0:y1,x0:x1])
    print('Same-shape batches and locked-test D flip/crop: OK',flush=True)

    # Exercise the real payload/save/load functions without updating any weights.
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-7)
    scaler = torch.amp.GradScaler('cpu',enabled=False)
    evaluation = train.EvaluationResult(metrics={},predictions=[])
    payload = train.checkpoint_payload(model,optimizer,scheduler,scaler,0,42,0,'best_loss',
        evaluation,{}, {}, {}, {'input_preprocessing':crop.preprocessing_metadata()},config)
    assert payload['input_preprocessing'] == architecture_metadata()['input_preprocessing']
    with tempfile.TemporaryDirectory(prefix='head40_crop55_check_') as temporary:
        path = Path(temporary)/'best_loss_stem733_crop55_ce.pth'
        train.atomic_torch_save(payload,path)
        loaded,saved = test.load_checkpoint_model(path,torch.device('cpu'))
        assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
        saved['input_preprocessing']['fraction'] = .5
        train.atomic_torch_save(saved,path)
        try:
            test.load_checkpoint_model(path,torch.device('cpu'))
        except ValueError:
            pass
        else:
            raise AssertionError('Wrong crop metadata was accepted')
    print('Checkpoint roundtrip and wrong-crop rejection: OK',flush=True)
    print(json.dumps(dict(parameter_count=parameter_count,training_started=False,
        train_count=len(train_loader.dataset),val_count=len(val_loader.dataset),test_count=len(test_loader.dataset))),flush=True)


if __name__ == '__main__':
    main()
