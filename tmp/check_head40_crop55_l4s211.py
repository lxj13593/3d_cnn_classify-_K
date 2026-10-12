"""Scoped CPU smoke check: no training run, no original data/result edits."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def file_module(name, relative):
    spec=importlib.util.spec_from_file_location(name,ROOT/relative)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def function_ast(path, name):
    tree=ast.parse(path.read_text(encoding='utf8'))
    return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name),include_attributes=False)


def main():
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    rels=[
        'model/resnet18_3d_head40_stem733_crop55_l4s211.py',
        'train_val/main_resnet18_head40_stem733_crop55_l4s211_ce_5fold.py',
        'test/test_resnet18_head40_stem733_crop55_l4s211_ce_3boards.py',
    ]
    for relative in rels:
        compile((ROOT/relative).read_text(encoding='utf8'),relative,'exec')
    from model.resnet18_3d_head40_stem733_crop55_l4s211 import resnet18_3d,architecture_metadata
    from model.resnet18_3d_head40_stem733_crop55 import resnet18_3d as control_model
    train=importlib.import_module('train_val.main_resnet18_head40_stem733_crop55_l4s211_ce_5fold')
    test=file_module('crop55_l4s211_test_check',rels[2])
    old_test_path=ROOT/'test/test_resnet18_head40_stem733_crop55_ce_3boards.py'
    old_train_path=ROOT/'train_val/main_resnet18_head40_stem733_crop55_ce_5fold.py'
    for name in ('train_one_epoch','evaluate','save_evaluation_artifacts','save_history','save_cross_validation_summary'):
        assert function_ast(ROOT/rels[1],name)==function_ast(old_train_path,name)
    for name in ('evaluate_model','grouped_metric_rows','ensemble_predictions','ensemble_metric_rows','mean_std_tables','save_excel','save_curves'):
        assert function_ast(ROOT/rels[2],name)==function_ast(old_test_path,name)
    assert train.CHECKPOINT_FILES==test.CHECKPOINT_FILES
    assert train.DEFAULT_MODEL_ROOT==test.DEFAULT_MODEL_ROOT
    assert train.DEFAULT_SPLIT_ROOT==test.DEFAULT_SPLIT_ROOT
    assert train.EXPERIMENT_NAME!='resnet18_head40_stem733_crop55_ce_5fold'
    sys.argv=[str(ROOT/rels[2])]
    assert test.parse_args().checkpoint_kinds==list(test.CHECKPOINT_ORDER)
    metadata=architecture_metadata()
    torch.manual_seed(42);control=control_model()
    torch.manual_seed(42);model=resnet18_3d()
    assert sum(p.numel() for p in model.parameters())==33143106
    assert all(torch.equal(value,model.state_dict()[key]) for key,value in control.state_dict().items())
    del control
    assert model.layer4[0].conv1.stride==(2,1,1)
    assert model.layer4[0].downsample[0].stride==(2,1,1)
    assert model.layer4[0].conv2.stride==(1,1,1)
    assert model.layer4[1].conv1.stride==(1,1,1)
    assert model.layer4[1].conv2.stride==(1,1,1)
    model.eval()
    for width,expected in [(16,(5,4,4)),(21,(5,6,6)),(27,(5,7,7))]:
        with torch.no_grad():
            features=model.forward_features(torch.randn(1,1,40,width,width))
            logits=model(torch.randn(1,1,40,width,width))
        assert tuple(features.shape)==(1,512,*expected)
        assert tuple(logits.shape)==(1,2) and torch.isfinite(logits).all()
        print(f'input 40x{width}x{width} -> Layer4 {expected}; classifier [1,2]',flush=True)
    model.train()
    loss=F.cross_entropy(model(torch.randn(1,1,40,16,16)),torch.tensor([1]))
    loss.backward()
    for parameter in [model.layer4[0].conv1.weight,model.layer4[0].downsample[0].weight,model.fc.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
    model.zero_grad(set_to_none=True);model.eval()
    print('Main path + shortcut + classifier finite backward gradients: OK',flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-3)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=1e-7)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    split_config=train.load_split_config(train.DEFAULT_SPLIT_ROOT)
    payload=train.checkpoint_payload(model,optimizer,scheduler,scaler,0,42,0,'best_loss',
        train.EvaluationResult(metrics={},predictions=[]),{}, {}, {},{},split_config)
    assert payload['model_metadata']==metadata
    with tempfile.TemporaryDirectory(prefix='crop55_l4s211_check_') as temporary:
        path=Path(temporary)/train.CHECKPOINT_FILES['best_loss']
        train.atomic_torch_save(payload,path)
        loaded,_=test.load_checkpoint_model(path,torch.device('cpu'))
        assert loaded.layer4[0].conv1.stride==loaded.layer4[0].downsample[0].stride==(2,1,1)
        assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
        del loaded
        # Weight shapes alone cannot distinguish these architectures.
        payload['model_metadata']['layer4_first_block_stride']=[1,1,1]
        train.atomic_torch_save(payload,path)
        try:
            test.load_checkpoint_model(path,torch.device('cpu'))
        except ValueError:
            pass
        else:
            raise AssertionError('Incorrect stride metadata accepted')
    print(json.dumps(dict(syntax_ok=True,complete_save_functions_preserved=True,
        default_all_three_checkpoint_kinds=True,parameter_count=33143106,
        seeded_initial_weights_match_control=True,checkpoint_roundtrip_ok=True,
        wrong_stride_checkpoint_rejected=True,real_training_or_test_started=False)),flush=True)


if __name__=='__main__':
    main()
