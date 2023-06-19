# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary
from copy import deepcopy
from mmseg.models.backbones import EfficientMultiheadAttention

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
from mmengine.model import is_model_wrapper

from mmseg.registry import RUNNERS

@RUNNERS.register_module()
class BoostRunner(Runner): #For boosting
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.model.eval() #freezing the model (just in case, please check self.model.training as False (if true, then it will be trained))

        print(f'here we are')


    def boost(self) -> nn.Module:
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

        if self.param_schedulers is not None:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        # initialize the model weights
        self._init_model_weights()
        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        # ---------------------- insert bottleneck and freeze weights --------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_shape = (1, 3, 512, 512)
        attn_module = "proj_drop"
        model = self.model

        shapes = get_attention_shapes(model.backbone, device, input_shape, attn_module)
        bottleneck_replacement_map = get_bottleneck_replacement_map(model.backbone, shapes, attn_module)
        model(torch.randn(input_shape).to(device)) # try the forward pass on the unmodified model
        for key, value in bottleneck_replacement_map.items():
            replace_layer(model.backbone, target=value[0], replacement=value[1])
        freeze_all_params_except_from(model, param_name="btn_alphas")

        

        #model(torch.rand((1, 3, 256, 300), device=device))
        torch.save(model.state_dict(), 'model_init.pth')
    
        # model = model.to(device)
        sum_ = summary(model, input_size=input_shape)
        # ---------------------- insert bottleneck and freeze weights --------------
        
        
        # Initiate inner count of `optim_wrapper`.
        self.optim_wrapper.initialize_count_status(
            self.model,
            self._train_loop.iter,  # type: ignore
            self._train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self._maybe_compile('train_step')
        
        self.dummy_train()
        
        model = self.train_loop.run()  # type: ignore
        self.call_hook('after_run')
        return model

    def dummy_train(self):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        # Assuming you have a model and dataloader defined

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Define the loss function
        criterion = nn.CrossEntropyLoss()

        # Define the optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        num_epochs = 2
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()  # Set the model to training mode
            
            for images, labels in self.train_dataloader:
                images = images.to(device)
                labels = labels.to(device)                
                # Forward pass
                outputs = self.model(images)
                
                # Compute the loss
                loss = criterion(outputs, labels)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Print the loss for each epoch
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

class Bottleneck(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.btn_alphas = nn.Parameter(torch.rand(shape), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, r):
        # resize necessary if input image shape different from alphas initialization process
        # if r.shape[-2:] != self.btn_alphas.shape[-2:]:
        #     btn_alphas_resized = F.interpolate(self.btn_alphas.unsqueeze(0), size=r.shape[1:], mode="bilinear").squeeze().to(r.device)
        #     return r * self.sigmoid(btn_alphas_resized)
        return r * self.sigmoid(self.btn_alphas)

def get_attention_shapes(model, device, input_shape, attn_module):
    shapes = []
    def get_attention(module, input, output):
        # shapes[module.name] = output.shape
        shapes.append(output.detach().cpu().shape)# is tuple (output, weights) if module is nn.MultiheadAttention
    if isinstance(attn_module, type):    
        hook_handles = [module.register_forward_hook(get_attention) for idx, (name, module) in enumerate(model.named_modules()) if isinstance(module, attn_module)]
    elif isinstance(attn_module, str):
        hook_handles = [module.register_forward_hook(get_attention) for idx, (name, module) in enumerate(model.named_modules()) if attn_module in name]
    else:
        raise TypeError
    
    model(torch.randn(input_shape).to(device))
    for handle in hook_handles:
        handle.remove()
    return shapes

def get_bottleneck_replacement_map(model, shapes, attn_module):
    shapes_cpy = deepcopy(shapes)
    bottlenecks_map = {}
    for name, module in model.named_modules():
        if isinstance(attn_module, type): 
            if isinstance(module, attn_module): 
                bottlenecks_map[name] = (module, nn.Sequential(module, Bottleneck(shape=shapes_cpy.pop(0)))) # (target, replacement)
        elif attn_module in name:
            bottlenecks_map[name] = (module, nn.Sequential(module, Bottleneck(shape=shapes_cpy.pop(0)))) # (target, replacement)
    assert len(shapes) == len(bottlenecks_map), "Number of bottlenecks doesn't fit the number of shapes!"
    return bottlenecks_map 


def replace_layer(model: nn.Module, target: nn.Module, replacement: nn.Module):
    """
    Replace a given module within a parent module with some third module
    Useful for injecting new layers in an existing model.
    ___

    When invoked from _run_training method:
     - target is a module taken from model.named_modules() that statisfies nn.Conv2d, nn.Linear, ...
     - replacement is a Sequential that consists of the target and it's corresponding bottleneck ...
    """
    def replace_in(model: nn.Module, target: nn.Module, replacement: nn.Module):
        # print("searching ", model.__class__.__name__)
        for name, submodule in model.named_children():
            # print("is it member?", name, submodule == target)
            if submodule == target:
                # we found it!
                if isinstance(model, nn.ModuleList):
                    # replace in module list
                    model[name] = replacement

                elif isinstance(model, nn.Sequential):
                    # replace in sequential layer
                    if name.isdigit():
                        model[int(name)] = replacement
                    else:
                        set_module_in_model(model, name, replacement)
                else:
                    # replace as member
                    model.__setattr__(name, replacement)

                # print("Replaced " + target.__class__.__name__ + " with "+replacement.__class__.__name__+" in " + model.__class__.__name__)
                return True

            elif len(list(submodule.named_children())) > 0:
                # print("Browsing {} children...".format(len(list(submodule.named_children()))))
                if replace_in(submodule, target, replacement):
                    return True
        return False
    
    if not replace_in(model, target, replacement):
        raise RuntimeError("Cannot substitute layer: Layer of type " + target.__class__.__name__ + " is not a child of given parent of type " + model.__class__.__name__)

def freeze_all_params_except_from(model, param_name):
    for name, param in model.named_parameters():
        if param_name not in name:
            param.requires_grad = False

def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = BoostRunner.from_cfg(cfg)
    else:
        # build customized runner from the registry""
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()
    # runner.test()


if __name__ == '__main__':
    main()
