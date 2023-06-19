import torch
from torch import nn
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
import numpy as np
from torchinfo import summary



class Bottleneck(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.btn_alphas = nn.Parameter(torch.rand(shape), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, r):
        return r * self.sigmoid(self.btn_alphas)

def get_attention_shapes(model, device):
    #shapes = {}
    shapes = []
    def get_attention(module, input, output):
        # shapes[module.name] = output.shape
        shapes.append(output.detach().cpu().shape)
        
    hook_handles = [module.register_forward_hook(get_attention) for idx, (name, module) in enumerate(model.named_modules()) if name.__contains__("attn_drop")] # name.endswith("attention")
    model(torch.randn(1, 3, 224, 224).to(device))
    for handle in hook_handles:
        handle.remove()
    return shapes

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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('facebookresearch/deit:main', 
        'deit_tiny_patch16_224', pretrained=True)
    model = model.to(device)
    
    shapes = get_attention_shapes(model, device)
    bottlenecks_map = {}
    for name, module in model.named_modules():
        if name.__contains__("attn_drop"):
            bottlenecks_map[name] = (module, nn.Sequential(module, Bottleneck(shape=shapes.pop(0)))) 

    print(model.training)
    # model.eval()
    for key in bottlenecks_map.keys():
        print(replace_layer(model, bottlenecks_map[key][0], bottlenecks_map[key][1]))
    print(model.training)
    freeze_all_params_except_from(model, param_name="btn_alphas")
    

    sum = summary(model, input_size=(1, 3, 224, 224))
    model(torch.randn(1, 3, 224, 224).to(device))
    print()

if __name__ == "__main__":
    print("---------------START---------------")
    main()


