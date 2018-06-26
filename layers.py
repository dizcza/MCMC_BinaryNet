import torch
import torch.nn as nn
import torch.nn.modules.conv
import torch.utils.data


def binarize_model(model: nn.Module, drop_layers=(nn.ReLU, nn.PReLU), keep_data=True) -> nn.Module:
    """
    :param model: net model
    :param drop_layers: remove these layers from the input model
    :param keep_data: keep original parameters data (True)
                      or re-sample (False) as two Gaussian peaks near 0.5 and -0.5
    :return: model with linear and conv layers wrapped in BinaryDecorator
    """
    if isinstance(model, BinaryDecorator):
        print("Layer is already binarized.")
        return model
    for name, child in list(model.named_children()):
        if isinstance(child, drop_layers):
            delattr(model, name)
            continue
        child_new = binarize_model(model=child, drop_layers=drop_layers, keep_data=keep_data)
        if child_new is not child:
            setattr(model, name, child_new)
    if isinstance(model, (nn.modules.conv._ConvNd, nn.Linear)):
        if hasattr(model, 'bias'):
            delattr(model, 'bias')
            model.register_parameter(name='bias', param=None)
        model = BinaryDecorator(model, as_two_peaks=not keep_data)
    return model


def compile_inference(model: nn.Module):
    for name, child in list(model.named_children()):
        compile_inference(child)
    if isinstance(model, BinaryDecorator):
        model.compile_inference()


class BinaryFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        tensor = tensor > 0
        tensor = 2 * tensor.type(torch.FloatTensor) - 1
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_output[input.ge(1)] = 0
        grad_output[input.le(-1)] = 0
        return grad_output


class BinaryDecorator(nn.Module):
    def __init__(self, layer: nn.Module, as_two_peaks=False):
        super().__init__()
        for param in layer.parameters():
            if as_two_peaks:
                data_peaks = 0.5 + 0.1 * torch.randn(param.data.shape)
                data_peaks[torch.rand(data_peaks.shape) > 0.5] *= -1
                if param.data.is_cuda:
                    data_peaks = data_peaks.cuda()
                param.data = data_peaks
            param.is_binary = True
        self.layer = layer
        self.is_inference = False

    def compile_inference(self):
        for param in self.layer.parameters():
            param.data.sign_()
        self.is_inference = True

    def forward(self, x):
        x = BinaryFunc.apply(x)
        if self.is_inference:
            x = self.layer(x)
        else:
            weight_full = self.layer.weight.data.clone()
            self.layer.weight.data.sign_()
            x = self.layer(x)
            self.layer.weight.data = weight_full
        return x

    def __repr__(self):
        tag = "[Binary]"
        if self.is_inference:
            tag += '[Compiled]'
        return tag + repr(self.layer)


class ScaleLayer(nn.Module):

    def __init__(self, size: int, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor(size).fill_(init_value))

    def forward(self, x):
        return self.scale * x

    def __repr__(self):
        return self.__class__.__name__ + f"(size={self.scale.numel()})"
