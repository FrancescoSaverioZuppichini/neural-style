import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.optim as optim
from torchvision.models import vgg19, vgg16
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize, ToPILImage


style = Image.open('./images/urlo.jpg')
content = Image.open('./images/paolo.jpeg')

RESIZE = 244
width, height = content.size
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# transform = Compose([Resize((RESIZE, RESIZE)), ToTensor()])
transform = Compose([Resize((RESIZE, RESIZE)), ToTensor(), Normalize(mean=mean, std=std)])


def loader(pil_im, transform=transform):
  return transform(pil_im).unsqueeze(0).to(device).float()

class NormalizeInverse(Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

unloader = Compose([NormalizeInverse(mean, std), ToPILImage()])  # reconvert into PIL image

def imshow(image, title=None, normalized=True):
    # image = image.cpu().clone()  # we clone the tensor to not do changes on it
    # image = image.squeeze(0)      # remove the fake batch dimension
    # image = unloader(image)
    # image = ToPILImage()(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array(
        (0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def __call__(self, x):
        return F.mse_loss(x, self.target)


class GradMatrix(nn.Module):
    def __init__(self, input):
        super().__init__()
        b, n, h, w = input.size()
        features = input.view(b * n, h * w)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize'
        # self.matrix = G.div(b * n * h * w)
        self.matrix = G
    def __call__(self):
        return self.matrix


class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.grad_target = GradMatrix(target.detach())()

    def __call__(self, x):
        grad_x = GradMatrix(x)()
        b, n, h, w = x.size()
        return F.mse_loss(grad_x, self.grad_target) / (b * n * h * w)


from functools import partial
from collections import OrderedDict


class StyleAndContentLoss(nn.Module):
    def __init__(self, module, style_layers, style_layers_weights):
        super().__init__()
        self.module = module
        self.style_layers = style_layers
        self.style_layers_weights = style_layers_weights
        self.features = OrderedDict({'inputs': [], 'styles': [], 'contents': []})
        self.register_forward_hooks()


    def hook(self, m, i, o):
        self.features[self.key].append(o)

    def register_forward_hooks(self):
        idx = 0
        for module in self.module.children():
            is_conv = type(module) is nn.Conv2d
            if is_conv:
                is_a_style_layer = idx in self.style_layers
                if is_a_style_layer:
                    module.register_forward_hook(self.hook)
                    print(f'[INFO] hook registered to {module}')
            idx += 1

    def forward(self, content, style):
        self.features = OrderedDict({'inputs': [], 'styles': [], 'contents': []})
        self.key = 'contents'
        self.module(content)
        self.key = 'styles'
        self.module(style)
        self.key = 'inputs'

    def compute_loss(self, device):
        content_loss = ContentLoss( self.features['contents'][4])(self.features['inputs'][4])
        style_loss =0

        for i, (input, content, style) in enumerate(zip(self.features['inputs'], self.features['contents'],  self.features['styles'])):
            # content_loss += ContentLoss(content)(input)
            style_loss += StyleLoss(style)(input) * self.style_layers_weights[i]


        self.features['inputs'] = []
        # print(f"[INFO] style={style_loss.item()} content={content_loss.item()}")
        # print(f"[INFO] {style_loss}")
        return (content_loss * 1e4) + (style_loss * 1e3)

    def __repr__(self):
        return str({k: [e.shape for e in v] for k, v in self.features.items()})

style_layers = [0, 5, 10, 19, 28]
style_layers_weights = [ 0.75, 0.5, 0.2, 0.2, 0.2]


content_x = loader(content)
style_x = loader(style)
content_weight = 1e4
style_weight = 1e2

cnn = vgg19(True).features
for param in cnn.parameters():
  param.requires_grad_(False)

for i, layer in enumerate(cnn):
  if isinstance(layer, torch.nn.MaxPool2d):
    cnn[i] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

cnn = cnn.to(device).eval()

criterion = StyleAndContentLoss(cnn, style_layers, style_layers_weights)
criterion(content_x, style_x)

# x = content_x.clone().requires_grad_(True).to(device)
x = torch.randn_like(content_x).requires_grad_(True).to(device)
optimizer = optim.Adam([x], 0.01)

for i in range(2000):
    optimizer.zero_grad()
    _ = cnn(x)
    loss = criterion.compute_loss(device)
    loss.backward()
    print(f"[INFO]{loss.item()}")

    optimizer.step()
    # if i % 50 == 0: imshow(x.detach())
    if i % 25 == 0: imshow(im_convert(x.detach()))
