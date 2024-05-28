import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class CosModel(nn.Module):
    def __init__(self):
        super(CosModel, self).__init__()
        model = InceptionResnetV1(
            pretrained="vggface2", classify=True, num_classes=1000
        )
        for param in model.parameters():
            param.requires_grad = False
        for param in model.block8.parameters():
            param.requires_grad = True
        for param in model.avgpool_1a.parameters():
            param.requires_grad = True
        for param in model.last_linear.parameters():
            param.requires_grad = True
        for param in model.last_bn.parameters():
            param.requires_grad = True

        model.logits = Identity()
        self.features = model

        self.my_weights = torch.nn.Parameter(torch.rand(1000, 512, requires_grad=True))

    def forward(self, x):
        feat = self.features(x)
        norm_feat = F.normalize(feat)
        norm_weights = F.normalize(self.my_weights)
        cos = norm_feat @ norm_weights.T
        return feat, cos
