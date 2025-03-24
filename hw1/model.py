import torch
import timm
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.swish = nn.SiLU()

    def forward(self, x):
        return self.fc1(x) * self.swish(self.fc2(x))


class CustomResNeXt101_32x8d_Simple(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomResNeXt101_32x8d_Simple, self).__init__()
        self.backbone = timm.create_model(
            "resnext101_32x8d.tv2_in1k",
            pretrained=True,
            cache_dir="./pretrained",
        )
        self.backbone.reset_classifier(0)
        embed_dim = self.backbone.num_features

        self.mlps = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, embed_dim),
            nn.Dropout(0.1),
            # nn.Linear(embed_dim, embed_dim),
            # nn.LayerNorm(embed_dim),
            # SwiGLU(embed_dim, embed_dim),
            # nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlps(x)
        return x


class CustomSEResNeXt50_32x4d_Simple(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomSEResNeXt50_32x4d_Simple, self).__init__()
        self.backbone = timm.create_model(
            "seresnext50_32x4d.gluon_in1k",
            pretrained=True,
            cache_dir="./pretrained",
        )
        self.backbone.reset_classifier(0)
        embed_dim = self.backbone.num_features

        self.mlps = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlps(x)
        return x


class CustomResNeXt50_32x4d_Simple(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomResNeXt50_32x4d_Simple, self).__init__()
        self.backbone = timm.create_model(
            "resnext50_32x4d.ra_in1k",
            pretrained=True,
            cache_dir="./pretrained",
        )
        self.backbone.reset_classifier(0)
        embed_dim = self.backbone.num_features

        self.mlps = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlps(x)
        return x


class CustomResNeXt26_Simple(nn.Module):
    def __init__(self, num_classes=100):
        super(CustomResNeXt26_Simple, self).__init__()
        self.backbone = timm.create_model(
            "resnext26ts.ra2_in1k",
            pretrained=True,
            cache_dir="./pretrained",
        )
        self.backbone.reset_classifier(0)
        embed_dim = self.backbone.num_features

        self.mlps = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            SwiGLU(embed_dim, embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlps(x)
        return x
