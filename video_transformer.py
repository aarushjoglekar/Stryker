import torch
import torch.nn as nn
from transformer import TransformerEncoder, TransformerEncoderLayer

class VideoTransformer(nn.Module):
    """
    Video Transformer (ViT) architecture.
    - Key components:
    1. Loads a pretrained backbone (e.g., DINOv2).
    2. Projection layer to match embed_dim.
    3. Transformer encoder
    4. MLP classification head.
    """
    def __init__(
        self,
        # backbone_name:str,
        # image_size=224,
        # patch_size=16,
        # in_chans=3,
        embed_dim=768,
        num_layers=6,
        num_heads=12,
        mlp_dim=3072,
        num_classes=17,
        dropout=0.0,
        num_frames_per_clip=180,
        frame_batch_size=30,
    ):
        super().__init__()

        self.frame_batch_size = frame_batch_size

        # 1. Load the pretrained backbone from pytorch hub
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14') # ViT-base
        # backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # ViT-base
        backbone.eval()
        self.backbone = backbone
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Projection layer to match embed_dim
        self.proj = nn.Linear(backbone.embed_dim, embed_dim)

        # 3. Append transformer encoder to the backbone

        ## 3-1. Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers, embed_dim)
        ### DEBUG
        # self.temp = nn.Linear(embed_dim, embed_dim)

        ## 3-2. Class token
        # - class_token: learnable vector prepended to the patch sequence to aggregate global image information
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.class_token, std=0.02)

        ## 3-3. Positional embeddings
        num_patches = num_frames_per_clip
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches + 1, embed_dim)) # +1 for class token
        # Initialize positional embeddings from normal distribution (std=0.02) like BERT
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Normalization before classification
        self.norm = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        

        # 4. Classification head
        self.head = nn.Linear(embed_dim, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, num_frames, 3, H, W]
        returns: [B, num_classes]

        1. Extract features from the backbone by aggregating frame features (cls token).
        2. Project to embed_dim.
        3. Pass through transformer encoder
        4. Classify using the class token.
        """
        
        B, num_frames, C, H, W = x.shape  

        # Reshape: combine batch and frames
        x = x.view(B * num_frames, C, H, W)   # [B*num_frames, 3, H, W]
        
        # Extract features from backbone (DINO ViT returns patch embeddings including CLS token)
        overallFeats = []
        for i in range(0, x.shape[0], self.frame_batch_size):
            with torch.no_grad():  # Backbone frozen
                feats = self.backbone(x[i:i + self.frame_batch_size])  # shape: [B*num_frames, backbone_embed_dim]
                overallFeats.append(feats)

        overallFeats = torch.cat(overallFeats, dim=0)  # [B*num_frames, backbone_embed_dim]

        # Reshape back to [B, num_frames, backbone_embed_dim]
        overallFeats = overallFeats.view(B, num_frames, -1)

        # overallFeats = []
        # for i in range(B):
        #     with torch.no_grad():
        #         feats = self.backbone(x[i])
        #         overallFeats.append(feats)

        # overallFeats = torch.stack(overallFeats)  # [B, num_frames, backbone_embed_dim]
        # overallFeats = overallFeats.view(B, num_frames, -1)  # [B, num_frames, backbone_embed_dim]

        # 2. Project to embed_dim
        x = self.proj(overallFeats) # [B, num_frames, embed_dim]
        x = self.norm2(x)

        # 3. Prepend class token
        cls_tokens = self.class_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 4. Add positional embeddings
        x = x + self.pos_embed

        # 5. Transformer forward + norm
        x = self.encoder(x)

        # 6. Classification on class token
        cls_out = self.norm(x[:, 0])
        logits = self.head(cls_out)

        return logits