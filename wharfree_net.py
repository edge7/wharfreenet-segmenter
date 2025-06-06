import torchvision.models as models
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from convlstm import ConvLSTM


class ChannelAttention(nn.Module):
    """Channel Attention sub-module"""

    def __init__(self, channels, reduction_ratio=16):  # Using standard reduction
        super().__init__()
        # Use Adaptive Pooling for variable input sizes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_channels = max(channels // reduction_ratio, 4)  # Ensure minimum channels

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),  # Use 1x1 Conv for MLP
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_weights = self.sigmoid(avg_out + max_out)
        return x * channel_weights


class SpatialAttention(nn.Module):
    """Spatial Attention sub-module - simplified from your Keras version"""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        # Pool along channel axis
        # Convolve the pooled features
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=padding, bias=False
        )  # Input is 2 channels (avg + max)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pool along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate
        pooled = torch.cat([avg_out, max_out], dim=1)
        # Convolve and apply sigmoid
        spatial_weights = self.sigmoid(self.conv(pooled))
        return x * spatial_weights


class CustomAttentionBlock(nn.Module):
    """Combines Channel and Spatial Attention (CBAM-like)"""

    def __init__(self, channels, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction_ratio)
        self.spatial_attn = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class LKA(nn.Module):
    """Large Kernel Attention module using depth-wise convolutions"""

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        # Parallel dilated convolutions with different rates
        self.conv_small = nn.Conv2d(dim, dim, 5, padding=4, groups=dim, dilation=2)
        self.conv_medium = nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3)
        self.conv_large = nn.Conv2d(dim, dim, 7, padding=12, groups=dim, dilation=4)

        self.conv_fuse = nn.Conv2d(4 * dim, dim, 1)
        self.activation = nn.GELU()

    def forward(self, x):
        u = x.clone()
        attn0 = self.conv0(x)
        attn_small = self.conv_small(x)
        attn_medium = self.conv_medium(x)
        attn_large = self.conv_large(x)

        attn = torch.cat([attn0, attn_small, attn_medium, attn_large], dim=1)
        attn = self.activation(attn)
        attn = self.conv_fuse(attn)

        return u * attn


class PositionalEncodingSpatioTemporal(nn.Module):
    """Learnable 2D positional encoding (factorized time + space)"""

    def __init__(self, embed_dim, seq_len_t, num_patches_n):
        super().__init__()
        self.seq_len_t = seq_len_t
        self.num_patches_n = num_patches_n
        self.embed_dim = embed_dim

        # Learnable embeddings for time and spatial patch sequence
        self.pos_embedding_t = nn.Parameter(torch.zeros(1, seq_len_t, 1, embed_dim))
        self.pos_embedding_n = nn.Parameter(torch.zeros(1, 1, num_patches_n, embed_dim))

        # Initialize (optional but good practice)
        nn.init.trunc_normal_(self.pos_embedding_t, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding_n, std=0.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, N, D)
        Returns:
            torch.Tensor: Input tensor with positional embeddings added. Shape (B, T, N, D).
        """
        b, t, n, d = x.shape
        assert t == self.seq_len_t and n == self.num_patches_n and d == self.embed_dim
        # Add positional embeddings with broadcasting
        # pos_embedding_t broadcasts over N, pos_embedding_n broadcasts over T
        return x + self.pos_embedding_t + self.pos_embedding_n


class SkipTemporalProcessor(nn.Module):
    """
    Applies the provided ConvLSTM to a sequence of skip connection features
    and returns a weighted combination of the processed features and the original central frame.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=(3, 3), num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        # ConvLSTM implementation expects hidden_dim potentially as a list for multi-layer
        self.hidden_dim_list = self._extend_for_multilayer(hidden_dim, num_layers)
        self.kernel_size_list = self._extend_for_multilayer(kernel_size, num_layers)
        self.num_layers = num_layers
        self.central_frame_idx = -1  # Will be determined dynamically

        # Use the provided ConvLSTM implementation
        self.conv_lstm = ConvLSTM(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim_list,  # Pass list
            kernel_size=self.kernel_size_list,  # Pass list
            num_layers=self.num_layers,
            batch_first=True,  # IMPORTANT: Our input is (B, T, C, H, W)
            bias=True,
            return_all_layers=False,  # We only need the output sequence of the last layer
        )

        # Determine the output channels from ConvLSTM's last layer
        last_layer_hidden_dim = self.hidden_dim_list[-1]

        # Output projection for ConvLSTM features
        self.output_conv = nn.Conv2d(last_layer_hidden_dim, input_dim, kernel_size=1)
        print(
            f"SkipTemporalProcessor: Adding 1x1 conv {last_layer_hidden_dim} -> {input_dim}"
        )

        # Feature fusion module to combine original and processed features
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(input_dim * 2, input_dim, kernel_size=1),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
        )


    def forward(self, x_seq):
        """
        Args:
            x_seq (torch.Tensor): Input sequence of shape (B, T, C, H, W)
        Returns:
            torch.Tensor: Processed features combining original and ConvLSTM features
                         Shape: (B, C_out, H, W)
        """
        b, t, c, h, w = x_seq.shape
        if self.central_frame_idx == -1:
            # Calculate central index once (assuming T is constant)
            self.central_frame_idx = t // 2

        # Get the original central frame features
        original_central_frame = x_seq[:, self.central_frame_idx].clone()

        # Pass the sequence through ConvLSTM
        layer_output_list, _ = self.conv_lstm(x_seq, hidden_state=None)

        # Get ConvLSTM output for the central frame
        last_layer_output_seq = layer_output_list[0]
        lstm_central_features = last_layer_output_seq[:, self.central_frame_idx]

        # Process ConvLSTM features
        processed_features = self.output_conv(lstm_central_features)
        concat_features = torch.cat([processed_features, original_central_frame], dim=1)
        combined_features = self.feature_fusion(concat_features)

        return combined_features

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        # Helper to ensure param is a list of length num_layers
        if not isinstance(param, list):
            param = [param] * num_layers
        elif len(param) != num_layers:
            # If list provided, ensure it has the correct length
            raise ValueError(
                f"Provided list parameter length ({len(param)}) does not match num_layers ({num_layers})"
            )
        return param


class EfficientNetV2SEncoder(nn.Module):
    def __init__(
        self, input_channels=1
    ):
        super().__init__()
        effnet_s = models.efficientnet_v2_s()

        if input_channels != 3:
            self.input_adapter = nn.Sequential(
                nn.Conv2d(input_channels, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 3, kernel_size=3, padding=1, bias=False),
            )
            # Initialize with He initialization
            for m in self.input_adapter.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
        else:
            self.input_adapter = nn.Identity()

        self.features = effnet_s.features

        self.skip_channels = [
            24,  # After features[0] (H/2)
            48,  # After features[1] (H/4)
            64,  # After features[2] (H/8)
            128,  # After features[4] (H/16)
            256,  # After features[6] (H/32)
        ]
        self.bottleneck_channels = 1280

    def forward(self, x):
        x = self.input_adapter(x)
        skips = []

        # --- Pass through features sequentially, storing outputs based on OBSERVED downsampling ---

        s0_out = self.features[0](x)  # H/2 (128x128), 24ch
        skips.append(s0_out)

        # Process features[1], but DON'T use its output as H/4 skip per observation
        s1_out = self.features[1](s0_out)  # Observed: Still H/2 (128x128), 48ch

        # Process features[2] - THIS output is observed as H/4
        s2_out = self.features[2](s1_out)  # Observed: H/4 (64x64), 64ch
        skips.append(s2_out)  # Use as H/4 skip

        # Process features[3] - THIS output is observed as H/8
        s3_feat3_out = self.features[3](s2_out)  # Observed: H/8 (32x32), 64ch
        skips.append(s3_feat3_out)  # Use as H/8 skip

        # Process features[4] - THIS output is observed as H/16
        s3_out = self.features[4](s3_feat3_out)  # Observed: H/16 (16x16), 128ch
        skips.append(s3_out)  # Use as H/16 skip

        # Process features[5] - No downsampling here
        s4_feat5_out = self.features[5](s3_out)  # H/16 (16x16), 128ch

        # Process features[6] - THIS output is observed as H/32
        s4_out = self.features[6](s4_feat5_out)  # Observed: H/32 (8x8), 256ch
        skips.append(s4_out)  # Use as H/32 skip

        # Process features[7] - Bottleneck
        bottleneck = self.features[7](s4_out)  # H/32 (8x8), 1280ch

        assert len(skips) == len(
            self.skip_channels
        ), f"Expected {len(self.skip_channels)} skips based on observed dimensions, but collected {len(skips)}"

        # Verify channel counts against the modified self.skip_channels
        for i, sk in enumerate(skips):
            assert (
                sk.shape[1] == self.skip_channels[i]
            ), f"OBSERVED Skip {i} channel mismatch: Expected {self.skip_channels[i]} (based on observation), Got {sk.shape[1]}"

        return bottleneck, skips


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        input_channels,  # C from bottleneck (e.g., 1280)
        input_height,  # H from bottleneck (e.g., H_img/32)
        input_width,  # W from bottleneck (e.g., W_img/32)
        seq_len,  # T (e.g., 9)
        embed_dim,  # D: Transformer internal dimension
        num_heads,  # Number of attention heads
        num_layers,  # Number of transformer encoder layers
        dim_feedforward,  # Hidden dim in FFN
        patch_size=2,
        dropout=0.1,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.patch_size = patch_size

        self.num_patches_h = input_height // patch_size
        self.num_patches_w = input_width // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w  # N

        # 1. Input Projection: Flatten spatial dims and project to embed_dim (D)
        self.input_proj = nn.Conv2d(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        assert (
            input_height % patch_size == 0 and input_width % patch_size == 0
        ), "Input H/W must be divisible by patch_size"

        # 2. Positional Encoding
        self.positional_encoding = PositionalEncodingSpatioTemporal(
            embed_dim, seq_len, self.num_patches
        )

        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",  # GELU is common in modern transformers
            batch_first=True,  # IMPORTANT: Input shape (B, T, D)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 4. Layer Normalization (applied before transformer)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # 5. Output Projection: Project central frame output back to spatial shape
        # We project back to the original bottleneck C*H*W dimension
        self.output_proj = nn.Linear(
            embed_dim, input_channels * patch_size * patch_size
        )
        self.unpatcher = Rearrange(
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.num_patches_h,
            w=self.num_patches_w,
            p1=patch_size,
            p2=patch_size,
            c=input_channels,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Bottleneck features from encoder.
                              Shape: (B * T, C, H, W)
        Returns:
            torch.Tensor: Processed features for the central frame, reshaped.
                          Shape: (B, C, H, W) - Ready for decoder input.
        """
        # --- Input Preparation ---
        # Get B and T
        bt, c, h, w = x.shape
        b = bt // self.seq_len
        t = self.seq_len
        assert c == self.input_channels, "Input channel mismatch"
        assert h == self.input_height, "Input height mismatch"
        assert w == self.input_width, "Input width mismatch"
        assert b * t == bt, "Batch size * seq_len doesn't match input batch dim"

        x = self.input_proj(x)  # (B*T, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (B*T, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B*T, num_patches, embed_dim) - Standard format

        # Reshape to (B, T, N, D)
        x = x.view(b, t, self.num_patches, self.embed_dim)

        # --- Temporal Processing ---
        # 3. Add Positional Encoding
        x = self.positional_encoding(x)  # Adds T and N embeddings

        # 4. Flatten T and N for Transformer sequence
        x = x.reshape(b, t * self.num_patches, self.embed_dim)

        # Apply LayerNorm and Dropout (common practice before transformer layers)
        x = self.layer_norm(x)
        transformer_output = self.transformer_encoder(x)  # (B, T*N, D)

        # --- Output Preparation ---
        # 6. Select Central Frame Patches
        # Get indices for the central time step t = T//2
        center_frame_start_idx = (t // 2) * self.num_patches
        center_frame_end_idx = center_frame_start_idx + self.num_patches
        central_frame_patches = transformer_output[
            :, center_frame_start_idx:center_frame_end_idx, :
        ]  # (B, N, D)

        # 7. Project back to Patch Dimension
        decoder_input_patched = self.output_proj(
            central_frame_patches
        )  # (B, N, patch_dim) where patch_dim = C'*P*P

        # 8. Unpatch / Fold back to Spatial Grid
        # Need shape (B, N, C'*P*P) -> (B, N, P*P*C') for Rearrange
        decoder_input = self.unpatcher(decoder_input_patched)  # (B, C', H, W)

        return decoder_input


class DecoderBlock(nn.Module):
    """
    UNet Decoder Block with optional LKA and CustomAttention.
    """

    def __init__(
        self,
        in_channels_up,
        in_channels_skip,
        out_channels,
        use_transpose=True,
        use_lka=True,
        use_custom_attention=True,
    ):  # Added flags
        super().__init__()
        self.use_transpose = use_transpose

        # Upsampling layer (same as before)
        if self.use_transpose:
            self.upsample = nn.ConvTranspose2d(
                in_channels_up, in_channels_up // 2, kernel_size=2, stride=2
            )
            conv_in_channels = (in_channels_up // 2) + in_channels_skip
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels_up, in_channels_up // 2, kernel_size=1),
            )
            conv_in_channels = (in_channels_up // 2) + in_channels_skip

        # Convolutional block (double conv - same as before)
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                conv_in_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # --- Optional Attention Modules ---
        self.lka = LKA(out_channels) if use_lka else nn.Identity()
        self.custom_attn = (
            CustomAttentionBlock(out_channels)
            if use_custom_attention
            else nn.Identity()
        )
        # ---------------------------------

    def forward(self, x_up, x_skip):
        x_up = self.upsample(x_up)
        if x_up.shape[2:] != x_skip.shape[2:]:
            x_up = F.interpolate(
                x_up, size=x_skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x_up, x_skip], dim=1)
        x = self.conv_block(x)

        # --- Apply Attention ---
        x = self.lka(x)
        x = self.custom_attn(x)
        # -----------------------
        return x


class UNetDecoder(nn.Module):
    def __init__(
        self,
        bottleneck_channels,
        skip_channels_list,
        decoder_channels,
        use_transpose=True,
        use_lka=True,
        use_custom_attention=True,
    ):  # Added flags
        super().__init__()
        skip_channels_reversed = list(reversed(skip_channels_list))
        assert len(skip_channels_list) == len(
            decoder_channels
        ), "Length of skip_channels_list and decoder_channels must be the same."

        self.decoder_blocks = nn.ModuleList()
        in_ch_up = bottleneck_channels
        for i in range(len(decoder_channels)):
            skip_ch = skip_channels_reversed[i]
            out_ch = decoder_channels[i]
            self.decoder_blocks.append(
                DecoderBlock(
                    in_ch_up,
                    skip_ch,
                    out_ch,
                    use_transpose=use_transpose,
                    use_lka=use_lka and i > 2,  # Pass flags down
                    use_custom_attention=use_custom_attention and i > 2,
                )  # Pass flags down
            )
            in_ch_up = out_ch

    def forward(self, x_bottleneck, skip_features_central):
        skips_reversed = list(reversed(skip_features_central))
        x = x_bottleneck
        aux_outputs = []
        for i, block in enumerate(self.decoder_blocks):
            skip = skips_reversed[i]
            x = block(x, skip)
            if i == 3:
                aux_outputs.append(x)
        return x, aux_outputs


class AuxHead(nn.Module):
    def __init__(self, in_ch, out_ch, up_factor=4):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=up_factor, mode="bilinear", align_corners=False
        )
        self.conv = nn.Conv2d(in_ch, out_ch, 1)  # 1×1, zero param headache

    def forward(self, x):
        return self.conv(self.up(x))


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor=4):
        super().__init__()
        self.upsample_factor = upsample_factor

        # ASPP-like module for better feature extraction
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, padding=1, dilation=1
        )
        self.conv3_d3 = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, padding=3, dilation=3
        )
        self.conv3_d6 = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, padding=6, dilation=6
        )

        self.bn1 = nn.BatchNorm2d(in_channels // 2)
        self.bn3 = nn.BatchNorm2d(in_channels // 2)
        self.bn3_d3 = nn.BatchNorm2d(in_channels // 2)
        self.bn3_d6 = nn.BatchNorm2d(in_channels // 2)

        self.relu = nn.ReLU(inplace=True)

        # Fusion and final prediction
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )

        self.conv_out = nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)

    def forward(self, x):
        if self.upsample_factor > 1:
            x = F.interpolate(
                x,
                scale_factor=self.upsample_factor,
                mode="bilinear",
                align_corners=False,
            )

        # Apply different convolutions and fuse
        x1 = self.relu(self.bn1(self.conv1(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x3_d3 = self.relu(self.bn3_d3(self.conv3_d3(x)))
        x3_d6 = self.relu(self.bn3_d6(self.conv3_d6(x)))

        x_cat = torch.cat([x1, x3, x3_d3, x3_d6], dim=1)
        x_fused = self.fusion(x_cat)

        return self.conv_out(x_fused)


class WharfreeUnet(nn.Module):
    def __init__(
        self,
        img_height: int,
        img_width: int,
        input_channels: int = 1,
        num_classes: int = 1,
        seq_len: int = 7,
        encoder_weights=None,
        # Transformer Hyperparameters
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256 * 2,
        transformer_dropout: float = 0.1,
        # Decoder Hyperparameters
        decoder_channels: list = None,
        use_transpose: bool = False,
        use_lka: bool = True,  # Flag to enable LKA in decoder
        use_custom_attention: bool = True,  # Flag to enable CustomAttention in decoder,
        use_skip_convlstm: bool = True,  # Flag to enable/disable
        skip_convlstm_hidden_channels: list = None,
        skip_convlstm_kernel_size: tuple = (3, 3),  # Kernel size for ConvLSTM cells
        skip_convlstm_num_layers: int = 2,  # Number of ConvLSTM layers per level
        use_temporal_transformer=True,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.img_height = img_height
        self.img_width = img_width

        # 1. Encoder
        self.encoder = EfficientNetV2SEncoder(
            input_channels=input_channels
        )
        if decoder_channels is None:
            decoder_channels = [skip for skip in self.encoder.skip_channels]
            decoder_channels.reverse()  # Reverse to match decoder order
        bottleneck_channels = self.encoder.bottleneck_channels
        self.skip_channels_list = (
            self.encoder.skip_channels
        )  # H/2, H/4, H/8, H/16, H/32


        self.bottleneck_h = img_height // 32
        self.bottleneck_w = img_width // 32

        # 2. Temporal Transformer
        if use_temporal_transformer:
            self.temporal_transformer = TemporalTransformer(
                input_channels=bottleneck_channels,
                input_height=self.bottleneck_h,
                input_width=self.bottleneck_w,
                seq_len=seq_len,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=transformer_dropout,
            )
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(
                    bottleneck_channels * 2,
                    bottleneck_channels * 2,
                    kernel_size=1,
                ),
                nn.BatchNorm2d(bottleneck_channels * 2),
                nn.ReLU(inplace=True),
            )
        self.use_temporal_transformer = use_temporal_transformer

        self.skip_temporal_processors = None
        self.use_skip_convlstm = use_skip_convlstm
        if self.use_skip_convlstm:
            print("Initializing Skip Connection ConvLSTMs...")
            self.skip_temporal_processors = nn.ModuleList()

            # Prepare hidden_dim list for ConvLSTM processors
            if skip_convlstm_hidden_channels is None:
                # Default: hidden channels = input channels for each level
                resolved_hidden_dims = self.skip_channels_list
            elif isinstance(skip_convlstm_hidden_channels, list) and len(
                skip_convlstm_hidden_channels
            ) == len(self.skip_channels_list):
                # Use provided list if length matches
                resolved_hidden_dims = skip_convlstm_hidden_channels
            else:
                raise ValueError(
                    f"skip_convlstm_hidden_channels must be None or a list of length {len(self.skip_channels_list)}"
                )

            # Prepare kernel_size list (ConvLSTM expects list for multi-layer, SkipTemporalProcessor handles extension)
            if isinstance(skip_convlstm_kernel_size, tuple):
                kernel_size_param = skip_convlstm_kernel_size  # Pass tuple directly
            elif isinstance(skip_convlstm_kernel_size, list):
                # If user provides list, ensure it's correct length if num_layers > 1
                if 1 < skip_convlstm_num_layers != len(skip_convlstm_kernel_size):
                    raise ValueError(
                        f"If providing list kernel_size, length must match skip_convlstm_num_layers ({skip_convlstm_num_layers})"
                    )
                kernel_size_param = skip_convlstm_kernel_size  # Pass list
            else:
                raise ValueError("skip_convlstm_kernel_size must be tuple or list")

            for i, skip_ch in enumerate(self.skip_channels_list[:4]):
                hidden_dim = (
                    resolved_hidden_dims[i] // 2 if i >= 2 else resolved_hidden_dims[i]
                )

                # Create the processor for this skip level
                processor = SkipTemporalProcessor(
                    input_dim=skip_ch,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size_param,  # Pass tuple or list
                    num_layers=skip_convlstm_num_layers,
                )
                self.skip_temporal_processors.append(processor)
        else:
            print("Skip Connection ConvLSTMs are DISABLED.")

        # 3. Decoder
        assert len(self.skip_channels_list) == len(
            decoder_channels
        ), f"Decoder channels list length ({len(decoder_channels)}) must match skip channels list length ({len(self.skip_channels_list)})"
        self.decoder = UNetDecoder(
            bottleneck_channels=(
                bottleneck_channels
                if not use_temporal_transformer
                else bottleneck_channels * 2
            ),
            skip_channels_list=self.skip_channels_list,
            decoder_channels=decoder_channels,
            use_transpose=use_transpose,
            use_lka=use_lka,
            use_custom_attention=use_custom_attention,
        )

        # 4. Segmentation Head
        # Decoder outputs features at H/2 resolution (since it has len(skip_channels_list) blocks)
        decoder_output_channels = decoder_channels[-1]
        head_upsample_factor = 2  # From H/2 to H
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_output_channels,
            out_channels=num_classes,
            upsample_factor=head_upsample_factor,
        )
        self.aux_head = AuxHead(decoder_channels[-2], num_classes, up_factor=4)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C_in, H, W)
                              where T is seq_len.
        Returns:
            torch.Tensor: Output segmentation logits for the central frame.
                          Shape: (B, num_classes, H, W)
        """
        b, t, c, h, w = x.shape
        if t != self.seq_len:
            raise ValueError(
                f"Input sequence length {t} does not match model's seq_len {self.seq_len}"
            )
        if h != self.img_height or w != self.img_width:
            # Optional: Add resize here, or raise error
            raise ValueError(
                f"Input spatial dimensions ({h}x{w}) do not match model's expected dimensions ({self.img_height}x{self.img_width})"
            )

        # Reshape for encoder: (B, T, C, H, W) -> (B*T, C, H, W)
        x_reshaped = x.view(b * t, c, h, w)

        # 1. Encoder Pass (Shared weights across time)
        # bottleneck_all: (B*T, C_bottle, H_bottle, W_bottle)
        # skips_all: List[(B*T, C_skip, H_skip, W_skip)] - H/2 down to H/32
        bottleneck_all, skips_all = self.encoder(x_reshaped)

        # Extract the central frame's bottleneck directly from encoder output
        # First reshape bottleneck_all from (B*T, C, H, W) to (B, T, C, H, W)
        bottleneck_reshaped = bottleneck_all.view(
            b,
            t,
            bottleneck_all.shape[1],
            bottleneck_all.shape[2],
            bottleneck_all.shape[3],
        )

        # Get central frame's bottleneck
        central_idx = t // 2
        central_frame_bottleneck = bottleneck_reshaped[:, central_idx]  # (B, C, H, W)

        if self.use_temporal_transformer:
            # 2. Temporal Transformer Pass
            transformer_bottleneck = self.temporal_transformer(bottleneck_all)

            # Combine features - direct residual from encoder-processed central frame
            combined_bottleneck = torch.cat(
                [transformer_bottleneck, central_frame_bottleneck],
                dim=1,
            )
            central_bottleneck = self.fusion_conv(combined_bottleneck)
        else:
            central_bottleneck = central_frame_bottleneck

        # 3. Prepare Skip Connections for Central Frame
        skips_central = []
        central_idx = t // 2  # Central frame index

        for i, s_all in enumerate(skips_all):
            # Reshape skip: (B*T, C_s, H_s, W_s) -> (B, T, C_s, H_s, W_s)
            _, c_s, h_s, w_s = s_all.shape
            s_all_reshaped = s_all.view(b, t, c_s, h_s, w_s)

            if (
                self.use_skip_convlstm
                and self.skip_temporal_processors is not None
                and i in (0, 1, 2, 3)
            ):
                # Process the full sequence s_all_reshaped with the corresponding processor
                processor = self.skip_temporal_processors[i]
                s_central_processed = processor(s_all_reshaped)
                # s_central_processed shape should be (B, C_s, H_s, W_s)
                # because SkipTemporalProcessor's output_conv projects back to C_s
                skips_central.append(s_central_processed)
            else:
                # Original behavior: just select the central frame's features
                s_central = s_all_reshaped[:, central_idx, :, :, :]
                skips_central.append(s_central)

        # 4. Decoder Pass
        # Takes central bottleneck and central skips
        # decoder_output: (B, C_decoder_last, H/2, W/2)
        decoder_output, aux_outputs = self.decoder(central_bottleneck, skips_central)

        # 5. Segmentation Head Pass
        # Upsamples H/2 -> H and predicts classes
        # logits: (B, num_classes, H, W)
        logits = self.segmentation_head(decoder_output)
        aux_logits = self.aux_head(aux_outputs[0])

        return logits, aux_logits

