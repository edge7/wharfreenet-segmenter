import torch


dependencies = ["torch", "einops", "torchvision"]


from wharfree_net import WharfreeUnet

IMG_HEIGHT = 320
IMG_WIDTH = 320
INPUT_CHANNELS = 1
NUM_CLASSES = 3
SEQ_LEN = 7
USE_CUSTOM_ATTENTION = False  # Always False

WEIGHTS_URLS = {
    "standard": "https://storage.googleapis.com/wharfreenet_models/A_baseline_final.pt",
    "lka": "https://storage.googleapis.com/wharfreenet_models/B_lka_only_final.pt",
    "conv_lstm": "https://storage.googleapis.com/wharfreenet_models/D_skip_lstm_final.pt",
    "transformer": "https://storage.googleapis.com/wharfreenet_models/C_transformer_final.pt",
    "conv_lstm_transformer": "https://storage.googleapis.com/wharfreenet_models/E_transfn_lstm_final.pt",
}


def _create_wharfree_unet(
    pretrained=False, progress=True, weights_url_key=None, **model_flags
):
    """
    Internal helper function to create a WharfreeUnet instance and load weights.
    """
    model = WharfreeUnet(
        img_height=IMG_HEIGHT,
        img_width=IMG_WIDTH,
        input_channels=INPUT_CHANNELS,
        num_classes=NUM_CLASSES,  # Pretrained weights will dictate actual num_classes
        seq_len=SEQ_LEN,
        use_custom_attention=USE_CUSTOM_ATTENTION,
        **model_flags,  # This will pass use_lka, use_skip_convlstm, use_temporal_transformer
    )

    if pretrained:
        if not weights_url_key or weights_url_key not in WEIGHTS_URLS:
            raise ValueError(
                f"Valid weights_url_key must be provided for pretrained=True. Got: {weights_url_key}"
            )

        weights_file_url = WEIGHTS_URLS[weights_url_key]

        state_dict = torch.hub.load_state_dict_from_url(
            weights_file_url, progress=progress, check_hash=False
        )
        model.load_state_dict(state_dict["model"])
    return model


# --- Entrypoints for each model version ---


def wharfree_unet_standard(pretrained=False, progress=True, **kwargs):
    """
    WharfreeUnet (Standard Configuration).
    Fixed params: img_height=320, img_width=320, input_channels=1, num_classes=3, seq_len=7.
    Flags: use_lka=False, use_skip_convlstm=False, use_temporal_transformer=False, use_custom_attention=False.
    Args:
        pretrained (bool): If True, returns a model pre-trained on a specific dataset.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _create_wharfree_unet(
        pretrained=pretrained,
        progress=progress,
        weights_url_key="standard" if pretrained else None,
        use_lka=False,
        use_skip_convlstm=False,
        use_temporal_transformer=False,
    )


def wharfree_unet_lka(pretrained=False, progress=True, **kwargs):
    """
    WharfreeUnet with Large Kernel Attention (LKA).
    Fixed params: img_height=320, img_width=320, input_channels=1, num_classes=3, seq_len=7.
    Flags: use_lka=True, use_skip_convlstm=False, use_temporal_transformer=False, use_custom_attention=False.
    Args:
        pretrained (bool): If True, returns a model pre-trained on a specific dataset.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _create_wharfree_unet(
        pretrained=pretrained,
        progress=progress,
        weights_url_key="lka" if pretrained else None,
        use_lka=True,
        use_skip_convlstm=False,
        use_temporal_transformer=False,
    )


def wharfree_unet_convlstm(pretrained=False, progress=True, **kwargs):
    """
    WharfreeUnet with Skip ConvLSTM.
    Fixed params: img_height=320, img_width=320, input_channels=1, num_classes=3, seq_len=7.
    Flags: use_lka=False, use_skip_convlstm=True, use_temporal_transformer=False, use_custom_attention=False.
    Args:
        pretrained (bool): If True, returns a model pre-trained on a specific dataset.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _create_wharfree_unet(
        pretrained=pretrained,
        progress=progress,
        weights_url_key="conv_lstm" if pretrained else None,
        use_lka=False,
        use_skip_convlstm=True,
        use_temporal_transformer=False,
    )


def wharfree_unet_transformer(pretrained=False, progress=True, **kwargs):
    """
    WharfreeUnet with Temporal Transformer.
    Fixed params: img_height=320, img_width=320, input_channels=1, num_classes=3, seq_len=7.
    Flags: use_lka=False, use_skip_convlstm=False, use_temporal_transformer=True, use_custom_attention=False.
    Args:
        pretrained (bool): If True, returns a model pre-trained on a specific dataset.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _create_wharfree_unet(
        pretrained=pretrained,
        progress=progress,
        weights_url_key="transformer" if pretrained else None,
        use_lka=False,
        use_skip_convlstm=False,
        use_temporal_transformer=True,
    )


def wharfree_unet_convlstm_transformer(pretrained=False, progress=True, **kwargs):
    """
    WharfreeUnet with Skip ConvLSTM and Temporal Transformer.
    Fixed params: img_height=320, img_width=320, input_channels=1, num_classes=3, seq_len=7.
    Flags: use_lka=False, use_skip_convlstm=True, use_temporal_transformer=True, use_custom_attention=False.
    Args:
        pretrained (bool): If True, returns a model pre-trained on a specific dataset.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    return _create_wharfree_unet(
        pretrained=pretrained,
        progress=progress,
        weights_url_key="conv_lstm_transformer" if pretrained else None,
        use_lka=False,
        use_skip_convlstm=True,
        use_temporal_transformer=True,
    )
