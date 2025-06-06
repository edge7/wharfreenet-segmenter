# Wharfree-Net Segmenter

This repository contains the segmenter model developed for the work:

**Wharfree-Net: A Hybrid and Interpretable System for Left Atrial Aortic Ratio Estimation in Veterinary Echocardiography**

The model aims to segment the aorta and left atrium in echocardiographic videos, specifically from the right parasternal short-axis view at the level of the aortic valve.

## Prerequisites

- Python 3.8+
- `uv` (a fast Python package installer and resolver)

## Getting Started

We recommend using `uv` for managing dependencies and running the project.

1.  **Install `uv`:**
    If you don't have `uv` installed, you can get it via pip:
    ```bash
    pip install uv
    ```

2.  **Set up the environment and install dependencies:**
    Navigate to the project's root directory (where `pyproject.toml` or `requirements.txt` is located) and run:
    ```bash
    uv sync
    ```
    This will create a virtual environment (if one isn't already active and managed by `uv`) and install all necessary dependencies.

3.  **Run the Demo:**
    You can run the `main.py` script to see the model in action.
    *   If you've already run `uv sync` and the environment is active, or if you want `uv` to manage the run in the correct environment:
        ```bash
        uv run python main.py
        ```
    *   Alternatively, after `uv sync`, you can activate the environment (e.g., `.venv/bin/activate` on Linux/macOS or `.venv\Scripts\activate` on Windows) and then run `python main.py`.

    **Modifying the Demo Video:**
    To test the model with a different video, you can modify `main.py`.
    Look for the following line:
    ```python
    # in main.py
    link = DATASET[0]  # Change this to download a different video
    ```
    You can change the index (e.g., `DATASET[1]`, `DATASET[2]`, etc.) or replace `DATASET[0]` with a direct URL to a compatible video file you wish to process.

## Standalone Usage with `torch.hub`

You can easily load and use the pre-trained models directly in your Python projects using `torch.hub`.

```python
import torch

# Define the model variant you want to use
# Available variants:
# - "wharfree_unet_standard"
# - "wharfree_unet_lka"
# - "wharfree_unet_convlstm"
# - "wharfree_unet_transformer"
# - "wharfree_unet_convlstm_transformer"
model_name = "wharfree_unet_convlstm" # Example

model = torch.hub.load(
    "edge7/wharfreenet-segmenter", # GitHub repository (user/repo_name)
    model_name,
    pretrained=True,
    source="github",
    # force_reload=True # Uncomment during development if you update the repo frequently
)

# Ensure the model is in evaluation mode if you're doing inference
model.eval()

# Example: Create a dummy input tensor
# (B, Seq, C, H, W) -> (1, 7, 1, 320, 320) for this model
dummy_input = torch.randn(1, 7, 1, 320, 320)

# Perform inference
# with torch.no_grad():
#     output = model(dummy_input)

# print(f"Output shape: {output.shape}") # Expected: (B, NumClasses, H, W) -> (1, 3, 320, 320)
```

Setting pretrained=True will automatically download the pre-trained weights for the aorta and left atrium segmentation task. These weights can also serve as an excellent starting point for transfer learning on related medical image segmentation tasks.


## Model Variants

The following model variants are available via `torch.hub` by specifying the `model_name` argument in `torch.hub.load()`:

*   `"wharfree_unet_standard"`: Baseline U-Net architecture.
*   `"wharfree_unet_lka"`: U-Net with Large Kernel Attention (LKA) modules.
*   `"wharfree_unet_convlstm"`: U-Net with ConvLSTM layers in the skip connections.
*   `"wharfree_unet_transformer"`: U-Net with a temporal transformer module at the bottleneck.
*   `"wharfree_unet_convlstm_transformer"`: U-Net combining both ConvLSTM skip connections and the temporal transformer.

Refer to the `hubconf.py` file in this repository and the original paper for more architectural details on each variant.
