import torch
from loguru import logger


import requests
import cv2
import numpy as np
from video_utility import get_roi


def get_torch_prediction(frames):
    frame = np.expand_dims(np.array(frames), axis=0) / 255.0
    tensor = torch.from_numpy(np.expand_dims(frame.astype(np.float32), axis=-1))
    tensor = tensor.permute(0, 1, 4, 2, 3)

    with torch.no_grad():
        logits, _ = model(tensor)
        probs = torch.softmax(logits, dim=1)

    return probs.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)


def create_sequences(video_path, sequence_length=7):
    """Create sequences of frames from video, taking the middle frame of each sequence."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error(f"Error: Could not open video {video_path}")
        return

    roi = get_roi(cap)
    cap.release()  # Release and reopen to reset position
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video info: {total_frames} frames, {fps} FPS")

    # Calculate how many complete sequences we can make (sliding window)
    num_sequences = total_frames - sequence_length + 1
    logger.info(
        f"Can create {num_sequences} complete sequences of {sequence_length} frames"
    )

    # Middle frame index in each sequence (0-indexed)
    middle_idx = sequence_length // 2
    logger.info(f"Middle frame index in each sequence: {middle_idx}")

    # Read all frames first
    all_frames = []
    all_original_frames = []  # Store original ROI frames for middle frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        y, y_end, x, x_end = roi
        original_frame = frame[y:y_end, x:x_end]  # ROI but original size
        original_frame_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

        resized_frame = cv2.resize(original_frame, (320, 320))
        resized_frame_gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        all_frames.append(resized_frame_gray)  # For sequences (320x320)
        all_original_frames.append(
            original_frame_gray
        )  # For middle frames (original ROI size)

    cap.release()

    sequences = []
    middle_frames = []

    # Create sliding window sequences
    for seq_start in range(num_sequences):
        sequence = []

        # Get sequence_length frames starting from seq_start
        for frame_idx in range(sequence_length):
            frame = all_frames[seq_start + frame_idx]  # Resized frame for sequence
            sequence.append(frame)

            # If this is the middle frame, store the original ROI size version
            if frame_idx == middle_idx:
                middle_frames.append(all_original_frames[seq_start + frame_idx])

        sequences.append(sequence)
        logger.info(f"Created sequence {seq_start + 1}/{num_sequences}")

    logger.info(f"Successfully created {len(sequences)} complete sequences")
    logger.info(f"Extracted {len(middle_frames)} middle frames")

    return sequences, middle_frames


def download_file(url, filename):
    """Download a file from URL and save it locally."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


DATASET = [
    "https://storage.googleapis.com/wharfreenet_models/cane6_fatto.avi",
    "https://storage.googleapis.com/wharfreenet_models/cane8Fatto.avi",
    "https://storage.googleapis.com/wharfreenet_models/cane47.avi",
    "https://storage.googleapis.com/wharfreenet_models/cane48.avi",
    "https://storage.googleapis.com/wharfreenet_models/cane49.avi",
    "https://storage.googleapis.com/wharfreenet_models/cane50.avi",
    "https://storage.googleapis.com/wharfreenet_models/gatto36fatto.avi",
    "https://storage.googleapis.com/wharfreenet_models/la_ao_gatto37.avi",
    "https://storage.googleapis.com/wharfreenet_models/la_ao_gatto39.avi",
    "https://storage.googleapis.com/wharfreenet_models/la_ao_gatto40.avi",
]
if __name__ == "__main__":
    model = torch.hub.load(
        ".",
        "wharfree_unet_convlstm",  # wharfree_unet_standard, wharfree_unet_lka, wharfree_unet_convlstm, wharfree_unet_transformer
        pretrained=True,
        source="local",
    )
    model.eval()
    link = DATASET[0]  # Change this to download a different video
    download_file(link, "video_test.avi")
    logger.info(f"Downloaded {link} to video_test.avi")
    sequences, middle_frames = create_sequences("video_test.avi")
    for seq, middle_frame in zip(sequences, middle_frames):
        probs = get_torch_prediction(seq)
        mask = np.argmax(probs[0], axis=-1)

        # Resize mask to middle_frame dimensions using nearest neighbor interpolation
        # This preserves the discrete class values in the mask
        target_height, target_width = middle_frame.shape[:2]
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

        # Create colored overlay for visualization
        # Convert mask to 3-channel color image
        colored_mask = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        colors = [
            [0, 0, 0],  # Class 0: Black
            [0, 255, 255],  # Class 1: Yellow
            [0, 165, 255],  # Class 2: Orange
        ]

        for class_id, color in enumerate(colors):
            colored_mask[mask_resized == class_id] = color

        # Convert middle_frame to 3-channel if it's grayscale
        if len(middle_frame.shape) == 2:
            middle_frame_3ch = cv2.cvtColor(middle_frame, cv2.COLOR_GRAY2BGR)
        else:
            middle_frame_3ch = middle_frame.copy()

        alpha = 0.1  # Transparency factor
        overlay = cv2.addWeighted(middle_frame_3ch, 1 - alpha, colored_mask, alpha, 0)

        cv2.imshow("Overlay", overlay)

        key = cv2.waitKey(220)
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()
