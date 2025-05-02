import os
import urllib.request
import argparse
import torch
from timm.models import create_model

# This script downloads the pretrained weights for FoundationStereo and its dependencies

HF_BASE_URL: str = "https://huggingface.co/datasets/steve-redefine/FoundationStereoWeights/resolve/main"
ROOT_DIR: str = "/FoundationStereo/pretrained_models"

def download_file(url: str, dest_path: str) -> None:
    """Download a file from a URL to a given local path using standard Python."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading {url} → {dest_path}")
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
        raise

def download_pretrained_weights(model_name: str) -> None:
    """Download FoundationStereo pretrained weights from Hugging Face based on the model name."""
    if model_name == "23-51-11":
        model_dir = os.path.join(ROOT_DIR, "23-51-11")
        download_file(f"{HF_BASE_URL}/23-51-11/model_best_bp2.pth", f"{model_dir}/model_best_bp2.pth")
        download_file(f"{HF_BASE_URL}/23-51-11/cfg.yaml", f"{model_dir}/cfg.yaml")

    elif model_name == "11-33-40":
        model_dir = os.path.join(ROOT_DIR, "11-33-40")
        download_file(f"{HF_BASE_URL}/11-33-40/model_best_bp2.pth", f"{model_dir}/model_best_bp2.pth")
        download_file(f"{HF_BASE_URL}/11-33-40/cfg.yaml", f"{model_dir}/cfg.yaml")

    elif model_name == "onnx":
        model_dir = os.path.join(ROOT_DIR, "onnx")
        download_file(f"{HF_BASE_URL}/onnx/foundation_stereo_23-51-11.onnx", f"{model_dir}/foundation_stereo_23-51-11.onnx")

    else:
        raise ValueError(f"❌ Unrecognized model name: {model_name}")


def download_torchhub_and_timm_models() -> None:
    """Preload Torch Hub (DINOv2) and timm (EdgeNeXt) model weights."""
    print("⬇️  Downloading DINOv2 repo from Torch Hub...")
    torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', source='github', trust_repo=True)

    print("⬇️  Downloading timm model (edgenext_small.usi_in1k)...")
    create_model('edgenext_small.usi_in1k', pretrained=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FoundationStereo weights and dependencies.")
    parser.add_argument(
        "--weights",
        type=str,
        choices=["23-51-11", "11-33-40", "onnx", ""],
        help="Which pretrained weights to download.  If empty, nothing, including dependency models, will be downloaded.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    weights = args.weights
    if weights == "":
        print("No pretrained weights selected. Skipping download.  Also not downloading Torch Hub and timm models.")
        exit(0)
    print(f"Selected pretrained model: {weights}")
    download_pretrained_weights(weights)
    download_torchhub_and_timm_models()
    print("✅ All model weights downloaded successfully.")
