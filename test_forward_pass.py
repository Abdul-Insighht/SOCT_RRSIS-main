import os
import sys
import glob
import argparse
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lib.soct_sam_model import SOCT_SAM


def find_sam3_ckpt(user_path: str = None) -> str:
    """Find sam3.pt across common local directories."""
    candidates = []
    if user_path:
        candidates.append(user_path)
    
    # Common local paths
    candidates.extend([
        "./pre-trained-weights/sam3.pt",
        "../sam3_pre_trained_weights/sam3.pt",
        "./checkpoints/sam3.pt",
        "../checkpoints/sam3.pt",
    ])
    
    for p in candidates:
        if os.path.isfile(p):
            return p
            
    return user_path or "./pre-trained-weights/sam3.pt"


def test_forward(ckpt_path: str = None):
    sam3_ckpt = find_sam3_ckpt(ckpt_path)
    print(f"Using SAM3 Checkpoint: {sam3_ckpt}")
    
    if not os.path.isfile(sam3_ckpt):
        raise FileNotFoundError(
            f"Could not find sam3.pt at '{sam3_ckpt}'. "
            f"Please specify via --sam3_ckpt <path_to_sam3.pt>."
        )

    print("Initializing Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device: {device}")

    model = SOCT_SAM(
        sam3_ckpt=sam3_ckpt,
        image_size=800,
        lora_rank=4,
        lora_alpha=8.0,
        use_dynamic_lora=True,
        use_contrastive_loss=True,
        use_multiscale_ot=True,
        use_ohem_loss=True,
        use_soct=True,
        lambda_ord=5.0,
        theta_anchor=0.1,
    )
    model = model.to(device).float()

    B = 2
    images = torch.randn(B, 3, 800, 800, device=device)
    captions = ["the small white plane to the left of the hangar", "a large ship on water"]
    
    print("Running Forward Pass...")
    try:
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images, captions)
            
        print("🎉 Forward pass successful!")
        print(f"Output mask shape: {outputs['pred_masks'].shape}")
        if 'target_occupancy' in outputs:
            print(f"Target occupancy maps: {len(outputs['target_occupancy'])} scales")
        return True
    except Exception as e:
        print(f"❌ Forward pass failed with error: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity test for SOCT-SAM forward pass")
    parser.add_argument("--sam3_ckpt", type=str, default=None, help="Path to sam3.pt checkpoint")
    args = parser.parse_args()
    
    test_forward(args.sam3_ckpt)
