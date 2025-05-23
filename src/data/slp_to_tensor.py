import torch
from slippi import Game
from typing import Dict, Any, List

def print_sample_data(frames: Dict[int, Any], max_samples: int = 5):
    printed = 0
    for frame_idx, frame in frames.items():
        if printed >= max_samples:
            break
        if frame_idx < 0 or not frame or not frame.ports.get(0) or not frame.ports.get(1):
            continue

        p1 = frame.ports[0].leader
        p2 = frame.ports[1].leader

        sample = {
            "frame": frame_idx,
            "p1": {
                "x": p1.position.x,
                "y": p1.position.y,
                "percent": p1.percent,
                "state": p1.action_state.name,
            },
            "p2": {
                "x": p2.position.x,
                "y": p2.position.y,
                "percent": p2.percent,
                "state": p2.action_state.name,
            }
        }
        print(f"Sample {printed + 1}: {sample}")
        printed += 1


def extract_tensor_data(slp_path: str) -> Dict[str, torch.Tensor]:
    game = Game(slp_path)
    frames = game.frames

    print_sample_data(frames, max_samples=5)

    p1_pos, p2_pos = [], []
    p1_percent, p2_percent = [], []

    for frame_idx, frame in frames.items():
        if frame_idx < 0 or not frame or not frame.ports.get(0) or not frame.ports.get(1):
            continue

        p1 = frame.ports[0].leader
        p2 = frame.ports[1].leader

        p1_pos.append([p1.position.x, p1.position.y])
        p2_pos.append([p2.position.x, p2.position.y])
        p1_percent.append([p1.percent])
        p2_percent.append([p2.percent])

    return {
        "p1_position": torch.tensor(p1_pos, dtype=torch.float32),
        "p2_position": torch.tensor(p2_pos, dtype=torch.float32),
        "p1_percent": torch.tensor(p1_percent, dtype=torch.float32),
        "p2_percent": torch.tensor(p2_percent, dtype=torch.float32),
    }


def save_tensor_data(tensors: Dict[str, torch.Tensor], output_path: str):
    torch.save(tensors, output_path)
    print(f"\nSaved tensor data to: {output_path}")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("slp_path", help="Path to .slp file")
    parser.add_argument("--out", default="tensor_output.pt", help="Output .pt file path")
    args = parser.parse_args()

    print(f"Loading .slp file: {args.slp_path}")
    tensors = extract_tensor_data(args.slp_path)
    save_tensor_data(tensors, args.out)

