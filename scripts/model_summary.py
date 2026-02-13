import argparse

import torch

from models.picnn_static import PICNN_static
from models.pinn import PINN
from models.pitcnn_latenttime import PITCNN_dynamic, PITCNN_dynamic_batchnorm, PITCNN_dynamic_latenttime1
from models.pitcnn_timefirst import PITCNN_dynamic_timefirst


def build_model(name, channels):
    if name == "picnn_static":
        return PICNN_static(loss_fn=torch.nn.MSELoss(), channels=channels)
    if name == "pitcnn_dynamic":
        return PITCNN_dynamic(c=channels)
    if name == "pitcnn_dynamic_latenttime1":
        return PITCNN_dynamic_latenttime1(c=channels)
    if name == "pitcnn_dynamic_timefirst":
        return PITCNN_dynamic_timefirst(c=channels)
    if name == "pitcnn_dynamic_batchnorm":
        return PITCNN_dynamic_batchnorm(c=channels)
    if name == "pinn":
        return PINN()
    raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Print parameter count and optional torchsummary for a model.")
    parser.add_argument(
        "--model",
        type=str,
        default="pitcnn_dynamic_latenttime1",
        choices=[
            "picnn_static",
            "pitcnn_dynamic",
            "pitcnn_dynamic_latenttime1",
            "pitcnn_dynamic_timefirst",
            "pitcnn_dynamic_batchnorm",
            "pinn",
        ],
    )
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, args.channels).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if args.model == "pinn":
        x = torch.randn(args.batch_size, 4, device=device)
        y = model(x)
    elif args.model == "picnn_static":
        x = torch.randn(args.batch_size, 1, 64, 32, 16, device=device)
        y = model(x)
    else:
        x = torch.randn(args.batch_size, 1, 64, 32, 16, device=device)
        t = torch.full((args.batch_size, 1), 0.5, device=device)
        y = model(x, t)
    print(f"Output shape: {tuple(y.shape)}")

    try:
        from torchsummary import summary

        if args.model == "pinn":
            summary(model, input_size=(4,), batch_size=args.batch_size, device=str(device))
        elif args.model == "picnn_static":
            summary(model, input_size=(1, 64, 32, 16), batch_size=args.batch_size, device=str(device))
        else:
            summary(model, input_size=[(1, 64, 32, 16), (1,)], batch_size=args.batch_size, device=str(device))
    except Exception as exc:
        print(f"torchsummary not available or failed: {exc}")


if __name__ == "__main__":
    main()
