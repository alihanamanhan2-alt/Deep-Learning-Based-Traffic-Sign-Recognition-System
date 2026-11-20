"""Write actual-model results only. Legacy TensorFlow entry point is deprecated."""
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Deprecated compatibility script. Use evaluate.py for PyTorch evaluation.",
    )
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="cnn")
    args = parser.parse_args()
    raise RuntimeError(
        f"test_model.py is deprecated. Run: python evaluate.py --model {args.model}"
    )


if __name__ == "__main__":
    main()
