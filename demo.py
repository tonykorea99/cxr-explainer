import argparse
import random
import time
from pathlib import Path


def _resolve_dataset(data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir)
    import kagglehub
    base = Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
    # Handle the chest_xray/chest_xray double-nesting that kagglehub produces
    for candidate in (base / "chest_xray" / "chest_xray", base / "chest_xray", base):
        if (candidate / "test").is_dir() or (candidate / "train").is_dir():
            return candidate
    return base


def _collect(dataset_root: Path, split: str) -> list[tuple[Path, str]]:
    split_dir = dataset_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    samples: list[tuple[Path, str]] = []
    for label in ("NORMAL", "PNEUMONIA"):
        label_dir = split_dir / label
        if not label_dir.is_dir():
            continue
        for p in sorted(label_dir.iterdir()):
            if p.suffix.lower() in (".jpeg", ".jpg", ".png"):
                samples.append((p, label))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly sample chest X-rays and run the full pipeline."
    )
    parser.add_argument("--data-dir", default=None,
                        help="Path to the chest_xray split root (skips kagglehub lookup)")
    parser.add_argument("--n", type=int, default=1,
                        help="Number of images to sample (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling (default: 42)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                        help="Dataset split to sample from (default: test)")
    parser.add_argument("--findings-only", action="store_true",
                        help="Print CheXagent findings only; skip Gemini explanation")
    args = parser.parse_args()

    dataset_root = _resolve_dataset(args.data_dir)
    all_images = _collect(dataset_root, args.split)
    if not all_images:
        print(f"No images found under {dataset_root / args.split}")
        return

    n = min(args.n, len(all_images))
    sampled = random.Random(args.seed).sample(all_images, n)

    from vision import extract_findings
    if not args.findings_only:
        from explain import explain

    SEP = "=" * 64

    for idx, (img_path, label) in enumerate(sampled):
        print(SEP)
        print(f"[{idx + 1}/{n}]  {img_path.name}  |  라벨: {label}")
        print()

        findings = extract_findings(str(img_path))
        print("[ CheXagent findings ]")
        print(findings)

        if not args.findings_only:
            print()
            print("[ 한국어 설명 ]")
            print(explain(findings))
            # Avoid hitting Gemini rate limit when sampling multiple images
            if idx < n - 1:
                time.sleep(1.5)

        print()

    print(SEP)


if __name__ == "__main__":
    main()
