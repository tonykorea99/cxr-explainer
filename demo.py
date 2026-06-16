import argparse
import math
import random
import time
from collections import Counter
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


def _sample(
    all_images: list[tuple[Path, str]],
    n: int,
    seed: int,
    label: str,
    balanced: bool,
) -> list[tuple[Path, str]]:
    rng = random.Random(seed)
    if label != "both":
        pool = [(p, l) for p, l in all_images if l == label]
        return rng.sample(pool, min(n, len(pool)))
    if balanced:
        normals = [(p, l) for p, l in all_images if l == "NORMAL"]
        pneumonias = [(p, l) for p, l in all_images if l == "PNEUMONIA"]
        k = math.ceil(n / 2)
        picked = rng.sample(normals, min(k, len(normals))) + rng.sample(pneumonias, min(k, len(pneumonias)))
        rng.shuffle(picked)
        return picked[:n]
    return rng.sample(all_images, min(n, len(all_images)))


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
    parser.add_argument("--label", default="both", choices=["NORMAL", "PNEUMONIA", "both"],
                        help="Class to sample from: NORMAL, PNEUMONIA, or both (default: both)")
    parser.add_argument("--balanced", action="store_true",
                        help="When --label=both, sample ceil(n/2) from each class")
    parser.add_argument("--findings-only", action="store_true",
                        help="Print CheXagent findings only; skip Gemini explanation")
    args = parser.parse_args()

    dataset_root = _resolve_dataset(args.data_dir)
    all_images = _collect(dataset_root, args.split)
    if not all_images:
        print(f"No images found under {dataset_root / args.split}")
        return

    counts = Counter(l for _, l in all_images)
    print(f"[pool] seed={args.seed}  NORMAL={counts['NORMAL']}  PNEUMONIA={counts['PNEUMONIA']}  total={len(all_images)}")

    sampled = _sample(all_images, args.n, args.seed, args.label, args.balanced)

    from vision import extract_findings
    if not args.findings_only:
        from explain import explain

    SEP = "=" * 64

    for idx, (img_path, label) in enumerate(sampled):
        print(SEP)
        print(f"[{idx + 1}/{len(sampled)}]  {img_path.name}  |  라벨: {label}")
        print()

        findings = extract_findings(str(img_path))
        print("[ CheXagent findings ]")
        print(findings)

        if not args.findings_only:
            print()
            print("[ 한국어 설명 ]")
            print(explain(findings))
            if idx < len(sampled) - 1:
                time.sleep(1.5)

        print()

    print(SEP)


if __name__ == "__main__":
    main()
