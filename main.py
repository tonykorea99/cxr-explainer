import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Chest X-ray findings explainer")
    parser.add_argument("image", help="Path to chest X-ray image (JPG/PNG)")
    parser.add_argument("-q", "--question", default=None, help="Optional follow-up question in Korean")
    parser.add_argument("--findings-only", action="store_true", help="Print raw CheXagent findings without Korean explanation")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    from vision import extract_findings
    print("[ 1/2 ] Extracting findings from X-ray ...")
    findings = extract_findings(str(image_path))
    print("\n--- CheXagent findings ---")
    print(findings)

    if args.findings_only:
        return

    from explain import explain
    print("\n[ 2/2 ] Generating Korean explanation ...")
    explanation = explain(findings, question=args.question)
    print("\n--- 설명 ---")
    print(explanation)


if __name__ == "__main__":
    main()
