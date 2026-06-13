import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import CHEXAGENT_MODEL, FINDINGS_PROMPT, MAX_NEW_TOKENS, MAX_FINDINGS_CHARS

# Matches CheXagent-2 grounding output: <|ref|>Label<|/ref|> <|box|>(x1,y1),(x2,y2)<|/box|>
_REF_BOX_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|box\|>\s*\((\d+),(\d+)\),\s*\((\d+),(\d+)\)\s*<\|/box\|>"
)


def _region(x1: int, y1: int, x2: int, y2: int) -> str:
    # Coordinates are on a 0–100 normalised grid.
    # Convention (standard PA chest X-ray): image-left = patient-right.
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1

    if w >= 50:
        # Box spans most of the image width → bilateral or central structure
        lat = "central/bilateral"
    elif cx < 40:
        lat = "patient's right"   # image-left = patient-right
    elif cx > 60:
        lat = "patient's left"    # image-right = patient-left
    else:
        lat = "central (cardiac/mediastinal)"

    if lat in ("central/bilateral", "central (cardiac/mediastinal)"):
        return lat

    vert = "upper" if cy < 33 else ("lower" if cy > 66 else "mid")
    return f"{lat}, {vert}"


def _parse_grounding(raw: str) -> str | None:
    """Convert grounding tokens to readable findings text.

    Returns None if no grounding tokens are found (prose fallback).
    """
    matches = _REF_BOX_RE.findall(raw)
    if not matches:
        return None

    seen: set[tuple[str, str]] = set()
    parts: list[str] = []
    for label, x1, y1, x2, y2 in matches:
        label = label.strip()
        region = _region(int(x1), int(y1), int(x2), int(y2))
        key = (label.lower(), region)
        if key not in seen:
            seen.add(key)
            parts.append(f"{label} ({region})")

    return "; ".join(parts)

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        print(f"[vision] Loading {CHEXAGENT_MODEL} ...")
        _tokenizer = AutoTokenizer.from_pretrained(
            CHEXAGENT_MODEL,
            trust_remote_code=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            CHEXAGENT_MODEL,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # SigLIP visual encoder loads images as float32 tensors.
        # Cast encoder weights to float32 (no mismatch inside the encoder),
        # then wrap encode() to return bfloat16 so the LM decoder sees
        # uniform bfloat16 inputs. Cannot use autocast here because the
        # attention forward explicitly disables it via @torch.autocast(enabled=False).
        _model.model.visual = _model.model.visual.float()
        _orig_encode = _model.model.visual.encode
        def _encode_bf16(image_paths, training=False):
            return _orig_encode(image_paths, training=training).to(torch.bfloat16)
        _model.model.visual.encode = _encode_bf16
        _model.eval()
        print("[vision] Model ready.")
    return _model, _tokenizer


def _condense(text: str) -> str:
    """Trim findings to MAX_FINDINGS_CHARS, keeping whole sentences."""
    if len(text) <= MAX_FINDINGS_CHARS:
        return text.strip()
    # Cut at last sentence boundary within the limit
    truncated = text[:MAX_FINDINGS_CHARS]
    last_period = max(truncated.rfind("."), truncated.rfind("。"))
    if last_period > 0:
        truncated = truncated[: last_period + 1]
    return truncated.strip()


def extract_findings(image_path: str) -> str:
    model, tokenizer = _load()

    query = tokenizer.from_list_format([
        {"image": str(image_path)},
        {"text": FINDINGS_PROMPT},
    ])
    conv = [
        {"from": "system", "value": "You are a helpful assistant."},
        {"from": "human", "value": query},
    ]
    input_ids = tokenizer.apply_chat_template(
        conv,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )[0]

    raw = tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)

    parsed = _parse_grounding(raw)
    return parsed if parsed is not None else _condense(raw)
