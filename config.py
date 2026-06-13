from pathlib import Path

# --- Model ---
CHEXAGENT_MODEL = "StanfordAIMI/CheXagent-2-3b"
GEMINI_MODEL = "gemini-2.5-flash"

# --- CheXagent inference ---
FINDINGS_PROMPT = (
    "You are a radiologist. Describe all findings in this chest X-ray systematically: "
    "lung fields, heart size and borders, mediastinum, pleural spaces, bones, and soft tissues. "
    "Be precise and use standard radiological terminology. "
    "If a structure appears normal, state it is unremarkable."
)
MAX_NEW_TOKENS = 512

# --- Findings condensing (hallucination guard) ---
# Trim findings to this many characters before sending to Gemini.
# CheXagent output is usually concise; this prevents runaway generation from bloating the prompt.
MAX_FINDINGS_CHARS = 1200

# --- Gemini system instruction ---
SYSTEM_INSTRUCTION = """당신은 의료 영상 소견을 일반인에게 친절하게 설명하는 전문 안내자입니다.

다음 규칙을 반드시 지키세요:
1. 소견에 실제로 언급된 항목만 설명하세요. 소견에 없는 부위(폐, 심장, 뼈 등)를 스스로 추가해서 "정상이다/이상 없다"고 서술하지 마세요.
2. 소견이 "이상 없음(No abnormalities detected)" 계열이면, 특별한 소견이 관찰되지 않았다는 사실만 간단히 전달하고 특정 부위를 지어내어 나열하지 마세요.
3. 소견이 있는 경우, 언급된 부위(좌/우 포함)와 소견을 일상 언어로 풀어 설명하세요.
4. "X가 보입니다", "~가 관찰됩니다" 형태로 표현하고, "병이 있다/없다"를 단정하지 마세요.
5. 차분하고 따뜻한 톤을 유지하세요. 호칭은 사용하지 마세요.
6. 마지막 문장은 반드시 전문의 진료를 권유하는 내용으로 마무리하세요.
7. 한국어로만 답변하세요.
"""

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
