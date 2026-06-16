# cxr-explainer

흉부 X-ray 이미지를 받아 CheXagent-2-3b로 소견을 추출하고, Gemini API로 일반인이 이해할 수 있는 한국어 설명을 생성하는 파이프라인.

> **주의:** 이 시스템의 출력은 AI 모델의 예측이며, 검증된 의학적 진단이 아닙니다. 소견 설명·안내 용도로만 사용하고, 진단 정확도를 보장하지 않습니다. 정확한 판독은 반드시 전문의에게 받으시기 바랍니다.

---

## 전체 처리 흐름

입력 이미지 → **CheXagent-2-3b** (vision.py)가 `<|ref|>라벨<|/ref|> <|box|>좌표<|/box|>` 형식의 그라운딩 출력을 생성 → vision.py 내 파서가 좌표를 해부학적 부위(좌/우, 상/하)로 변환해 "라벨 (부위)" 형태의 영어 소견 텍스트로 정리 → **Gemini 2.5 Flash** (explain.py)가 해당 소견을 일반인 대상 한국어 설명으로 변환 → 출력.

---

## 프로젝트 구조

```
cxr-explainer/
├── config.py          # 모델명·프롬프트·Gemini 시스템 인스트럭션 등 모든 설정
├── vision.py          # CheXagent 로드, 소견 추출, grounding 파싱·부위 변환
├── explain.py         # Gemini 클라이언트, 한국어 설명 생성
├── main.py            # 단일 이미지 CLI (python main.py <image>)
├── demo.py            # 데이터셋에서 랜덤 샘플링 후 파이프라인 실행
├── requirements.txt   # Python 의존성 (torch/torchvision 제외)
├── .env.example       # API 키 템플릿 (GEMINI_API_KEY=)
├── .env               # 실제 API 키 — git에 포함되지 않음
└── samples/           # 테스트용 이미지 — git에 포함되지 않음
```

| 파일 | 역할 |
|------|------|
| `config.py` | 모델명(`StanfordAIMI/CheXagent-2-3b`, `gemini-2.5-flash`), 추론 프롬프트, Gemini 시스템 인스트럭션, 소견 최대 길이 등 설정을 한 곳에서 관리 |
| `vision.py` | CheXagent 싱글턴 로드(bfloat16, device_map=auto) + `extract_findings(image_path)`. grounding 토큰 파싱 및 좌표→부위 변환 포함. 산문형 출력은 원문 반환 |
| `explain.py` | Gemini Client 싱글턴 + `explain(findings, question=None)`. `.env`에서 API 키 로드 |
| `main.py` | 이미지 경로를 인자로 받아 vision → explain 순서로 실행하는 CLI |
| `demo.py` | kagglehub 데이터셋 또는 지정 경로에서 이미지를 랜덤 샘플링해 파이프라인 검증 |

---

## 동작 방식

**1. CheXagent-2-3b (vision.py)**

흉부 X-ray를 입력받아 이상 소견을 그라운딩 형식으로 출력합니다.

- 출력 형식: `<|ref|>라벨<|/ref|> <|box|>(x1,y1),(x2,y2)<|/box|>`
- vision.py가 정규식으로 (라벨, 좌표) 쌍을 추출해 **"라벨 (부위)"** 텍스트로 변환합니다.
- 부위 변환 시 흉부 PA X-ray 표준 컨벤션 적용: **image-left = patient-right** (이미지 왼쪽 = 환자 오른쪽).
- 좌표(0–100 정규화 그리드) → 부위 매핑 규칙:
  - 박스 폭 ≥ 50 → `central/bilateral`
  - 중심 x < 40 → `patient's right`
  - 중심 x > 60 → `patient's left`
  - 그 외 → `central (cardiac/mediastinal)`
  - 수직: cy < 33 → `upper`, cy > 66 → `lower`, 그 외 → `mid`
- 그라운딩 토큰이 없는 산문형 출력은 원문 그대로 다음 단계로 넘깁니다.

**2. Gemini 2.5 Flash (explain.py)**

CheXagent 소견을 일반인이 이해할 수 있는 한국어로 설명합니다.

- 언급된 부위(좌/우 포함)와 소견만 설명 (없는 내용 생성 금지)
- 단정 표현 금지 ("병이 있다/없다" 대신 "~이 관찰됩니다" 형태)
- 마지막에 전문의 진료 권유

---

## 새 환경에서 처음부터 설치할 때

### 0. 저장소 받기

```bash
git clone <repo-url>
cd cxr-explainer
```

### 1. conda 환경 생성 및 활성화

Python 3.13은 `tokenizers 0.19.x` 바이너리 wheel을 지원하지 않으므로 3.12를 사용합니다.

```bash
conda create --prefix ~/cxr-explainer/.venv python=3.12 -y
conda activate ~/cxr-explainer/.venv
```

### 2. 의존성 설치

torch와 torchvision은 CUDA 빌드를 별도로 설치합니다. `torch==2.7.1+cu128`은 CUDA 12.8용으로 컴파일되었지만 CUDA 13.x 드라이버와 하위 호환됩니다.

```bash
# torch/torchvision: cu128 빌드
pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 나머지 의존성 (transformers==4.40.0 고정 포함)
pip install -r requirements.txt
```

> `transformers`를 4.40.0보다 높은 버전으로 올리면 CheXagent-2-3b의 `trust_remote_code` custom 코드가 깨질 수 있습니다. 버전을 고정해 두세요.

### 3. Gemini API 키 설정

[Google AI Studio](https://aistudio.google.com/)에서 API 키를 발급받은 뒤:

```bash
cp .env.example .env
# .env 파일을 열어 GEMINI_API_KEY=<발급받은 키> 입력
```

### 4. 테스트 이미지 준비

`samples/` 폴더와 `CheXagent-2-3b` 모델 가중치는 git에 포함되지 않습니다.

**방법 A — kagglehub로 공개 데이터셋 다운로드 (권장)**

```bash
# Kaggle 계정이 있다면 ~/.kaggle/kaggle.json에 토큰을 넣은 뒤:
python - <<'EOF'
import kagglehub, shutil, pathlib
base = pathlib.Path(kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia"))
# kagglehub 버전에 따라 chest_xray가 한 겹 또는 두 겹으로 중첩될 수 있음
inner = base / "chest_xray" / "chest_xray"
src = (inner if inner.is_dir() else base / "chest_xray") / "test"
shutil.copy(next((src / "NORMAL").iterdir()), "samples/normal.jpeg")
shutil.copy(next((src / "PNEUMONIA").iterdir()), "samples/pneumonia.jpeg")
print("Done:", list(pathlib.Path("samples").iterdir()))
EOF
```

> `demo.py`는 데이터셋에서 직접 이미지를 뽑으므로 `samples/`에 복사하지 않아도 됩니다.

**방법 B — 본인 X-ray 사용**

JPEG 또는 PNG 파일을 `samples/`에 복사합니다.

```bash
cp /path/to/my_xray.jpg samples/
```

---

## 실행

### 바로 실행 

> 이미 셋업된 서버(`/home/hail/cxr-explainer`)를 사용하는 경우, 클론·환경 생성·의존성 설치·API 키 입력·데이터 다운로드가 모두 불필요합니다.


```bash
cd ~/cxr-explainer
conda activate ~/cxr-explainer/.venv
python main.py samples/pneumonia.jpeg   # 단일 이미지 전체 파이프라인(비정상하나 sample에 있는거 꺼내서 씀)
python demo.py --label PNEUMONIA      # 폐렴 케이스만
python demo.py --balanced --n 2       # 정상 1 + 폐렴 1
python demo.py --seed 7               # 다른 랜덤 한 장 (숫자 바꿀 때마다 다른 이미지)                          
```

---

### main.py — 단일 이미지

```bash
# 전체 파이프라인 (CheXagent 소견 → Gemini 한국어 설명)
python main.py samples/pneumonia.jpeg

# 추가 질문 포함
python main.py samples/pneumonia.jpeg -q "왼쪽 폐쪽이 특히 걱정되는데 괜찮을까요?" # q 뒤에 따옴표 안에 질문 넣으면 그 질문에 맞게끔 답해주는것도 만들어봄

# CheXagent 소견만 출력 (Gemini 호출 없음)
python main.py samples/pneumonia.jpeg --findings-only
```

### demo.py — 데이터셋 랜덤 샘플

kagglehub 캐시(또는 `--data-dir` 지정 경로)에서 이미지를 자동으로 뽑아 파이프라인을 실행합니다.

```bash
# 기본: test split에서 1장 (seed=42)
python demo.py

# 3장 샘플링 (재현 가능)
python demo.py --n 3 --seed 7

# train split에서 5장, Gemini 호출 없이
python demo.py --n 5 --split train --findings-only

# 데이터셋 경로 직접 지정
python demo.py --data-dir /path/to/chest_xray --n 2
```

첫 실행 시 CheXagent-2-3b 모델(~12 GB)을 HuggingFace Hub에서 자동 다운로드합니다.

---

## 알려진 한계

- CheXagent-2-3b는 학습 데이터 분포에 따라 특정 소견에 편향될 수 있으며, 모델 예측이 항상 정확하지 않습니다.
- 좌표 → 부위 변환은 휴리스틱이며, 모델이 출력한 바운딩 박스 정확도에 의존합니다.
- 본 시스템은 소견 내용을 사용자에게 설명하는 용도이며, 진단 도구로 사용해서는 안 됩니다.
