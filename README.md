# cxr-explainer

흉부 X-ray 이미지를 받아 CheXagent로 소견을 추출하고, Gemini API로 일반인이 이해할 수 있는 한국어 설명을 생성하는 파이프라인.

## 설치

```bash
cd ~/cxr-explainer
pip install -r requirements.txt
```

## API 키 설정

```bash
cp .env.example .env
# .env 파일을 열어 GEMINI_API_KEY 값을 입력
```

## 실행

```bash
# 기본 실행 (소견 추출 + 한국어 설명)
python main.py samples/test.jpg

# 추가 질문 포함
python main.py samples/test.jpg -q "왼쪽 폐 부분이 특히 걱정됩니다."

# CheXagent 영어 소견만 출력 (Gemini 호출 없음)
python main.py samples/test.jpg --findings-only
```

## 파일 구조

| 파일 | 역할 |
|------|------|
| `config.py` | 모델명, 프롬프트, 시스템 인스트럭션 등 모든 설정 |
| `vision.py` | CheXagent 로드 및 `extract_findings(image_path)` |
| `explain.py` | Gemini 클라이언트 및 `explain(findings, question)` |
| `main.py` | CLI 진입점, vision → explain 파이프라인 실행 |

## 모델

- **Vision**: `StanfordAIMI/CheXagent-2-3b` (HuggingFace Hub에서 자동 다운로드)
- **Explain**: `gemini-2.5-flash` (Google AI Studio API)
