# Project Repository

이 레포지토리는 개인적인 프로젝트들을 정리하고 관리하기 위한 공간입니다. 자연어 처리(NLP) 모델을 직접 구현하거나 논문 기반 실험을 수행한 결과물을 코드로 담아두고 있습니다. 현재는 BERT 기반의 욕설 탐지 모델 프로젝트가 포함되어 있습니다.



## 포함된 프로젝트

### 1. bert-profanity-detection-model

BERT를 활용한 욕설 감지 모델입니다. 사전 학습된 BERT를 파인튜닝하여 성능을 향상시킵니다.

#### 파일 구성

- `A01_Preprocessing.py`  
  데이터 로딩 및 텍스트 전처리 수행

- `A02_CustomDataset.py`  
  PyTorch용 커스텀 데이터셋 클래스 정의

- `A03_Train.py`  
  모델 구성 및 학습 로직 정의

- `A04_TrainRun.py`  
  모델 학습 실행을 위한 메인 스크립트

- `A05_TestRun.py`  
  학습된 모델의 테스트 실행 스크립트

- `results/fine_tuned_model/`  
  모델 학습 결과 디렉토리  
  - `config.json` : 학습된 모델의 설정 정보  
  - `tokenizer_config.json` : 토크나이저 설정  
  - `special_tokens_map.json` : 특수 토큰 매핑 정보  
  - `vocab.txt` : 토크나이저 단어 사전

---

## 확장 계획

- 다양한 사전학습 언어 모델 실험 (RoBERTa, DistilBERT 등)
- 텍스트 분류 외 감정 분석, 질의응답, 요약 등의 태스크 추가
- 모델 성능 비교 및 정량적 평가 자동화 기능 개발
- GUI 또는 REST API 기반 배포 기능 추가

---

## 라이선스 및 목적

이 레포지토리는 개인 학습, 실험 및 연구 목적으로 작성되었습니다.  
상업적 용도나 외부 공개 배포를 위한 목적이 아니며,  
오픈소스 모델의 사용 및 재현을 통해 연구를 이해하고 학습하는 데 목적이 있습니다.
