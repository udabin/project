import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


class Train:

    # text와 label을 입력받고, 토크나이저를 사용하여 토큰화 진행
    def __init__(self, save_path, train_batch_size, eval_batch_size, epochs, weight_decay):
        # 토크나이저 로드
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.save_path = save_path
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)


    def __getitem__(self, idx):
        # 토큰화된 텍스트의 딕셔너리를 텐서형태로 변환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


    def train_model(self, train_texts, train_labels, test_texts, test_labels):
        
        # 훈련 및 테스트 데이터셋 생성
        train_dataset = CustomDataset(train_texts, train_labels)
        test_dataset = CustomDataset(test_texts, test_labels)

        # BERT 모델 로드
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # 훈련 설정
        training_args = TrainingArguments(
            output_dir='./results',          # 결과 저장 경로
            evaluation_strategy="epoch",     # 에폭마다 평가
            per_device_train_batch_size=self.train_batch_size,  # 배치 크기
            per_device_eval_batch_size=self.eval_batch_size,
            num_train_epochs=self.epochs,              # 학습 에폭 수
            weight_decay=self.weight_decay,               # 가중치 감쇠 (regularization)
        )

        # Trainer 설정
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        # 모델 학습
        trainer.train()

        # 모델 평가
        # trainer.evaluate()

        # 모델/토크나이저 저장
        self.model.save_pretrained(self.save_path)
        self.tokenizer.save_pretrained(self.save_path)


    def predict(self, text):
        # 예측을 위한 모델이 없는 경우, 에러 처리
        if self.model is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 train_model을 호출하세요.")

        # 입력 데이터를 토큰화하고 텐서로 변환
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # 모델을 CUDA로 옮김 (CUDA가 가능한 경우)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(device)

        # 입력 데이터도 CUDA로 옮김
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # 모델을 평가 모드로 설정 (드롭아웃 등을 끔)
        self.model.eval()

        # 예측 수행
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # logits에서 가장 높은 값을 가진 클래스 예측
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1)

        # 예측 결과 출력
        ans = "비속어" if prediction == 1 else "비속어 아님"
        
        return ans
