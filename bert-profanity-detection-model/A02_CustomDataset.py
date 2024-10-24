import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset


class CustomDataset:


    # text와 label을 입력받고, 토크나이저를 사용하여 토큰화 진행
    def __init__(self, texts, labels):

        # 토크나이저 로드
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  
        # truncation : 문장의 길이가 지정된 길이보다 길면 잘라냄 (max_length)
        # padding : 짧은 문장은 패딩으로 길이를 동일하게 맞춤
        # encodings 에 토큰화된 텍스트가 저장됨
        self.encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=5)
        self.labels = labels
      

    def __getitem__(self, idx):

        # 토큰화된 텍스트의 딕셔너리를 텐서형태로 변환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


    def __len__(self):
        return len(self.labels)
