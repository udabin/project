# run

from transformers import BertTokenizer, BertForSequenceClassification

save_path = './results/fine_tuned_model'

# 저장된 모델과 토크나이저 불러오기
model = BertForSequenceClassification.from_pretrained(save_path)
tokenizer = BertTokenizer.from_pretrained(save_path)

tr = Train(save_path, train_batch_size, eval_batch_size, epochs, weight_decay)
ans = tr.predict("시발")

print(ans)