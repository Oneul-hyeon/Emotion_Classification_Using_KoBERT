import warnings
warnings.filterwarnings('ignore')

# Setting Library
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from bert_utils import BERTClassifier, BERTDataset

# koBERT
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

# Transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# KoBERT로부터 model, vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()

# 모델 경로 지정
model_path = 'model_all.pth'
model = torch.load(model_path)

max_len = 64
batch_size = 32
output = {0 : 'angry',
          1 : 'sadness',
          2 : 'fear',
          3 : 'disgust',
          4 : 'neutral',
          5 : 'happiness',
          6 : 'surprise'}

# 모델 출력 함수
def emotion_predict(text) :
  # 맞춤법 및 이모티콘 교정
#   text = correct_spelling(text)
  # KoBERT 모델의 입력 데이터 생성
  data = [text, '0']
  dataset = [data]
  # 문장 토큰화
  tokenizer = get_tokenizer()
  tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
  test_data = BERTDataset(dataset,0, 1, tok, max_len, True, False)
  # torch 형식으로 변환
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=5)
  model.eval()

  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
      token_ids = token_ids.long().to(device)
      segment_ids = segment_ids.long().to(device)

      valid_length = valid_length
      label = label.long().to(device)

      out = model(token_ids, valid_length, segment_ids)

      test_eval = []
      for i in out: # out = model(token_ids, valid_length, segment_ids)
          logits = i
          logits = logits.detach().cpu().numpy()

      return output[np.argmax(logits)]