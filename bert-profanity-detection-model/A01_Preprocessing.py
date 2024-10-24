import pandas as pd
import re
from sklearn.model_selection import train_test_split
#from torch.utils.data import Dataset


class Preprocessing:


  def __init__(self, bword_data, gword_data):

    self.bword_data = bword_data
    self.gword_data = gword_data 


  def make_data(self):

    # gword_data에 숫자 제거해주기
    gword_data = self.gword_data.values.tolist( )
    g_data = sum(gword_data, [])

    g_data_lst = list()
    for i in g_data:
      tmp = re.sub(r'\d', '', i)
      g_data_lst.append(tmp)
    g_data_df = pd.DataFrame(g_data_lst, columns=['word'])

    # 라벨 추가하기
    self.bword_data['label'] = 1
    g_data_df['label'] = 0

    # 데이터 병합
    data_ = pd.concat([self.bword_data, g_data_df], axis=0)

    # 데이터 중복 제거
    data_.drop_duplicates(subset=['word'], keep='first', inplace=True, ignore_index=True)

    # max_length 찾기
    max_length = data_['word'].str.len().max()

    # 데이터 shuffle
    data_shuffle = data_.sample(frac=1).reset_index(drop=True)

    # 데이터 셋 분리
    train_texts, test_texts, train_labels, test_labels = train_test_split(data_shuffle['word'].tolist(), data_shuffle['label'].tolist(), test_size=0.2)

    return train_texts, test_texts, train_labels, test_labels, max_length