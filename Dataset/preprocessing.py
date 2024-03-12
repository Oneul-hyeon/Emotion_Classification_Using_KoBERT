# Setting Library
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Data Load
data1 = pd.read_csv('/mnt/data3/emotion_classification/Dataset/4차년도.csv', encoding = 'CP949')
data2 = pd.read_csv('/mnt/data3/emotion_classification/Dataset/5차년도.csv', encoding = 'CP949')
data3 = pd.read_csv('/mnt/data3/emotion_classification/Dataset/5차년도_2차.csv', encoding = 'CP949')
data = pd.concat([data1, data2, data3])

# Label Encoding
label_encoding_info = {}
data['상황'].loc[data['상황']=="anger"] = "angry"
data['상황'].loc[data['상황']=="sad"] = "sadness"
labels = list(data['상황'].unique())
for i in range(len(labels)) :
  label_encoding_info[labels[i]] = i
  data.loc[data['상황'] == labels[i], '상황'] = i

# Create KoBERT Dataset
dataset = []
for text, label in zip(data['발화문'], data['상황'])  :
    data = []
    data.append(text)
    data.append(str(label))
    dataset.append(data)

# Data Split
train_data, test_data = train_test_split(dataset, test_size = 0.2, random_state = 42)

# Save Dataset
with open('/mnt/data3/emotion_classification/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)
with open('/mnt/data3/emotion_classification/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)
with open('/mnt/data3/emotion_classification/label_encoding_info.pkl', 'wb') as f:
    pickle.dump(label_encoding_info, f)