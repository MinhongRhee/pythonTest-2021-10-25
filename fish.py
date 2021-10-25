# 1. 라이브러리 설치
# pip install numpy - 넘파이 설치
# pip install pandas - 판다스 설치
# pip3 install mariadb SQLAlchemy - SQLAlchemy 설치

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import sqlalchemy as db
import matplotlib.pyplot as plt

# 2. fish.csv 파일 로드하기
# 도미의 길이
bl = pd.read_csv('C:/workspace/pythonwork/test20211025/csv/bream_length.csv', header=None);
bream_length = bl.to_numpy().flatten()

# 도미의 무게
bw = pd.read_csv('C:/workspace/pythonwork/test20211025/csv/bream_weight.csv', header=None);
bream_weight = bw.to_numpy().flatten()

# 빙어의 길이
sl = pd.read_csv('C:/workspace/pythonwork/test20211025/csv/smelt_length.csv', header=None);
smelt_length = sl.to_numpy().flatten()

# 빙어의 무게
sw = pd.read_csv('C:/workspace/pythonwork/test20211025/csv/smelt_weight.csv', header=None);
smelt_weight = sw.to_numpy().flatten()

# 도미 데이터
bream_data = np.column_stack((bream_length, bream_weight));

# 빙어 데이터
smelt_data = np.column_stack((smelt_length, smelt_weight));

# 3. 도미와 빙어 데이터 시각화
plt.scatter(bream_data[:,0], bream_data[:,1])
plt.scatter(smelt_data[:,0], smelt_data[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# 4. 2차원 배열 만들기 및 shape 확인 - 도미 35, 빙어 14
fish_data = np.concatenate((bream_data, smelt_data))
print(fish_data.shape) # [[길이, 무게]]

#  5. 타겟 데이터 만들기 - 도미 : 1, 빙어 : 0 으로 구분 !!
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
# print(fish_target.shape) - (49, ) -> fish_data와 fish_target을 합치기 위해서는 (49,1)로 reshape 해야한다.
fish_target = fish_target.reshape((-1,1)) # (49,1)
fishes = np.hstack((fish_target, fish_data))
print(fishes)

# 6. shuffle을 이용하여 데이터 섞기
index = np.arange(49) # 도미 : 0~34, 빙어 : 35~48
np.random.shuffle(index)
print(index)

# 7. 테스트 데이터와 훈련 데이터로 구분
# 훈련 데이터 - train data,
train_input = fish_data[index[:35]]
train_target = fish_target[index[:35]]

# 테스트 데이터 - test data
test_input = fish_data[index[35:]]
test_target = fish_target[index[35:]]

# 8. 훈련 데이터를 matplot으로 시각화
plt.scatter(train_input[:,0], train_input[:,1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()


# engine
engine = db.create_engine("mariadb+mariadbconnector://python:python1234@127.0.0.1:3306/pythondb")

# 9. 훈련 데이터 DB 저장 - 테이블명 : train
train_data = pd.DataFrame(fishes[index[:35]], columns=["train_target","train_lentgh", "train_weight"])
train_data.to_sql("train", engine, index=False, if_exists="replace" )

# 10. 테스트 데이터 DB 저장 - 테이블명 : test
test_data = pd.DataFrame(fishes[index[35:]], columns=["train_target","train_lentgh", "train_weight"])
test_data.to_sql("test", engine, index=False, if_exists="replace" )
