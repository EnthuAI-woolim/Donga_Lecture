import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#%%
### Load datasets
'''
- kc_house_data: 미국 워싱턴주 시애틀(South King County)을 포함한 킹카운티(King County) 지역의 주택 가격 데이터를 포함한 공개 데이터셋
1. id: 각 주택 거래에 대한 고유 식별자.
2. date: 주택이 판매된 날짜.
3. price: 주택의 판매 가격 (종속 변수, 목표 값).
4. bedrooms: 침실의 개수.
5. bathrooms: 욕실의 개수 (부분 욕실도 포함).
6. sqft_living: 주택의 실내 면적 (평방 피트).
7. sqft_lot: 주택의 대지 면적 (평방 피트).
8. floors: 주택의 층수.
9. waterfront: 주택이 해안가에 위치해 있는지 여부 (1: 해안가, 0: 해안가 아님).
10. view: 주택에서의 전망의 질을 나타내는 지표 (0~4).
11. condition: 주택의 전반적인 상태 (1~5).
12. grade: 주택의 건축 품질과 디자인 등급 (1~13).
13. sqft_above: 지상 층의 면적 (평방 피트).
14. sqft_basement: 지하 층의 면적 (평방 피트).
15. yr_built: 주택이 지어진 연도.
16. yr_renovated: 주택이 개보수된 연도.
17. zipcode: 우편번호.
18. lat: 주택의 위도.
19. long: 주택의 경도.
20. sqft_living15: 2015년 기준으로, 인근 15개의 주택의 평균 실내 면적 (평방 피트).
21. sqft_lot15: 2015년 기준으로, 인근 15개의 주택의 평균 대지 면적 (평방 피트).
'''
# Dataset 다운로드
# !wget "https://dongaackr-my.sharepoint.com/:x:/g/personal/sjkim_donga_ac_kr/ET2udlQfsxRAsvnlEtgzfi0B3HAMAmqP_Y2WRsbYrTvYaA?e=t6809f&download=1" -O kc_house_data.csv

#%%
# Load dataset file
data = pd.read_csv('./Data/kc_house_data.csv')
print(data)
print(type(data))   # DataFrame
#%%
### Dataset 전처리
# 예측 수행에 불필요한 열 삭제
new_data = data.drop(['id', 'date'], axis=1)

# 값이 존재하지 않는 행 삭제
new_data = new_data.dropna()

# 데이터셋 가우시안 정규화
data_normalized = (new_data - new_data.mean()) / new_data.std()
print(data_normalized)

#%%
### Dataset 특성 파악

# 각 데이터에 대한 시각화
fixed_column = 'price'
print(len(data_normalized))
used_columns = data_normalized.columns.drop(fixed_column) # price를 제외한 컬럼명들 생성(price는 타겟 컬럼이기 때문에)

#%%
num_vars = len(used_columns)
print(num_vars)
rows = (num_vars + 2) // 3  # 3열 기준으로 필요한 행 계산

fig, axs = plt.subplots(rows, 3, figsize=(15, rows * 5))

# 각 열에 대해 scatter plot을 생성하는 반복문
for i, used_column in enumerate(used_columns):
    row = i // 3
    col = i % 3
    axs[row, col].scatter(data_normalized[used_column], data_normalized[fixed_column], color='blue')
    axs[row, col].set_title(f'{used_column} vs {fixed_column}')
    axs[row, col].set_xlabel(used_column)
    axs[row, col].set_ylabel(fixed_column)

# 남은 빈 subplot 제거
for i in range(num_vars, rows * 3):
    fig.delaxes(axs.flatten()[i])

plt.tight_layout()
plt.show()

#%%
# 데이터의 각 열에 대한 상관관계 확인
correlation_matrix = data_normalized.corr()
print(correlation_matrix)

#%%
### 집 가격 예측에 사용할 데이터 지정

# 예측에 사용할 데이터들에 대한 2차원 행렬 변환
x = np.array(data_normalized[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']])
y = np.array(data_normalized[['price']])

# Train dataset / Test dataset 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Train dataset 형상 확인
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#%%
### LSM

# cf) θ=(XT⋅X)−1⋅(XT⋅Y)

def LSM(x, y):
  bias = np.ones([x_train.shape[0], 1])     # bias를 그냥 1로 설정
  X = np.hstack([x, bias])

  XT = X.T
  XTX = np.dot(XT, X)
  XTX_inverse = np.linalg.inv(XTX)
  XTY = np.dot(XT, y)
  theta = np.dot(XTX_inverse, XTY)

  return theta

#%%
theta = LSM(x_train, y_train)
print(theta)

#%%
### Prediction

# Test dataset을 이용한 예측
bias = np.ones([x_test.shape[0], 1])
x_input = np.hstack([x_test, bias])
y_hat = np.dot(x_input, theta)

#%%
# 정답 데이터와 예측 데이터 간 차이 계산
def MSE (pred, label):
  error = pred - label
  mse = np.mean(error ** 2)
  return mse

print(MSE(y_hat, y_test))

#%%
# # 시각화 - 입력 변수가 하나일 경우
# fig = plt.figure(figsize=(8,6))
# plt.scatter(x_test, y_test, color='b', marker='o', s=30)
# plt.plot(x_test, y_hat, color='r')
# plt.show()
print(x_test)
#%%
# 시각화
# input 중 인덱스가 1인 변수에 대한 예측값과 해당 변수의 정답값을 그래프로 나타낸 것임(여러가지 변수 중 하나만을 확인한 것임)
fig = plt.figure(figsize=(8,6))
plt.scatter(x_test[:, 1], y_test, color='b', marker='o', s=30)
plt.plot(x_test[:, 1], y_hat, color='r')
plt.show()

### 결과값
# loss: 0.3213859288171691