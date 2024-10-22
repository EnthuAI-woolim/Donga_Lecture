import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

### Dataset 다운로드
# !wget "https://dongaackr-my.sharepoint.com/:x:/g/personal/sjkim_donga_ac_kr/ET2udlQfsxRAsvnlEtgzfi0B3HAMAmqP_Y2WRsbYrTvYaA?e=t6809f&download=1" -O kc_house_data.csv

# Load dataset file
data = pd.read_csv('kc_house_data.csv')

### Dataset 전처리
# 예측 수행에 불필요한 열 삭제
new_data = data.drop(['id', 'date'], axis=1)

# 값이 존재하지 않는 행 삭제
new_data = new_data.dropna()

# 데이터셋 가우시안 정규화
data_normalized = (new_data - new_data.mean()) / new_data.std()

### Dataset 특성 파악
# 각 데이터에 대한 시각화
fixed_column = 'price'
print(len(data_normalized))
used_columns = data_normalized.columns.drop(fixed_column) # price 컬럼 제거

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

# 데이터의 각 열에 대한 상관관계 확인
correlation_matrix = data_normalized.corr()

### 집 가격 예측에 사용할 데이터 지정
# 예측에 사용할 데이터들에 대한 2차원 행렬 변환
x = np.array(data_normalized[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']])
y = np.array(data_normalized[['price']])

# Train dataset / Test dataset 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Train dataset 형상 확인
print(x_train.shape)
print(y_train.shape)

### LSM
def LSM(x, y):
  bias = np.ones([x_train.shape[0], 1])
  X = np.hstack([x, bias])

  XT = X.T
  XTX = np.dot(XT, X)
  XTX_inverse = np.linalg.inv(XTX)
  XTY = np.dot(XT, y)
  theta = np.dot(XTX_inverse, XTY)

  return theta

theta = LSM(x_train, y_train)
print(theta)

### Predicition
# Test dataset을 이용한 예측
bias = np.ones([x_test.shape[0], 1])
x_input = np.hstack([x_test, bias])
y_hat = np.dot(x_input, theta)

# 정답 데이터와 예측 데이터 간 차이 계산
def MSE (pred, label):
  error = pred - label
  mse = np.mean(error ** 2)
  return mse

print(MSE(y_hat, y_test))

# # 시각화 - 입력 변수가 하나일 경우
# fig = plt.figure(figsize=(8,6))
# plt.scatter(x_test, y_test, color='b', marker='o', s=30)
# plt.plot(x_test, y_hat, color='r')
# plt.show()

# 시각화
fig = plt.figure(figsize=(8,6))
plt.scatter(x_test[:, 1], y_test, color='b', marker='o', s=30)
plt.plot(x_test[:, 1], y_hat, color='r')
plt.show()

### 결과값
# loss: 0.3213859288171691