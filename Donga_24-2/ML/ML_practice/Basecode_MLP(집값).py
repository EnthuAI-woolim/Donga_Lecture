import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Dataset 다운로드
# !wget "https://dongaackr-my.sharepoint.com/:x:/g/personal/sjkim_donga_ac_kr/ET2udlQfsxRAsvnlEtgzfi0B3HAMAmqP_Y2WRsbYrTvYaA?e=t6809f&download=1" -O kc_house_data.csv

# Load dataset file
data = pd.read_csv('kc_house_data.csv')

# 예측 수행에 불필요한 열 삭제
new_data = data.drop(['id', 'date'], axis=1)

# 값이 존재하지 않는 행 삭제
new_data = new_data.dropna()

# 데이터셋 가우시안 정규화
data_normalized = (new_data - new_data.mean()) / new_data.std()

# 각 데이터에 대한 시각화
fixed_column = 'price'
used_columns = data_normalized.columns.drop(fixed_column)

num_vars = len(used_columns)
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

# 예측에 사용할 데이터들에 대한 2차원 행렬 변환
x = np.array(data_normalized[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']])
y = np.array(data_normalized[['price']])

# Train dataset / Test dataset 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)

# Train dataset 형상 확인
print(x_train.shape)
print(y_train.shape)

class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(16, 16, bias=True)
    self.fc2 = nn.Linear(16, 8, bias=True)
    self.fc3 = nn.Linear(8, 1, bias=True)
    self.sigmoid = nn.Sigmoid()
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    y = self.fc3(x)

    return y

batch_size = 64
learning_rate = 0.01
training_epochs = 50
loss_function = nn.MSELoss()
network = MLP()
optimizer = torch.optim.SGD(network.parameters(), lr = learning_rate)

network.train()

for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = len(x_train) // batch_size

  for batch in range(total_batch):
    input = torch.tensor(x_train[batch * batch_size : (batch * batch_size) + batch_size, :], dtype=torch.float32)
    label = torch.tensor(y_train[batch * batch_size : (batch * batch_size) + batch_size, :], dtype=torch.float32)

    pred = network(input)

    loss = loss_function(pred, label)
    optimizer.zero_grad() # gradient 초기화
    loss.backward()
    optimizer.step()

    avg_cost += loss / total_batch

  print('Epoch: %d Loss = %f'%(epoch+1, avg_cost))

print('Learning finished')

for param in network.parameters():
    print(param)

# Test dataset을 이용한 예측
input = torch.tensor(x_test, dtype=torch.float32)
network.eval()
with torch.no_grad(): # test에서는 기울기 계산 제외
    y_hat = network(input)
y_hat = y_hat.detach().numpy()

# 정답 데이터와 예측 데이터 간 차이 계산
def MSE (pred, label):
  error = pred - label
  mse = np.mean(error ** 2)
  return mse

print(MSE(y_hat, y_test))

# 시각화
fig = plt.figure(figsize=(8,6))
plt.scatter(x_test[:, 1], y_test, color='b', marker='o', s=30)
plt.plot(x_test[:, 1], y_hat, color='r')
plt.show()





