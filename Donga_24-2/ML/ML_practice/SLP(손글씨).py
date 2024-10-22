### 패키지 선언
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader

### Dataset 선언
# Training dataset 다운로드
mnist_train = dataset.MNIST(root = "./", # 데이터셋을 저장할 위치
                            train = True,
                            transform = transform.ToTensor(),
                            download = True)
# Testing dataset 다운로드
mnist_test = dataset.MNIST(root = './',
                            train = False,
                            transform = transform.ToTensor(),
                            download = True)

### MNIST 데이터셋 형상 확인
import matplotlib.pyplot as plt
# print(len(mnist_train))     # training dataset 개수 확인

first_data = mnist_train[0]
# print(first_data[0].shape)  # 첫번째 data의 형상 확인
# print(first_data[1])        # 첫번째 data의 정답 확인

# plt.imshow(first_data[0][0,:,:], cmap='gray')
# plt.show()

first_img = first_data[0]
# print(first_img.shape)

# view(): 평탄화 시키는 함수 - SLP클래스 안에서 사용됨
first_img = first_img.view(-1, 28*28) # 이미지 평탄화 수행 2D -> 1D
# print(first_img.shape)

### Single Layer Perceptron 모델 정의
class SLP (nn.Module):
  def __init__ (self):
    super(SLP, self).__init__()
    # SLP의 입력노드는 784개, 출력노드는 10개
    # 참고: torch.nn.Linear = Fully connected layer
    self.fc1 = nn.Linear(in_features=784, out_features=10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, 28*28) # 이미지 평탄화
    y = self.relu(self.fc1(x))
    return y

'''
Hyper-parameters 지정
- 참고: torch.nn.CrossEntropyLoss()
   1. 예측 값들에 대해 자동으로 softmax 적용: 여러 개의 입력을 받아 각각의 확률 값으로 출력
   2. 정답 값과 예측 값을 이용해 cross entropy loss 측정
- small batch size: 자주 모델이 업데이트->모델이 더 잘 일반화됨. 하지만, 계산 시간이 김.
- big batch size: 적은 횟수로 업데이트->더 빠른 계산을 가능. 하지만, 일반화 성능이 떨어질 수 있음.
'''
batch_size = 32                                # 한번에 학습할 데이터의 개수
learning_rate = 0.1                            # 가중치를 얼마나 변경시킬지 정하는 상수
training_epochs = 40                           # 학습 횟수(1 Epoch: 전체 데이터셋에 대해 1회 학습)
loss_function = nn.CrossEntropyLoss()          # 학습 모델이 얼마나 잘못 예측하고 있는지는 표현하는 지표, nn에 접근가능

'''
network의 하위 파라미터들(weight, bias)
- network.fc.weight
- network.fc.weight.grad
- network.fc.bias
- network.fc.bias.grad
'''
network = SLP()
# Optimizer: SGD - Loss function을 최소로 만들기 위한 가중치, 편향을 찾는 알고리즘 - 파라미터 넘겨주는 걸로 정의됨
optimizer = torch.optim.SGD(network.parameters(), lr = learning_rate)

# Batch 단위 학습을 위해 DataLoader() 사용
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

### Perceptron 학습을 위한 반복문 선언
'''
- 6만개의 이미지를 한번에 넣기에는 너무 많음
- 그래서 배치 사이즈만큼 나눠서 한번에 32개의 이미지를 훈련시키는 과정을 총 1875번해서 6만개의 이미지를 훈련시킴
- training_epochs를 통해 위 과정을 40번 함
- 따라서 파라미터들이 업데이트 되는 횟수: 32 x 60000/32개 x 40 번
'''
for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = len(data_loader)

  for img, label in data_loader:

    # SLP클래스의 forward함수가 실행됨
    pred = network(img)                 # 입력 이미지에 대해 forward pass

    loss = loss_function(pred, label)   # 예측 값, 정답을 이용해 loss 계산

    # 기울기를 저장하는 network.fc.weight.grad 속성의 값을 0으로 설정
    # 기울기를 저장하는 network.fc.bias.grad 속성의 값을 0으로 설정
    # cf) 각 배치마다 따로 기울기를 계산해야되서, 안해주면 각 배치에 대해 누적됨
    optimizer.zero_grad()               # 파리미터들의 gradient 초기화(weight, bias)

    # 각 파라미터들의 grad속성에 편미분 통해 얻은 gradient를 저장
    loss.backward()                     # 모든 weight와 bias에 대해 편미분 값 계산

    # weight, bias 업데이트
    # w의 기울기가 음수라는 건 x축 기준으로 기울기가 0이 되는 x값보다 작으므로 -1을 곱해서 w에 특정비율만큼 더해주는 것임
    # w의 기울기가 양수라는 건 x축 기준으로 기울기가 0이 되는 x값보다 크므로 -1을 곱해서 w에 특정비율만큼 빼주는 것임
    # weight = weight − (learning_rate × weight.grad)
    # bias = bias − (learning_rate × bias.grad)
    optimizer.step()                    # 각 파라미터에 대해 저장된 기울기를 사용하여 파라미터를 업데이트

    avg_cost += loss / total_batch      # 모든 배치에 대한 평균 loss값 계산

  # print('Epoch: %d Loss = %f'%(epoch+1, avg_cost))

# print('Learning finished')
# 총 32 x 60000/32개 x 40 번의 파라미터(weight, bias) 업데이트

### 학습이 완료된 모델을 이용해 정답률 확인
with torch.no_grad(): # test에서는 PyTorch에서 자동 미분 기능을 비활성화
  img_test = mnist_test.data.float()
  label_test = mnist_test.targets

  prediction = network(img_test) # 전체 test data를 한번에 계산

  # 예측 값이 가장 높은 숫자(0~9)와 정답데이터가 일치한 지 확인
  correct_prediction = torch.argmax(prediction, 1) == label_test
  accuracy = correct_prediction.float().mean()
  # print('Accuracy:', accuracy.item())

### Weight parameter 저장하기/불러오기
# torch.save(network.state_dict(), "./slp_mnist.pth")

### 저장된 파라미터 로드
network.load_state_dict(torch.load("./slp_mnist.pth"))

### 파라미터 출력
# for name, param in network.named_parameters():
#     print(f"Parameter name: {name}")
#     print(f"Parameter value: {param.data}")
#     print(f"Gradient: {param.grad}")
#     print()  # 줄 바꿈

### 결과값
'''
1.
- self.fc1 = nn.Linear(in_features=784, out_features=10)
- batch_size = 100
- training_epochs = 15
> Accuracy: 0.8906999826431274
'''
