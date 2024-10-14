### 패키지 선언
import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader

# ### Dataset 선언
# # Training dataset 다운로드
# mnist_train = dataset.MNIST(root = "./", # 데이터셋을 저장할 위치
#                             train = True,
#                             transform = transform.ToTensor(),
#                             download = True)
# # Testing dataset 다운로드
# mnist_test = dataset.MNIST(root = "./",
#                             train = False,
#                             transform = transform.ToTensor(),
#                             download = True)
#
# ### MNIST 데이터셋 형상 확인
# import matplotlib.pyplot as plt
# # print(len(mnist_train))     # training dataset 개수 확인
#
# first_data = mnist_train[0]
# # print(first_data[0].shape)  # 첫번째 data의 형상 확인
# # print(first_data[1])        # 첫번째 data의 정답 확인
#
# # plt.imshow(first_data[0][0,:,:], cmap='gray')
# # plt.show()
#
# first_img = first_data[0]
# # print(first_img.shape)
#
# first_img = first_img.view(-1, 28*28) # 이미지 평탄화 수행 2D -> 1D
# # print(first_img.shape)
#
# ### Multi Layer Perceptron 모델 정의
# class MLP (nn.Module):
#   def __init__ (self):
#     super(MLP, self).__init__()
#     # MLP의 입력은 784개
#     # 1번째 layer 출력 10개
#     # 2번째 layer 출력 10개
#     # 참고: torch.nn.Linear = Fully connected layer
#     self.fc1 = nn.Linear(in_features=784, out_features=100)
#     self.fc2 = nn.Linear(in_features=100, out_features=10)
#     self.sigmoid = nn.Sigmoid()
#
#   def forward(self, x):
#     x = x.view(-1, 28*28) # 이미지 평탄화
#     y = self.sigmoid(self.fc1(x))
#     y = self.fc2(y)
#     return y
#
# ### Hyper-parameters 지정
# # 참고: torch.nn.CrossEntropyLoss()
# #   1. 예측 값들에 대해 자동으로 softmax 적용: 여러 개의 입력을 받아 각각의 확률 값으로 출력
# #   2. 정답 값과 예측 값을 이용해 cross entropy loss 측정
# batch_size = 100
# learning_rate = 0.1
# training_epochs = 15 # 학습 횟수
# loss_function = nn.CrossEntropyLoss() # CEE
# network = MLP()
# optimizer = torch.optim.SGD(network.parameters(), lr = learning_rate) # Optimizer: SGD
#
# data_loader = DataLoader(dataset=mnist_train,
#                          batch_size=batch_size,
#                          shuffle=True,
#                          drop_last=True)
#
# ### Perceptron 학습을 위한 반복문 선언
# for epoch in range(training_epochs):
#   avg_cost = 0
#   total_batch = len(data_loader)
#
#   for img, label in data_loader:
#     pred = network(img)                 # 입력 이미지에 대해 forward pass
#
#     loss = loss_function(pred, label)   # 예측 값, 정답을 이용해 loss 계산
#     optimizer.zero_grad()               # gradient 초기화
#     loss.backward()                     # 모든 weight에 대해 편미분 값 계산
#     optimizer.step()                    # 파라미터 업데이트
#
#     avg_cost += loss / total_batch      # 모든 배치에 대한 평균 loss값 계산
#
#   print('Epoch: %d Loss = %f'%(epoch+1, avg_cost))
#
# print('Learning finished')
#
# ### 학습이 완료된 모델을 이용해 정답률 확인
# network = network.to('cpu')
# with torch.no_grad(): # test에서는 기울기 계산 제외
#
#   img_test = mnist_test.data.float()
#   label_test = mnist_test.targets
#
#   prediction = network(img_test) # 전체 test data를 한번에 계산
#
#   # 예측 값이 가장 높은 숫자(0~9)와 정답데이터가 일치한 지 확인
#   correct_prediction = torch.argmax(prediction, 1) == label_test
#   accuracy = correct_prediction.float().mean()
#   print('Accuracy:', accuracy.item())
#
#   # ppt보고 다시해보기 - mnist_test.data[0] 때문에
#   # prediction_num = torch.argmax(prediction, 1)
#   # print(prediction)
#   # print(prediction_num)
#   # plt.imshow(mnist_test.data[0], cmap="gray")
#   # plt.show()
#
# ### Weight parameter 저장하기/불러오기
# # torch.save(network.state_dict(), "./mlp_mnist.pth")
#
# ### 저장된 파라미터 로드
# network.load_state_dict(torch.load("./slp_mnist.pth"))
#
# ### 파라미터 출력
# for name, param in network.named_parameters():
#     print(f"Parameter name: {name}")
#     print(f"Parameter value: {param.data}")
#     print(f"Gradient: {param.grad}")
#     print()  # 줄 바꿈