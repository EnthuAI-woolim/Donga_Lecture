import torch
import numpy as np
import torch.nn as nn
import torchvision.datasets as dataset
import torchvision.transforms as transform
from torch.utils.data import DataLoader

# Training dataset 다운로드
cifar10_train = dataset.CIFAR10(root = './', # 데이터셋을 저장할 위치
                            train = True,
                            transform = transform.ToTensor(),
                            download = True)
# Testing dataset 다운로드
cifar10_test = dataset.CIFAR10(root = './',
                            train = False,
                            transform = transform.ToTensor(),
                            download = True)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 1024, bias=True)
        self.fc2 = nn.Linear(1024, 512, bias=True)
        self.fc3 = nn.Linear(512, 256, bias=True)
        self.fc4 = nn.Linear(256, 128, bias=True)
        self.fc5 = nn.Linear(128, 10, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 3072) # 이미지 평탄화
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        y = self.fc5(x)
        return y

print(cifar10_train.data[0])

batch_size = 32
learning_rate = 0.1
training_epochs = 20
loss_function = nn.CrossEntropyLoss()
network = MLP()
optimizer = torch.optim.SGD(network.parameters(), lr = learning_rate)

data_loader = DataLoader(dataset=cifar10_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

# network = network.to("cuda:0")  # gpu메모리 사용
network = network.to("cpu")   # cpu메모리 사용

network.train()
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)


    for img, label in data_loader:
        # img = img.to("cuda:0")
        # label = label.to("cuda:0")
        img = img.to("cpu")
        label = label.to("cpu")

        pred = network(img)

        loss = loss_function(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_cost += loss / total_batch
    print('Epoch: %d, lr = %f, Loss = %f' %(epoch+1, optimizer.param_groups[0]['lr'], avg_cost))
print('Learning finished')

network = network.to("cpu")
network.eval()
with torch.no_grad(): # test에서는 기울기 계산 제외

  img_test = torch.tensor(np.transpose(cifar10_test.data,(0,3,1,2))) / 255.
  label_test = torch.tensor(cifar10_test.targets)

  prediction = network(img_test) # 전체 test data를 한번에 계산

  correct_prediction = torch.argmax(prediction, 1) == label_test
  accuracy = correct_prediction.float().mean()
  print('Accuracy:', accuracy.item())

## 결과값
'''
1.
- 함수 정의
  - self.fc1 = nn.Linear(3072, 1024, bias=True)
  - self.fc2 = nn.Linear(1024, 512, bias=True)
  - self.fc3 = nn.Linear(512, 256, bias=True)
  - self.fc4 = nn.Linear(256, 128, bias=True)
  - self.fc5 = nn.Linear(128, 10, bias=True)
  - self.relu = nn.ReLU()
- batch_size = 64
- learning_rate = 0.1
- training_epochs = 30
> Accuracy: 0.5103999972343445

2.
- 함수 정의
  - self.fc1 = nn.Linear(3072, 4096, bias=True)
  - self.fc2 = nn.Linear(4096, 1024, bias=True)
  - self.fc3 = nn.Linear(1024, 256, bias=True)
  - self.fc4 = nn.Linear(256, 10, bias=True)
  - self.relu = nn.ReLU()
- batch_size = 100
- learning_rate = 0.1
- training_epochs = 30
> Accuracy: 0.5407000184059143
'''