### import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# ### Load datasets
# # Load dataset file
# data = pd.read_csv('Data/kc_house_data.csv', usecols=['price', 'sqft_living'])
# # print(data)
#
# # Numpy 배열로 전환
# data_np = np.array(data)
#
# x = data_np[:, 1]          # sqft_living
# y = data_np[:, 0]          # price
#
# # Dataset 정규화
# x_mean = np.mean(x)
# y_mean = np.mean(y)
# x_std = np.std(x)
# y_std = np.std(y)
#
# x = (x-x_mean)/x_std
# y = (y-y_mean)/y_std
#
# # 2차원 행렬 변환
# x = np.expand_dims(x, 1)
# y = np.expand_dims(y, 1)
#
# # Train dataset / Test dataset 분할
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
#
# # print(x_train.shape)
# # print(y_train.shape)
#
# # 시각화
# # fig = plt.figure(figsize=(8,6))
# # plt.scatter(x_test, y_test, color='b', marker='o', s=30)
# # plt.show()
#
# ### Least Square Method
# # θ = (X^T⋅X)^−1(X^T⋅Y) - 노트 4주차_2차시에 행렬θ 유도 과정 정리해 놓음
# def LSM(x, y):
#   # 행렬 X에 bias 열 추가
#   bias = np.ones_like(x)                  # x 행렬과 똑같은 크기의 1로 채워진 행렬 생성
#   X = np.hstack([x, bias])                # 행렬 가로 쌓기
#   # x = np.array([[-3], [-1], [1], [3]])
#   # y = np.array([[-1], [-1], [3], [3]])
#
#   # X transepose 생성 - 전치 행렬
#   XT = X.T
#
#   # X^T * X 생성 - 행렬 곱
#   XTX = np.dot(XT, X)
#
#   # (X^T * X)^-1 생성 - 역 행렬
#   XTX_inverse = np.linalg.inv(XTX)
#
#   # X^T * Y 생성 - 행렬 곱
#   XTY = np.dot(XT, y)
#
#   # theta 계산 - 행렬 곱
#   theta = np.dot(XTX_inverse, XTY)
#
#   return theta[0], theta[1]       # theta[0]: weight, theta[1]: bias
#
# w, b = LSM(x_train, y_train)
# # print(w, b)
#
# ### Prediction
# # Test dataset을 이용한 예측
# y_ = []
# for i in x_test:
#   y_.append(i*w + b)
#
# # 시각화
# # fig = plt.figure(figsize=(8,6))
# # plt.scatter(x_test, y_test, color='b', marker='o', s=30)
# # plt.plot(x_test, y_, color='r')
# # plt.show()
#
# ### Gradient Decent Method
# def GDM(x, y):
#   # 하이퍼 파라미터 설정
#   learning_rate = 0.1
#   n_iters = 100
#
#   # w, b 초기값 설정
#   w = 0
#   b = 0
#
#   # 행렬 X에 bias 열 추가
#   bias = np.ones_like(x)
#   X = np.hstack([x, bias])
#
#   for i in range(n_iters):
#     # [[w],
#     #  [b]] 행렬 생성
#     theta = np.array([w, b])
#     theta = theta.reshape(2, 1)
#
#     # y_hat 계산
#     y_hat = np.dot(X, theta)
#
#     # dw, db 계산 - w, b에 대해 각각 편미분한 값 계산
#     dw = 2/x.shape[0] * sum((y - y_hat) * -x)
#     db = 2/x.shape[0] * sum((y - y_hat) * -1)
#
#     # w, b 업데이트
#     w = w - learning_rate * dw
#     b = b - learning_rate * db
#
#     # print("w ", w, "\b", b)
#
#   return w, b
#
# w, b = GDM(x_train, y_train)
# # print(w, b)
#
# ### Prediction
# # Test dataset을 이용한 예측
# y_ = []
# for i in x_test:
#   y_.append(i*w + b)
#
# # 시각화
# # fig = plt.figure(figsize=(8,6))
# # plt.scatter(x_test, y_test, color='b', marker='o', s=30)
# # plt.plot(x_test, y_, color='r')
# # plt.show()
