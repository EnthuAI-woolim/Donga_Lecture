import pandas as pd
import numpy as np


#%%
# #%% : 셸을 나눔 
# Ctrl + Enter :  현재 셸만 실행(주피터와 같이 해당 셸만 실행 가능)
print(1)
#%%
print(2)
a=[10, 20, 30]

#%%
### 판다스 개요

## 판다스 Series 객체 생성

# 코드 2-1
age = pd.Series([25, 34, 19, 45, 60])
age
type(age)

data = ['spring', 'summer', 'fall', 'winter']
season = pd.Series(data)
season
type(season)
season.iloc[2]

#%%
## 판다스 DataFrame 객체 생성

# 코드 2-2
score = pd.DataFrame([[85, 96],
                      [73, 69],
                      [78, 50]])
score
type(score)

score.index
score.columns

score.iloc[1, 1]

#%%
## numpy 배열 <-> pandas 배열
# numpy의 1차원 배열 <-> pandas의 Series

w_np = np.array([65.4,71.3, np.nan, 57.8])  # numpy 1차원 배열
weight = pd.Series(w_np)                    # numpy 배열을 pandas Series로
weight

# 판다스 시리즈를 넘파이로 변환
w_np2 = pd.Index.to_numpy(weight)           # pandas Series를 numpy 배열로
w_np2

#%%
# 넘파이 2차원 배열로부터 데이터프레임 생성
s_np = np.array([[85, 96],
                 [73, 69],
                 [78, 50]])
s_np

score2 = pd.DataFrame(s_np)                 # 넘파이 배열을 데이터프레임으로
score2                                      # 판다스 데이터프레임

# 데이터프레임을 넘파이 2차원 배열로 변환
score_np = score2.to_numpy()                # 데이터프레임을 넘파이 배열로
score_np                                    # 넘파이 2차원 배열

#%%
## 행과 열에 레이블을 부여하는 방법

# 코드 2-3
# 시리즈에 레이블 부여
age = pd.Series([25, 34, 19, 46])
age.index = ['John','Jane','Tom', 'Luke']   # 행에 레이블 부여
print(age.iloc[2])                          # 절대위치에 의한 인덱싱
print(age.loc['Tom'])                       # 레이블에 의한 인덱싱

# 데이터프레임에 레이블 부여
score = pd.DataFrame([[85, 96, 40, 95],
                      [73, 69, 45, 80],
                      [78, 50, 60, 90]])

score                                       # 레이블 부여 전
score.index = ['John','Jane','Tom']         # 행에 레이블 부여
score.columns = ['KOR','ENG','MATH','SCI']  # 열에 레이블 부여
score

# 인덱싱으로 데이터 하나만 가져오기
print(age.iloc[2, 1])                          # 절대위치에 의한 인덱싱
print(age.loc['Tom', 'ENG'])                       # 레이블에 의한 인덱싱

# 레이블(컬럼)로 데이터들 가져오기
score.KOR
# type(score.KOR)               # 타입: Series

score['KOR']
# type(score['KOR'])            # 타입: Series

score[['KOR', 'ENG']]
# type(score[['KOR', 'ENG']])   # 타입: DataFrame

#%%

# 코드 2-4
# !! 인덱스는 중복을 허용하지 않지만, 컬럼명은 중복을 허용한다.
# 인덱스는 시스템에 의해서 자동 관리고, 레이블 인덱스는 사용자가 임의로 지정할 수 있으며, 중복이 존재할 수 있다.
age = pd.Series([25, 34, 19, 45, 60])
age.index = ['John','Jane','Tom','Micle','Tom']
age

age.iloc[3]
age.loc['Tom']

#%%

# 코드 2-5
population = pd.Series([523, 675, 690, 720, 800])
population.index = [10, 20, 30, 40, 50]
population

# population.iloc[20]       # 에러발생
population.loc[20]