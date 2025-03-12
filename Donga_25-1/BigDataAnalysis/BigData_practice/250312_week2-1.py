import pandas as pd
import numpy as np

#%%
# 데이터프레임에 레이블 부여
score = pd.DataFrame([[85, 96, 40, 95],
                      [73, 69, 45, 80],
                      [78, 50, 60, 90]])

score                                       # 레이블 부여 전
score.index = ['John','Jane','Tom']         # 행에 레이블 부여
score.columns = ['KOR','ENG','MATH','SCI']  # 열에 레이블 부여
score     

#%%
# 인덱싱으로 데이터 하나만 가져오기
# score.iloc[2,1] # 절대위치에 의한 인덱싱
# score.loc['Tom','ENG'] # 레이블에 의한 인덱싱                                  # 레이블 부여 후

# 레이블(컬럼)로 데이터들 가져오기
score.KOR
# type(score.KOR)               # 타입: Series

score['KOR']
# type(score['KOR'])            # 타입: Series

score[['KOR', 'ENG']]
# type(score[['KOR', 'ENG']])   # 타입: DataFrame



#%%
# 코드 2-4
# !! 인덱스명은 중복을 허용하지 않지만, 컬럼명은 중복을 허용한다.
age = pd.Series([25, 34, 19, 45, 60])
age.index = ['John','Jane','Tom','Micle','Tom']
age

age.iloc[3]
age.loc['Tom']

#%%
### 1. 시리즈 정보 확인하기
# 데이터 입력
#code1
temp = pd.Series([-0.8, -0.1, 7.7, 13.8, 18.0, 22.4,
                  25.9, 25.3, 21.0, 14.0, 9.6, -1.4])
temp  # temp의 내용 확인

temp.size # 배열의 크기(행의 갯수)
len(temp) # 배열의 크기(행의 갯수)

# ⭐⭐⭐⭐⭐
temp.shape # 배열의 형태
temp.dtype # 배열의 자료형
#%%

# 월 이름을 레이블 인덱스로 지정하기
temp.index  # 인덱스 내용 확인
temp.index = ['1월', '2월', '3월', '4월',  # 월 이름을 인덱스로 지정
              '5월', '6월', '7월', '8월',
              '9월', '10월', '11월', '12월']
temp  # temp의 내용 확인

temp.iloc[2]

temp.loc[['4월', '6월', '8월']] # 불연속
temp.loc['6월':'9월'] # 연속적   

temp.loc[temp >= 15] # 값이 15 이상인 시리즈(행)만 출력됨
type(temp.loc[temp >= 15]) # 시리즈임

temp.loc[(temp >= 15) & (temp < 25)] # 조건이 두개 이상일 경우 & 하나만 사용하면 비트연산을 함(1 or 0)
temp.loc[(temp < 15) | (temp >= 25)]

march = temp.loc['3월']
temp.loc[temp < march]

# where()는 메소드로 시리지에만 사용가능
temp.where(temp >= 15) # 조건에 맞지 않는 시리지의 값은 NaN로 처리됨(결측치)
temp.where(temp >= 15).dropna()

### 2. 시리즈 객체의 산술 연산
temp + 1

# 통계 관련 메서드
temp.sum()
temp.abs()
temp.describe()
score.describe()

upperMean = [temp.where(temp >= temp.mean()).dropna().index]
temp.loc[temp >= temp.mean()].index

upperMean

#%%
### 3. 시리즈 객체 내용 변경
salary = pd.Series([20, 15, 18, 30])  # 레이블 인덱스가 없는 시리즈 객체
score = pd.Series([75, 80, 90, 60],
                  index=['KOR', 'ENG', 'MATH', 'SOC'])
salary
score

score.loc['PHY'] = 50
score.shape

# 값의 추가(레이블 인덱스가 없는 경우)
next_idx = salary.size
salary.loc[next_idx] = 33
salary

## _append() 메서드를 이용한 추가
new = pd.Series({'MUS':95})
score._append(new)          # 값 변경 없음
score = score._append(new)  # 값 변경
score

salary._append(pd.Series([66]), ignore_index=True)          # 값 변경 없음
salary = salary._append(pd.Series([66]), ignore_index=True)  # 값 변경

# cf) score.drop('PHY', inplace=True) : inplace=True는 원본에 바로 업데이트
score.drop('PHY', inplace=True) # 값 변경
score

## 시리즈 객체 복사
score_1 = [1, 2]
score_2 = score_1       # score_1, score_2는 같은 테이블을 바라보고 있음

score_3 = score_1.copy()    # 값이 같은 별개의 테이블 생성

