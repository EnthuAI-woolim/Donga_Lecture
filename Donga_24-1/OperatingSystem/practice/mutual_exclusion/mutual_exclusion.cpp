#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#define THREAD_PER_NUM 25

// 코드상 서로 임계영역으로 진입을 양보
// 늦게 진입한 스레드가 임계영역에 먼저 진입하게 되는 구조

void setVariable();

// 스레드가 공유하는 데이터 
int threadId[2];      // 스레드의 id 변수
bool flag[2];         // 플래그 설정
int turn = 0;         // P0 = 0 / P1 = 1

// 계산을 위한 변수
int num[100];
int startNum = 1;

int main() {
  pthread_t tid[2];

  setVariable();

  // 스레드 S0
  flag[0] = true;               // S0의 임계영역 진입 표시
  while (flag[1] == true) {     // S1의 임계영역 진입 여부 확인
    if (turn == 1) {            // S1의 turn 확인
      flag[0] = false;          // 프래그를 재설정하여 P1에 진입 순서 양보
      while (turn == 1) {       // turn이 바뀔 때까지 대기

      }
      flag[0] = true;           // S0이 임계영역에 재진입 시도
    }
  }

  /* 임계 영역 */
  turn = 1;                     // S1에게 turn을 넘김
  flag[0] = false;              // S0의 임계영역 사용 완료
  /* 나머지 영역 */             // S0이 나머지 영역 수행

  
  // 스레드 S1
  flag[1] = true;               // S1의 임계영역 진입 표시
  while (flag[0] == true) {     // S0의 임계영역 진입 여부 확인
    if (turn == 0) {            // S0의 turn 확인
      flag[1] = false;          // 프래그를 재설정하여 S0에 진입 순서 양보
      while (turn == 0) {       // turn이 바뀔 때까지 대기

      }
      flag[1] = true;           // S1이 임계영역에 재진입 시도
    }
  }

  /* 임계 영역 */
  turn = 0;                     // S0에게 turn을 넘김
  flag[1] = false;              // S1의 임계영역 사용 완료
  /* 나머지 영역 */             // S1이 나머지 영역 수행

  
    
  return 0;
}

void setVariable() {
  threadId[0] = 0;
  threadId[1] = 1;
  
  flag[0] = false;
  flag[1] = false;

  for (int i = 0; i < 100: i++)
    num[i] = i+1;
}

void* thread_func(void* thread_id) {
  int threadId = *(int*)thread_id;
  
  // 스레드 S0
  flag[0] = true;               // S0의 임계영역 진입 표시
  while (flag[1] == true) {     // S1의 임계영역 진입 여부 확인
    if (turn == 1) {            // S1의 turn 확인
      flag[0] = false;          // 프래그를 재설정하여 P1에 진입 순서 양보
      while (turn == 1) {       // turn이 바뀔 때까지 대기

      }
      flag[0] = true;           // S0이 임계영역에 재진입 시도
    }
  }

  /* 임계 영역 */
  turn = 1;                     // S1에게 turn을 넘김
  flag[0] = false;              // S0의 임계영역 사용 완료
  /* 나머지 영역 */             // S0이 나머지 영역 수행
  
}
