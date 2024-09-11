#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

#define NUM_THREAD 4
#define THREAD_PER_NUM 25

void setNum();                     // 변수 초기화 함수
void printNum();                   // 숫자 출력 함수
void* func(void* arg);             // 각 스레드에서 실행할 함수

enum thread_statement { idle, want_in, in_cs }; 
volatile int flag[NUM_THREAD];     
volatile int turn = 0;             // turn 변수를 volatile로 선언  

// 계산을 위한 변수
int num[100];                      // 1-100까지의 숫자 변수
int startIndex[NUM_THREAD];        // 시작 인덱스 배열
 
int main() {
  pthread_t tids[NUM_THREAD];

  setNum();

 for (int i = 0; i < NUM_THREAD; i++) {
    pthread_create(&tids[i], NULL, func, (void*)(intptr_t)i);
  }

  for (int i = 0; i < NUM_THREAD; i++) {
    pthread_join(tids[i], NULL);
  }

  return 0;
}

void setNum() {
  for (int i = 0; i < NUM_THREAD; i++) 
    flag[i] = 0;

  for (int i = 0; i < 100; i++)
    num[i] = 1 + i;

  for (int i = 0; i < NUM_THREAD; i++)
    startIndex[i] = i * THREAD_PER_NUM;
}

void printNum(int thread_id) {
  int start = startIndex[thread_id];
  int end = startIndex[thread_id] + THREAD_PER_NUM;
  int count = startIndex[thread_id] / THREAD_PER_NUM + 1;

  printf("\nthread %d : ", count);
  for (start; start < end; start++) {
    printf("%4d", num[start] * 3);
  }
  printf("\n");
}

void* func(void* arg) {
  int thread_id = (intptr_t)arg;   
  int j;

  do {
    /* 임계 영역 진입시도 1단계 */
    flag[thread_id] = want_in;                          // 현재 스레드가 임계영역 진입 표시

    while (turn != thread_id) {                   // turn이 현재 스레드가 아닌 경우 반복, turn이 현재 스레드면 탈출 
      if (flag[turn] == idle) turn = thread_id;      // 다음 스레드가 임계영역에 없을 경우, turn을 현재 스레드로 설정
    }                                             

    /* 임계 영역 진입시도 2단계 */
    j = 0;                                        // 카운트 변수 j 초기화
    while ((j < NUM_THREAD) && (j == thread_id || flag[j] != in_cs)) {  
        j++;                                                    
    }
                            
    // index가 0부터 스레드 숫자 만큼 반복할 때, while문이 끝나는 동시에 do~while문을 빠져나올려면
    // while문이 스레드 숫자만큼(4번) 반복되어야 된다. (그래야 j => 4가 되면서 do~while문 탈출가능)
    // 그럴려면, 경우의 수가
    // 1. 4번중 3번이 j == thread => false가 나오고 flag[j] != 2 => true가 나와야되고, 
    // 2. 4번중 1번이 j == thread => true가 나오고 flag[j] != 2 => false가 나와야됨.
    // 즉, 다른 스레드의 플래그가 모두 0이고, 현재 스레드의 플래그만 2일 경우, do~while문 탈출 가능하다.

    // 만약, 경우의 수 1번에서 flag[j] != 2 => false일 경우가 나오면 (경우의 수 2번은 무조건 나오는 나옴)
    // while문을 바로 탈출하고, do~while문은 탈출하지 못해
    // 임계 지역 진입시도 1단계로 다시 돌아간다.

  } while (j >= NUM_THREAD);  // 다른 플래그가 모두 0인 경우에는 탈출 => 어떤 스레드도 임계영역에 들어가 있지 않은 경우
  
  flag[thread_id] = in_cs;    // 현재 스레드가 임계영역 진입 표시

  printNum(thread_id);

  flag[thread_id] = idle;              // 임계 영역을 빠져나옴

  return NULL;
}

