#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <cstdint>

#define NUM_THREAD 4
#define THREAD_PER_NUM 25

void setNum();                  // 변수 초기화 함수
void printNum(int startIndex);  // 숫자 출력 함수
void* func(void* index);        // 각 스레드에서 실행할 함수
void wait();
void signal();

sem_t semaphore;                // 세마포 변수
int mutex = 1;

// 계산을 위한 변수
int num[100];                   // 1-100까지의 숫자 변수

 
int main() {
  pthread_t tids[NUM_THREAD];
  sem_init(&semaphore, 0, NUM_THREAD);   // return :: 0 -> success, others -> fail

  setNum();

 for (int i = 0; i < NUM_THREAD; i++) {
    pthread_create(&tids[i], NULL, func, (void*)(intptr_t)i);
  }

  for (int i = 0; i < NUM_THREAD; i++) {
    pthread_join(tids[i], NULL);
  }

  sem_destroy(&semaphore); // 사용이 끝난 세마포어를 제거

  return 0;
}

void setNum() {
  for (int i = 0; i < 100; i++)
    num[i] = 1 + i;
}

void printNum(int startIndex) {
  int start = startIndex * THREAD_PER_NUM;
  int end = start + THREAD_PER_NUM;

  printf("\nthread %d :", startIndex + 1);
  for (start; start < end; start++) {
    printf("%4d", num[start] * 3);
  }
  printf("\n");
}

void wait() {
  while (mutex <= 0) {}
  mutex--;
}

void signal() {
  mutex++;
}

void* func(void* index) {
  int startIndex = (intptr_t)index;
  
  wait();

  printNum(startIndex);  

  signal();

  return NULL;
}

