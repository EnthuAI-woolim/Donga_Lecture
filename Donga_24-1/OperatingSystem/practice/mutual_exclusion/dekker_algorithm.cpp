#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>

#define THREAD_PER_NUM 50

void* func0(void* arg);
void* func1(void* arg);
void setNum();                     // 변수 초기화 함수
void printNum();                   // 숫자 출력 함수

bool flag[2] = { false, false };   // 플래그 변수
int turn = 0;                      // turn 변수

// 계산을 위한 변수
int num[100];                      // 1-100까지의 숫자 변수
int startIndex = 0;                // 시작 인덱스 변수
 
int main() {
  pthread_t tid1, tid2;

  setNum();

  pthread_create(&tid1, NULL, func0, NULL);
  pthread_create(&tid2, NULL, func1, NULL);

  pthread_join(tid1, NULL);
  pthread_join(tid2, NULL);

  return 0;
}

void setNum() {
  for (int i = 0; i < 100; i++)
    num[i] = 1+i;
}

void printNum() {
  int start = startIndex;
  int end = startIndex + THREAD_PER_NUM;
  int count = start / THREAD_PER_NUM + 1;

  printf("thread %d : \n", count);
  for (start; start < end; start++) {
    printf("%4d", num[start] * 3);
    startIndex++;
  }
  printf("\n");
}

void* func0(void* arg) {
  flag[0] = true;               
  while (flag[1] == true) {
    if (turn == 1) {
        flag[0] = false;
          
        while (turn == 1) {}
        flag[0] = true;
    }
  }

  printNum();

  turn = 1;
  flag[0] = false;

  return NULL;
}

void* func1(void* arg) {
  flag[1] = true;
  while (flag[0] == true) {
    if (turn == 0) {
        flag[1] = false;

        while (turn == 0) {}
        flag[1] = true;
    }
  }

  printNum();

  turn = 0;
  flag[1] = false;
  
  return NULL;
}

