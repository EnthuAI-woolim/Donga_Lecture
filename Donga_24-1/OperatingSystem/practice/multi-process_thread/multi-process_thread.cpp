#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>
#include <pthread.h>

#define PROCESS_NUM 4
#define THREAD_NUM 16

void settingStartNum();
void* multiply(void* ranges);
int genProcessThread();

// 처음 숫자, 마지막 숫자, 곱할 숫자를 저장할 2차배열
int ranges[16][3];

int main() {
  clock_t start_time = clock();

  settingStartNum();
  genProcessThread();

  clock_t end_time = clock(); 
  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("프로세스 종료까지 소요된 시간: %.7f 초\n", elapsed_time);
    
  return 0;
}

// 시작 숫자 세팅 함수
void settingStartNum() {
  // 2차 배열 초기화
  for (int i = 0; i < THREAD_NUM; i++) {
    ranges[i][0] = (i % 4 == 0) ? 1 : (i % 4 == 1) ? 26 : (i % 4 == 2) ? 51 : 76;
    ranges[i][1] = (i % 4 == 0) ? 25 : (i % 4 == 1) ? 50 : (i % 4 == 2) ? 75 : 100;
    ranges[i][2] = (i < 4) ? 3 : (i < 8) ? 5 : (i < 12) ? 7 : 9;      
  }
}

// 배열을 매개변수로 받아와 계산 처리
void* multiply(void* ranges) {
  int* range = (int*)ranges;
  int start = range[0];
  int end = range[1];
  int factor = range[2];

  for (int i = start; i <= end; i++) {
      printf("%5d ", i * factor);
  }
  printf("\n");
  return 0;
}

// fork 실행 함수
int genProcessThread() {
  pid_t pid1, pid2, pid3;
  pthread_t tid[THREAD_NUM];

  pid1 = fork();

  // 프로세스 오류 처리
  if (pid1 < 0) {
    printf("fork 실패\n");
    return 1;
  } 

  if (pid1 == 0) {
    pid3 = fork();

    // 프로세스 오류 처리
    if (pid2 < 0) {
      printf("fork 실패\n");
      return 1;
    }

    if (pid2 == 0) {
      // 프로세스 오류 처리
      if (pid2 < 0) {
      printf("fork 실패\n");
      return 1;
      }

      printf("Process 1\n");
       // 스레드 생성 및 실행
      for (int i = 0; i < 4; i++) {
        printf("Thread %d : %d(%d) ~ %d(%d)\n", i+1, ranges[i][0]*ranges[i][2], ranges[i][0], ranges[i][1]*ranges[i][2], ranges[i][1]);
        int result = pthread_create(&tid[i], NULL, multiply, (void*)ranges[i]);
        if (result != 0) {
            fprintf(stderr, "스레드 생성 실패\n");
            return 1;
        }
      }
      // for문을 통해 각 스레드의 종료를 기다림
      for (int i = 0; i < 4; i++)
        pthread_join(tid[i], NULL);
      
    } else {

      printf("Process 2\n");
      // 스레드 생성 및 실행
      for (int i = 4; i < 8; i++) {
        printf("Thread %d : %d(%d) ~ %d(%d)\n", i+1, ranges[i][0]*ranges[i][2], ranges[i][0], ranges[i][1]*ranges[i][2], ranges[i][1]);
        int result = pthread_create(&tid[i], NULL, multiply, (void*)ranges[i]);
        if (result != 0) {
            fprintf(stderr, "스레드 생성 실패\n");
            return 1;
        }
      }
      // for문을 통해 각 스레드의 종료를 기다림
      for (int i = 4; i < 8; i++)
        pthread_join(tid[i], NULL);
      
    }
  } else {
    pid3 = fork();

    // 프로세스 오류 처리
    if (pid3 < 0) {
      printf("fork 실패\n");
      return 1;
    }

    if (pid3 == 0) {
      // 프로세스 오류 처리
      if (pid3 < 0) {
        printf("fork 실패\n");
        return 1;
      }

      printf("Process 3\n");
      // 스레드 생성 및 실행
      for (int i = 8; i < 12; i++) {
        printf("Thread %d : %d(%d) ~ %d(%d)\n", i+1, ranges[i][0]*ranges[i][2], ranges[i][0], ranges[i][1]*ranges[i][2], ranges[i][1]);
        int result = pthread_create(&tid[i], NULL, multiply, (void*)ranges[i]);
        if (result != 0) {
            fprintf(stderr, "스레드 생성 실패\n");
            return 1;
        }
      }

      // for문을 통해 각 스레드의 종료를 기다림
      for (int i = 8; i < 12; i++)
        pthread_join(tid[i], NULL);
    
    } else {

      printf("Process 4\n");
      // 스레드 생성 및 실행
      for (int i = 12; i < 16; i++) {
        printf("Thread %d : %d(%d) ~ %d(%d)\n", i+1, ranges[i][0]*ranges[i][2], ranges[i][0], ranges[i][1]*ranges[i][2], ranges[i][1]);
        int result = pthread_create(&tid[i], NULL, multiply, (void*)ranges[i]);
        if (result != 0) {
            fprintf(stderr, "스레드 생성 실패\n");
            return 1;
        }
      }

      // for문을 통해 각 스레드의 종료를 기다림
      for (int i = 12; i < 16; i++)
        pthread_join(tid[i], NULL);
    }
  }
  
  // 모든 자식 프로세스가 완료될 때까지 대기
  for (int i = 0; i < PROCESS_NUM-1; i++) 
    wait(NULL);

  return 0;
}





