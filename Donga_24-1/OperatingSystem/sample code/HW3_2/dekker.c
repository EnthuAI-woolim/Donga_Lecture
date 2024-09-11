#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>

// 공유 변수
bool flag[2] = {false, false};
int turn = 0;
int num = 0;

// thread1 실행 함수
void *func1(void *args){
    // tid1이 임계영역에 들어갈 의사가 있음으로 바꿈
    flag[0] = true;
    // tid2가 임계영역에 들어갈 의사가 있을 때까지 반복
    while(flag[1] == true){
        // tid2 차례일 때
        if(turn == 1){
            // tid1은 의사 없음으로 바꿈
            flag[0] = false;
            while(turn == 1){
                // 바쁜 대기
            }
            // tid1 flag를 true로 다시 바꿔줌.
            flag[0] = true;
        }
    }
    
    /* 임계 영역 */
    for (num = 1; num <= 50; num++){
        printf("Thread 1: %d * 3 = %d\n", num, (num*3));
    }
    // 후처리
    turn = 1;
    flag[0] = false;
    /* 나머지 영역 */
    pthread_exit(NULL);   
}

// thread2 실행 함수
void *func2(void *args){
    // tid2가 임계영역에 들어갈 의사가 있음으로 바꿈
    flag[1] = true;
    // tid1이 임계영역에 들어갈 의사가 있을 때까지 반복
    while(flag[0] == true){
        // tid1 차례일 때
        if(turn == 0){
            // tid2은 의사 없음으로 바꿈
            flag[1] = false;
            while(turn == 0){
                // 바쁜 대기
            }
            // tid2 flag를 true로 다시 바꿔줌.
            flag[1] = true;
        }
    }

    /* 임계 영역 */
    for (num = 51; num <= 100; num++){
        printf("Thread 2: %d * 3 = %d\n", num, (num*3));
    }
    // 후처리
    turn = 0;
    flag[1] = false;
    /* 나머지 영역 */
    pthread_exit(NULL);
}

int main(){
    pthread_t tid1, tid2;

    // thread1 생성
    if(pthread_create(&tid1, NULL, func1, NULL)  != 0){
        fprintf(stderr, "thread1 create error\n");
    }
    // thread2 생성
    if(pthread_create(&tid2, NULL, func2, NULL)  != 0){
        fprintf(stderr, "thread2 create error\n");
    }
    // thread 자원 회수
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);
}