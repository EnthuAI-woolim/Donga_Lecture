#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <stdbool.h>

// 공유 변수
int flag[4] = {0, 0, 0, 0};    // idle == 0, want-in == 1, in-CS == 2
int turn = 0;
int num = 0;

// thread1 실행 함수
void *func1(void *args){
    /* 임계 영역 진입시도 1단계 */
    flag[0] = 1;
    while(turn != 0){
        if(flag[turn] == 0)
            turn = 0;
    }

    /* 임계 영역 진입시도 2단계 */
    flag[0] = 2;
    int j = 0;
    while((j < 4) && (j == 0 || flag[j] != 2)){
        j++;
    }
    /* 임계 영역 */
    for (num = 1; num <= 25; num++){
        printf("Thread 1: %d * 3 = %d\n", num, (num*3));
    }
    flag[0] = 0;
    /* 나머지 영역 */
    pthread_exit(NULL);   
}

// thread2 실행 함수
void *func2(void *args){
    /* 임계 영역 진입시도 1단계 */
    flag[1] = 1;
    while(turn != 1){
        if(flag[turn] == 0)
            turn = 1;
    }

    /* 임계 영역 진입시도 2단계 */
    flag[1] = 2;
    int j = 0;
    while((j < 4) && (j == 1 || flag[j] != 2)){
        j++;
    }
    /* 임계 영역 */
    for (num = 26; num <= 50; num++){
        printf("Thread 2: %d * 3 = %d\n", num, (num*3));
    }
    flag[1] = 0;
    /* 나머지 영역 */
    pthread_exit(NULL); 
}

// thread3 실행 함수
void *func3(void *args){
    /* 임계 영역 진입시도 1단계 */
    flag[2] = 1;
    while(turn != 2){
        if(flag[turn] == 0)
            turn = 2;
    }

    /* 임계 영역 진입시도 2단계 */
    flag[2] = 2;
    int j = 0;
    while((j < 4) && (j == 2 || flag[j] != 2)){
        j++;
    }
    /* 임계 영역 */
    for (num = 51; num <= 75; num++){
        printf("Thread 3: %d * 3 = %d\n", num, (num*3));
    }
    flag[2] = 0;
    /* 나머지 영역 */
    pthread_exit(NULL); 
}

// thread4 실행 함수
void *func4(void *args){
    /* 임계 영역 진입시도 1단계 */
    flag[3] = 1;
    while(turn != 3){
        if(flag[turn] == 0)
            turn = 3;
    }

    /* 임계 영역 진입시도 2단계 */
    flag[3] = 2;
    int j = 0;
    while((j < 4) && (j == 3 || flag[j] != 2)){
        j++;
    }
    /* 임계 영역 */
    for (num = 76; num <= 100; num++){
        printf("Thread 4: %d * 3 = %d\n", num, (num*3));
    }
    flag[3] = 0;
    /* 나머지 영역 */
    pthread_exit(NULL); 
}

int main(){
    pthread_t tid1, tid2, tid3, tid4;

    if(pthread_create(&tid1, NULL, func1, NULL)  != 0){
        fprintf(stderr, "thread1 create error\n");
    }

    if(pthread_create(&tid2, NULL, func2, NULL)  != 0){
        fprintf(stderr, "thread2 create error\n");
    }

    if(pthread_create(&tid3, NULL, func3, NULL)  != 0){
        fprintf(stderr, "thread3 create error\n");
    }

    if(pthread_create(&tid4, NULL, func4, NULL)  != 0){
        fprintf(stderr, "thread4 create error\n");
    }

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);
    pthread_join(tid3, NULL);
    pthread_join(tid4, NULL);
}