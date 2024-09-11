#include <semaphore.h>
#include <pthread.h>
#include <stdio.h>

// 공유 변수
sem_t semaphore;
int user = 1;
int num = 0;

// thread1 실행 함수
void *func1(void *args){
    // 임계영역에 아무도 없을 때
    if (user != 0){
        // tid1이 들어가고 user 값을 하나 줄여줌
        user--;
        // 임계영역 진입 (다른 thread들 대기하게 함)
        sem_wait(&semaphore);

        /* 임계 영역*/
        for (num = 1; num <= 25; num++){
            printf("Thread 1: %d * 3 = %d\n", num, (num*3));
        }
        // 후처리
        user++;
        // 임계 영역에서 나옴(대기 중인 thread 깨우려고 신호 보냄)
        sem_post(&semaphore);
    } 
    // 임계 영역에 누군가 있을 때
    else if(user == 0){
        while(user == 0){
            // 바쁜 대기
        }
        /* 임계 영역 */
        for (num = 1; num <= 25; num++){
            printf("Thread 1: %d * 3 = %d\n", num, (num*3));
        }
        // 후처리
        user++;
        // 임계 영역에서 나옴(대기 중인 thread 깨우려고 신호 보냄)
        sem_post(&semaphore);
    }
    pthread_exit(NULL);
}

// thread2 실행 함수
void *func2(void *args){
    // 임계영역에 아무도 없을 때
    if (user != 0){
        // tid2가 들어가고 user 값을 하나 줄여줌
        user--;
        // 임계영역 진입 (다른 thread들 대기하게 함)
        sem_wait(&semaphore);

        /* 임계 영역 */
        for (num = 26; num <= 50; num++){
            printf("Thread 2: %d * 3 = %d\n", num, (num*3));
        }
        // 후처리
        user++;
        // 임계 영역에서 나옴(대기 중인 thread 깨우려고 신호 보냄)
        sem_post(&semaphore);
    } 
    // 임계 영역에 누군가 있을 때
    else if(user == 0){
        while(user == 0){
            // 바쁜 대기
        }
        /* 임계 영역 */
        for (num = 26; num <= 50; num++){
            printf("Thread 2: %d * 3 = %d\n", num, (num*3));
        }
        // 후처리
        user++;
        // 임계 영역에서 나옴(대기 중인 thread 깨우려고 신호 보냄)
        sem_post(&semaphore);
    }
    pthread_exit(NULL);
}

// thread3 실행 함수
void *func3(void *args){
    // 임계영역에 아무도 없을 때
    if (user != 0){
        // tid3이 들어가고 user 값을 하나 줄여줌
        user--;
        // 임계영역 진입 (다른 thread들 대기하게 함)
        sem_wait(&semaphore);

        /* 임계 영역 */
        for (num = 51; num <= 75; num++){
            printf("Thread 3: %d * 3 = %d\n", num, (num*3));
        }
        // 후처리
        user++;
        // 임계 영역에서 나옴(대기 중인 thread 깨우려고 신호 보냄)
        sem_post(&semaphore);
    }
    // 임계 영역에 누군가 있을 때
    else if(user == 0){
        while(user == 0){
            // 바쁜 대기
        }
        /* 임계 영역 */
        for (num = 51; num <= 75; num++){
            printf("Thread 3: %d * 3 = %d\n", num, (num*3));
        }
        // 후처리
        user++;
        // 임계 영역에서 나옴(대기 중인 thread 깨우려고 신호 보냄)
        sem_post(&semaphore);
    }
    pthread_exit(NULL);
}

// thread4 실행 함수
void *func4(void *args){
    // 임계영역에 아무도 없을 때
    if (user != 0){
        // tid4가 들어가고 user 값을 하나 줄여줌
        user--;
        // 임계영역 진입 (다른 thread들 대기하게 함)
        sem_wait(&semaphore);

        /* 임계 영역 */
        for (num = 76; num <= 100; num++){
            printf("Thread 4: %d * 3 = %d\n", num, (num*3));
        }
        // 후처리
        user++;
        // 임계 영역에서 나옴(대기 중인 thread 깨우려고 신호 보냄)
        sem_post(&semaphore);
    }
    // 임계 영역에 누군가 있을 때
    else if(user == 0){
        while(user == 0){
            // 바쁜 대기
        }
        /* 임계 영역 */
        for (num = 76; num <= 100; num++){
            printf("Thread 4: %d * 3 = %d\n", num, (num*3));
        }
        // 후처리
        user++;
        // 임계 영역에서 나옴(대기 중인 thread 깨우려고 신호 보냄)
        sem_post(&semaphore);
    }
    pthread_exit(NULL);
}

int main(){
    pthread_t tid1, tid2, tid3, tid4;

    // 세마포어 객체 초기화(임계영역에는 1개의 thread만 들어갈 수 있음)
    sem_init(&semaphore, 0, 1);

    // 스레드 생성
    printf("Semaphore test Start!\n");
    pthread_create(&tid1, NULL, func1, NULL);
    pthread_create(&tid2, NULL, func2, NULL);
    pthread_create(&tid3, NULL, func3, NULL);
    pthread_create(&tid4, NULL, func4, NULL);

    // 스레드 조인
    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);
    pthread_join(tid3, NULL);
    pthread_join(tid4, NULL);

    printf("모든 스레드가 계산을 끝냈습니다.\n");

    // 세마포어 객체 소멸
    sem_destroy(&semaphore);

    return 0;
}