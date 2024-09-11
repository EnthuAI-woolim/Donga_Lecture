#include <semaphore.h>
#include <pthread.h>
#include <stdio.h>

sem_t semaphore;
int num = 4;

void* threadF0(void* arg) {
    sem_wait(&semaphore); // wait 함수로 queue에 넣는다. 그리고 block()
    printf("<Thread 0> STRT.\n");
    
    for (int i = 1; i <= 25; i++) {
        printf("%d ", i * 3);
    }
    printf("\n<Thread 0> END.\n");
    sem_post(&semaphore); // sign 함수는 깨워서 실행시키는 함수
    return NULL;
}

void* threadF1(void* arg) {
    sem_wait(&semaphore); 
    printf("<Thread 1> STRT.\n");
    for (int i = 26; i <= 50; i++) {
        printf("%d ", i * 3);
    }
    printf("\n<Thread 1> END.\n");
    sem_post(&semaphore);
    return NULL;
}

void* threadF2(void* arg) {
    sem_wait(&semaphore);
    printf("<Thread 2> STRT.\n");
    for (int i = 51; i <= 75; i++) {
        printf("%d ", i * 3);
    }
    printf("\n<Thread 2> END.\n");
    sem_post(&semaphore);
    return NULL;
}

void* threadF3(void* arg) {
    sem_wait(&semaphore);
    printf("<Thread 3> STRT.\n");
    for (int i = 76; i <= 100; i++) {
        printf("%d ", i * 3);
    }
    printf("\n<Thread 3> END.\n");
    sem_post(&semaphore);
    return NULL;
}

int main() {
    pthread_t thread0, thread1, thread2, thread3;
    sem_init(&semaphore, 0, 1); // 1이어야 상호배제
    printf("<START>\n");

    pthread_create(&thread0, NULL, threadF0, NULL);
    pthread_create(&thread1, NULL, threadF1, NULL);
    pthread_create(&thread2, NULL, threadF2, NULL);
    pthread_create(&thread3, NULL, threadF3, NULL);

    pthread_join(thread0, NULL);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    printf("<END>\n");
    sem_destroy(&semaphore); // Destroy the semaphore
    return 0;
}
