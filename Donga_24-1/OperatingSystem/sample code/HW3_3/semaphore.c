#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

int num[100];
Sem_t semaphore;

void* multiply(void* arg) {
    int tid = *(int*)arg;
    int start = tid * 25;
    int end = start + 25 - 1;

    sem_wait(&semaphore);

    if (start < 100) {
        for (int i = start; i <= end && i < 100; i++) {
            printf("thread %d: %d * 3 = %d\n", tid, num[i], num[i] * 3);
        }
    }
    sem_post(&semaphore);

    pthread_exit(NULL);
}

int main() {
    pthread_t threads[4];
    int thread_ids[4];

    sem_init(&semaphore, 0, 1);

    for (int i = 0; i < 100; i++) {
        num[i] = i + 1;
    }

    for (int i = 0; i < 4; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, multiply, (void*)&thread_ids[i]);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }
    sem_destroy(&semaphore);

    return 0;
}