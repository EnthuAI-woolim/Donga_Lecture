#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int n[100];
int flag[4];
int turn = -1;

void* multiply_thread(void* ti) {
    long tid = (long)ti;
    int start = tid * 25;
    int end = start + 25 - 1;

    flag[tid] = 1; 
    while (1) {
        int j;
        for (j = 0; j < 4; j++) {
            if (j != tid && flag[j] == 2) {
                break;
            }
        }
        if (j == 4) {
            break;
        }
    }

    flag[tid] = 2;
    turn = tid;

    for (int i = start; i <= end; i++) {
        printf("thread %ld: %d * 3 = %d\n", tid, n[i], n[i] * 3);
    }

    flag[tid] = 0;
    pthread_exit(NULL);
}

int main() {
    pthread_t threads[4];

    for (int i = 0; i < 100; i++) {
        n[i] = i + 1;
    }

    for (long i = 0; i < 4; i++) {
        pthread_create(&threads[i], NULL, multiply_thread, (void*)i);
    }

    for (int i = 0; i < 4; i++) {
        pthread_join(threads[i], NULL);
    }

    pthread_exit(NULL);
}