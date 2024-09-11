#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>

int numbers[100];
int turn;
bool flag[2] = { false, false };

void* multiply_numbers(void* arg) {
    long thread_id = (long)arg; 
    int start = thread_id * 50;
    int end = (thread_id + 1) * 50-1;

    flag[thread_id] = true;

    while (flag[1 - thread_id]) {
        if (turn != thread_id) {
            flag[thread_id] = false;
            while (turn != thread_id);
            flag[thread_id] = true;
        }
    }

    for (int i = start; i <= end; i++) {
        printf("thread %ld: %d * 3 = %d\n", thread_id, numbers[i], numbers[i] * 3);
    }

    turn = 1 - thread_id;
    flag[thread_id] = false;

    return NULL;
}


int main() {
    pthread_t threads[2];

    for (int i = 0; i < 100; i++) {
        numbers[i] = i + 1;
    }


    for (long i = 0; i < 2; i++) {
        pthread_create(&threads[i], NULL, multiply_numbers, (void*)i);
    }

    for (long i = 0; i < 2; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}

