#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>

int numbers[100];
int turn;
bool flag[2] = { false, false };

void* peterson(void* arg) {
    int thread_id = *(int*)arg;
    int start = thread_id * 50;
    int end = (thread_id + 1) * 50 -1;

    for (int i = start; i <= end; ++i) {
        flag[thread_id] = true;
        turn = 1 - thread_id;

        while (flag[1 - thread_id] && turn == 1 - thread_id) {
        }

        // critical section
        printf("thread %d: %d * 3 = %d\n", thread_id, numbers[i], numbers[i] * 3);

        flag[thread_id] = false;
    }

    return NULL;
}


int main() {
    pthread_t threads[2];
    int thread_ids[2] = { 0, 1 };

    for (int i = 0; i < 100; i++) {
        numbers[i] = i + 1;
    }

    for (int i = 0; i < 2; ++i) {
        pthread_create(&threads[i], NULL, peterson, &thread_ids[i]);
    }

    for (int i = 0; i < 2; ++i) {
        pthread_join(threads[i], NULL);
    }

    return 0;
}
