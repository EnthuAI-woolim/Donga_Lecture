#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>

bool flag[2] = { false, false }; // 처음에는 flag (들어오고 싶다는 의사) 는 false로 초기화
int turn = 0; // 0번 (쓰레드 0) 부터 시작하겠음.

void* first_thread_fc(void* args) {
    flag[0] = true;
    while (flag[1]) {
        if (turn == 1) {
            flag[0] = false;
            while (turn == 1) {
                // busy wait
            }
            flag[0] = true; // 재진입 시도
        }
    }

    // 임계영역
    printf("<Thread 1> STRT\n");
    for (int i = 1; i <= 50; i++) {
        printf("%d ", i * 3);
    }
    printf("\n<Thread 1> END\n");

    turn = 1;
    flag[0] = false;
    return NULL;
}

void* second_thread_fc(void* args) {
    flag[1] = true;
    while (flag[0]) {
        if (turn == 0) {
            flag[1] = false;
            while (turn == 0) {
                // busy wait
            }
            flag[1] = true; // 재진입 시도
        }
    }

    // 임계영역
    printf("<Thread 2> STRT\n");
    for (int i = 51; i <= 100; i++) {
        printf("%d ", i * 3);
    }
    printf("\n<Thread 2> END\n");

    turn = 0;
    flag[1] = false;
    return NULL;
}

int main() {
    pthread_t th0, th1;
    
    pthread_create(&th0, NULL, first_thread_fc, NULL);
    pthread_create(&th1, NULL, second_thread_fc, NULL);

    pthread_join(th0, NULL);
    pthread_join(th1, NULL);

    return 0;
}
