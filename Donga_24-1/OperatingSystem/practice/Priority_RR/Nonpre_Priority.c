#include "./Nonpre_priority.h"

// 기본 프로세스 환경에서 해야됨
// 안되면 스레드 환경에서 해도됨
// 일단 스레드 변수와 실행함수는 주석처리해 놓음

int main() {
    pthread_t threads[5];

    // 큐 구조체 초기화
    Queue* q = (Queue*)malloc(sizeof(Queue));

    // 큐 초기화
    initQueue(q);


    // 프로세스 정보 입력 받아 구조체에 저장
    Process* processes = (Process*)malloc(PROCESS_NUM * sizeof(Process));
    inputProcesses(processes);

    enqueueProcesses(q, processes);

    printGanttChart(q);
    calculateAverageTimes(q);

    // 스레드 생성 및 실행
    for (int i = 0; i < PROCESS_NUM; ++i) {
        pthread_create(&threads[i], NULL, printMultiplication, (void*)q);
        pthread_join(threads[i], NULL);
    }

    free(processes);
    free(q);
    

    return 0;
}
