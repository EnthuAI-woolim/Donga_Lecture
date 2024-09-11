#include "./mlfq.h"
#include <unistd.h>
#include <sys/wait.h>

Process* currentProcess;   // 현재 프로세스  
Process* nextProcess;
// int index = 0;
        
int id;
int start;
int count;
int turn = 0;


int startTime[PROCESS_NUM] = {0};
int remain_one_process = 0;

int main() {
    pid_t pid1, pid2, pid3, pid4;
    int processNum, status;
    pthread_t threads[5];
    
    // 큐 구조체 초기화
    Queue* q = (Queue*)malloc(Q_NUM * sizeof(Queue));
    initQueue(q);
    // initQueue(q2);

    // 프로세스 정보를 저장하는 구조체 초기화
    Process* processes = (Process*)malloc(PROCESS_NUM * sizeof(Process));
    inputProcesses(processes);
    sortProcesses(processes);
    
  
    executeMLFQ(q, processes);



    // // 스레드 생성 및 실행
    // for (int i = 0; i < PROCESS_NUM; ++i) {
    //     processNums[i] = i;
    //     void* args[] = { q2, ganttInfo, &processNums[i] };
    //     pthread_create(&threads[i], NULL, printMultiplication, (void*)args);
    //     // pthread_join(threads[i], NULL);
    // }
    
    //  // 스레드가 종료될 때까지 대기
    // for (int i = 0; i < PROCESS_NUM; ++i) {
    //     pthread_join(threads[i], NULL);
    // }

    free(q);
    free(processes);
    

    return 0;
}
