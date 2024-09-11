#include <stdio.h>
#include <stdlib.h>
#include <semaphore.h>
#include <pthread.h>

#define Q_NUM 3
#define PROCESS_NUM 3
#define TIME_QUANTUM 5 


enum process_statement { idle, want_in, in_cs }; 
int flag[PROCESS_NUM];     
int turn; 
int completed_processes = 0;            
int currentTime = 0;
int endTime = 0;


// 프로세스 구조체 정의
typedef struct Process {
    int id;             // 프로세스 ID
    int arrivalTime;    // 도착 시간
    int burstTime;      // 실행 시간
    int executeNum;     // 실행 횟수
    struct Process* next;  // 다음 프로세스를 가리키는 포인터
} Process;

// 큐 구조체 정의
typedef struct Queue {
    int id;
    Process* front;     // 큐의 맨 앞 요소를 가리키는 포인터
    Process* rear;      // 큐의 맨 뒤 요소를 가리키는 포인터
    pthread_mutex_t mutex; // 뮤텍스 변수
} Queue;

typedef struct GanttInfo {
    int startTime;
    int endTime;
    int execute;
} GanttInfo;

// 큐 초기화 함수
void initQueue(Queue* q) {
  for (int i = 0; i < 3; ++i) {
    q[i].id = i+1;
    q[i].front = NULL;
    q[i].rear = NULL;
    pthread_mutex_init(&q->mutex, NULL); // 뮤텍스 초기화
  }
}

// 프로세스 정보 입력 받기 함수
void inputProcesses(Process *processes) {
    for (int i = 0; i < PROCESS_NUM; ++i) {
        printf("====== P%d ======\n", i + 1);
        processes[i].id = i + 1;
        processes[i].arrivalTime = 0;
        printf("실행 시간 : "); scanf("%d", &processes[i].burstTime);
        processes[i].executeNum = 0;
        processes[i].next = NULL;
        printf("\n");
    }
}

// 간트정보 초기화 함수
void initGanttInfo(GanttInfo* ganttInfo) {
    for (int i = 0; i < PROCESS_NUM; ++i) {
        ganttInfo[i].startTime = 0;
        ganttInfo[i].endTime = 0;
        ganttInfo[i].execute = 0;
    }
}

// 도착 시간이 빠른 순서대로 정렬
void sortProcesses(Process* processes) {
    // 도착 시간이 빠른 순서대로 정렬
    for (int i = 0; i < PROCESS_NUM - 1; ++i) 
        for (int j = i + 1; j < PROCESS_NUM; ++j) 
            if (processes[i].arrivalTime > processes[j].arrivalTime) {
                Process temp = processes[i];
                processes[i] = processes[j];
                processes[j] = temp;
            }
}

// 독립적이지만 동일한 데이터를 가진 구조체 생성 함수
Process* createIndependentCopy(Process* process) {
    Process* copy = (Process*)malloc(PROCESS_NUM * sizeof(Process));
    if (copy == NULL) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < PROCESS_NUM; i++) {
        copy[i].id = process[i].id;
        copy[i].arrivalTime = process[i].arrivalTime;
        copy[i].burstTime = process[i].burstTime;
        copy[i].next = process[i].next;
    }
    return copy;
}

// // 큐에 프로세스 추가하는 함수
// void enqueue(Queue* q, Process* newProcess) {  
//     if (q->rear == NULL) {
//         q->front = newProcess;
//         q->rear = newProcess;
//         q->rear->next = newProcess;
//     } else {
//         q->rear->next = newProcess; // 큐에서 마지막 프로세스의 다음 프로세스로 설정
//         q->rear = newProcess;       // 큐의 마지막 프로세스로 설정
//         q->rear->next = q->front;
//     }
// }

// 간트차트 출력 함수
void printGanttChart(Queue* q, GanttInfo* ganttInfo) {
    int currentTime = 0;
    int completed_processes = 0;
    int remain_one_process = 0;
    
    while (completed_processes < PROCESS_NUM) {
        Process* currentProcess = q->front;   // 현재 프로세스  
        Process* nextProcess = q->front->next;
        int index = currentProcess->id - 1;

        if (remain_one_process == 0) ganttInfo[index].startTime = currentTime;
        
        if (currentProcess->burstTime - TIME_QUANTUM > 0) {             // 실행시간이 남아 있는 경우
            q->front = nextProcess;
            q->rear = currentProcess;

            currentProcess->burstTime -= TIME_QUANTUM;

            ganttInfo[index].endTime = currentTime + TIME_QUANTUM;
            if (completed_processes != 4) ganttInfo[index].execute++;

            if (q->front->id == q->rear->id) remain_one_process = 1;
        } else {                                                        // 실행시간이 남아 있지 않는 경우
            q->front = nextProcess;
            q->rear->next = nextProcess;

            ganttInfo[index].endTime = currentTime + currentProcess->burstTime;

            currentProcess->burstTime -= currentProcess->burstTime;

            if (currentProcess->burstTime == 0) completed_processes++;
        }
        printf("P%d (%d-%d)\n", currentProcess->id, currentTime, ganttInfo[index].endTime);
        currentTime = ganttInfo[index].endTime;
    }
    printf("\n");
}

void printAverageTimes(Process *processes, GanttInfo* ganttInfo) {
    int totalReturnTime = 0;    // 총 반환시간
    int totalWaitingTime = 0;   // 총 대기시간

    printf("    반환시간 대기시간\n");
    for (int i = 0; i < PROCESS_NUM; ++i) {
        int returnTime = ganttInfo[i].endTime - processes[i].arrivalTime;
        int waitingTime = ganttInfo[i].startTime - processes[i].arrivalTime - (ganttInfo[i].execute) * TIME_QUANTUM;
        totalReturnTime += returnTime;
        totalWaitingTime += waitingTime;
        printf("P%d: %7d  %7d\n", i+1, returnTime, waitingTime);
    }
    float averageReturnTime = (float)totalReturnTime / PROCESS_NUM;
    float averageWaitingTime = (float)totalWaitingTime / PROCESS_NUM;

    printf("평균 반환시간: %.2f\n", averageReturnTime);
    printf("평균 대기시간: %.2f\n\n", averageWaitingTime);
}

void dijkstra(int processNum, int id, int start, int count) {
    flag[processNum] = want_in;
    int j;

    do {
        /* 임계 영역 진입시도 1단계 */
        while (turn != processNum) {                        // 자신의 turn일 때까지 대기
            if (flag[turn] == idle) turn = processNum;      // 현재 실행 중인 프로세스의 차례가 끝나면, turn을 자신으로 함
        }                                             

        /* 임계 영역 진입시도 2단계 */
        j = 0;                                       
        while ((j < PROCESS_NUM) && (j == processNum || flag[j] != in_cs)) {  
            j++;                                                    
        }

    } while (j < PROCESS_NUM);     // 다른 프로세스 모두 in_cs가 아닌 경우(4) + 자신인 경우(1) = 5 
                                    //      => 어떤 스레드도 임계영역에 들어가 있지 않은 경우
    flag[processNum] = in_cs;    

    if (id == processNum) {
        for (start; start <= count; start++) {
            printf("P%d: %d x %d = %d\n", id, start, id, start*id);
            fflush(stdout); // 버퍼를 즉시 비워 출력 순서 보장
        }
    }
    
    flag[processNum] = idle;              // 임계 영역을 빠져나옴
}

// 큐에 프로세스 추가하는 함수 (실행시간과 실행횟수가 수정된 프로세스 전달됨)
void enqueue(Queue* q, Process* newProcess) {  
    newProcess->next = NULL;

    if (q->front == NULL) {
        q->front = newProcess;
        q->rear = newProcess;
        // if (newProcess->executeNum >= 3 && newProcess->burstTime > 2) {
        //     printf("==============1\n");
        //     q->rear->next = newProcess;
        // }

    } else {
        q->rear->next = newProcess; // 큐에서 마지막 프로세스의 다음 프로세스로 설정
        q->rear = newProcess;       // 큐의 마지막 프로세스로 설정
        // if (newProcess->executeNum >= 3 && newProcess->burstTime > 2) {
        //     printf("==============2\n");
        //     q->rear->next = q->front;
        // }
            
    }
}

// 큐에서 요소를 제거하고 반환하는 함수
Process* dequeue(Queue* q) {
    if (q->front == NULL) {
        printf("Queue is empty\n");
        return NULL; // 큐가 비어 있을 때의 오류 값
    }

    Process* temp = q->front;
    q->front = q->front->next;
    if (q->front == NULL) {
        q->rear = NULL;
    }

    return temp;
}

void executeMLFQ(Queue* q, Process* processes) {
    Queue* q1 = &q[0];
    Queue* q2 = &q[1];
    Queue* q3 = &q[2];
    Queue* currentQ = q1;
    Process* currentProcess;
    
    // 도착시간이 전부 0이므로 그냥 프로세스 번호순으로 큐에 추가
    for (int i = 0; i < PROCESS_NUM; ++i) enqueue(currentQ, &processes[i]);
    
    // q1에 프로세스 전부 추가된 상태
    

    while (completed_processes < Q_NUM) {

        if (currentQ->id == 1) {
            printf("Q1: ");
            while (currentQ->front != NULL) {
                currentTime = endTime;
                currentProcess = dequeue(currentQ);
                currentProcess->burstTime -= 1;
                currentProcess->executeNum++;
                endTime += 1;
                if (currentProcess->burstTime == 0) completed_processes++;

                enqueue(q2, currentProcess);
                printf("P%d (%d-%d)\n", currentProcess->id, currentTime, endTime);
            }
            currentQ = q2;

        } else if (currentQ->id == 2) {
            printf("Q2: ");
            while (currentQ->front != NULL) {
                currentTime = endTime;
                currentProcess = dequeue(currentQ);
                currentProcess->burstTime -= 2;
                currentProcess->executeNum++;
                endTime += 2;
                if (currentProcess->burstTime == 0) completed_processes++;

                enqueue(q3, currentProcess);
                printf("P%d (%d-%d)\n", currentProcess->id, currentTime, endTime);
            }

            currentQ = q3;
        } else if (currentQ->id == 3) {
            printf("Q3: ");
            while (currentQ->front != NULL) {
                currentTime = endTime;
                currentProcess = dequeue(currentQ);
                currentProcess->executeNum++;
                if (currentProcess->burstTime < 4) {
                    endTime += currentProcess->burstTime;
                    currentProcess->burstTime = 0;
                } else {
                    currentProcess->burstTime -= 4;
                    endTime += 4;
                }
                
                if (currentProcess->burstTime == 0) completed_processes++;

                if (currentProcess->burstTime == 1) enqueue(q1, currentProcess);
                else if (1 < currentProcess->burstTime == 2) enqueue(q2, currentProcess);
                else if (currentProcess->burstTime > 2) enqueue(q3, currentProcess);

                printf("P%d (%d-%d)\n", currentProcess->id, currentTime, endTime);
            }

            currentQ = q1;
        }
    }

}

// 곱하기 출력 함수
void* printMultiplication(void* args) {
    Queue* q = ((Queue**)args)[0];
    GanttInfo* ganttInfo = ((GanttInfo**)args)[1];
    int processNum = *((int*)((int**)args)[2]);
    
    int currentTime = 0;
    int completed_processes = 0;
    int startTime[PROCESS_NUM] = {0};
    int remain_one_process = 0;

    while (completed_processes < PROCESS_NUM) {
        // pthread_mutex_lock(&(q->mutex));

        Process* currentProcess = q->front;   // 현재 프로세스  
        Process* nextProcess = q->front->next;
        int index = currentProcess->id - 1;

        int id = currentProcess->id;
        int start = startTime[id - 1] + 1;
        int count;
        turn = id;

        if (remain_one_process == 0) ganttInfo[index].startTime = currentTime;
        
        if (currentProcess->burstTime - TIME_QUANTUM > 0) {             // 실행시간이 남아 있는 경우
            q->front = nextProcess;
            q->rear = currentProcess;

            currentProcess->burstTime -= TIME_QUANTUM;

            count = TIME_QUANTUM+ startTime[id - 1];
            

            ganttInfo[index].endTime = currentTime + TIME_QUANTUM;
            if (completed_processes != 4) ganttInfo[index].execute++;

            if (q->front->id == q->rear->id) remain_one_process = 1;
        } else {                                                        // 실행시간이 남아 있지 않는 경우
            q->front = nextProcess;
            q->rear->next = nextProcess;

            count = currentProcess->burstTime + startTime[id - 1];

            ganttInfo[index].endTime = currentTime + currentProcess->burstTime;

            currentProcess->burstTime -= currentProcess->burstTime;

            if (currentProcess->burstTime == 0) completed_processes++;
        }
        currentTime = ganttInfo[index].endTime;

        if (processNum == id -1) {
            for (start; start <= count; start++) {
                printf("P%d: %d x %d = %d\n", id, start, id, start*id);
                fflush(stdout); // 버퍼를 즉시 비워 출력 순서 보장
            }
        }
        

        startTime[id - 1] = start - 1;
    }
    // pthread_mutex_unlock(&q->mutex); // 큐에 대한 뮤텍스 잠금 해제
    return NULL;
}

// // 곱하기 출력 함수
// void* printMultiplication(void* args) {
//     Queue* q = ((Queue**)args)[0];
//     GanttInfo* ganttInfo = ((GanttInfo**)args)[1];
//     int processNum = *((int*)((int**)args)[2]);

//     pthread_mutex_lock(&(q->mutex));

//     while (completed_processes < PROCESS_NUM) {
//         Process* currentProcess = q->front;   // 현재 프로세스  
//         Process* nextProcess = q->front->next;
//         int index = currentProcess->id - 1;

//         int id = currentProcess->id;
//         int start = startTime[id - 1] + 1;
//         int count;
//         turn = id;

//         if (remain_one_process == 0) ganttInfo[index].startTime = currentTime;
        
//         if (currentProcess->burstTime - TIME_QUANTUM > 0) {             // 실행시간이 남아 있는 경우
//             q->front = nextProcess;
//             q->rear = currentProcess;

//             currentProcess->burstTime -= TIME_QUANTUM;

//             count = TIME_QUANTUM+ startTime[id - 1];
            

//             ganttInfo[index].endTime = currentTime + TIME_QUANTUM;
//             if (completed_processes != 4) ganttInfo[index].execute++;

//             if (q->front->id == q->rear->id) remain_one_process = 1;
            
//         } else {                                                        // 실행시간이 남아 있지 않는 경우
//             q->front = nextProcess;
//             q->rear->next = nextProcess;

//             count = currentProcess->burstTime + startTime[id - 1];

//             ganttInfo[index].endTime = currentTime + currentProcess->burstTime;

//             currentProcess->burstTime -= currentProcess->burstTime;

//             if (currentProcess->burstTime == 0) completed_processes++;
//         }
//         currentTime = ganttInfo[index].endTime;

//         pthread_mutex_unlock(&q->mutex);
//         for (start; start <= count; start++) {
//             printf("P%d: %d x %d = %d\n", id, start, id, start*id);
//             fflush(stdout); // 버퍼를 즉시 비워 출력 순서 보장
//         }
        
        
        
        
//         startTime[id - 1] = start - 1;
//     }

//     return NULL;
// }
