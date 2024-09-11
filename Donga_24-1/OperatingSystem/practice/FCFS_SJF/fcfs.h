#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define PROCESS_NUM 5

// 프로세스 구조체 정의
typedef struct Process {
    int id;             // 프로세스 ID
    int arrivalTime;    // 도착 시간
    int burstTime;      // 실행 시간
    struct Process* next;  // 다음 프로세스를 가리키는 포인터
} Process;

// 큐 구조체 정의
typedef struct Queue {
    Process* front;     // 큐의 맨 앞 요소를 가리키는 포인터
    Process* rear;      // 큐의 맨 뒤 요소를 가리키는 포인터
    pthread_mutex_t mutex; // 뮤텍스 변수
} Queue;

// 큐 초기화 함수
void initQueue(Queue* q) {
    q->front = NULL;
    q->rear = NULL;
    pthread_mutex_init(&q->mutex, NULL); // 뮤텍스 초기화
}

// 프로세스 정보 입력 받기 함수
void inputProcesses(Process *processes) {
    for (int i = 0; i < PROCESS_NUM; ++i) {
        printf("====== P%d ======\n", i + 1);
        processes[i].id = i + 1;
        printf("도착 시간 : "); scanf("%d", &processes[i].arrivalTime);
        printf("실행 시간 : "); scanf("%d", &processes[i].burstTime);
        processes[i].next = NULL;
        printf("\n");
    }
}

// 큐에 프로세스 추가 함수
void enqueue(Queue* q, Process* newProcess) {
    newProcess->next = NULL;

    if (q->rear == NULL) {
        q->front = newProcess;
        q->rear = newProcess;
    } else {
        q->rear->next = newProcess; // 큐에서 마지막 프로세스의 다음 프로세스로 설정
        q->rear = newProcess;       // 큐의 마지막 프로세스로 설정
    }
}

// 도착 시간이 빠른 순서대로 정렬 후, enqueue함수 실행
void enqueueProcesses(Queue* q, Process processes[]) {
    // 버블 정렬을 사용하여 도착 시간이 빠른 순서대로 정렬
    for (int i = 0; i < PROCESS_NUM - 1; ++i) 
        for (int j = i + 1; j < PROCESS_NUM; ++j) 
            if (processes[i].arrivalTime > processes[j].arrivalTime) {
                Process temp = processes[i];
                processes[i] = processes[j];
                processes[j] = temp;
            }
        
    // 정렬된 순서대로 큐에 프로세스 추가
    for (int i = 0; i < PROCESS_NUM; ++i) 
        enqueue(q, &processes[i]);
}

// 큐에서 프로세스 제거 함수
Process* dequeue(Queue* q) {
    if (q->front == NULL) {
        printf("큐가 비어있습니다..\n");
        return NULL;
    }

    Process* temp = q->front;
    q->front = q->front->next;
    if (q->front == NULL) q->rear = NULL;

    return temp;
}

// 간트차트 출력 함수
void printGanttChart(Queue* q) {
    int currentTime = 0;
    Process* currentProcess = q->front; // 큐의 맨 앞 프로세스를 가리키는 포인터

    printf(" 간트 차트\n");
    // 큐가 비어있지 않은 동안 실행 시간 출력
    while (currentProcess != NULL) {
        int startTime = currentTime;
        int endTime = currentTime + currentProcess->burstTime;
        printf("P%d (%d-%d)\n", currentProcess->id, startTime, endTime);
        currentTime = endTime;
        currentProcess = currentProcess->next; // 다음 프로세스로 이동
    }
    printf("\n");
}

// 평균 반환시간과 평균 대기시간 계산 함수
void calculateAverageTimes(Queue* q) {
    int totalReturnTime = 0;    // 총 반환시간
    int totalWaitingTime = 0;   // 총 대기시간
    int currentTime = 0;        // 시작시간
    int idx;
    int returnTime[PROCESS_NUM] = {0};
    int waitingTime[PROCESS_NUM] = {0};

    Process* currentProcess = q->front; 

    printf("    반환시간 대기시간\n");
    // 각 프로세스의 반환시간과 대기시간 계산

    while (currentProcess != NULL) {

        idx = currentProcess->id - 1;

        // 대기시간 
        waitingTime[idx] = currentTime - currentProcess->arrivalTime;

        // 반환시간 
        returnTime[idx] = currentTime - currentProcess->arrivalTime + currentProcess->burstTime;
        
        // 현재 실행시간 설정
        currentTime += currentProcess->burstTime;

        // 다음 프로세스로 이동
        currentProcess = currentProcess->next;
    }

    // 총 반환시간과 대기시간 설정
    for (int i = 0; i < PROCESS_NUM; ++i) {
        totalReturnTime += returnTime[i];
        totalWaitingTime += waitingTime[i];
    }

    for (int i = 0; i < PROCESS_NUM; ++i)
        printf("P%d: %2d       %2d\n", i+1, returnTime[i], waitingTime[i]);

    // 평균 반환시간과 평균 대기시간 계산
    float averageReturnTime = (float)totalReturnTime / PROCESS_NUM;
    float averageWaitingTime = (float)totalWaitingTime / PROCESS_NUM;

    // 결과 출력
    printf("평균 반환시간: %.2f\n", averageReturnTime);
    printf("평균 대기시간: %.2f\n\n", averageWaitingTime);
}

// 곱하기 출력 함수
void* printMultiplication(void* arg) {
    Queue* q = (Queue*)arg;
    int startTime[PROCESS_NUM] = {0};
    
    // 큐에서 프로세스를 꺼내서 실행하고 실행 시간 출력
    while (1) {
        Process* currentProcess;

        pthread_mutex_lock(&(q->mutex));

        // 큐가 비어있는지 확인 후 작업 진행
        if (q->front != NULL) {
            currentProcess = dequeue(q);
            pthread_mutex_unlock(&q->mutex);
            
            int id = currentProcess->id;
            int start = startTime[id - 1] + 1;
            int count = currentProcess->burstTime + startTime[id - 1];

            for (start; start <= count; start++) {
                printf("P%d: %d x %d = %d\n", id, start, id, start*id);
                fflush(stdout); // 버퍼를 즉시 비워 출력 순서 보장
            }
            startTime[id - 1] = start - 1;
            
        } else {
            pthread_mutex_unlock(&q->mutex);
            break; // 큐가 비어있으면 작업을 중단하고 스레드 종료
        }
    }

    pthread_exit(NULL);
}

