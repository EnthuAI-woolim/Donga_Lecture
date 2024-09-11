#include "./RR.h"
#include <unistd.h>
#include <sys/wait.h>

Process* currentProcess;   // 현재 프로세스  
Process* nextProcess;
// int index = 0;
        
int id;
int start;
int count;
int turn = 0;

int currentTime = 0;
int completed_processes = 0;
int startTime[PROCESS_NUM] = {0};
int remain_one_process = 0;

int main() {
    pid_t pid1, pid2, pid3, pid4;
    int processNum, status;
    pthread_t threads[5];
    
    // 큐 구조체 초기화
    Queue* q1 = (Queue*)malloc(sizeof(Queue));
    Queue* q2 = (Queue*)malloc(sizeof(Queue));
    initQueue(q1);
    initQueue(q2);

    // 프로세스 정보를 저장하는 구조체 초기화
    Process* processes1 = (Process*)malloc(PROCESS_NUM * sizeof(Process));
    inputProcesses(processes1);
    sortProcesses(processes1);
    Process* processes2 = createIndependentCopy(processes1);

    // 간트차트 정보를 저장하는 구조체 초기화
    GanttInfo ganttInfo[PROCESS_NUM];
    initGanttInfo(ganttInfo);

    for (int i = 0; i < PROCESS_NUM; ++i) {
        enqueue(q1, &processes1[i]);
        enqueue(q2, &processes2[i]);
    }

    // 간트차트 및 평균시간 출력
    printGanttChart(q1, ganttInfo);
    printAverageTimes(processes1, ganttInfo);
    // 구구단 출력
    // printMultiplication(q2, ganttInfo, processNum);

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


    
    // printMultiplication(q2, ganttInfo);
    //시작 전 기본은 idle임 모두 초기화
	for (int i = 0; i < 4; i++) {
		flag[i] = idle;
	}

    pid1 = fork();
    pid2 = fork();
    processNum = 0;

    if (pid1 == 0) {
        processNum += 2;
        if(pid2 == 0) {
            processNum++;
            pid3 = fork();
            if (pid3 == 0) {
                processNum++;
            }    // 프로세스 4
            // 프로세스 3
        }
        // 프로세스 2
    } else {
        if (pid2 == 0) {
            processNum++;
        }    // 프로세스 1
        // 프로세스 0
    }

    // printf("processNum : %d\n", processNum);    

    while (completed_processes < PROCESS_NUM) {
        flag[processNum] = want_in;
        int j; 
        turn = q2->front->id - 1;
        // printf("flag%d : %d\n", processNum, flag[processNum]);
    
        while (1) {
            
            /* 임계 영역 진입시도 1단계 */
            while (turn != processNum) {                        // 자신의 turn일 때까지 대기
                if (flag[turn] == idle) turn = processNum;
            }  
            /* 임계 영역 진입시도 2단계 */
            j = 0;                                       
            while ((j < PROCESS_NUM) && (j == processNum || flag[j] != in_cs)) {  
                j++;                                                    
            }

            if (j >= PROCESS_NUM) break;
        }

        flag[processNum] = in_cs;
        
    
                                                   
        // printf("turn : %d\nprocessNum : %d\n", turn, processNum);
            
  

        currentProcess = q2->front;   // 현재 프로세스  
        nextProcess = q2->front->next;
        int index = currentProcess->id - 1;
        

        id = currentProcess->id;
        start = startTime[id - 1] + 1;
        count;

        if (remain_one_process == 0) ganttInfo[index].startTime = currentTime;
        
        if (currentProcess->burstTime - TIME_QUANTUM > 0) {             // 실행시간이 남아 있는 경우
            q2->front = nextProcess;
            q2->rear = currentProcess;
            

            currentProcess->burstTime -= TIME_QUANTUM;

            count = TIME_QUANTUM + startTime[id - 1];
            

            ganttInfo[index].endTime = currentTime + TIME_QUANTUM;
            if (completed_processes != 4) ganttInfo[index].execute++;

            if (q2->front->id == q2->rear->id) remain_one_process = 1;
        } else {                                                        // 실행시간이 남아 있지 않는 경우
            q2->front = nextProcess;
            q2->rear->next = nextProcess;

            count = currentProcess->burstTime + startTime[id - 1];

            ganttInfo[index].endTime = currentTime + currentProcess->burstTime;

            currentProcess->burstTime -= currentProcess->burstTime;

            if (currentProcess->burstTime == 0) completed_processes++;
        }
        currentTime = ganttInfo[index].endTime;

        if (id - 1 == processNum) {
            for (start; start <= count; start++) {
                printf("P%d: %d x %d = %d\n", id, start, id, start*id);
                fflush(stdout); // 버퍼를 즉시 비워 출력 순서 보장
            }
        }

        startTime[id - 1] = start - 1;

        flag[processNum] = idle;              // 임계 영역을 빠져나옴
        // turn = q2->front->id - 1;
        // printf("turn : %d\n", turn);

    }


     // 자식 프로세스 정리
    if (pid1 == 0 || pid2 == 0 || pid3 == 0) {
        exit(0);
    } else {
        while (wait(&status) > 0);
    }




    free(processes1);
    free(processes2);
    free(q1);
    free(q2);
    

    return 0;
}
