#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

int params[16];

void* printing(void* ps)
{
    int PID = *(int *)ps / 4;
    int TID = *(int *)ps % 4;
    // printf("PID : %d, TID : %d\n", PID, TID); 

    for(int num = 1 + (TID * 25); num <= 25 + (TID * 25); num++)
        printf("Process %d's Thread %d's Output : %d\n", PID, TID, (3 + (PID * 2)) * num);
    
    pthread_exit(NULL);
}

int main()
{
    pid_t ROOT = getpid();
    pid_t R = fork();
    pid_t C = fork();

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    pid_t pid = getpid();

    int p_r = pid % ROOT;
    pthread_t tid[4];

    for(int i = 0; i < 16; i++)
        params[i] = i;

    for(int i = 0; i < 4; i++)
    {
        int* p = &params[(p_r * 4) + i];
        pthread_create(&tid[i], NULL, printing, p);
    }

    for(int i = 0; i < 4; i++)
        pthread_join(tid[i], NULL);

    if (C == 0)
        return 0;
    else if(R == 0)
    {
        while(wait(NULL) != -1);
        return 0;
    }
    else
        while(wait(NULL) != -1);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    long spend_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    printf("총 소요 시간 : %.3lf\n", (double)spend_time/1000000);
    
    return 0;
}