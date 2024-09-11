#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

int main()
{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for(int i = 1; i <= 100; i++)
        printf("Single Process (* 3) : %d\n", i * 3);
    for(int i = 1; i <= 100; i++)
        printf("Single Process (* 5) : %d\n", i * 5);
    for(int i = 1; i <= 100; i++)
        printf("Single Process (* 7) : %d\n", i * 7);
    for(int i = 1; i <= 100; i++)
        printf("Single Process (* 9) : %d\n", i * 9);

    clock_gettime(CLOCK_MONOTONIC, &end);
    long spend_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    printf("총 소요 시간 : %.3lf\n", (double)spend_time/1000000);

    return 0;
}