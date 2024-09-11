#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>


int main() {
    pid_t pid1, pid2, pid3;
    int i=1, start, end;
    struct timeval p_start_time, p_end_time;
    struct timeval start_time, end_time;
    double t = 0.0;
    
    // fork() 함수를 사용하여 8개의 프로세스를 생성하고 
    // 작업을 나누어서 진행하여 결과 출력 및 시간 확인
    
    gettimeofday(&p_start_time, NULL);
    int status;

    pid1 = fork();
    pid2 = fork();
    pid3 = fork();

    if(pid1 == 0) {
        i = i * 5;
        if (pid2 == 0) i = i + 2;
        if (pid3 == 0) i++;
    } 
    else { 
        if( pid2 == 0) i = i + 2;
        if(pid3 == 0) i++;
    }

    printf("\nprocess id: %d\n", i);
    int j = (1000 / 8) * i;
    for(int k = j - (1000/8) + 1; k <= j; k++) {
        printf("%d ", k * 7);
    }
    printf("\n");
    gettimeofday(&p_end_time, NULL);
    
    t = t+ ((p_end_time.tv_sec - p_start_time.tv_sec) + (p_end_time.tv_usec - p_start_time.tv_usec)/ 1000000);

    if(pid3 == 0) exit(2);
    else if(pid2 == 0) {
        while(wait(&status) != pid3) continue;
        exit(2);
    } else if(pid1 == 0) {
        while(wait(&status) != pid2) continue;
        exit(2);
    } else {
        while(wait(&status) != pid1) continue;
    }
    
    
    printf("\nTotal multi process execution time: %lf seconds\n\n", t);
    


    // for문을 이용하여 일반적인 단일 프로세스 환경에서 같은 작업을 수행 후 출력 
    gettimeofday(&start_time, NULL);
    for(int i=1; i<1001;i++){
        printf("%d ", i*7);
    }
    gettimeofday(&end_time, NULL);

    printf("\nTotal single process execution time: %lf seconds\n", 
           (double)((end_time.tv_sec - start_time.tv_sec) + 
            end_time.tv_usec - start_time.tv_usec) / 1000000);

    return 0;
}