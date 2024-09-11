#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/time.h>


int main() {
    pid_t pid1, pid2, pid3, pid4, pid5;
    int i=1;
    struct timeval p_start_time, p_end_time;
    struct timeval start_time, end_time;
    double t = 0.0;
    
    // fork() 함수를 사용하여 10개의 프로세스를 생성하고 작업을 나누어서 진행하여 결과 출력 및 시간 확인
    
    gettimeofday(&p_start_time, NULL);
    int status;

    pid1 = fork();
    pid2 = fork();
    pid3 = fork();

    // 내가 다시 정리한 코드
    // if (pid1 == 0) {
    //     i += 4;
    //     if (pid2 == 0) {
    //         i += 2;
    //         pid4 = fork();
    //         if (pid3 == 0) i += 2;
    //         if (pid4 == 0) i += 1;
    //     }
    //     if (pid3 == 0) i += 1;
    // } else {
    //     if (pid2 == 0) i += 2;
    //     if (pid3 == 0) i += 1;
    // }

    if(pid1 == 0) {
        i = i * 5; //process num 5
        if (pid2 == 0) {
            i = i + 2; //process num 7
            if (pid3 == 0) {
                pid5 = fork();
                if(pid5 == 0)ni += 1; //process num 9
                else i += 2; //process num 10
            }
            else {
                pid4 = fork();
                if (pid4 == 0) i ++; //process num 8
            }
        } 

        if (pid3 == 0) i++; //process num 6
        
    } 
    else { // pid1 != 0
        if( pid2 == 0) i = i + 2;
        
        if(pid3 == 0) i++; //process num 2
    }

    printf("\nprocess id: %d\n", i);
    int j = (1000 / 10) * i;
    for(int k = j - (1000/10) + 1; k <= j; k++) {
        printf("%d ", k * 7);
    }
    printf("\n");
    gettimeofday(&p_end_time, NULL);
    
    t = t+ ((p_end_time.tv_sec - p_start_time.tv_sec) + (p_end_time.tv_usec - p_start_time.tv_usec)/ 1000000);

    
    if(pid3 == 0) {
        exit(2);
    }
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