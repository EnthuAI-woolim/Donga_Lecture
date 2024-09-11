#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <time.h>

typedef struct Thread{
    int start;
    int end;
    int var;
} Thread;

double multi_time = 0;

void *func(void *arg){
    Thread t = *((Thread *)arg);
    
    for (int i=t.start; i<=t.end; i++){
        printf("%d ", i*t.var);
    } 
    printf("\n");
    
    pthread_exit(NULL);
}

int main(){
    clock_t start_time = clock();
    int i=1;
    pid_t pid1 = fork();
    pid_t pid2 = fork();
    if(pid1==0){
        i*=3;
    }
    if(pid2==0){
        i++;
    }
    printf("프로세스: %d", i);

    int var = i*2+1;
    
    pthread_t tids[4];
    Thread t[4];
    
    for(int j=0; j<4; j++){
        t[j].end = (j+1) * 100/4;
        t[j].start = t[j].end - 100/4 + 1;
        t[j].var = var;
        printf("스레드: %d\n", j+1);
        pthread_create(&tids[j], NULL, func, (void*)&t[j]);
    }
    for(int j=0; j<4; j++){
        pthread_join(tids[j], NULL);
    }

    int status;
    if(pid2==0)
        exit(2);

    else if(pid1==0){
        while(wait(&status)!=pid2) continue;
        exit(2);
    }
    else
        while(wait(&status)!=pid1) continue;

    clock_t end_time = clock();
    multi_time += (((double)(end_time - start_time)) / CLOCKS_PER_SEC);
    printf("Multi_process time: %f seconds\n", multi_time);
    return 0;
}

