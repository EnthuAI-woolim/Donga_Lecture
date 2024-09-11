#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <time.h>

int process_count = 0;

void multiply(int start_num, int end_num) {
    for (int i = start_num; i <= end_num; i++) {
        printf("%d x 7 = %d\n", i, i * 7);
    }
}

int main() {
    clock_t start_time, end_time;
    double work_time;
    start_time = clock();

    for (int i = 0; i < 8; i++) {
        pid_t pid = fork();
        process_count++;
        if (pid == 0) {
            int start_num = (i * 125) + 1;
            int end_num = (i + 1) * 125;
            multiply(start_num, end_num);
            exit(0);
        }
    }
    for (int i = 0; i < 8; i++) {
        wait(NULL);
    }


    end_time = clock();
    work_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("\n------------------------------------\n");
    printf("Process 8 working time : %f\n", work_time);
    printf("Number of processes used : %d\n\n", process_count);

}