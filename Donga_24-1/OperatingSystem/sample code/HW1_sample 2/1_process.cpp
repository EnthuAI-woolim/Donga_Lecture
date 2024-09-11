#include <stdio.h>
#include <time.h>

void multiply(int start_num, int end_num) {
    for (int i = start_num; i <= end_num; i++) {
        printf("%d x 7 = %d\n", i, i * 7);
    }
}
int main() {
    time_t start_time, end_time;
    double work_time;
    start_time = clock();

    multiply(1, 1000);

    end_time = clock();
    work_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("\n------------------------------------\n");
    printf("Only for-Repetition working time : %f\n\n", work_time);
}

