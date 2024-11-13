#include <stdio.h>
#include <stdlib.h>

#define MAX_NUMBERS 10000

int readFile(const char *filename, int *A, int max_numbers);
void writeFile(const char *filename, int *A, int n);

int main() {
    int A[MAX_NUMBERS];
    int n = readFile("input.txt", A, MAX_NUMBERS);
    if (n < 0) {
        printf("파일을 읽는 데 문제가 발생했습니다.\n");
        return 1;
    }

    // Selection Sort
    for (int i = 0; i < n; ++i) {
        int min = i;
        for (int j = i+1; j < n; ++j) {
            if (A[j] < A[min]) {
                min = j;
            }
        }
        int temp = A[min];
        A[min] = A[i];
        A[i] = temp;
    }

    writeFile("selection_output.txt", A, n);

    return 0;
}


int readFile(const char *filename, int *A, int max_numbers) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("파일을 열 수 없습니다.\n");
        return -1;
    }

    int n = 0;
    while (fscanf(file, "%d", &A[n]) == 1) {
        n++;
        if (n >= max_numbers) {
            printf("저장 가능한 최대 숫자 개수를 초과했습니다.\n");
            break;
        }
    }

    fclose(file);
    return n;
}

void writeFile(const char *filename, int *A, int n) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("결과 파일을 열 수 없습니다.\n");
        return;
    }

    for (int i = 0; i < n; ++i) {
        fprintf(file, "%d\n", A[i]);
    }

    fclose(file);
    printf("정렬된 숫자들이 %s에 저장되었습니다.\n", filename);
}
