#include <stdio.h>
#include <stdlib.h>

#define MAX_NUMBERS 10000

int readFile(const char *filename, int *A, int max_numbers);
void writeFile(const char *filename, int *A, int n);
void DownHeap(int A[], int n, int i);
void BuildHeap(int A[], int n);

int main() {
    int A[MAX_NUMBERS];
    int n = readFile("input.txt", A, MAX_NUMBERS);
    if (n < 0) {
        printf("파일을 읽는 데 문제가 발생했습니다.\n");
        return 1;
    }

    BuildHeap(A, n);

    for (int i = n; i > 1; --i) {
        int temp = A[i];
        A[i] = A[1];
        A[1] = temp;
        
        DownHeap(A, i-1, 1);
    }

    writeFile("Heap_output.txt", A, n);

    return 0;
}



void DownHeap(int A[], int n, int i) {
    int bigger = i;
    int l = i * 2;
    int r = i * 2 + 1;
    
    if (l <= n && A[l] > A[bigger]) bigger = l;
    if (r <= n && A[r] > A[bigger]) bigger = r;

    if (bigger != i) {
        int temp = A[bigger];
        A[bigger] = A[i];
        A[i] = temp;
        DownHeap(A, n, bigger);
    }
}

void BuildHeap(int A[], int n) {
    for (int i = n / 2; i >= 1; --i) {
        DownHeap(A, n, i);
    }
}

void writeFile(const char *filename, int *A, int n) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("결과 파일을 열 수 없습니다.\n");
        return;
    }

    for (int i = 1; i <= n; ++i) {
        fprintf(file, "%d\n", A[i]);
    }

    fclose(file);
    printf("%s을 생성하였습니다.\n", filename);
}

int readFile(const char *filename, int *A, int max_numbers) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("파일을 열 수 없습니다.\n");
        return -1;
    }

    A[0] = -1;
    int n = 1;
    while (fscanf(file, "%d", &A[n]) == 1) {
        n++;
        if (n >= max_numbers) {
            printf("저장 가능한 최대 숫자 개수를 초과했습니다.\n");
            break;
        }
    }

    fclose(file);
    return n-1;
}


