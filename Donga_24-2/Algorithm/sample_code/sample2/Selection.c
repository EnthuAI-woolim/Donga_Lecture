#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int midOfThree(int *arr, int a, int b, int c);
void swap(int *arr, int x, int y);
int selection(int *arr, int p, int q, int k);

int main(void) {
    int size = 0;
    int capacity = 1;
    int *arr = (int*)malloc(sizeof(int) * capacity);

    // 파일 입력
    FILE *fp = fopen("./inupt_sort.txt", "r");
    int input = 0;
    while (fscanf(fp, "%d", &input) != EOF) {
        if (size >= capacity) {
            capacity *= 2;
            int *temp = realloc(arr, sizeof(int) * capacity);
            arr = temp;
        }

        arr[size] = input;
        size++;
    }
    fclose(fp);

    // 50번째 작은 숫자
    clock_t start_fifty = clock(); // 수행 시간 측정
    printf("50번째 작은 숫자: %d\n", selection(arr, 0, size, 50));
    printf("수행 시간: %lf초\n", (double) (clock() - start_fifty) / CLOCKS_PER_SEC);


    // 70번째 작은 숫자
    clock_t start_seventy = clock(); // 수행 시간 측정
    printf("70번째 작은 숫자: %d\n", selection(arr, 0, size, 70));
    printf("수행 시간: %lf초\n", (double) (clock() - start_seventy) / CLOCKS_PER_SEC);

    return 0;
}

// 세 원소의 중간값의 인덱스를 반환
int midOfThree(int *arr, int a, int b, int c) {
    if (arr[a] >= arr[b]) {
        if (arr[b] >= arr[c]) {
            return b;
        } else if (arr[a] <= arr[c]) {
            return a;
        } else {
            return c;
        }
    } else if (arr[a] >= arr[c]) {
        return a;
    } else if (arr[b] >= arr[c]) {
        return c;
    } else {
        return b;
    }
}

void swap(int *arr, int x, int y) {
    int temp = arr[x];
    arr[x] = arr[y];
    arr[y] = temp;
}

int selection(int *arr, int p, int q, int k) {
    int m = floor((p + q) / 2);
    int pivot = midOfThree(arr, p, m, q); // pivot 선택

    // pivot과 p 변경
    swap(arr, pivot, p);
    pivot = p;
    // pivot과 값들을 비교하면서 자리 이동
    int left = p+1, right = q;
    while (left <= right) {
        while (left <= right && arr[left] < arr[pivot]) left++;
        while (left <= right && arr[right] > arr[pivot]) right--;

        if (left <= right) {
            swap(arr, left, right);
            left++;
            right--;
        }
    }

    // pivot 이동
    swap(arr, pivot, right);

    int S = (right - 1) - p + 1;
    if (k <= S) return selection(arr, p, right - 1, k);
    else if (k == S + 1) return arr[right+1];
    else return selection(arr, right + 1, q, k - S - 1);
}