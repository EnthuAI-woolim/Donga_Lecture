#include <stdio.h>
#include <stdlib.h>

int* readNumbersFromFile(const char* filename, int* size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "파일을 열 수 없습니다!\n");
        return NULL;
    }

    int* numbers = NULL;
    int num;
    *size = 0;
    int capacity = 10;
    numbers = (int*)malloc(capacity * sizeof(int));
    if (numbers == NULL) {
        fprintf(stderr, "메모리 할당 실패!\n");
        fclose(file);
        return NULL;
    }

    while (fscanf(file, "%d", &num) == 1) {
        if (*size >= capacity) {
            capacity *= 2;
            numbers = (int*)realloc(numbers, capacity * sizeof(int));
            if (numbers == NULL) {
                fprintf(stderr, "메모리 할당 실패!\n");
                fclose(file);
                return NULL;
            }
        }
        numbers[*size] = num;
        (*size)++;
    }
    
    fclose(file);
    return numbers;
}

void writeArrayToFile(const char* filename, int* arr, int size) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "파일을 열 수 없습니다!\n");
        return;
    }

    for (int i = 0; i < size; ++i) fprintf(file, "%d\n", arr[i]);

    fclose(file);
}

void swap(int* arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

int partition(int *arr, int first, int r) {
    int mid = (first + r) / 2;
    int pivot = arr[mid];
    int l = first + 1;

    swap(arr, first, mid);
    while (l <= r) {
        while (l <= r && arr[l] < pivot) l++;
        while (l <= r && arr[r] > pivot) r--;

        if (l < r) {
            swap(arr, l, r);
            l++;
            r--;
        }
    }
    swap(arr, first, r);
    return r;
}

void quickSelect(int* arr, int l, int r, int k) {
    int pivot = partition(arr, l, r);
    int s_len = pivot - l;

    if (k <= s_len) quickSelect(arr, l, pivot - 1, k);
    else if (k > s_len + 1) quickSelect(arr, pivot + 1, r, k - s_len - 1);
}