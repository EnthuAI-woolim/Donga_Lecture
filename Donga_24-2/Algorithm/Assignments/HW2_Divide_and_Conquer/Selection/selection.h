#include <assert.h>
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
    int capacity = 10; // 초기 용량 설정
    numbers = (int*)malloc(capacity * sizeof(int));
    if (numbers == NULL) {
        fprintf(stderr, "메모리 할당 실패!\n");
        fclose(file);
        return NULL;
    }

    while (fscanf(file, "%d", &num) == 1) {
        if (*size >= capacity) {
            capacity *= 2; // 용량 두 배로 증가
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

    for (int i = 0; i < size; ++i) {
        fprintf(file, "%d\n", arr[i]);
    }

    fclose(file);
}

void swap(int *first, int *second)
{
    int temp = *first;
    *first = *second;
    *second = temp;
}

void selectionSort(int *arr, int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        int min_index = i;
        for (int j = i + 1; j < size; j++)
        {
            if (arr[min_index] > arr[j])
            {
                min_index = j;
            }
        }
        if (min_index != i)
        {
            swap(arr + i, arr + min_index);
        }
    }
}