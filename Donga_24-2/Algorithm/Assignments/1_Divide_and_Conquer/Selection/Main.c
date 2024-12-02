#include "./SelectionSort.h"
#include <time.h>
#include <string.h>

int main(int argc, const char *argv[]) {
    int size;
    int *arr1, *arr2;
    clock_t start, end;

    arr1 = readNumbersFromFile("../input_sort.txt", &size);
    if (arr1 == NULL || size == 0) return 1;

    arr2 = (int*)malloc(size * sizeof(int));
    if (arr2 == NULL) {
        fprintf(stderr, "메모리 할당 실패!\n");
        free(arr1);
        return 1;
    }
    memcpy(arr2, arr1, size * sizeof(int));
    

    start = clock();
    quickSelect(arr1, 0, size - 1, 50);
    end = clock();
    printf("50번째 : %d\n", arr1[49]);
    printf("50번째 소요 시간: %lf ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000.0);
    printf("\n");

    start = clock();
    quickSelect(arr2, 0, size - 1, 70);
    end = clock();
    printf("70번째 : %d\n", arr2[69]);
    printf("70번째 소요 시간: %lf ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000.0);

    free(arr1);
    free(arr2);
    return 0;
}
