#include "./SelectionSort.h"
#include <time.h>

int main(int argc, const char *argv[]) {
    int size;
    clock_t start, end;

    int* arr = readNumbersFromFile("../input_sort.txt", &size);
    if (arr == NULL || size == 0) return 1;
    
    start = clock(); 
    selectionSort(arr, size); 
    end = clock();
    printf("MS : %lf\n", (double)(end - start) / CLOCKS_PER_SEC * 1000.0);

    printf("50번째 : %d\n", arr[49]);
    printf("70번째 : %d\n", arr[69]);

    free(arr); 
    return 0;
}
