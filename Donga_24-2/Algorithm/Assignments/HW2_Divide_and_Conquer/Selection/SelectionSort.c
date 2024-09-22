#include "./selection.h"

int main(int argc, const char *argv[])
{
    int size;
    int* arr = readNumbersFromFile("../input_sort.txt", &size);
    if (arr == NULL || size == 0) {
        return 1; // 파일이 비어 있거나 읽기 실패
    }

    selectionSort(arr, size);
    writeArrayToFile("../output_selection_sort.txt", arr, size);

    printf("50번째 : %d\n", arr[49]);
    printf("70번째 : %d\n", arr[69]);

    free(arr); 
    return 0;
}
