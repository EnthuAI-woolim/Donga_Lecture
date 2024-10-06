#include "./MergeSort.h"
#include <ctime>

int main() {
    clock_t start, end;
    std::vector<int> numbers = readNumbersFromFile("../input_sort.txt");
    if (numbers.empty()) {
        return 1;
    }

    int size = numbers.size();
    int *arr = new int[size];
    for (int i = 0; i < size; ++i) arr[i] = numbers[i];

    start = clock();
    mergeSort(arr, 0, size - 1);
    end = clock();

    printf("running time : %lfms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    
    writeArrayToFile("../output_merge_sort.txt", arr, size);

    delete[] arr;
    return 0;
}
