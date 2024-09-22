#include "./merge.h"

int main() {
    std::vector<int> numbers = readNumbersFromFile("../input_sort.txt");
    if (numbers.empty()) {
        return 1;
    }

    int size = numbers.size();
    int *arr = new int[size];
    for (int i = 0; i < size; ++i) {
        arr[i] = numbers[i];
    }

    mergeSort(arr, 0, size - 1);
    
    writeArrayToFile("../output_merge_sort.txt", arr, size);

    delete[] arr;
    return 0;
}
