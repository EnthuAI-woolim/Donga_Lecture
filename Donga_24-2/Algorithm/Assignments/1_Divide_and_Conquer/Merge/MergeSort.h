#include <iostream>
#include <fstream>
#include <vector>

std::vector<int> readNumbersFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다!" << std::endl;
        return {};
    }

    std::vector<int> numbers;
    int num;
    while (file >> num) {
        numbers.push_back(num);
    }
    file.close();
    
    return numbers;
}

void writeArrayToFile(const std::string& filename, int* arr, int size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "파일을 열 수 없습니다!" << std::endl;
        return;
    }

    for (int i = 0; i < size; ++i) {
        file << arr[i] << std::endl;
    }

    file.close(); 
}

void merge(int *arr, int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int *L = new int[n1], *R = new int[n2];

    for (i = 0; i < n1; i++) L[i] = arr[l + i];
    for (j = 0; j < n2; j++) R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 || j < n2) {
        if (j >= n2 || (i < n1 && L[i] <= R[j])) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    delete[] L;
    delete[] R;
}

void mergeSort(int *arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void show(int *arr, int size) {
    std::cout << "Sorted array : ";
    for (int i = 0; i < size; i++) std::cout << arr[i] << " ";
    std::cout << "\n";
}