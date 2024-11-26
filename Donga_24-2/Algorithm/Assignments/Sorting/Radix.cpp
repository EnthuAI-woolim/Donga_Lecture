#include <iostream>
#include <fstream>

#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>

#define MAX_NUMBERS 10000
#define BASE 10

extern int readFile(const std::string &filename, int *A, int max_numbers);
extern void writeFile(const std::string &filename, int *A, int n);

// 최대 숫자가 몇자리 숫자인지 구하는 함수
int findK(int* A, int n) {
    int max = 0;
    for (int i = 0; i < n; ++i) {
        if (A[i] > max) {
            max = A[i];
        }
    }
    int k = 0;
    while (max) {
        k++;
        max /= 10;
    }

    return k;
}

int aaa(int* A, int n) {
    int max = 0;
    for (int i = 0; i < n; ++i) {
        if (A[i] > max) {
            max = A[i];
        }
    }
    int exp = 1;
    while (max / exp >= 10) {
        exp *= 10;
    }

    return exp;
}

void radix_lsd(int* A, int n) {
    int k = findK(A, n);
    // Radix Sort
    for (int i = k-1; i >= 0; --i) {
        // 임시 저장할 변수, 각 자리의 숫자들의 갯수 저장 변수
        int temp[n], count[BASE] = {0};

        // 각 자리 숫자가 몇개 있는지 카운트 - O(r)
        for (int j = 0; j < n; ++j) {
            int t = std::pow(10, i);
            count[(A[j] % (10 * t)) / t]++;
        }

        // 카운트된 갯수에 따라 각 자리수의 인덱스 설정 - O(1)
        for (int j = 1; j < BASE; ++j) {
            count[j] += count[j - 1];
        }

        // 각 자리수에 맞는 순서에 저장 - O(n)
        for (int j = n - 1; j >= 0; --j) {
            int t = std::pow(10, i);
            int digit = (A[j] % (10 * t)) / t;
            temp[--count[digit]] = A[j];
        }

        for (int j = 0; j < n; ++j) {
            A[j] = temp[j];
        }
    }
}

// 특정 자리수를 기준으로 배열 정렬 (MSD)
void msdRadixSort(int* arr, int left, int right, int divisor) {
    if (left >= right || divisor == 0) {
        return; // 더 이상 정렬할 필요 없음
    }

    int count[BASE] = {0}; // 각 자릿수 빈도수
    int n = right - left + 1;
    int* temp = new int[n]; // 임시 배열 동적 할당

    // 현재 자리수를 기준으로 빈도수 계산
    for (int i = left; i <= right; ++i) {
        int digit = (arr[i] / divisor) % BASE;
        count[digit]++;
    }

    // 누적 합 계산
    for (int i = 1; i < BASE; ++i) {
        count[i] += count[i - 1];
    }

    // 현재 자리수를 기준으로 정렬
    for (int i = right; i >= left; --i) {
        int digit = (arr[i] / divisor) % BASE;
        temp[--count[digit]] = arr[i];
    }

    // 정렬된 결과를 원래 배열에 복사
    for (int i = 0; i < n; ++i) {
        arr[left + i] = temp[i];
    }

    delete[] temp; // 임시 배열 해제

    // 각 자릿수 범위를 기준으로 재귀 호출
    int start = left;
    for (int i = 0; i < BASE; ++i) {
        int end = left + count[i] - 1;
        if (start <= end) {
            msdRadixSort(arr, start, end, divisor / BASE);
        }
        start = end + 1;
    }
}

// MSD Radix Sort 진입점
void radixMSD(int* arr, int size) {
    if (size <= 0) return;

    // 배열의 최대값 찾기
    int maxVal = *std::max_element(arr, arr + size);

    // 최대값의 자릿수만큼 시작 divisor 설정
    int divisor = 1;
    while (maxVal / divisor >= 10) {
        divisor *= 10;
    }

    // MSD 기반 정렬 시작
    msdRadixSort(arr, 0, size - 1, divisor);
}


int main() {
    // int A[MAX_NUMBERS];
    // int n = readFile("input.txt", A, MAX_NUMBERS);
    // if (n < 0) {
    //     std::cout << "파일을 읽는 데 문제가 발생했습니다.\n";
    //     return 1;
    // }

    // radix_lsd(A, n);

    // writeFile("radix_lsd_output.txt", A, n);

    int B[MAX_NUMBERS];
    int n = readFile("input.txt", B, MAX_NUMBERS);
    if (n < 0) {
        std::cout << "파일을 읽는 데 문제가 발생했습니다.\n";
        return 1;
    }
    

    radixMSD(B, n);

    writeFile("radix_msd_output.txt", B, n);


    return 0;
}


int readFile(const std::string &filename, int *A, int max_numbers) {
    std::ifstream file(filename);
    if (!file) {
        std::cout << "파일을 열 수 없습니다.\n";
        return -1;
    }

    int n = 0;
    while (file >> A[n]) {
        n++;
        if (n >= max_numbers) {
            std::cout << "저장 가능한 최대 숫자 개수를 초과했습니다.\n";
            break;
        }
    }

    file.close();
    return n;
}

void writeFile(const std::string &filename, int *A, int n) {
    std::ofstream file(filename);
    if (!file) {
        std::cout << "결과 파일을 열 수 없습니다.\n";
        return;
    }

    for (int i = 0; i < n; ++i) {
        file << A[i] << "\n";
    }

    file.close();
    std::cout << filename << "을 생성하였습니다.\n";
}