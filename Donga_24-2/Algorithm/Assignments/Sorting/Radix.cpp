#include <iostream>
#include <fstream>

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

int main() {
    int A[MAX_NUMBERS];
    int count[BASE];
    int n = readFile("input.txt", A, MAX_NUMBERS);
    if (n < 0) {
        std::cout << "파일을 읽는 데 문제가 발생했습니다.\n";
        return 1;
    }
    int k = findK(A, n);

    
    // Radix Sort
    for (int i = 0; i < k; ++i) {
        // 임시 저장할 변수, 각 자리의 숫자들의 갯수 저장 변수
        int temp[n], count[10] = {0};

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

    writeFile("radix_lsd_output.txt", A, n);


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