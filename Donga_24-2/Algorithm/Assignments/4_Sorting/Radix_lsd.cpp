#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>

#define MAX_NUMBERS 10000
#define BASE 10

int readFile(const std::string &filename, int *A, int max_numbers) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "파일을 열 수 없습니다.\n";
        return -1;
    }

    int n = 0;
    while (file >> A[n]) {
        n++;
        if (n >= max_numbers) {
            std::cerr << "저장 가능한 최대 숫자 개수를 초과했습니다.\n";
            break;
        }
    }

    file.close();
    return n;
}

void writeFile(const std::string &filename, int *A, int n) {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "결과 파일을 열 수 없습니다.\n";
        return;
    }

    for (int i = 0; i < n; ++i) {
        file << A[i] << "\n";
    }

    file.close();
    std::cout << filename << "을 생성하였습니다.\n";
}

void radix_lsd(int* A, int n) {
    int max_value = *std::max_element(A, A + n); // 배열에서 최대값 찾기
    int exp = 1; // 자리수 표현 (1의 자리부터 시작)

    // 최대값의 자리수만큼 반복
    while (max_value / exp > 0) {
        int temp[n];
        int count[BASE] = {0};

        // 현재 자리수의 숫자 개수 카운트
        for (int i = 0; i < n; ++i) {
            int digit = (A[i] / exp) % BASE;
            count[digit]++;
        }

        // 누적합 계산 (자리수의 실제 인덱스 계산)
        for (int i = 1; i < BASE; ++i) {
            count[i] += count[i - 1];
        }

        // 정렬된 값을 임시 배열에 저장 (역순으로 처리하여 안정성 유지)
        for (int i = n - 1; i >= 0; --i) {
            int digit = (A[i] / exp) % BASE;
            temp[--count[digit]] = A[i];
        }

        // 정렬된 값을 원래 배열로 복사
        for (int i = 0; i < n; ++i) {
            A[i] = temp[i];
        }

        exp *= BASE; // 다음 자리수로 이동
    }
}

int main() {
    int A[MAX_NUMBERS];
    int n = readFile("input.txt", A, MAX_NUMBERS);
    if (n < 0) {
        std::cerr << "파일을 읽는 데 문제가 발생했습니다.\n";
        return 1;
    }

    radix_lsd(A, n);

    writeFile("radix_lsd_output.txt", A, n);

    return 0;
}
