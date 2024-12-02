#include <iostream>
#include <fstream>
#include <string>

#define MAX_NUMBERS 10000

extern int readFile(const std::string &filename, int *A, int max_numbers);
extern void writeFile(const std::string &filename, int *A, int n);

int main() {
    int A[MAX_NUMBERS];
    int n = readFile("input.txt", A, MAX_NUMBERS);
    if (n < 0) {
        std::cout << "파일을 읽는 데 문제가 발생했습니다.\n";
        return 1;
    }

    // Insertion Sort
    for (int i = 1; i < n; ++i) {
        int CurrentElement = A[i];
        int j = i - 1;

        while(j >= 0 && A[j] > CurrentElement) {
            A[j+1] = A[j];
            j--;
        }
        A[j+1] = CurrentElement;
    }

    writeFile("insertion_output.txt", A, n);


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
