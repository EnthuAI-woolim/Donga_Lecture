#include <iostream>
#include <fstream>
#include <cmath>
#include <unordered_map>

#define MAX_NUMBERS 10000
using namespace std;

int readFile(const string &filename, int *A, int max_numbers) {
    ifstream file(filename);
    if (!file) {
        cout << "파일을 열 수 없습니다.\n";
        return -1;
    }
    int n = 0;
    while (file >> A[n]) {
        n++;
        if (n >= max_numbers) {
            cout << "저장 가능한 최대 숫자 개수를 초과했습니다.\n";
            break;
        }
    }
    file.close();
    return n;
}

void writeFile(const string &filename, int *A, int n) {
    ofstream file(filename);
    if (!file) {
        cout << "결과 파일을 열 수 없습니다.\n";
        return;
    }
    for (int i = 0; i < n; ++i) {
        file << A[i] << "\n";
    }
    file.close();
    cout << filename << "을 생성하였습니다.\n";
}

int digit_at(int x, int d) {
    return (int)(x / pow(10, d - 1)) % 10;
}

void MSD_sort(int* arr, int lo, int hi, int d) {
    if (hi <= lo) return;

    int count[12] = { 0 };
    unordered_map<int, int> temp;

    for (int i = lo; i <= hi; i++) count[digit_at(arr[i], d) + 2]++;
    for (int r = 0; r < 11; r++) count[r + 1] += count[r];
    for (int i = lo; i <= hi; i++) temp[count[digit_at(arr[i], d) + 1]++] = arr[i];
    for (int i = lo; i <= hi; i++) arr[i] = temp[i - lo];
    for (int r = 0; r < 10; r++) MSD_sort(arr, lo + count[r], lo + count[r + 1] - 1, d - 1);
}

int getMax(int arr[], int n) {
    int mx = arr[0];
    for (int i = 1; i < n; i++) if (arr[i] > mx) mx = arr[i];
    return mx;
}

void radixsort(int* arr, int n) {
    int d = floor(log10(abs(getMax(arr, n)))) + 1;
    MSD_sort(arr, 0, n - 1, d);
}

int main() {
    int A[MAX_NUMBERS];
    int n = readFile("input.txt", A, MAX_NUMBERS);
    if (n < 0) {
        cout << "파일을 읽는 데 문제가 발생했습니다.\n";
        return 1;
    }

    radixsort(A, n);

    writeFile("radix_msd_output.txt", A, n);

    return 0;
}
