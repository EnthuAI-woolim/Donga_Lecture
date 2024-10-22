#include <iostream>
#include <fstream>
#include <cmath>

void MergeSort(int* Array, int p, int q);

int main(void) {
    // txt 파일 읽기
    std::string ifname = "inupt_sort.txt";
    std::string ofname = "output_merge_sort.txt";
    std::ifstream file(ifname);

    // 배열 생성 (동적 할당)
    int *targetArray = new int[1];

    // 동적으로 배열의 크기를 늘리면서 입력 받음
    int capacity = 1, size = 0, number = 0;
    while(file >> number) {
        if (capacity < size) {
            int* newArray = new int[capacity * 2];
            capacity *= 2;
            for(int i = 0; i < size; i++) {
                newArray[i] = targetArray[i];
            }
            delete [] targetArray;
            targetArray = newArray;
            newArray = NULL;
        }
        targetArray[size] = number;
        size++;
    }
    file.close();

    // MergeSort 실행
    clock_t start = clock();
    MergeSort(targetArray, 0, size-1);
    std::cout << "수행시간: " << (double) (clock() - start) / CLOCKS_PER_SEC << "\n"; // 실행 시간 측정 및 출력

    // 파일 출력
    std::ofstream ofile(ofname);
    for (int i = 0; i < size; i++) {
        ofile << targetArray[i] << "\n";
    }

    ofile.close();

    return 0;
}

void MergeSort(int* Array, int p, int q) {
    if (p < q) {
        int k = floor((p + q) / 2);

        MergeSort(Array, p, k); // 분할 - 재귀 호출 (A)
        MergeSort(Array, k+1, q); // 분할 - 재귀 호출 (B)

        // 합병
        int i = p, j = k + 1, n = 0; // i, j를 분할한 배열의 첫번째 인덱스를 가리키도록 설정
        // n을 임시 배열의 포인터로 사용
        int sorted[p+q+1]; // 정렬을 위한 임시 배열
        while (i <= k && j <= q) { // i, j가 각 배열의 마지막 인덱스를 가리킫 때까지 반복
            if (Array[i] <= Array[j]) sorted[n++] = Array[i++];
            else sorted[n++] = Array[j++];
        }

        if (i > k) { // 남은 부분에 대한 처리
            while (j <= q) sorted[n++] = Array[j++];
        } else {
            while (i <= k) sorted[n++] = Array[i++];
        }

        for (i = p, n = 0; i <= q; i++, n++) {
            Array[i] = sorted[n];
        }
    }
}