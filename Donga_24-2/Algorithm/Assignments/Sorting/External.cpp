#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstddef>

#define MEMORY_SIZE 100 // 한 번에 읽을 숫자의 개수

extern void readFile(const std::string &filename, std::vector<int> &inputData);
extern void writeFile(const std::string &filename, int *A, int n);




int internalSort(const std::vector<int> &inputData, std::vector<std::vector<int>> &HDD1) {
    size_t totalSize = inputData.size(); // 전체 데이터 크기
    size_t numBlock = (totalSize + MEMORY_SIZE - 1) / MEMORY_SIZE; // MEMORY 크기로 나눌 때 필요한 청크 개수 계산

    for (size_t i = 0; i < numBlock; ++i) {
        // 시작 인덱스와 끝 인덱스 계산
        size_t start = i * MEMORY_SIZE;
        size_t end = std::min(start + MEMORY_SIZE, totalSize);

        // MEMORY 만큼 데이터를 복사
        std::vector<int> block(inputData.begin() + start, inputData.begin() + end);

        // 내부 정렬 수행
        std::sort(block.begin(), block.end());

        // 정렬된 청크를 HDD1에 추가
        HDD1.push_back(block);
    }
    return static_cast<int>(numBlock);
}

// 보조기억장치에서 처리하는 것들은 일반 메모리 할당
// 메모리에서 처리되는 데이터들은 동적 메모리 할당
int main() {
    std::vector<int> inputData;
    std::vector<std::vector<int>> HDD1;
    std::vector<std::vector<int>> HDD2;
    readFile("input.txt", inputData);

    // MEMORY만큼 읽어 들인후 내부정렬하는 함수하고  HDD1에 저장
    int numBlock = internalSort(inputData, HDD1);

    // while(HDD에 저장된 블록 수 > 1)
    // 입력 HDD에서 블록을 2개씩 선택함
    // 각각의 블록으로부터 데이터를 부분적으로 메모리에 읽어들임(각각 50개씩)
    // 각각을 비교하여 더 작은 수를 출력 HDD에 저장 and 빠져나간 공간에 입력 HDD에서부터 해당 블록에서 새로운 데이터를 읽어들임
    // 위 과정을 진행 후 
    // if HDD에 저장된 블록 수가 홀수일시(1개) 
    //      마지막 블록은 그대로 출력 HDD에 저장
    //      입력과 출력 HDD의 역할을 바꾼다.
    while (numBlock > 1) {
        std::vector<std::vector<int>> memory;
        int idx1 = 0;
        int idx2 = 1;
        int blockSize = MEMORY_SIZE / 2;
        
        while(HDD1[0].size() == 0 and HDD2[1].size() == 0)
        // 블록 제거하는 코드
        HDD1.erase(HDD1.begin() + idx1);
        HDD1.erase(HDD1.begin() + idx2);
    }


    

    return 0;
}

void readFile(const std::string &filename, std::vector<int> &inputData) {
    std::ifstream file(filename);
    if (!file) {
        std::cout << "파일을 열 수 없습니다.\n";
        return;
    }

    int num;
    while (file >> num) {
        inputData.push_back(num);
    }

    
    file.close();
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