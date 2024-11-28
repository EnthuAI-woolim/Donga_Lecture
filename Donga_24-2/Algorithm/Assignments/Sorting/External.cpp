#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstddef>

#define MEMORY_SIZE 100 // 한 번에 읽을 숫자의 개수

extern void readFile(const std::string &filename, std::vector<int> &inputData);
extern void writeFile(const std::string &filename, std::vector<int> &outputData);

void internalSort(const std::vector<int> &inputData, std::vector<std::vector<int>> &HDD1) {
    size_t totalSize = inputData.size(); // 전체 데이터 크기
    size_t numBlock = (totalSize + MEMORY_SIZE - 1) / MEMORY_SIZE; // MEMORY 크기로 나눌 때 필요한 청크 개수 계산

    for (size_t i = 0; i < numBlock; ++i) {
        size_t start = i * MEMORY_SIZE;
        size_t end = std::min(start + MEMORY_SIZE, totalSize);

        std::vector<int> block(inputData.begin() + start, inputData.begin() + end);

        std::sort(block.begin(), block.end());

        HDD1.push_back(block);
    }
}

int main() {
    int blockSize = MEMORY_SIZE / 2;
    std::vector<int> inputData;
    std::vector<std::vector<int>> inputHDD;
    std::vector<std::vector<int>> outputHDD;
    std::vector<int> block;
    printf("==");
    readFile("input.txt", inputData);
    
    internalSort(inputData, inputHDD);
    
    while (inputHDD.size() > 1) {
        std::vector<std::vector<int>> memory(2);

        while (!inputHDD[0].empty() || !inputHDD[1].empty()) {
            
            for (int i = 0; i < 2; ++i) {
                while (memory[i].size() < blockSize && !inputHDD[i].empty()) {
                    memory[i].push_back(inputHDD[i][0]);
                    inputHDD[i].erase(inputHDD[i].begin());
                }
            }

            if (!memory[0].empty() || (memory[1].empty() && memory[0][0] <= memory[1][0])) {
                block.push_back(memory[0][0]);
                memory[0].erase(memory[0].begin());
                if (!inputHDD[0].empty()) {
                    memory[0].push_back(inputHDD[0][0]);
                    inputHDD[0].erase(inputHDD[0].begin());
                }
            } else {
                block.push_back(memory[1][0]);
                memory[1].erase(memory[1].begin());
                if (!inputHDD[1].empty()) {
                    memory[1].push_back(inputHDD[1][0]);
                    inputHDD[1].erase(inputHDD[1].begin());
                }
            }

            if (inputHDD[0].empty() && inputHDD[1].size()) {
                outputHDD.push_back(block);
                block.clear();
                inputHDD.erase(inputHDD.begin(), inputHDD.begin() + 2);
            }

            

            // if (!block.empty()) {
            //     outputHDD.push_back(block);
            //     block.clear();
            // }
        }

        outputHDD.push_back(inputHDD[0]);
        inputHDD.clear();

        inputHDD = outputHDD;
        outputHDD.clear();
    }

    std::vector<int> finalResult = inputHDD[0];
    writeFile("output.txt", finalResult);

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

void writeFile(const std::string &filename, std::vector<int> &outputData) {
    std::ofstream file(filename);
    if (!file) {
        std::cout << "파일을 열 수 없습니다.\n";
        return;
    }

    for (const int &value : outputData) {
        file << value << "\n";
    }

    file.close();
    std::cout << filename << "을 생성하였습니다.\n";
}
















// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cstddef>

// #define MEMORY_SIZE 100 // 한 번에 읽을 숫자의 개수

// extern void readFile(const std::string &filename, std::vector<int> &inputData);
// extern void writeFile(const std::string &filename, std::vector<int> &outputData);




// void internalSort(const std::vector<int> &inputData, std::vector<std::vector<int>> &HDD1) {
//     size_t totalSize = inputData.size(); // 전체 데이터 크기
//     size_t numBlock = (totalSize + MEMORY_SIZE - 1) / MEMORY_SIZE; // MEMORY 크기로 나눌 때 필요한 청크 개수 계산

//     for (size_t i = 0; i < numBlock; ++i) {
//         // 시작 인덱스와 끝 인덱스 계산
//         size_t start = i * MEMORY_SIZE;
//         size_t end = std::min(start + MEMORY_SIZE, totalSize);

//         // MEMORY 만큼 데이터를 복사
//         std::vector<int> block(inputData.begin() + start, inputData.begin() + end);

//         // 내부 정렬 수행
//         std::sort(block.begin(), block.end());

//         // 정렬된 청크를 HDD1에 추가
//         HDD1.push_back(block);
//     }
// }

// // 보조기억장치에서 처리하는 것들은 일반 메모리 할당
// // 메모리에서 처리되는 데이터들은 동적 메모리 할당
// int main() {
//     int blockSize = MEMORY_SIZE / 2;
//     std::vector<int> inputData;
//     std::vector<std::vector<int>> inputHDD;
//     std::vector<std::vector<int>> outputHDD;
//     std::vector<int> block;
//     readFile("input.txt", inputData);

//     // MEMORY만큼 읽어 들인후 내부정렬하는 함수하고 inputHDD에 저장
//     internalSort(inputData, inputHDD);

//     // while(HDD에 저장된 블록 수 > 1)
//     // 입력 HDD에서 블록을 2개씩 선택함
//     // 각각의 블록으로부터 데이터를 부분적으로 메모리에 읽어들임(각각 50개씩)
//     // 각각을 비교하여 더 작은 수를 출력 HDD에 저장 and 빠져나간 공간에 입력 HDD에서부터 해당 블록에서 새로운 데이터를 읽어들임
//     // 위 과정을 진행 후 
//     // if HDD에 저장된 블록 수가 홀수일시(1개) 
//     //      마지막 블록은 그대로 출력 HDD에 저장
//     //      입력과 출력 HDD의 역할을 바꾼다.
//      // 외부 정렬 시작
//     while (inputHDD.size() > 1) {
//         std::vector<std::vector<int>> memory(2);

//         // 입력 HDD에서 블록 2개씩 읽어 병합
//         while (!inputHDD[0].empty() || !inputHDD[1].empty()) {
//             // Memory 채우기
//             for (int i = 0; i < 2; ++i) {
//                 while (memory[i].size() < blockSize && !inputHDD[i].empty()) {
//                     memory[i].push_back(inputHDD[i][0]);
//                     inputHDD[i].erase(inputHDD[i].begin());
//                 }
//             }

//             // 병합 작업

//             // 두 블록의 가장 작은 값 비교
//             if (!memory[0].empty() && (memory[1].empty() || memory[0][0] <= memory[1][0])) {
//                 block.push_back(memory[0][0]);
//                 memory[0].erase(memory[0].begin());
//                 if (!inputHDD.empty() && !inputHDD[0].empty()) {
//                     memory[0].push_back(inputHDD[0][0]);
//                     inputHDD[0].erase(inputHDD[0].begin());
//                 }
//             } else if (!memory[1].empty()) {
//                 block.push_back(memory[1][0]);
//                 memory[1].erase(memory[1].begin());
//                 if (!inputHDD.empty() && !inputHDD[1].empty()) {
//                     memory[1].push_back(inputHDD[1][0]);
//                     inputHDD[1].erase(inputHDD[1].begin());
//                 }
//             }

//             // 입력 HDD에서 두 블록을 처리한 후 삭제
//             if (inputHDD.size() > 1) {
//                 inputHDD.erase(inputHDD.begin(), inputHDD.begin() + 2);  // 두 블록을 처리한 후 입력에서 제거
//             }

//             // 만약 inputHDD가 하나만 남았다면, 해당 블록을 outputHDD로 이동
//             if (inputHDD.size() == 1) {
//                 outputHDD.push_back(inputHDD[0]);
//                 inputHDD.clear();  // inputHDD를 비운다
//             }

//             // block에 데이터가 있으면 outputHDD에 추가
//             if (!block.empty()) {
//                 outputHDD.push_back(block);
//                 block.clear();  // block 비우기
//             }
            
//         }

//         // 역할 교체: inputHDD와 outputHDD
//         inputHDD = outputHDD;
//         outputHDD.clear();
//     }

//     // 최종 정렬 결과 출력
//     std::vector<int> finalResult = inputHDD[0]; // 최종 정렬된 데이터
//     writeFile("output.txt", finalResult);
    


    

//     return 0;
// }

// void readFile(const std::string &filename, std::vector<int> &inputData) {
//     std::ifstream file(filename);
//     if (!file) {
//         std::cout << "파일을 열 수 없습니다.\n";
//         return;
//     }

//     int num;
//     while (file >> num) {
//         inputData.push_back(num);
//     }

    
//     file.close();
// }

// void writeFile(const std::string &filename, std::vector<int> &outputData) {
//     std::ofstream file(filename);
    
//     // 파일 열기 실패 시 에러 처리
//     if (!file) {
//         std::cout << "파일을 열 수 없습니다.\n";
//         return;
//     }

//     // 벡터의 모든 데이터를 파일에 작성
//     for (const int &value : outputData) {
//         file << value << "\n";
//     }

//     file.close();
//     std::cout << filename << "을 생성하였습니다.\n";
// }