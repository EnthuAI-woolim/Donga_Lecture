#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstddef>

#define MEMORY_SIZE 100

extern void readFile(const std::string &filename, std::vector<int> &inputData);
extern void writeFile(const std::string &filename, std::vector<int> &outputData);
extern void internalSort(const std::vector<int> &data, std::vector<std::vector<int>> &inputHDD);


int main() {
    int memoryBlockSize = MEMORY_SIZE / 2;
    std::vector<int> data;
    std::vector<std::vector<int>> inputHDD;
    std::vector<std::vector<int>> outputHDD;

    std::vector<std::vector<int>> memory(2);
    std::vector<int> block;
    
    readFile("input.txt", data);
    internalSort(data, inputHDD);

    
    while (inputHDD.size() > 1) {
        
        for (int i = 0; i < 2; ++i) {
            while (memory[i].size() < memoryBlockSize && !inputHDD[i].empty()) {
                memory[i].push_back(inputHDD[i][0]);
                inputHDD[i].erase(inputHDD[i].begin());
            }
        }

        if (memory[1].empty() || (!memory[0].empty() && memory[0][0] <= memory[1][0])) {
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

        if (memory[0].empty() && memory[1].empty()) {
            outputHDD.push_back(block);
            block.clear();
            inputHDD.erase(inputHDD.begin(), inputHDD.begin() + 2);
        }
        
        if (inputHDD.size() < 2) {
             if (inputHDD.size() == 1) {
                outputHDD.push_back(inputHDD[0]);
            }
            
            inputHDD = outputHDD;
            outputHDD.clear();
        } 
        
    }


    writeFile("external_output.txt", inputHDD[0]);

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


void internalSort(const std::vector<int> &data, std::vector<std::vector<int>> &inputHDD) {
    size_t totalSize = data.size(); // 전체 데이터 크기
    size_t numBlock = (totalSize + MEMORY_SIZE - 1) / MEMORY_SIZE; // MEMORY 크기로 나눌 때 필요한 청크 개수 계산

    for (size_t i = 0; i < numBlock; ++i) {
        size_t start = i * MEMORY_SIZE;
        size_t end = std::min(start + MEMORY_SIZE, totalSize);

        std::vector<int> block(data.begin() + start, data.begin() + end);

        std::sort(block.begin(), block.end());

        inputHDD.push_back(block);
    }
}