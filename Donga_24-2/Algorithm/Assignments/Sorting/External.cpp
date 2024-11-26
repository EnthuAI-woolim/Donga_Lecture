#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#define MEMORY 100 // 한 번에 읽을 숫자의 개수

// 파일에서 숫자 100개씩 읽어오는 함수
void readNumbers(std::string& filename, std::vector<int>& input) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
        return;
    }

    int number; // 파일에서 읽을 데이터를 저장할 변수
    std::vector<int> chunk; // 100개의 데이터를 임시로 저장할 벡터

    while (file >> number) {
        chunk.push_back(number); // 데이터를 chunk에 추가

        // chunk가 100개가 되면 input에 추가하고 초기화
        if (chunk.size() == MEMORY) {
            input.insert(input.end(), chunk.begin(), chunk.end());
            chunk.clear(); // chunk를 비워 다음 데이터를 받을 준비
        }   
    }

    // 남아 있는 데이터를 input에 추가 (마지막 100개 이하의 데이터)
    if (!chunk.empty()) {
        input.insert(input.end(), chunk.begin(), chunk.end());
    }

    file.close();
}


int main() {
    std::string inputfile = "input.txt";
    std::string outputfile = "external_output.txt";
    std::vector<int> input; // 100개 크기의 배열 동적 할당
    readNumbers(inputfile, input); // 숫자 읽기

    

    

    return 0;
}