#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>
#include <iomanip>

void readDimensions(const std::string& filename, std::vector<int>& d) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "파일을 열 수 없습니다." << std::endl;
        return;
    }

    std::string line;
    int cols = 0;
    int count = 0;
    bool first_m_row = true;

    while (std::getline(file, line)) {
        int num;
        int option = line.find("]]") != std::string::npos ? 1 : line.find(']') != std::string::npos ? 2 : 0;
        bool first_line = false;

        // = 이 있는 줄이다 (첫째줄이다)
        size_t equal_pos = line.find('=');
        if (equal_pos != std::string::npos) {
            first_line = true;
            line = line.substr(equal_pos + 1);
        }

        line.erase(std::remove(line.begin(), line.end(), '['), line.end());
        line.erase(std::remove(line.begin(), line.end(), ']'), line.end());

        std::stringstream ss(line);
        while (ss >> num) count++;

        if (option == 1) {
            if (first_line) cols = count;
            
            if (first_m_row) {
                d.push_back(count / cols); // 행
                first_m_row = false;
            }
            
            d.push_back(cols); // 열
            cols = 0;
            count = 0;
        } else if (option == 2) {
            
            if (cols == 0) cols = count;

        }
    }

    file.close();
}


int main() {
    std::string filename = "matrix_input.txt";
    std::vector<int> d;

    readDimensions("matrix_input.txt", d);

    int n = d.size() - 1;
    std::vector<std::vector<int>> C(n + 1, std::vector<int>(n + 1, 0));

    for (int L = 1; L <= n - 1; ++L) { // L: 곱해지는 행렬의 갯수
        for (int i = 1; i <= n - L; ++i) { // i: 앞에 곱해지는 행렬의 첫번째 행렬의 인덱스(뒤에 곱하는 행렬 크기가 클수록 i의 범위는 작아짐)
            int j = i + L; // j: 뒤에 곱해지는 행렬의 마지막 행렬의 인덱스를 구함
            C[i][j] = INT_MAX;
            for (int k = i; k <= j - 1; ++k) { // k:앞에 곱해지는 행렬의 마지막 행렬의 인덱스 범위
                int temp = C[i][k] + C[k + 1][j] + d[i - 1] * d[k] * d[j];
                C[i][j] = std::min(C[i][j], temp);
            }
        }
    }

    std::cout << "n : " << n << std::endl;
    std::cout << "d : ";
    for (int i = 0; i < n+1; ++i) std::cout << d[i] << " ";
    std::cout << std::endl << std::endl;

    std::cout << "Minimum multiplications : " << C[1][n] << std::endl;
    std::cout << std::setw(5) << "C";
    for (int i = 1; i <= n; ++i) std::cout << std::setw(5) << i;
    std::cout << std::endl;

    for (int i = 1; i <= n; ++i) {
        std::cout << std::setw(5) << i;
        for (int j = 1; j <= n; ++j) {
            if (i > j) std::cout << std::setw(5) << " ";
            else std::cout << std::setw(5) << C[i][j];
        }
        std::cout << std::endl;
    }


    return 0;
}
