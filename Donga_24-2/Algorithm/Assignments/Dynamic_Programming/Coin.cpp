#include <iostream>
#include <climits>
#include <vector>
#include <iomanip>

int main() {
    int n = 20;
    int d[] = {16, 10, 5, 1};
    int k = sizeof(d) / sizeof(d[0]);
    std::vector<int> C(n + 1, INT_MAX);

    C[0] = 0;
    for (int j = 1; j <= n; ++j) 
        for (int i = 0; i < k; ++i) 
            if (d[i] <= j && C[j - d[i]] + 1 < C[j]) 
                C[j] = C[j - d[i]] + 1;


    std::cout << std::setw(3) << "j";
    for (int j = 0; j <= n; ++j) 
        std::cout << std::setw(3) << j;
    std::cout << "\n";

    std::cout << std::setw(3) << "c";
    for (int i = 0; i <= n; ++i) 
        std::cout << std::setw(3) << C[i];
    std::cout << "\n";

    std::cout << "\nMin coins for change " << n << ": " << C[n] << std::endl;

    return 0;
}

// 현재의 거스름돈에서 들어갈 수 있는 가장 큰 동전 단위를 뺀 나머지 거스름돈의 최소 동전 갯수를 구하고,
// 거기에 1을 더해주는 값을 현재 거스름돈의 최소 동전 갯수로 설정

// d (동전 단위): 16, 10, 5, 1  /  n (현재 거스름돈): 15
// 1. n에 따른 들어갈 수 있는 최대 d를 구함
// 2. n에서 d를 뺀 거스름돈 금액인 n-d의 값에 1을 더한 값이 n 값이 됨 

// EX
// n = 15일때, d = 10

// 1.
// n - d = 15 - 10 = 5
// n = 5일때의 최소 동전 수는 메모리에 저장되어 있음
// n = 5일때의 값 + 1(d = 10) => n = 15일 때의 최소 동전 갯수

// 2.
// n - d = 15 - 5 = 10
// n = 10일때의 최소 동전 수는 메모리에 저장되어 있음
// n = 10일때의 값 + 1(d = 10) => n = 15일 때의 최소 동전 갯수

// 3.
// n - d = 15 - 1 = 14
// n = 14일때의 최소 동전 수는 메모리에 저장되어 있음
// n = 14일때의 값 + 1(d = 10) => n = 15일 때의 최소 동전 갯수

// => 위  1, 2, 3번을 전부 했을 때, 제일 작은 값이 n = 15에 저장됨

// cf) j - d[i] == 현재 거스름돈 - 들어갈 수 있는 가장 큰 동전 단위 == 가장 큰 동전 단위를 뺀 나머지 거스름돈 금액 
