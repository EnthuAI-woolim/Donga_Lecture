#include <iostream>
#include <climits>
#include <vector>
#include <iomanip>

int main() {
    int d[100], n, k = 0;

    std::cout << "Enter the coin values (end with a negative number):\n";
    while (true) {
        int coin;
        std::cin >> coin;
        if (coin < 0) break;
        d[k++] = coin;
    }

    std::cout << "Enter the Change (n): ";
    std::cin >> n;


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
    std::cout << "\n\n";

    return 0;
}
