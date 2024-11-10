#include <vector>
#include <climits>

#define MIN(a, b) ((a) < (b) ? (a) : (b))

int main() {
    int n = 0;
    int D[n+1][n+1];  // (n + 1) x (n + 1) 크기의 배열 생성
    
    // 초기화
    for (int i = 0; i <= n; i++) 
        for (int j = 0; j <= n; j++) 
            D[i][j] = INT_MAX;

    for (int k = 1; k <= n; ++k) {
        for (int i = 1; i <= n; ++i) { // i != k
            if (i == k) continue;
            for (int j = 1; j <= n; ++j) {
                if (j == k || j == i) continue;
                D[i][j] = MIN( D[i][k] + D[k][j], D[i][j] );
            }
        }
    }


}