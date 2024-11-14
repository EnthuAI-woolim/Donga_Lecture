#include <stdio.h>
#include <time.h>

#define N 10
#define INF 10000
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define TRUE 1
#define FALSE 0
int dis[N][N];
int visited[N];

const char* cities[] = {
    "서울", "천안", "원주", "강릉", "논산",
    "대전", "대구", "포항", "광주", "부산"
};

// 서울 0, 천안 1, 원주 2, 강릉 3, 논산 4, 대전 5, 대구 6, 포항 7, 광주 8, 부산 9
int D[N][N] = {
    {0, 12, 15, INF, INF, INF, INF, INF, INF, INF},
    {12, 0, INF, INF, 4, 10, INF, INF, INF, INF},
    {15, INF, 0, 21, INF, INF, 7, INF, INF, INF},
    {INF, INF, 21, 0, INF, INF, INF, 25, INF, INF},
    {INF, 4, INF, INF, 0, 3, INF, INF, 13, INF},
    {INF, 10, INF, INF, 3, 0, 10, INF, INF, INF},
    {INF, INF, 7, INF, INF, 10, 0, 19, INF, 9},
    {INF, INF, INF, 25, INF, INF, 19, 0, INF, 5},
    {INF, INF, INF, INF, 13, INF, INF, INF, 0, 15},
    {INF, INF, INF, INF, INF, INF, 9, 5, 15, 0}       
};


void dijkstra(int s, int n) {
    for (int i = 0; i < n; i++) {
        visited[i] = FALSE;
        dis[s][i] = INF;
    }
    dis[s][s] = 0;

    for (int i = 0; i < n - 1; i++) {
        int min = INF, u;

        for (int j = 0; j < n; j++) {
            if (dis[s][j] < min && visited[j] == FALSE) {
                min = dis[s][j];
                u = j;
            }
        }
        
        visited[u] = TRUE;

        for (int j = 0; j < n; j++) 
            if (visited[j] == FALSE && D[u][j] != INF && dis[s][u] + D[u][j] < dis[s][j]) 
                dis[s][j] = dis[s][u] + D[u][j]; 
    }
}

void printGraph(int D[N][N], const char* cities[]) {
    printf("      ");
    for (int i = 0; i < N; i++) 
        printf("%4s ", cities[i]);
    
    printf("\n");
    for (int i = 0; i < N; i++) {
        printf("%4s ", cities[i]);
        for (int j = 0; j < N; j++) {
            if (j > i) printf("%4d ", D[i][j]);
            else printf("     ");
        }
        printf("\n");
    }
}


int main() {
    clock_t start1, end1, start2, end2;

    // Dijkstra - 새로운 dis[][]에 가중치 저장
    start1 = clock();
    for (int start = 0; start < N; start++) 
        dijkstra(start, N);
    end1 = clock();

    // Floyd - 기존 D[][]에 가중치 업데이트
    start2 = clock();
    for (int k = 0; k < N; ++k) { 
        for (int i = 0; i < N; ++i) { 
            if (i == k) continue;
            for (int j = 0; j < N; ++j) { 
                if (j == k || j == i) continue;
                D[i][j] = MIN(D[i][j], D[i][k] + D[k][j]);
            }
        }
    }
    end2 = clock();

    printf("\nDijkstra Algorithm\n");
    printGraph(dis, cities);
    printf("\nRunning Time: %lf ms\n\n", (double)(end1 - start1) / CLOCKS_PER_SEC * 1000.0);

    printf("\nFloyd-Warshall Algorithm\n");
    printGraph(D, cities);
    printf("\nRunning Time: %lf ms\n\n", (double)(end2 - start2) / CLOCKS_PER_SEC * 1000.0);
    

    return 0;
}


// 각 경유 노드(k)마다 
// 경우의 수가 나올 수 있는 모든 출발(i), 도착(j)을 전부 업데이트해 나가는 알고리즘
// D[i][j] vs D[i][k]->D[k][j] 두 가중치 중 더 작은 값을 D[i][j]에 저장하는 방식

// k: 경유 노드 선택 - h idx 설정
// i: 출발 노드 선택 - h idx 설정
// j: 도착 노드 선택 - w idx 설정
