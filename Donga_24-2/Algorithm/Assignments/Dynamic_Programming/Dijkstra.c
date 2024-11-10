#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define TRUE 1
#define FALSE 0
#define N 10
#define INF 2147483647

int dis[N][N];
int visited[N];

const char* cities[] = {
    "서울", "천안", "원주", "강릉", "논산",
    "대전", "대구", "포항", "광주", "부산"
};

// 서울 0, 천안 1, 원주 2, 강릉 3, 논산 4, 대전 5, 대구 6, 포항 7, 광주 8, 부산 9
int weight[N][N] = {
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


int update(int dis[], int n, int visited[]) {
    int min = INF, min_pos;
    
    for (int i = 0; i < n; i++) {
        if (dis[i] < min && visited[i] == FALSE) {
            min = dis[i];
            min_pos = i;
        }
    }

    return min_pos;
}

void Shortest_Path_Dijkstra(int s, int n) {
    for (int i = 0; i < n; i++) {
        visited[i] = FALSE;
        dis[s][i] = INF;
    }
    dis[s][s] = 0;

    for (int i = 0; i < n - 1; i++) {
        int u = update(dis[s], n, visited);
        visited[u] = TRUE;

        for (int j = 0; j < n; j++) 
            if (visited[j] == FALSE && weight[u][j] != INF && dis[s][u] + weight[u][j] < dis[s][j]) 
                dis[s][j] = dis[s][u] + weight[u][j]; 
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


int main(void) {
    clock_t start, end;

    start = clock();
    
    for (int start = 0; start < N; start++) 
        Shortest_Path_Dijkstra(start, N);
    
    end = clock();

    // 결과 출력
    printf("\nDijkstra Algorithm\n");
    printGraph(dis, cities);
    printf("\nRunning Time: %lf ms\n\n", (double)(end - start) / CLOCKS_PER_SEC * 1000.0);

    return 0;
}