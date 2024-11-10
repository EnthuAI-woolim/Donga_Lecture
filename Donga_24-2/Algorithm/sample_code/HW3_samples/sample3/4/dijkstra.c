#include <stdio.h>
#include <limits.h>
#include <time.h>

#define N 10 // 도시 개수
#define INF INT_MAX

// 거리 행렬
int graph[N][N] = {
    {0, 12, INF, INF, 15, INF, INF, INF, INF, INF},  // 서울
    {12, 0, 4, 10, INF, INF, INF, INF, INF, INF},    // 천안
    {INF, 4, 0, 3, INF, INF, INF, INF, 13, INF},    // 논산
    {INF, 10, 3, 0, INF, INF, 10, INF, INF, INF},   // 대전
    {15, INF, INF, INF, 0, 21, 7, INF, INF, INF},  // 원주
    {INF, INF, INF, INF, 21, 0, INF, 25, INF, INF},  // 강릉
    {INF, INF, INF, 10, INF, INF, 0, 19, INF, 9},     // 대구
    {INF, INF, INF, INF, INF, 25, 19, 0, INF, 5},   // 포항
    {INF, INF, 13, INF, INF, INF, INF, INF, 0, 15},  // 광주
    {INF, INF, INF, INF, INF, INF, 9, 5, 15, 0}    // 부산
};

void dijkstra(int start, int dist[]) {
    int visited[N] = { 0 };

    // 시작점 거리 초기화
    for (int i = 0; i < N; i++) {
        dist[i] = INF;
    }
    dist[start] = 0;

    for (int count = 0; count < N - 1; count++) {
        int min = INF, u;

        // 방문하지 않은 점 중 최단 거리의 점 선택
        for (int v = 0; v < N; v++) {
            if (!visited[v] && dist[v] < min) {
                min = dist[v];
                u = v;
            }
        }

        // 선택한 점 방문 처리
        visited[u] = 1;

        // 거리 갱신
        for (int v = 0; v < N; v++) {
            if (!visited[v] && graph[u][v] != INF && dist[u] != INF && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
            }
        }
    }
}

void printDistances(int allDistances[N][N]) {
    printf("      서울  천안  논산  대전  원주  강릉  대구  포항  광주  부산\n");
    printf("----------------------------------------------------------------\n");
    for (int i = 0; i < N; i++) {
        printf("%s ", i == 0 ? "서울" : i == 1 ? "천안" : i == 2 ? "논산" : i == 3 ? "대전" :
            i == 4 ? "원주" : i == 5 ? "강릉" : i == 6 ? "대구" : i == 7 ? "포항" :
            i == 8 ? "광주" : "부산");
        for (int j = 0; j < N; j++) {
            if (allDistances[i][j] == INF) {
                printf("  INF ");
            }
            else {
                printf("%5d ", allDistances[i][j]);
            }
        }
        printf("\n");
    }
}

int main() {
    int allDistances[N][N];
    struct timespec start_time, end_time;
    double time_taken;

    // 시작 시간 기록
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // 각 도시를 출발점으로 하여 최단 거리 계산
    for (int i = 0; i < N; i++) {
        dijkstra(i, allDistances[i]);
    }

    // 종료 시간 기록
    clock_gettime(CLOCK_MONOTONIC, &end_time);

    // 실행 시간 계산 (밀리초로 변환)
    time_taken = (end_time.tv_sec - start_time.tv_sec) * 1e3;
    time_taken += (end_time.tv_nsec - start_time.tv_nsec) / 1e6;

    // 최단 거리 출력
    printDistances(allDistances);

    // 실행 시간 출력
    printf("\nDijkstra Shortest Path algorithm Running Time: %.3f ms\n", time_taken);

    return 0;
}

