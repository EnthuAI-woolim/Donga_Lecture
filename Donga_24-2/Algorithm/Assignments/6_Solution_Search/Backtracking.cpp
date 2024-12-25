#include <iostream>
#include <vector>
#include <climits> // INT_MAX 사용

using namespace std;

#define INF INT_MAX

// 전역 변수: 최적 경로와 최소 비용
int bestCost = INF;
vector<int> bestTour;

void BacktrackTSP(vector<vector<int>>& graph, vector<int>& currentTour, vector<bool>& visited, int currentCost, int level) {
    int n = graph.size(); // 그래프 크기

    // 완전한 해에 도달한 경우
    if (level == n) {
        // 마지막 노드에서 시작 노드로 돌아가는 비용 추가
        int returnCost = graph[currentTour[level - 1]][currentTour[0]];
        if (returnCost != INF && currentCost + returnCost < bestCost) {
            bestCost = currentCost + returnCost;
            bestTour = currentTour;
            bestTour.push_back(currentTour[0]); // 경로를 닫음 (마지막 -> 시작)
        }
        return;
    }

    // 가능한 다음 노드 탐색
    for (int v = 0; v < n; ++v) {
        if (!visited[v] && graph[currentTour[level - 1]][v] != INF) {
            // 방문할 수 있는 노드 추가
            visited[v] = true;
            currentTour[level] = v;

            // 백트래킹
            BacktrackTSP(graph, currentTour, visited, currentCost + graph[currentTour[level - 1]][v], level + 1);

            // 백트래킹 후 복원
            visited[v] = false;
        }
    }
}

int main() {
    // 그래프 초기화
    vector<vector<int>> graph = {
        {0,  2,  7,  3, 10},  // A와 다른 노드 간의 가중치
        {2,  0,  3,  5,  4},  // B와 다른 노드 간의 가중치
        {7,  3,  0,  6,  1},  // C와 다른 노드 간의 가중치
        {3,  5,  6,  0,  9},  // D와 다른 노드 간의 가중치
        {10, 4,  1,  9,  0}   // E와 다른 노드 간의 가중치
    };

    int n = graph.size();
    vector<int> currentTour(n); // 현재 경로
    vector<bool> visited(n, false); // 방문 여부

    // 시작점 설정 (노드 0부터 시작)
    currentTour[0] = 0;
    visited[0] = true;

    // 백트래킹 호출
    BacktrackTSP(graph, currentTour, visited, 0, 1);

    // 결과 출력
    if (bestCost != INF) {
        cout << "최소 비용: " << bestCost << endl;
        cout << "최적 경로: ";
        for (int node : bestTour) {
            cout << node << " ";
        }
        cout << endl;
    } else {
        cout << "해결 불가능한 경로입니다." << endl;
    }

    return 0;
}
