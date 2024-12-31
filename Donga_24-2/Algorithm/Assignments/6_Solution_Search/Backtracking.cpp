#include <iostream>
#include <vector>
#include <climits> // INT_MAX

using namespace std;

#define INF INT_MAX


int bestDis = INF;
vector<int> bestTour;

void BacktrackTSP(vector<vector<int>>& graph, vector<int>& currentTour, vector<bool>& visited, int currentDis, int level) {
    int n = graph.size(); 

    // 완전한 해에 도달한 경우
    if (level == n) {
        int returnDis = graph[currentTour[level - 1]][currentTour[0]]; // 마지막 노드에서 시작 노드로 돌아가는 거리
        if (returnDis != INF && currentDis + returnDis < bestDis) {
            bestDis = currentDis + returnDis;
            bestTour = currentTour;
            bestTour.push_back(currentTour[0]); // 경로를 닫음 (마지막 -> 시작)
        }
        return;
    }

    // 가능한 다음 노드 탐색
    for (int v = 0; v < n; ++v) {
        if (!visited[v] && graph[currentTour[level - 1]][v] != INF) { // 방문하지 않았고 경로가 존재하면
            visited[v] = true;
            currentTour[level] = v;

            BacktrackTSP(graph, currentTour, visited, currentDis + graph[currentTour[level - 1]][v], level + 1);

            visited[v] = false; // 백트래킹 후 복원
        }
    }
}

int main() {
    // 그래프 초기화
    vector<vector<int>> graph = {
        {0,  2,  7,  3, 10},
        {2,  0,  3,  5,  4},
        {7,  3,  0,  6,  1},
        {3,  5,  6,  0,  9},
        {10, 4,  1,  9,  0}
    };

    int n = graph.size();
    vector<int> currentTour(n);
    vector<bool> visited(n, false);

    currentTour[0] = 0; // 시작점(A)
    visited[0] = true;

    BacktrackTSP(graph, currentTour, visited, 0, 1);

    if (bestDis != INF) {
        cout << "최적 경로: ";
        for (int node : bestTour) {
            cout << static_cast<char>('A' + node) << " ";
        }
        cout << endl;
        cout << "거리: " << bestDis << endl;
    } else {
        cout << "해결 불가능한 경로입니다." << endl;
    }

    return 0;
}
