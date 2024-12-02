#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <limits>
#include <queue>

using namespace std;

const int INF = numeric_limits<int>::max();
const int FALSE = 0;
const int TRUE = 1;

// 두 도시 간 거리 계산
double calculateDistance(pair<int, int> p1, pair<int, int> p2) {
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

// 최솟값을 가지는 노드를 찾는 함수
int update(vector<double> &dis, int n, vector<bool> &visited) {
    double min = INF;
    int min_pos = -1;

    for (int i = 0; i < n; i++) {
        if (dis[i] < min && !visited[i]) {
            min = dis[i];
            min_pos = i;
        }
    }

    return min_pos;
}

// 다익스트라 알고리즘
void Shortest_Path_Dijkstra(char start, map<char, pair<int, int>> &edges) {
    int n = edges.size();
    map<char, int> indexMap;
    vector<vector<double>> weight(n, vector<double>(n, INF));
    vector<double> dis(n, INF);
    vector<bool> visited(n, false);

    // 도시 이름을 인덱스로 매핑
    int idx = 0;
    for (auto &[city, coord] : edges) {
        indexMap[city] = idx++;
    }

    // 그래프 가중치 초기화
    for (auto &[city1, coord1] : edges) {
        for (auto &[city2, coord2] : edges) {
            if (city1 != city2) {
                weight[indexMap[city1]][indexMap[city2]] = calculateDistance(coord1, coord2);
            }
        }
    }

    // 시작 노드 초기화
    int startIdx = indexMap[start];
    dis[startIdx] = 0;

    // 다익스트라 알고리즘 실행
    for (int i = 0; i < n - 1; i++) {
        int u = update(dis, n, visited);
        if (u == -1) break;  // 더 이상 방문할 노드가 없는 경우 종료
        visited[u] = true;

        for (int v = 0; v < n; v++) {
            if (!visited[v] && weight[u][v] != INF && dis[u] + weight[u][v] < dis[v]) {
                dis[v] = dis[u] + weight[u][v];
            }
        }
    }

    // 결과 출력
    cout << "Shortest distances from " << start << ":\n";
    for (auto &[city, idx] : indexMap) {
        cout << start << " -> " << city << ": ";
        if (dis[idx] == INF) {
            cout << "INF\n";
        } else {
            cout << dis[idx] << "\n";
        }
    }
}

int main() {
    // 입력 데이터
    map<char, pair<int, int>> edges = {
        {'A', {0, 3}}, {'B', {7, 5}}, {'C', {6, 0}}, {'D', {4, 3}},
        {'E', {1, 0}}, {'F', {5, 3}}, {'H', {4, 1}}, {'G', {2, 2}}
    };

    // 시작점 A에서 최단 거리 계산
    Shortest_Path_Dijkstra('A', edges);

    return 0;
}
