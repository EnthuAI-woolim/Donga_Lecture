#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <set>
#include <limits>
#include <algorithm>

using namespace std;

// 도시 좌표 데이터
map<char, pair<int, int>> coordinates = {
    {'A', {0, 3}}, {'B', {7, 5}}, {'C', {6, 0}}, {'D', {4, 3}},
    {'E', {1, 0}}, {'F', {5, 3}}, {'H', {4, 1}}, {'G', {2, 2}}
};

// 두 도시 간 거리 계산 함수
double calculateDistance(pair<int, int> p1, pair<int, int> p2) {
    return sqrt(pow(p1.first - p2.first, 2) + pow(p1.second - p2.second, 2));
}

// 그래프 생성 함수
vector<vector<double>> buildGraph() {
    int n = coordinates.size();
    vector<vector<double>> graph(n, vector<double>(n, 0));
    map<char, int> indexMap;
    int idx = 0;
    for (auto &[city, coord] : coordinates) {
        indexMap[city] = idx++;
    }

    for (auto &[city1, coord1] : coordinates) {
        for (auto &[city2, coord2] : coordinates) {
            if (city1 != city2) {
                graph[indexMap[city1]][indexMap[city2]] = calculateDistance(coord1, coord2);
            }
        }
    }
    return graph;
}

// 프림 알고리즘으로 MST 생성
vector<pair<int, int>> primMST(vector<vector<double>> &graph) {
    int n = graph.size();
    vector<double> key(n, numeric_limits<double>::max());
    vector<bool> inMST(n, false);
    vector<int> parent(n, -1);
    vector<pair<int, int>> mstEdges;

    key[0] = 0;  // 시작 노드(A)로 고정
    for (int count = 0; count < n - 1; ++count) {
        double minKey = numeric_limits<double>::max();
        int u = -1;

        // MST에 포함되지 않은 노드 중 가장 작은 키 값 선택
        for (int v = 0; v < n; ++v) {
            if (!inMST[v] && key[v] < minKey) {
                minKey = key[v];
                u = v;
            }
        }
        inMST[u] = true;

        // 선택된 노드와 연결된 모든 노드의 키 값 갱신
        for (int v = 0; v < n; ++v) {
            if (graph[u][v] && !inMST[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }

    // MST 간선 저장
    for (int i = 1; i < n; ++i) {
        mstEdges.push_back({parent[i], i});
    }
    return mstEdges;
}

// DFS를 이용해 MST 순회 및 도시 순서 생성
void dfs(int node, vector<vector<int>> &adjList, vector<bool> &visited, vector<int> &path) {
    visited[node] = true;
    path.push_back(node);
    for (int neighbor : adjList[node]) {
        if (!visited[neighbor]) {
            dfs(neighbor, adjList, visited, path);
        }
    }
}

// MST 기반 TSP 도시 순서 생성
pair<vector<int>, double> tspFromMST(vector<pair<int, int>> &mstEdges, vector<vector<double>> &graph, int n) {
    vector<vector<int>> adjList(n);
    for (auto &[u, v] : mstEdges) {
        adjList[u].push_back(v);
        adjList[v].push_back(u);
    }

    vector<bool> visited(n, false);
    vector<int> path;
    dfs(0, adjList, visited, path);
    path.push_back(0);  // 시작 도시로 복귀

    // 이동 거리 계산
    double totalDistance = 0;
    for (size_t i = 0; i < path.size() - 1; ++i) {
        totalDistance += graph[path[i]][path[i + 1]];
    }
    return {path, totalDistance};
}

// 메인 함수
int main() {
    vector<vector<double>> graph = buildGraph();
    vector<pair<int, int>> mstEdges = primMST(graph);

    cout << "MST Edges:\n";
    for (auto &[u, v] : mstEdges) {
        cout << u << " - " << v << " (" << graph[u][v] << ")\n";
    }

    auto [tspPath, totalDistance] = tspFromMST(mstEdges, graph, graph.size());

    cout << "\nTSP Approximation Path and Distances:\n";
    char cityNames[] = {'A', 'B', 'C', 'D', 'E', 'F', 'H', 'G'};
    for (size_t i = 0; i < tspPath.size() - 1; ++i) {
        cout << cityNames[tspPath[i]] << " -> " << cityNames[tspPath[i + 1]]
             << " (Distance: " << graph[tspPath[i]][tspPath[i + 1]] << ")\n";
    }
    cout << "Total Distance: " << totalDistance << endl;

    return 0;
}
