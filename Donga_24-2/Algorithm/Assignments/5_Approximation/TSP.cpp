#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <set>
#include <queue>

using namespace std;

struct Node {
    string name;
    int x, y;
};

struct Edge {
    int src, dest;
    double weight;
};

// 우선순위 큐를 사용하여 Prim 알고리즘으로 MST를 찾기
vector<Edge> prim(const vector<Node>& nodes) {
    int numVertices = nodes.size();
    vector<vector<pair<int, double>>> adjList(numVertices); // 인접 리스트 (노드, 가중치)
    
    // 모든 노드 간의 간선 생성
    for (int i = 0; i < numVertices; i++) {
        for (int j = i + 1; j < numVertices; j++) {
            double weight = sqrt(pow(nodes[i].x - nodes[j].x, 2) + pow(nodes[i].y - nodes[j].y, 2));
            adjList[i].push_back({j, weight});
            adjList[j].push_back({i, weight});
        }
    }

    // Prim 알고리즘을 위한 우선순위 큐 (가중치, 노드번호)
    priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
    vector<bool> inMST(numVertices, false);  // MST에 포함된 노드를 체크
    vector<Edge> mstEdges;
    vector<double> key(numVertices, numeric_limits<double>::infinity());
    vector<int> parent(numVertices, -1);

    pq.push({0, 0});  // 시작 노드 (A)
    key[0] = 0;

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (inMST[u]) continue;
        inMST[u] = true;

        for (const auto& neighbor : adjList[u]) {
            int v = neighbor.first;
            double weight = neighbor.second;

            if (!inMST[v] && key[v] > weight) {
                key[v] = weight;
                pq.push({key[v], v});
                parent[v] = u;
            }
        }
    }

    // MST 간선 생성
    for (int i = 1; i < numVertices; ++i) {
        mstEdges.push_back({parent[i], i, key[i]});
    }

    return mstEdges;
}

// 두 노드 간의 거리 계산 함수
double calculateDistance(const Node& a, const Node& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

// DFS 함수 - 인접 리스트에서 깊이를 고려한 탐색
void dfs(int node, const vector<vector<int>>& adjList, vector<bool>& visited, vector<int>& path, vector<int>& depth) {
    visited[node] = true;
    path.push_back(node);

    // 인접 노드를 깊이가 낮은 순으로 탐색 (깊이 기반 오름차순 정렬)
    vector<pair<int, int>> sortedNeighbors;
    for (int neighbor : adjList[node]) {
        if (!visited[neighbor]) {
            sortedNeighbors.push_back({depth[neighbor], neighbor});
        }
    }

    // 깊이 기반 오름차순 정렬
    sort(sortedNeighbors.begin(), sortedNeighbors.end(), [](const pair<int, int>& a, const pair<int, int>& b) {
        return a.first < b.first;  // 깊이가 낮은 순서대로 정렬
    });

    // 정렬된 인접 노드를 깊이 우선 탐색
    for (const auto& p : sortedNeighbors) {
        int neighbor = p.second;
        dfs(neighbor, adjList, visited, path, depth);
    }
}

// TSP 경로 찾기 함수
vector<int> findTSPPath(const vector<Node>& nodes, const vector<Edge>& mst) {
    // 인접 리스트 생성
    vector<vector<int>> adjList(nodes.size());
    for (const Edge& edge : mst) {
        adjList[edge.src].push_back(edge.dest);
        adjList[edge.dest].push_back(edge.src);
    }

    // 깊이 계산 (각 노드가 트리에서의 깊이를 가질 수 있도록 BFS 또는 DFS로 계산)
    vector<int> depth(nodes.size(), -1);
    queue<int> q;
    q.push(0);  // A부터 시작 (0번 노드)
    depth[0] = 0;  // A의 깊이는 0

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        for (int neighbor : adjList[node]) {
            if (depth[neighbor] == -1) {  // 아직 깊이를 계산하지 않은 노드
                depth[neighbor] = depth[node] + 1;
                q.push(neighbor);
            }
        }
    }

    // 노드 방문 여부와 경로를 추적
    vector<bool> visited(nodes.size(), false);
    vector<int> path;
    int startNode = 0; // A의 인덱스가 0번이라고 가정

    // DFS를 통한 경로 탐색 (깊이 우선 탐색)
    dfs(startNode, adjList, visited, path, depth);

    // 경로에서 중복된 노드 제거 (TSP에서는 각 노드를 한 번만 방문)
    set<int> visitedNodes;
    vector<int> tspPath;
    for (int node : path) {
        if (visitedNodes.find(node) == visitedNodes.end()) {
            visitedNodes.insert(node);
            tspPath.push_back(node);
        }
    }

    return tspPath;
}

int main() {
    vector<Node> nodes = {
        {"A", 0, 3}, {"B", 7, 5}, {"C", 6, 0}, {"D", 4, 3},
        {"E", 1, 0}, {"F", 5, 3}, {"H", 4, 1}, {"G", 2, 2}
    };

    vector<Edge> mst = prim(nodes); 

    // TSP 경로 구하기
    vector<int> tspPath = findTSPPath(nodes, mst);

    // 이동 순서와 이동 거리 출력
    double totalDistance = 0.0;
    cout << "TSP Path:" << endl;
    for (size_t i = 0; i < tspPath.size(); ++i) {
        cout << nodes[tspPath[i]].name;
        if (i < tspPath.size() - 1) {
            double dist = calculateDistance(nodes[tspPath[i]], nodes[tspPath[i + 1]]);
            totalDistance += dist;
            cout << " -> ";
        }
    }

    // 마지막 노드에서 시작 노드로 돌아오는 거리 계산
    double returnDist = calculateDistance(nodes[tspPath.back()], nodes[tspPath[0]]);
    totalDistance += returnDist;
    cout << " -> " << nodes[tspPath[0]].name;  // 시작 노드로 돌아오는 부분 출력

    cout << endl;
    cout << "Total Distance: " << totalDistance << endl;

    return 0;
}
