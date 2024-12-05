#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <set>

using namespace std;

struct Node {
    string name;
    int x, y;
};

struct Edge {
    int src, dest;
    double weight;
};

// DisjointSet 클래스 정의
class DisjointSet {
public:
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }

    int find(int u) {
        if (parent[u] != u)
            parent[u] = find(parent[u]);
        return parent[u];
    }

    void unionSets(int u, int v) {
        int rootU = find(u);
        int rootV = find(v);

        if (rootU != rootV) {
            if (rank[rootU] > rank[rootV]) {
                parent[rootV] = rootU;
            } else if (rank[rootU] < rank[rootV]) {
                parent[rootU] = rootV;
            } else {
                parent[rootV] = rootU;
                rank[rootU]++;
            }
        }
    }

private:
    vector<int> parent;
    vector<int> rank;
};

// Kruskal 함수
vector<Edge> kruskal(const vector<Node>& nodes) {
    int numVertices = nodes.size();
    DisjointSet ds(numVertices);

    vector<Edge> edges;

    // 모든 노드 간의 간선 생성
    for (int i = 0; i < numVertices; i++) {
        for (int j = i + 1; j < numVertices; j++) {
            double weight = sqrt(pow(nodes[i].x - nodes[j].x, 2) + pow(nodes[i].y - nodes[j].y, 2));
            edges.push_back({i, j, weight});
        }
    }

    // 간선 정렬 (가중치 오름차순)
    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        return a.weight < b.weight;
    });

    vector<Edge> mst;

    // 간선들을 하나씩 검사하며 MST에 추가
    for (const Edge& edge : edges) {
        if (ds.find(edge.src) != ds.find(edge.dest)) {
            mst.push_back(edge);
            ds.unionSets(edge.src, edge.dest);
        }
    }

    return mst;
}

// DFS 함수
void dfs(int node, const vector<vector<int>>& adjList, vector<bool>& visited, vector<int>& path) {
    visited[node] = true;
    path.push_back(node);

    // 인접 노드를 오름차순으로 탐색 (작은 번호의 노드부터 방문)
    for (int neighbor : adjList[node]) {
        if (!visited[neighbor]) {
            dfs(neighbor, adjList, visited, path);
        }
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

    // DFS를 통한 경로 탐색
    vector<bool> visited(nodes.size(), false);
    vector<int> path;
    int startNode = 0; // A의 인덱스가 0번이라고 가정

    dfs(startNode, adjList, visited, path);

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

// 두 노드 간의 거리 계산 함수
double calculateDistance(const Node& a, const Node& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int main() {
    vector<Node> nodes = {
        {"A", 0, 3}, {"B", 7, 5}, {"C", 6, 0}, {"D", 4, 3},
        {"E", 1, 0}, {"F", 5, 3}, {"H", 4, 1}, {"G", 2, 2}
    };

    vector<Edge> mst = kruskal(nodes);

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
