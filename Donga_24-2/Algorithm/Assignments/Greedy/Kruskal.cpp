#include <iostream>
#include <vector>
#include <algorithm>

struct Edge {
    int src, dest, weight;
};

// Union-Find 구조체
class DisjointSet {
public:
    DisjointSet(int n) {
        parent.resize(n);
        rank.resize(n, 0);
        for (int i = 0; i < n; i++) {
            parent[i] = i; // 초기화: 각 노드의 부모를 자기 자신으로 설정
        }
    }

    int find(int u) {
        if (parent[u] != u) // 경로 압축
            parent[u] = find(parent[u]);
        return parent[u];
    }

    void unionSets(int u, int v) {
        int rootU = find(u);
        int rootV = find(v);

        if (rootU != rootV) {
            // Union by rank
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
    std::vector<int> parent;
    std::vector<int> rank;
};

// Kruskal 알고리즘
void kruskal(std::vector<Edge>& edges, int numVertices) {
    // 간선을 가중치에 따라 정렬
    std::sort(edges.begin(), edges.end(), [](Edge a, Edge b) {
        return a.weight < b.weight;
    });

    DisjointSet ds(numVertices); // Union-Find 초기화
    std::vector<Edge> mst; // 최소 신장 트리를 저장할 벡터

    for (const Edge& edge : edges) {
        int u = edge.src;
        int v = edge.dest;

        // 두 노드가 다른 집합에 속하면 추가
        if (ds.find(u) != ds.find(v)) {
            mst.push_back(edge);
            ds.unionSets(u, v); // 두 노드를 같은 집합으로 union
        }
    }

    // 결과 출력
    std::cout << "Minimum Spanning Tree:\n";
    for (const Edge& edge : mst) {
        std::cout << edge.src << " -- " << edge.dest << " == " << edge.weight << "\n";
    }
}

int main() {
    std::vector<Edge> edges = {
        {0, 1, 4},
        {0, 2, 1},
        {0, 3, 4},
        {1, 2, 3},
        {1, 3, 8},
        {1, 4, 3},
        {2, 3, 2},
        {2, 4, 1},
        {3, 4, 5},
        {3, 5, 7},
        {4, 5, 0},
    };

    int numVertices = 6; // 그래프의 정점 수
    kruskal(edges, numVertices); // Kruskal 알고리즘 호출
    return 0;
}
