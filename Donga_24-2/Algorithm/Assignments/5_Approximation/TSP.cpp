#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

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


int main() {
    vector<Node> nodes = {
        {"A", 0, 3}, {"B", 7, 5}, {"C", 6, 0}, {"D", 4, 3},
        {"E", 1, 0}, {"F", 5, 3}, {"H", 4, 1}, {"G", 2, 2}
    };

    vector<Edge> mst = kruskal(nodes);

    // // MST 출력
    // cout << "Edges in the Minimum Spanning Tree:" << endl;
    // for (const Edge& edge : mst) {
    //     cout << nodes[edge.src].name << " - " << nodes[edge.dest].name
    //          << " (Weight: " << edge.weight << ")" << endl;
    // }

    return 0;
}