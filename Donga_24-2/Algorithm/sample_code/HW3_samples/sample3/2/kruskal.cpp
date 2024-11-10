#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono> // 시간 측정을 위한 라이브러리

using namespace std;
using namespace std::chrono;

// 간선 구조체 정의
struct Edge {
    int u, v, weight;
    bool operator<(const Edge& other) const {
        return weight < other.weight; // 가중치 오름차순 정렬
    }
};

// 부모 찾기 (Union-Find 알고리즘)
int findParent(vector<int>& parent, int v) {
    if (v == parent[v])
        return v;
    return parent[v] = findParent(parent, parent[v]); // 경로 압축을 사용해 부모 찾기
}

// 두 집합을 합치는 연산 (Union-Find)
void unionSets(vector<int>& parent, vector<int>& rank, int u, int v) {
    int parent_u = findParent(parent, u);
    int parent_v = findParent(parent, v);
    if (parent_u != parent_v) {
        if (rank[parent_u] < rank[parent_v])
            parent[parent_u] = parent_v;
        else if (rank[parent_u] > rank[parent_v])
            parent[parent_v] = parent_u;
        else {
            parent[parent_v] = parent_u;
            rank[parent_u]++;
        }
    }
}

// Kruskal 알고리즘
vector<Edge> KruskalMST(int V, vector<Edge>& L) {
    // 가중치 오름차순으로 간선들을 정렬
    sort(L.begin(), L.end());

    // 트리 T를 초기화
    vector<Edge> T; // 최소 신장 트리 저장 벡터

    // 부모와 랭크 배열 초기화
    vector<int> parent(V), rank(V, 0);
    for (int i = 0; i < V; i++)
        parent[i] = i; // 모든 정점의 부모를 자기 자신으로 초기화

    // while (T의 간선 수 < n-1)
    while (T.size() < V - 1) {
        // L에서 가장 작은 가중치를 가진 간선 e를 가져오고, e를 L에서 제거
        Edge e = L.front(); // 가장 작은 가중치의 간선 선택
        L.erase(L.begin()); // 그 간선을 리스트에서 제거

        // if (간선 e가 T에 추가되어 사이클을 만들지 않으면)
        if (findParent(parent, e.u) != findParent(parent, e.v)) {
            // e를 T에 추가
            T.push_back(e);
            unionSets(parent, rank, e.u, e.v); // 두 정점을 같은 집합으로 병합
        }
        // 사이클이 생기면 간선을 버림 (자동으로 처리됨, 특별한 동작 불필요)
    }

    // T는 최소 신장 트리
    return T;
}

int main() {
    int V = 6; // 정점의 수

    // 간선 리스트 (a:0, b:1, c:2, d:3, e:4, f:5로 정의)
    vector<Edge> L = {
        {0, 1, 8}, {0, 3, 2}, {0, 4, 4},
        {1, 2, 1}, {1, 3, 4}, {1, 5, 2},
        {2, 5, 1}, {3, 4, 3}, {3, 5, 7},
        {4, 5, 9}
    };

    // 실행 시간 측정
    auto start = high_resolution_clock::now();

    // Kruskal 알고리즘 실행
    vector<Edge> mst = KruskalMST(V, L);

    auto end = high_resolution_clock::now();

    // 시간을 초 단위로 변환 (소수점 포함)
    auto duration = duration_cast<nanoseconds>(end - start);
    double seconds = duration.count() / 1e9; // 나노초를 초 단위로 변환

    cout << "Minimum Spanning Tree (MST):" << endl;
    for (const Edge& edge : mst) {
        cout << "(" << edge.u << ", " << edge.v << ", " << edge.weight << ")" << endl;
    }
    cout << "Kruskal algorithm running time: " << seconds << " s" << endl;

    return 0;
}
