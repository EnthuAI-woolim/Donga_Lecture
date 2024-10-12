#include <iostream>
#include <vector>
#include <sstream>
#include <algorithm>
#include <ctime>


struct Edge {
    int src, dest, weight;
};


class DisjointSet {
public:
    // 초기화
    DisjointSet(int n) {
        parent.resize(n);   // 각 노드의 root 노드
        rank.resize(n, 0);  // 각 노드의 트리 높이
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    } 

    // 입력 노드의 root 노드를 찾음
    int find(int u) {
        if (parent[u] != u)     // 초기 세팅 값이 아니면(자기자신) 즉, 한 번 수정되었다면
            parent[u] = find(parent[u]);    // 재귀함수를 통해 자신의 root 노드를 찾음
        return parent[u];
    }

    void unionSets(int u, int v) {
        int rootU = find(u);
        int rootV = find(v);

        if (rootU != rootV) {
            if (rank[rootU] > rank[rootV]) {    // rank가 더 높은 트리에 더 낮은 트리를 붙임
                parent[rootV] = rootU;
            } else if (rank[rootU] < rank[rootV]) {
                parent[rootU] = rootV;
            } else {    // rank가 같다면 노드의 부여번호(0~5)가 더 낮은 트리아래에 다른 트리 붙임
                parent[rootV] = rootU;
                rank[rootU]++;  // root 노드의 rank 높임
            }
        }
    }

private:
    std::vector<int> parent;
    std::vector<int> rank;
};

std::vector<Edge> kruskal(std::vector<Edge>& edges, int numVertices) {
    // 가중치를 기준으로 오름차순
    std::sort(edges.begin(), edges.end(), [](Edge a, Edge b) {
        return a.weight < b.weight;
    });

    DisjointSet ds(numVertices); 
    std::vector<Edge> mst;

    for (const Edge& edge : edges) {
        int u = edge.src;
        int v = edge.dest;

        // 두 노드의 root 노드가 서로 다르다면 추가(사이클이 안된다는 의미)
        if (ds.find(u) != ds.find(v)) {
            mst.push_back(edge);
            ds.unionSets(u, v);
        }
    }

    return mst;
}

void inputEdges(std::vector<Edge>& edges) {
    std::string input;
    int src, dest, weight;

    std::cout << "Enter the edges in the format (src, dest, weight):\n";
    std::cout << "(Enter \'end\' to finish input)\n";
    while (true) {
        std::getline(std::cin, input);
        if (input == "end") {
            std::cout << "\n";
            break;
        }

        input.erase(remove(input.begin(), input.end(), '('), input.end());
        input.erase(remove(input.begin(), input.end(), ')'), input.end());

        std::istringstream iss(input);
        if (iss >> src && iss.ignore() && iss >> dest && iss.ignore() && iss >> weight) {
            edges.push_back({src, dest, weight});
        } else {
            std::cout << "Invalid input. Please enter in the correct format.\n";
        }
    }
}


int main() {
    clock_t start, end;
    std::vector<Edge> edges, mst;

    inputEdges(edges);
    int numVertices = edges.size();
    
    start = clock();
    mst = kruskal(edges, numVertices);
    end = clock();

    std::cout << "Minimum Spanning Tree:\n";
    for (const Edge& edge : mst) {
        std::cout << "(" << edge.src << " ," << edge.dest << " ," << edge.weight << ")" << "\n";
    }
    printf("running time : %lfms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000);
    return 0;
}

// std::vector<Edge> edges = {
//         {1, 2, 1},
//         {2, 5, 1},
//         {1, 5, 2},
//         {0, 3, 2},
//         {3, 4, 3},
//         {0, 4, 4},
//         {1, 3, 4},
//         {3, 5, 7},
//         {0, 1, 8},
//         {4, 5, 9},
//     };