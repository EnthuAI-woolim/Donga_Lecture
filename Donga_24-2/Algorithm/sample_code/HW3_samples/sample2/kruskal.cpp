#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <ctime>
#include <cstdio>  // printf를 사용하기 위해 추가

/*
유니온 파인드 자료구조 
Union-Find는 서로 중복되지 않는 부분 집합들로 나눠진 원소들에 대한 정보를
저장하고 조작하는 자료구조
1. Find: 어떤 원소가 주어졌을 때, 이 원소가 속한 집합을 찾습니다.
2. Union: 두 개의 집합을 하나로 합칩니다.

간선의 두 정점이 같은 집합에 속해 있지 않다면 (find 함수로 확인):
    두 집합을 합칩니다 (unite 함수 사용).
    이 간선을 최소 신장 트리에 추가합니다.
같은 집합에 속해 있다면, 이 간선을 추가하면 사이클이 생기므로 무시합니다.
 */

using namespace std;

class UnionFind
{
    private:
        /* 각 원소의 부모 저장 */
        vector<int> parent;
    public:
        /* 생성자 : 초기화 */
        UnionFind(int n): parent(n) {
                for (int i =0; i< n; i++){
                    parent[i] = i;
                }
            }
        /* x가 속한 집합의 루트를 찾는다.*/
        int find(int x) {
            // 루트 노드가 아니라면 루트 노드를 찾을 때까지 재귀적으로 탐색 
            if (parent[x] != x){
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        /* 서로 다른 두 개의 집합을 하나의 집합으로 병합하는 연산 */
        void unite(int x, int y) { 
            int rx = find(x);
            int ry = find(y);
            /* 루트가 다르다면 한 쪽의 루트를 다른 쪽의 자식으로 만듦*/
            if(rx != ry) { 
                parent[ry] = rx;
            }
        }
};

// 정렬에서 사용할 비교 함수
bool cmp(const vector<int>& a, const vector<int>& b) { 
    return a[2] < b[2];
}

int main(){
    clock_t start, end;
    start = clock();

    // 그래프의 엣지를 저장할 벡터
    vector<vector<int>> v;

    // 연결된 것 저장할 벡터
    vector<vector<int>> connected; 

    queue<vector<int>> q;
    // 10번 입력 받기 
    for (int i = 0; i < 10; i++) {
        int start, end, weight;
        cin >> start >> end >> weight;
        v.push_back({start, end, weight});
    }

    /* vector input 오름차순 정렬 */
    sort(v.begin(),v.end(),cmp);

    for (int i = 0; i < v.size();i++){
        q.push(v[i]);
    }
    /* 노드 6개 초기화*/
    UnionFind uf(6);

    for (const auto& e : v) {
        int u = e[0]; // v
        int v = e[1]; // vertex
        int l = e[2]; // length 

        if (uf.find(u) != uf.find(v)){
            uf.unite(u,v);
            connected.push_back(e);
        }
    }

    end = clock();
    double time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("런타임: %.6f 초\n", time);  // 소수점 이하 6자리까지 표시

    for (const auto& e : connected) {
        if (e[0] > e[1]) {
                    cout << "(" << e[1] << "," << e[0] << "," << e[2] << ")" << endl;
                } else {
                    cout << "(" << e[0] << "," << e[1] << "," << e[2] << ")" << endl;
                }
    }
   
}