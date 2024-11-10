#include <cstdio>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>

std::vector<int> parent;
struct line { //간선들을 저장할 때 사용 할 구조체 선언 (시작, 끝, 가중치로 구성)
    int a;
    int b;
    int weight;
};
int find(int a) { //union-find에서 같은 부모를 가지는지 확인할 때 사용하는 find 함수(같은 집합에 속하는지)
    if(parent[a] == a) return a;
    return parent[a] = find(parent[a]);
}
void union_merge(int a, int b) { //같은 집합에 속하지 않았다면 같은 집합에 속하게 만듬
    a = find(a);
    b = find(b);
    if(a != b) parent[b] = a;
}
void kruskal(std::vector<line>& lines) { //크루스칼 함수
    std::vector<line> mst; //MST를 저장 할 벡터
    int sum = 0; //전체 가중치의 합

    for(int i=0; i<lines.size(); i++) { //전체 간선에 대해 반복
        int a = lines[i].a;
        int b = lines[i].b;
        if(find(a) != find(b)) { //두 개의 노드가 서로 다른 집합에 속해있따면
            sum += lines[i].weight; //가중치를 합한다.
            union_merge(a, b);  //union-find
            mst.push_back(lines[i]);  //MST에 간선을 추가한다.
            //printf("선택된 간선: (%d, %d) 가중치: %d\n", a, b, lines[i].weight); //디버깅용
        }

    }

    for(line i : mst) {
        printf("(%d, %d, %d)\n", i.a, i.b, i.weight);
    }
}
int main() {
    std::vector<line> lines;
    lines.push_back({1, 2, 1});
    lines.push_back({2, 5, 1});
    lines.push_back({1, 5, 2});
    lines.push_back({0, 3, 2});
    lines.push_back({3, 4, 3});
    lines.push_back({0, 4, 4});
    lines.push_back({1, 3, 4});
    lines.push_back({3, 5, 7});
    lines.push_back({0, 1, 8});
    lines.push_back({4, 5, 9});
    for(int i=0; i<6; i++){
        parent.push_back(i);  //parent를 자기 자신으로 초기화
    }
    std::sort(lines.begin(), lines.end(), [](const line& lhs, const line& rhs) {
        return lhs.weight < rhs.weight;  //weight 비교
    }); 
    clock_t start, end;
    double duration;

    start = clock();
    kruskal(lines);
    end = clock();
    duration = (double)(end - start) / CLOCKS_PER_SEC;

    printf("running time: %lf\n", duration);
    


    return 0;
}