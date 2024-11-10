#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define CITIES 10
#define INF __INT_MAX__

int graph[CITIES][CITIES]; //도시들별 연결 상태를 저장할 2차원 그래프

void Dijkstra(int start) { //다익스트라 함수
    int visited[CITIES]; //해당 도시의 방문을 확인 할 배열 false==방문x   true==방문o
    int D[CITIES]; //가중치를 저장 할 배열 (거리)

    for(int i=0; i<CITIES; i++) { //모든도시는 아직 방문하지 않았고 거리는 무한대로 초기화
        visited[i] = 0;
        D[i] = INF;
    }
    D[start] = 0; //시작지점은 거리 0으로 설정

    for(int i=0; i<CITIES; i++) {
        int min = INF; //최소거리를 저장 할 변수 (초기값은 제일 큰 값)
        int minNode = -1; //최소거리인 노드 (도시)

        for(int j=0; j<CITIES; j++) {
            if(visited[j] == 0 && D[j] < min) { //아직 방문하지 않았으면서 최소거리인 도시이면
                min = D[j]; //min 업데이트
                minNode = j; //도시 업데이트
            }
        }
        if(minNode == -1)  //더이상 방문 할 도시가 없다는 뜻이므로 반복할 필요가 없다.
            break;
        visited[minNode] = 1; //선택된 도시는 최소거리인 도시이기 때문에 방문했다고 표시한다. 
        for(int j=0; j<CITIES; j++) { //간선완화
            if(graph[minNode][j] != INF && !visited[j] && (D[minNode] + graph[minNode][j] < D[j])) { //두 도시가 간선으로 이어져있으며, 아직 방문하지 않았고, 현재 그 도시까지 가는 최소 거리보다 새로 생긴 경로가 더 거리가 짧다면 D를 갱신한다.
                D[j] = D[minNode] + graph[minNode][j];

            }
        }
    }
    //좌측 도시 출력용 switch 문
    switch(start) {
        case 0: printf("서울"); break;
        case 1: printf("원주"); break;
        case 2: printf("강릉"); break;
        case 3: printf("대구"); break;
        case 4: printf("포항"); break;
        case 5: printf("부산"); break;
        case 6: printf("천안"); break;
        case 7: printf("논산"); break;
        case 8: printf("대전"); break;
        case 9: printf("광주"); break;
    }
    for(int i=0; i<10; i++) {
        printf("%7d", D[i]);
    }
}

int main() {
    for(int i=0; i<CITIES; i++) {
        for(int j=0; j<CITIES; j++) {
            if(i == j) graph[i][j] = 0; //도시 본인은 0
            else graph[i][j] = INF; //그 외는 최댓값으로 초기화 한다.
        }
    }
    //0 - 서울   1-원주   2-강릉  3-대구  4-포항  5-부산  6-천안  7-논산 8-대전 9-광주
    //양방향 그래프이기 때문에 [0][1]에 값을 넣어준다면 [1][0]에도 값을 넣어줘야 함.
    graph[0][1] = 15; //서울 - 원주
    graph[1][0] = 15;
    graph[0][6] = 12; //서울 - 천안
    graph[6][0] = 12;
    graph[1][2] = 21; //원주 - 강릉
    graph[2][1] = 21;
    graph[1][3] = 7;  //원주 - 대구
    graph[3][1] = 7;
    graph[2][4] = 25; //강릉 - 포항
    graph[4][2] = 25;
    graph[3][4] = 19; //대구 - 포항
    graph[4][3] = 19;
    graph[4][5] = 5;  //포항 - 부산
    graph[5][4] = 5;
    graph[3][5] = 9;  //대구 - 부산
    graph[5][3] = 9;
    graph[6][7] = 4;  //천안 - 논산
    graph[7][6] = 4;
    graph[6][8] = 10; //천안 - 대전
    graph[8][6] = 10;
    graph[7][8] = 3;  //논산 - 대전
    graph[8][7] = 3;
    graph[8][3] = 10; //대전 - 대구
    graph[3][8] = 10;
    graph[7][9] = 13; //논산 - 광주
    graph[9][7] = 13;
    graph[9][5] = 15; //광주 - 부산
    graph[5][9] = 15;
    clock_t start, end;
    start = clock();
    printf("        서울   원주   강릉   대구   포항   부산   천안   논산   대전   광주\n");
    for(int i=0; i<10; i++) {
        Dijkstra(i);
        printf("\n");
    }
    end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("running time: %lf\n", duration);


}
