#include <stdio.h>
#include <stdbool.h>
#include <wchar.h>
#include <locale.h>
#include <time.h>

#define INF 99999
#define CITIES 10

int main() { 
    setlocale(LC_ALL, "");  // 한글 출력을 위한 설정

    clock_t start, end;
    double cpu_time_used;

    start = clock(); // 시간 측정시작

    enum city {서울,천안,논산,광주,부산,대구,대전,원주,강릉,포항};
    const wchar_t* city_names[] = {L"서울", L"천안", L"논산", L"광주", L"부산", L"대구", L"대전", L"원주", L"강릉", L"포항"};
            for(int k=0; k<CITIES; k++) { 
            wprintf(L"%ls ", city_names[k]); 
        }
    for(int j =0 ; j<CITIES; j++) { 
        // 주어진 출발점 시작 
        int start_point = j;

        // 거리 배열 초기화 (출발지는 0임)
        int D[CITIES];          // 전역 nodes
        int graph[CITIES][CITIES] = {0};
        bool V[CITIES] = {false};    // visited

        // 서울 제외 , 무한대로 초기화 
        for (int i = 0; i < CITIES; i++) {
            D[i] = (i == start_point) ? 0 : INF;
        }
        
        // 그래프 초기화 (모든 거리를 INF로 설정)
        for (int i = 0; i < CITIES; i++) {
            for (int j = 0; j < CITIES; j++) {
                if (i != j) graph[i][j] = INF;
            }
        }

        // 주어진 간선 정보 입력
        graph[서울][천안] = graph[천안][서울] = 12;
        graph[천안][논산] = graph[논산][천안] = 4;
        graph[논산][광주] = graph[광주][논산] = 13;
        graph[광주][부산] = graph[부산][광주] = 15;
        graph[부산][대구] = graph[대구][부산] = 9;
        graph[논산][대전] = graph[대전][논산] = 3;
        graph[천안][대전] = graph[대전][천안] = 10;
        graph[서울][원주] = graph[원주][서울] = 15;
        graph[원주][강릉] = graph[강릉][원주] = 21;
        graph[대전][대구] = graph[대구][대전] = 10;
        graph[원주][대구] = graph[대구][원주] = 7;
        graph[강릉][포항] = graph[포항][강릉] = 25;
        graph[포항][부산] = graph[부산][포항] = 5;
        graph[대구][포항] = graph[포항][대구] = 19;

        int visited_count = 0;
        // printf("초기 거리 배열:\n");
        for(int k=0; k<CITIES; k++) { 
            // printf("%d ", D[k]); 
        }
        // printf("\n\n");

        // 점이 10개니까 V사이즈가 10개되면 그만
        while(visited_count < CITIES){
            int min = INF;
            int min_city = -1;

            // 방문하지 않은 점들중에 최소인점 선택 // 
            for(int i=0; i< CITIES ;i++) { 
                if(!V[i] && min > D[i]) { 
                    min = D[i];
                    min_city = i;
                }
            } 
            // printf("선택된 도시: %d, 거리: %d\n", min_city, min);

            /* 간선완화 */    
            // graph에서 min_city 와 연결된 점 찾기. 
            int min_city_with_connected[CITIES][2] = {0};
            int connected_count = 0;
                    for(int i=0; i < CITIES; i++) { 
                        if(graph[min_city][i] != INF) { 
                            min_city_with_connected[connected_count][0] = i;
                            min_city_with_connected[connected_count][1] = graph[min_city][i];
                            connected_count++;
                } 
            }
            // printf("연결된 도시 수: %d\n", connected_count);

                // 연결된 점 업데이트해주기 
                for(int i=0; i<connected_count;i++){ 
                    int min_city_c = min_city_with_connected[i][0];
                    if(D[min_city_c] > min_city_with_connected[i][1] + D[min_city]) {  // 기존보다 더 작아야 업데이트해야함.
                        D[min_city_c] = min_city_with_connected[i][1] + D[min_city]; //  0 + 12 ;
                        // printf("업데이트 된 거리: %d\n", D[min_city_c]);
                    }
            }
            
            // 처음엔 시작점일거임.
            if(min_city == j) { 
                D[j] = 0;
                V[j] = true;
                // printf(L"시작 도시 %ls 처리\n", city_names[j]);
            }
            else { 
                V[min_city] = true;
            }
            visited_count ++;


            // printf("현재 거리 배열:\n");
            for(int k=0; k<CITIES; k++) { 
                // printf("%d ", D[k]); 
            }
            // printf("\n\n");
        }
        // 결과 출력
        wprintf(L"\n%ls", city_names[j]);

        for(int k=0; k<CITIES; k++) { 
            if(D[k] == INF) {
                wprintf(L"INF ");
            } else {
                wprintf(L"%3d ", D[k]); 
            }
        }

    }
    end = clock(); // 시간 측정 종료
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    wprintf(L"\n실행 시간: %f 초\n", cpu_time_used);
        
    return 0;
}