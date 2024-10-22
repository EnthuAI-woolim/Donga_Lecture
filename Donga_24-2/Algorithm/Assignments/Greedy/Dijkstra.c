#include <stdio.h>
#include <stdlib.h>

#define MAX 20
#define INF 999

int mat[MAX][MAX];
int V;

int dist[MAX];

int q[MAX];
int qp = 0;

void enqueue(int v) { q[qp++] = v; }

int cf(const void *a, const void *b)
{
    int *x = (int *)a;
    int *y = (int *)b;
    return *y - *x;
}

int dequeue()
{
    qsort(q, qp, sizeof(int), cf);
    return q[--qp];
}

int queue_has_something() { return (qp > 0); }

int visited[MAX];
int vp = 0;

void dijkstra(int s)
{
    dist[s] = 0;
    int i;
    for (i = 0; i < V; ++i)
    {
        if (i != s)
        {
            dist[i] = INF;
        }
        enqueue(i);
    }
    while (queue_has_something())
    {
        int u = dequeue();
        visited[vp++] = u;
        for (i = 0; i < V; ++i)
        {
            if (mat[u][i])
            {
                if (dist[i] > dist[u] + mat[u][i])
                {
                    dist[i] = dist[u] + mat[u][i];
                }
            }
        }
    }
}

int main(int argc, char const *argv[])
{
    printf("Enter the number of vertices: ");
    scanf(" %d", &V);
    printf("Enter the adj matrix: ");
    int i, j;
    for (i = 0; i < V; ++i)
    {
        for (j = 0; j < V; ++j)
        {
            scanf(" %d", &mat[i][j]);
        }
    }

    dijkstra(0);

    printf("\nNode\tDist\n");
    for (i = 0; i < V; ++i)
    {
        printf("%d\t%d\n", i, dist[i]);
    }

    return 0;
}

// // 서울, 천안, 논산, 광주, 대전, 원주, 대구, 강릉, 부산, 포항

// 0, 12, 0, 0, 0, 15, 0, 0, 0, 0
// 12, 0, 4, 0, 10, 0, 0, 0, 0, 0
// 0, 4, 0, 13, 3, 0, 0, 0, 0, 0
// 0, 0, 13, 0, 0, 0, 0, 0, 15, 0
// 0, 10, 3, 0, 0, 0, 10, 0, 0, 0
// 15, 0, 0, 0, 0, 0, 7, 21, 0, 0
// 0, 0, 0, 0, 10, 7, 0, 0, 9, 19
// 0, 0, 0, 0, 0, 21, 0, 0, 0, 25
// 0, 0, 0, 15, 0, 0, 9, 0, 0, 5
// 0, 0, 0, 0, 0, 0, 19, 25, 5, 0


// #include <stdio.h>
// #include <stdlib.h>
// #include <limits.h>

// #define MAX_VERTICES 100  // 최대 정점 수 정의

// typedef struct {
//     int vertex;
//     double distance;
//     int previousVertex;
// } DistanceModel;

// typedef struct {
//     int numVertices;
//     double adjMatrix[MAX_VERTICES][MAX_VERTICES];
// } DirectedWeightedGraph;

// // 그래프 초기화 함수
// void initializeGraph(DirectedWeightedGraph* graph, int vertices) {
//     graph->numVertices = vertices;
//     for (int i = 0; i < vertices; i++) {
//         for (int j = 0; j < vertices; j++) {
//             graph->adjMatrix[i][j] = (i == j) ? 0 : __DBL_MAX__;
//         }
//     }
// }

// // 간선 추가 함수
// void addEdge(DirectedWeightedGraph* graph, int src, int dest, double weight) {
//     graph->adjMatrix[src][dest] = weight;
// }

// // 최소 미방문 인접 정점을 찾는 함수
// int getMinimalUnvisitedAdjacentVertex(DirectedWeightedGraph* graph, int* visited, DistanceModel* distArray, int numVertices) {
//     double minDistance = __DBL_MAX__;
//     int minVertex = -1;

//     for (int i = 0; i < numVertices; i++) {
//         if (!visited[i] && distArray[i].distance < minDistance) {
//             minDistance = distArray[i].distance;
//             minVertex = i;
//         }
//     }
//     return minVertex;
// }

// // Dijkstra 알고리즘
// void dijkstra(DirectedWeightedGraph* graph, int startVertex, DistanceModel* distArray) {
//     int numVertices = graph->numVertices;
//     int visited[MAX_VERTICES] = {0};  // 방문한 정점 기록

//     // 거리 배열 초기화
//     for (int i = 0; i < numVertices; i++) {
//         distArray[i].vertex = i;
//         distArray[i].distance = __DBL_MAX__;
//         distArray[i].previousVertex = -1;
//     }
//     distArray[startVertex].distance = 0;

//     int currentVertex = startVertex;

//     for (int i = 0; i < numVertices - 1; i++) {
//         visited[currentVertex] = 1;

//         // 인접 정점의 거리 업데이트
//         for (int j = 0; j < numVertices; j++) {
//             double adjDistance = graph->adjMatrix[currentVertex][j];
//             if (!visited[j] && adjDistance != __DBL_MAX__ && 
//                 distArray[currentVertex].distance + adjDistance < distArray[j].distance) {
//                 distArray[j].distance = distArray[currentVertex].distance + adjDistance;
//                 distArray[j].previousVertex = currentVertex;
//             }
//         }

//         // 다음 최소 거리의 정점을 찾음
//         currentVertex = getMinimalUnvisitedAdjacentVertex(graph, visited, distArray, numVertices);
//         if (currentVertex == -1) break;
//     }
// }

// // 결과 출력 함수
// void printShortestPaths(DistanceModel* distArray, int numVertices, int startVertex) {
//     printf("Shortest paths from vertex %d:\n", startVertex);
//     for (int i = 0; i < numVertices; i++) {
//         printf("To vertex %d: Distance = %.2lf, Previous vertex = %d\n", 
//                i, distArray[i].distance, distArray[i].previousVertex);
//     }
// }

// int main() {
//     DirectedWeightedGraph graph;
//     int numVertices = 5;

//     initializeGraph(&graph, numVertices);

//     addEdge(&graph, 0, 1, 10);
//     addEdge(&graph, 0, 3, 5);
//     addEdge(&graph, 1, 2, 1);
//     addEdge(&graph, 1, 3, 2);
//     addEdge(&graph, 2, 4, 4);
//     addEdge(&graph, 3, 1, 3);
//     addEdge(&graph, 3, 2, 9);
//     addEdge(&graph, 3, 4, 2);
//     addEdge(&graph, 4, 2, 6);

//     DistanceModel distArray[MAX_VERTICES];
//     int startVertex = 0;

//     dijkstra(&graph, startVertex, distArray);

//     printShortestPaths(distArray, numVertices, startVertex);

//     return 0;
// }
