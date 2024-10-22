import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

class Edge {
    int s, d, w;
    
    public Edge(int s, int d, int w) {
        this.s = s;
        this.d = d;
        this.w = w;
    }
}

public class Prim {
    // 그래프를 인접 리스트로 표현
    private List<List<Edge>> graph;
    private boolean[] visited;

    public Prim(int numNodes) {
        graph = new ArrayList<>();
        visited = new boolean[numNodes];
        
        // 인접 리스트 초기화
        for (int i = 0; i < numNodes; i++) {
            graph.add(new ArrayList<>());
        }
    }

    // 그래프에 간선 추가
    public void addEdge(int s, int d, int w) {
        graph.get(s).add(new Edge(s, d, w)); // get(행)
        graph.get(d).add(new Edge(d, s, w));
    }

    // Prim 알고리즘을 수행하여 최소 신장 트리를 찾음
    public List<Edge> primMST(int startNode) {
        List<Edge> mst = new ArrayList<>();

        // 가중치가 가장 작은 간선을 우선적으로 처리하기 위해 사용되는 최소 힙
        PriorityQueue<Edge> minHeap = new PriorityQueue<>((e1, e2) -> e1.w - e2.w); // 가중치를 기준으로 정렬되는 PriorityQueue

        visited[startNode] = true; // 시작 노드를 방문 처리
        minHeap.addAll(graph.get(startNode)); // 시작 노드에서 갈 수 있는 간선을 힙에 추가

        // 아직 처리할 수 있는 간선이 남아 있으면 and 저장된 간선의 수가 n-1개 미만이면
        while (!minHeap.isEmpty() && mst.size() < graph.size() - 1) {
            Edge edge = minHeap.poll(); // PriorityQueue에서 가장 우선 순위가 높은 요소를 제거하고 반환
            int nextNode = edge.d; // 선택된 간선의 도착 노드

            if (!visited[nextNode]) {
                visited[nextNode] = true; // 도착 노드를 방문 처리
                mst.add(edge); // 최소 신장 트리에 간선 추가

                // 새롭게 방문한 노드에서 갈 수 있는 간선들을 힙에 추가
                for (Edge adjEdge : graph.get(nextNode)) 
                    if (!visited[adjEdge.d]) 
                        minHeap.offer(adjEdge); // 방문하지 않은 간선만 힙에 추가
                    
                
            }
        }
        // p.34 - 예시 그래프
        // 시작 노드가 2이면, 처음 while문을 시작할때 
        // while문 시작시 - minHeap: (2, 1, 1), (2, 5, 1)이 추가됨
        // poll()실행 후 - minHeap: d=1인 간선 없어짐
        // for문 실행 후 - minHeap: (2, 5, 1), (1, 5, 2), (1, 0, 3), (1, 3, 4)이 됨
        // -> 간선하나 추가하고, 해당 간선의 dest가 src값인 간선들을 minHeap에 추가
        //    minHeap은 가중치가 작은 순으로 정렬되어있음
        // 특정 노드가 mst에 추가될 때는 weight가 가장 작은 노드와 연결된 간선이 먼저 추가 됨.
        // cf) Prim방식은 점을 하나씩 트리에 추가하는 방식이기 때문에 사이클이 생길 경우가 없음.

        return mst; // 최소 신장 트리 반환
    }

    public static void main(String[] args) {
        Prim prim = new Prim(6); // 6개의 노드를 가진 그래프

        // 간선을 (s, d, w) 형식으로 추가
        prim.addEdge(1, 2, 1);
        prim.addEdge(2, 5, 1);
        prim.addEdge(1, 5, 2);
        prim.addEdge(0, 3, 2);
        prim.addEdge(0, 1, 3);
        prim.addEdge(1, 3, 4);
        prim.addEdge(0, 4, 4);
        prim.addEdge(3, 4, 5);
        prim.addEdge(3, 5, 7);
        prim.addEdge(4, 5, 9);

        // Prim 알고리즘을 이용해 MST를 구하고 결과 출력
        int startNode = 2; // 시작 노드를 2로 설정
        List<Edge> mst = prim.primMST(startNode);
        System.out.println("Minimum Spanning Tree:");
        for (Edge edge : mst) {
            System.out.printf("(%d, %d, %d)\n", edge.s, edge.d, edge.w);
        }
    }
}
