import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

class Edge {
    int n1, n2, w;
    
    public Edge(int n1, int n2, int w) {
        this.n1 = n1;
        this.n2 = n2;
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
    public void addEdge(int n1, int n2, int w) {
        graph.get(n1).add(new Edge(n1, n2, w));
        graph.get(n2).add(new Edge(n2, n1, w));
    }

    // Prim 알고리즘을 수행하여 최소 신장 트리를 찾음
    public List<Edge> primMST(int startNode) {
        List<Edge> mst = new ArrayList<>();
        PriorityQueue<Edge> minHeap = new PriorityQueue<>((e1, e2) -> e1.w - e2.w);

        // 시작 노드에서 출발
        visited[startNode] = true;
        minHeap.addAll(graph.get(startNode));

        while (!minHeap.isEmpty() && mst.size() < graph.size() - 1) {
            Edge edge = minHeap.poll();
            int nextNode = edge.n2;

            if (!visited[nextNode]) {
                visited[nextNode] = true;
                mst.add(edge);

                // 새롭게 방문한 노드에서 갈 수 있는 간선들을 힙에 추가
                for (Edge adjEdge : graph.get(nextNode)) {
                    if (!visited[adjEdge.n2]) {
                        minHeap.offer(adjEdge);
                    }
                }
            }
        }
        return mst;
    }

    public static void main(String[] args) {
        Prim prim = new Prim(6); // 6개의 노드를 가진 그래프

        // 간선을 (n1, n2, w) 형식으로 추가
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
            System.out.printf("(%d, %d, %d)\n", edge.n1, edge.n2, edge.w);
        }
    }
}
