import java.util.*;

class Edge {
    int src, dest, weight;

    Edge(int src, int dest, int weight) {
        this.src = src;
        this.dest = dest;
        this.weight = weight;
    }
}

public class Prim {
    private static final int V = 6; // 정점 수
    private static final int INF = Integer.MAX_VALUE;

    public static List<Edge> primMST(List<Edge> edges) {
        List<Edge> mst = new ArrayList<>(); // 최소 신장 트리를 저장할 리스트
        int[] D = new int[V]; // 최소 가중치를 저장하는 배열
        boolean[] inTree = new boolean[V]; // 트리에 포함 여부를 확인하는 배열
        int[] parent = new int[V]; // 각 정점의 부모 노드 저장

        Arrays.fill(D, INF);  // 모든 정점의 가중치를 무한대로 초기화
        D[2] = 0;  // 시작점을 2로 설정하여 가중치를 0으로 설정
        parent[2] = -1; // 시작점의 부모는 없음

        long start = System.nanoTime();

        for (int count = 0; count < V; count++) {
            int vmin = -1;
            int minWeight = INF;

            // T에 속하지 않은 각 점 v에 대하여, D[v]가 최소인 점 vmin을 선택
            for (int v = 0; v < V; v++) {
                if (!inTree[v] && D[v] < minWeight) {
                    minWeight = D[v];
                    vmin = v;
                }
            }

            // 최소 가중치 간선 (u, vmin)을 T에 추가
            inTree[vmin] = true;
            if (parent[vmin] != -1) {
                mst.add(new Edge(parent[vmin], vmin, D[vmin]));
            }

            // T에 속하지 않은 각 점 w에 대해 D[w] 갱신
            for (Edge edge : edges) {
                int u = edge.src;
                int v = edge.dest;
                int weight = edge.weight;

                if ((u == vmin && !inTree[v]) || (v == vmin && !inTree[u])) {
                    int w = (u == vmin) ? v : u;
                    if (weight < D[w]) {
                        D[w] = weight;
                        parent[w] = vmin;
                    }
                }
            }
        }

        long end = System.nanoTime();
        double durationSeconds = (double) (end - start) / 1_000_000_000; // 초 단위로 변환

        // 결과 출력
        System.out.println("Minimum Spanning Tree (MST):");
        for (Edge e : mst) {
            System.out.println("(" + e.src + ", " + e.dest + ", " + e.weight + ")");
        }
        System.out.println("Prim algorithm Running time: " + durationSeconds + " s");

        return mst;
    }

    public static void main(String[] args) {
        // 간선 리스트 (a:0, b:1, c:2, d:3, e:4, f:5로 정의)
        List<Edge> edges = Arrays.asList(
            new Edge(0, 1, 3), new Edge(0, 3, 2), new Edge(0, 4, 4),
            new Edge(1, 2, 1), new Edge(1, 3, 4), new Edge(1, 5, 2),
            new Edge(2, 5, 1), new Edge(3, 4, 5), new Edge(3, 5, 7),
            new Edge(4, 5, 9)
        );

        primMST(edges);
    }
}

