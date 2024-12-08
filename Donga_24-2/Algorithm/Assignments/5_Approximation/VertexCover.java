import java.util.*;

class VertexCover {
    private int V;  // 노드 수
    private List<List<Integer>> adj;  // 인접 리스트

    // 그래프 초기화
    VertexCover(int v) {
        V = v;
        adj = new ArrayList<>(v);
        for (int i = 0; i < v; ++i) {
            adj.add(new ArrayList<>());
        }
    }

    // 엣지 추가 메서드
    public void addEdge(int u, int[] neighbors) {
        for (int v : neighbors) {
            adj.get(u).add(v);  // u 노드와 연결된 v 노드를 추가
            adj.get(v).add(u);  // v 노드와 연결된 u 노드를 추가 (양방향)
        }
    }

    // Set Cover 알고리즘을 통해 극대 매칭을 찾는 메서드
    public List<String> maxMatching() {
        boolean[] covered = new boolean[V];  // 각 노드가 커버되었는지 여부를 추적하는 배열
        List<String> selectedEdges = new ArrayList<>();
        
        // 각 간선의 양 끝 노드에 연결된 간선 수를 계산
        List<int[]> edgeList = new ArrayList<>();
        for (int u = 0; u < V; u++) {
            for (int v : adj.get(u)) {
                // 간선 (u, v)가 중복되지 않도록 처리
                if (u < v) {
                    edgeList.add(new int[]{u, v});
                }
            }
        }

        // 간선 리스트를 양 끝 노드의 연결 수가 많은 순으로 정렬
        edgeList.sort((edge1, edge2) -> {
            int count1 = adj.get(edge1[0]).size() + adj.get(edge1[1]).size();
            int count2 = adj.get(edge2[0]).size() + adj.get(edge2[1]).size();
            return Integer.compare(count2, count1);  // 내림차순 정렬
        });

        // 정렬된 간선 리스트에서 선택하여 커버할 노드를 찾아나감
        for (int[] edge : edgeList) {
            int u = edge[0];
            int v = edge[1];
            // u와 v가 모두 커버되지 않았다면 이 간선을 선택
            if (!covered[u] && !covered[v]) {
                selectedEdges.add(convertNodeToString(u) + "-" + convertNodeToString(v));  // 노드를 문자로 변환하여 간선 추가
                covered[u] = true;  // u 노드 커버
                covered[v] = true;  // v 노드 커버
            }
        }

        return selectedEdges;
    }

    // 노드 번호를 A, B, C,... 와 같은 문자로 변환하는 메서드
    private String convertNodeToString(int node) {
        return String.valueOf((char) ('A' + node));
    }

    // Driver 메서드
    public static void main(String[] args) {
        VertexCover graph = new VertexCover(16); // 노드 수

        // 엣지 연결
        graph.addEdge(0, new int[] {0, 4});   // A - B, E
        graph.addEdge(1, new int[] {0, 2, 4, 5, 6});   // B - A, C, F
        graph.addEdge(2, new int[] {1, 3, 6});   // C - B, D, G
        graph.addEdge(3, new int[] {2, 6, 7});   // D - C, H

        graph.addEdge(4, new int[] {0, 1, 5, 8, 9});   // E - A, F, I
        graph.addEdge(5, new int[] {1, 4, 6, 9});   // F - B, E, G, J
        graph.addEdge(6, new int[] {1, 2, 3, 5, 7, 9, 10, 11});   // G - C, F, H, K
        graph.addEdge(7, new int[] {3, 6, 11});   // H - D, G, L

        graph.addEdge(8, new int[] {4, 9, 12});   // I - E, J, M
        graph.addEdge(9, new int[] {4, 5, 6, 8, 10, 12, 13, 14});   // J - F, I, K, N
        graph.addEdge(10, new int[] {6, 9, 11, 14});   // K - G, J, L, O
        graph.addEdge(11, new int[] {6, 7, 10, 14, 15});   // L - H, K, P

        graph.addEdge(12, new int[] {8, 9, 13});   // M - I, N
        graph.addEdge(13, new int[] {9, 12, 14});   // N - J, M, O
        graph.addEdge(14, new int[] {9, 10, 11, 13, 15});   // O - K, N, P
        graph.addEdge(15, new int[] {11, 14});   // P - L, O

        // 극대 매칭 찾기
        System.out.println("Maximal matching edges");
        List<String> matchingEdges = graph.maxMatching();
        for (String edge : matchingEdges) {
            System.out.println(edge);
        }
    }
}
