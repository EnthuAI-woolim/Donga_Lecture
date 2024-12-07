import java.util.*;

// 이 클래스는 인접 리스트를 사용하여 무방향 그래프를 나타냅니다.
class VertexCover {
    private int V;  // 노드 수
    private LinkedList<Integer>[] adj;  // 인접 리스트
    private List<String> edges;  // 각 엣지 번호에 연결된 노드들을 저장

    // 그래프 초기화
    VertexCover(int v) {
        V = v;
        adj = new LinkedList[v];
        edges = new ArrayList<>();
        for (int i = 0; i < v; ++i) {
            adj[i] = new LinkedList<Integer>();
        }
    }

    // 엣지 번호를 추가하는 메서드 (노드 번호 기준으로)
    public void addEdge(int v, int[] edgeNumbers) {
        for (int edgeIndex : edgeNumbers) {
            // 엣지가 존재하지 않으면 새로 추가
            if (edgeIndex >= edges.size()) {
                edges.add(v + "-" + edgeIndex);  // 엣지 번호와 연결된 노드 정보 추가
            } else {
                // 기존 엣지에 추가하기
                String[] edge = edges.get(edgeIndex).split("-");
                int u = Integer.parseInt(edge[0]);
                edges.set(edgeIndex, u + "-" + v);  // 연결된 노드를 엣지 번호로 저장
            }
        }
    }

    // Greedy Set Cover Algorithm을 사용하여 최소 정점 커버를 구하는 방법
    public Set<Integer> greedySetCover() {
        Set<Integer> vertexCover = new HashSet<>();
        Set<Integer> uncoveredEdges = new HashSet<>();

        // 모든 엣지를 uncoveredEdges에 추가
        for (int i = 0; i < edges.size(); i++) {
            uncoveredEdges.add(i);
        }

        // 커버할 엣지가 없을 때까지 반복
        while (!uncoveredEdges.isEmpty()) {
            Map<Integer, Integer> vertexCoverage = new HashMap<>();

            // 각 엣지가 커버되는 노드를 찾기
            for (Integer edgeIndex : uncoveredEdges) {
                String[] edge = edges.get(edgeIndex).split("-");
                int u = Integer.parseInt(edge[0]);
                int v = Integer.parseInt(edge[1]);

                // 각 노드가 커버하는 엣지 수를 계산
                vertexCoverage.put(u, vertexCoverage.getOrDefault(u, 0) + 1);
                vertexCoverage.put(v, vertexCoverage.getOrDefault(v, 0) + 1);
            }

            // 최대 커버된 엣지를 가진 노드 찾기
            int bestVertex = -1;
            int maxCoverage = 0;
            for (Map.Entry<Integer, Integer> entry : vertexCoverage.entrySet()) {
                if (entry.getValue() > maxCoverage) {
                    maxCoverage = entry.getValue();
                    bestVertex = entry.getKey();
                }
            }

            // 해당 노드를 커버 집합에 추가
            vertexCover.add(bestVertex);

            // 이 노드로 커버되는 모든 엣지를 uncoveredEdges에서 제거
            final int finalBestVertex = bestVertex;  // 최종적으로 선택된 노드
            uncoveredEdges.removeIf(edgeIndex -> {
                String[] edge = edges.get(edgeIndex).split("-");
                return edge[0].equals(String.valueOf(finalBestVertex)) || edge[1].equals(String.valueOf(finalBestVertex));
            });
        }

        return vertexCover;
    }

    // Maximal Matching을 구하는 방법
    public List<String> maximalMatching() {
        List<String> matching = new ArrayList<>();
        Set<Integer> matchedVertices = new HashSet<>();

        // 모든 엣지에 대해 가능한 한 매칭을 시도
        for (int i = 0; i < edges.size(); i++) {
            String[] edge = edges.get(i).split("-");
            int u = Integer.parseInt(edge[0]);
            int v = Integer.parseInt(edge[1]);

            // 두 노드가 아직 매칭되지 않았고, 실제로 연결된 엣지일 경우에만 매칭을 추가
            if (!matchedVertices.contains(u) && !matchedVertices.contains(v) && isConnected(u, v)) {
                // 이 엣지를 매칭에 추가
                matching.add((char) ('A' + u) + "-" + (char) ('A' + v));
                matchedVertices.add(u);
                matchedVertices.add(v);
            }
        }

        return matching;
    }

// 엣지가 실제로 연결된 엣지인지 확인하는 메서드
private boolean isConnected(int u, int v) {
    return adj[u].contains(v);
}


    // Driver method
    public static void main(String[] args) {
        VertexCover graph = new VertexCover(16);  // 노드 수
        
        // 각 노드에 대해 엣지 번호를 연결
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
        
        // Maximal Matching을 구합니다
        List<String> matching = graph.maximalMatching();

        // 매칭된 엣지 출력
        System.out.println("Maximal Matching (Edges): " + matching);
    }
}
