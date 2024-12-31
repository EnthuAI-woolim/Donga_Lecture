import java.util.*;

class BranchAndBound {
    
    static int numNodes = 5; // 노드 수

    static int[] bestTour = new int[numNodes + 1]; // 최적 경로
    static boolean[] visited = new boolean[numNodes]; // 방문 여부
    static int bestDistance = Integer.MAX_VALUE; // 최소 거리

    // 현재 경로를 최적 경로에 복사
    static void copyToBestTour(int[] currentTour) {
        System.arraycopy(currentTour, 0, bestTour, 0, numNodes);
        bestTour[numNodes] = currentTour[0]; // 경로 닫기
    }

    // 최소 간선 비용 찾기
    static int getFirstMinEdge(int[][] graph, int node) {
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < numNodes; i++) {
            if (graph[node][i] < min && node != i) min = graph[node][i];
        }
        return min;
    }

    // 두 번째 최소 간선 비용 찾기
    static int getSecondMinEdge(int[][] graph, int node) {
        int first = Integer.MAX_VALUE, second = Integer.MAX_VALUE;
        for (int i = 0; i < numNodes; i++) {
            if (node == i) continue;
            if (graph[node][i] <= first) { second = first; first = graph[node][i]; }
            else if (graph[node][i] <= second) second = graph[node][i];
        }
        return second;
    }

    // Branch and Bound 방식으로 재귀적으로 TSP 해결
    static void BranchAndBoundTSPRecursively(int[][] graph, int currentBound, int currentWeight, int level, int[] currentTour) {
        if (level == numNodes) { // 모든 노드를 방문했을 때
            if (graph[currentTour[level - 1]][currentTour[0]] != 0) {
                int currentTotalDistance = currentWeight + graph[currentTour[level - 1]][currentTour[0]];
                if (currentTotalDistance < bestDistance) {
                    copyToBestTour(currentTour); // 최적 경로 갱신
                    bestDistance = currentTotalDistance;
                }
            }
            return;
        }

        for (int i = 0; i < numNodes; i++) {
            if (graph[currentTour[level - 1]][i] != 0 && !visited[i]) { // 방문하지 않은 노드
                int temp = currentBound;
                currentWeight += graph[currentTour[level - 1]][i];

                if (level == 1) { // 첫 번째 레벨에서는 두 최소 간선 사용
                    currentBound -= ((getFirstMinEdge(graph, currentTour[level - 1]) + getFirstMinEdge(graph, i)) / 2);
                } else {
                    currentBound -= ((getSecondMinEdge(graph, currentTour[level - 1]) + getFirstMinEdge(graph, i)) / 2);
                }

                if (currentBound + currentWeight < bestDistance) { // 최적 경로 가능성 확인
                    currentTour[level] = i;
                    visited[i] = true;
                    BranchAndBoundTSPRecursively(graph, currentBound, currentWeight, level + 1, currentTour); // 재귀 호출
                }

                // 백트래킹
                currentWeight -= graph[currentTour[level - 1]][i];
                currentBound = temp;

                Arrays.fill(visited, false);
                for (int j = 0; j < level; j++) visited[currentTour[j]] = true;
            }
        }
    }

    static void BranchAndBoundTSP(int[][] graph) {
        int[] currentTour = new int[numNodes + 1];
        int currentBound = 0;

        Arrays.fill(currentTour, -1); // 경로 초기화
        Arrays.fill(visited, false); // 방문 여부 초기화

        for (int i = 0; i < numNodes; i++) {
            currentBound += (getFirstMinEdge(graph, i) + getSecondMinEdge(graph, i));
        }
        currentBound = currentBound / 2;

        visited[0] = true;
        currentTour[0] = 0; // 시작점(A)

        BranchAndBoundTSPRecursively(graph, currentBound, 0, 1, currentTour);
    }

    public static void main(String[] args) {
        int[][] graph = {
            {0,  2,  7,  3, 10},
            {2,  0,  3,  5,  4},
            {7,  3,  0,  6,  1},
            {3,  5,  6,  0,  9},
            {10, 4,  1,  9,  0}
        };

        BranchAndBoundTSP(graph);

        // 결과 출력
        System.out.printf("최적 경로 : ");
        for (int i = 0; i <= numNodes; i++) {
            System.out.printf("%c ", (char) ('A' + bestTour[i]));
        }
        System.out.println();
        System.out.printf("거리 : %d\n", bestDistance);
    }
}
