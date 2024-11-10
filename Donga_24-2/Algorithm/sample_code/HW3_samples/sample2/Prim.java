import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;
import java.util.Scanner;

public class Prim {

    ArrayList<int[]> input_edges = new ArrayList<int[]>();

    // D는 노드를 저장하는 곳 
    // 무한대로 초기화
    ArrayList<int[]> D = new ArrayList<int[]>(){{
        add(new int[]{0,99999});
        add(new int[]{1,99999});
        add(new int[]{2,99999});
        add(new int[]{3,99999});
        add(new int[]{4,99999});
        add(new int[]{5,99999});
    }}; 
    
    ArrayList<int[]> output_edges = new ArrayList<int[]>(); // T

    public void run_prim() {
        long startTime = System.nanoTime(); // 시작 시간 측정

        // 입력 받기
        Scanner scanner = new Scanner(System.in);
        for (int i = 0; i < 10; i++) {
            int start = scanner.nextInt();
            int end = scanner.nextInt();
            int weight = scanner.nextInt();
            input_edges.add(new int[]{start, end, weight});
        }

        Random random = new Random();
        int startNode = random.nextInt(6); // 0 ~ 5 사이의 랜덤한 수 선택
        D.get(startNode)[1] = 0; 

        boolean[] visited = new boolean[6]; // 방문한 노드를 저장하는 배열
        visited[startNode] = true; // 시작 노드는 방문했음

        for (int i = 0; i < 5; i++) { // 5개의 간선을 선택 (6개 노드 - 1)
            int minWeight = Integer.MAX_VALUE;
            int minNode = -1;
            int[] minEdge = null; // 최소 가중치를 가진 간선

            for (int[] edge : input_edges) {
                int node1 = edge[0];
                int node2 = edge[1];
                int weight = edge[2];
                // 방문한 노드에 인접한 노드 중 가장 가중치가 작은 간선 찾기
                if ((visited[node1] && !visited[node2]) || (visited[node2] && !visited[node1])) {
                    if (weight < minWeight) {
                        minWeight = weight;
                        if (visited[node1]) {
                            minNode = node2;
                        } 
                        else {
                            minNode = node1;
                        }
                        minEdge = edge; 
                    } 
                } 
            } 
            // 최소 가중치를 가진 간선을 찾았다면
            if (minEdge != null) {
                visited[minNode] = true;   // 방문한 노드에 인접한 노드를 방문했다고 표시
                System.out.println("(" + minEdge[0] + ", " + minEdge[1] + ", " + minEdge[2] + ")");
                output_edges.add(minEdge); // 최소 가중치를 가진 간선을 추가    
            }
        }

        long endTime = System.nanoTime();      // 종료 시간 측정
        long duration = (endTime - startTime); // 실행 시간 계산 (나노초 단위)
        
        System.out.println("실행 시간: " + duration / 1000000.0 + " 밀리초");

    }
  
    public static void main(String[] args) {
        Prim prim = new Prim();
        prim.run_prim();
    }
}
