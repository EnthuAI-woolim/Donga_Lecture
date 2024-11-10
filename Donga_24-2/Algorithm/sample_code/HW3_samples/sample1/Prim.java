import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

public class Prim {
    public static class Node implements Comparable<Node>{ //간선의 정보를 담을 Node 클래스 선언
        int a;
        int b;
        int weight;
    
        public Node(int a, int b, int weight) { //생성자
            this.a = a;
            this.b = b;
            this.weight = weight;
        }
    
        @Override
        public int compareTo(Node o) { //weight를 기준으로 오름차순 정렬
            return this.weight - o.weight;
        }
    }

    public static List<Node>[] graph; //그래프
    static StringBuilder sb; //출력때 사용 할 스트링빌더
    public static void main(String[] args) {
        sb = new StringBuilder();
        ArrayList<Node> nodes = new ArrayList<>();
        //각 간선들에 대한 정보를 ArrayList에 저장한다.
        nodes.add(new Node(1, 2, 1));
        nodes.add(new Node(2, 5, 1));
        nodes.add(new Node(1, 5, 2));
        nodes.add(new Node(0, 3, 2));
        nodes.add(new Node(3, 4, 3));
        nodes.add(new Node(0, 4, 4));
        nodes.add(new Node(1, 3, 4));
        nodes.add(new Node(3, 5, 7));
        nodes.add(new Node(0, 1, 8));
        nodes.add(new Node(4, 5, 9));
        //그래프를 선언한다.
        graph = new ArrayList[7];
        for(int i=0; i<7; i++) {
            graph[i] = new ArrayList<>(); //각 그래프에 연결리스트 선언
        }
        for(Node node : nodes) { //전체 간선들을 그래프에 저장한다. 양방향 간선들이므로 주의해서 저장한다.
            graph[node.a].add(new Node(node.a, node.b, node.weight));
            graph[node.b].add(new Node(node.b, node.a, node.weight));
        }
        long start = System.nanoTime();
        prim(1);
        long end = System.nanoTime();
        long duration = end - start;
        sb.append("running time: ").append((double)duration/1000000000);
        System.out.println(sb.toString());
    }
    public static void prim(int start) {

        boolean[] visited = new boolean[6]; //방문했는지 확인할 떄 사용하는 visited 배열 선언
        int[] D = new int[6]; //가중치를 저장 할 D 배열 선언
        for(int i=0; i<6; i++) {
            D[i] = Integer.MAX_VALUE; //초기값은 int의 가장 큰 값으로 초기화한다.
        }
        D[start] = 0; //시작지점의 가중치는 0으로 초기화 한다.
        for(int i=0; i<6; i++) { //아직 아무 노드도 방문하지 않았으므로 모든 간선들에 대해 false로 초기화 한다.
            visited[i] = false;
        }

        Queue<Node> pq = new PriorityQueue<>(); //우선순위 큐 선언. 현재 큐에 들어가있는 간선들 중 weight가 가장 낮은걸 꺼낼 떄 사용하면 시간복잡도가 O(logn)으로 줄어든다.
        pq.add(new Node(start, start, 0)); //시작지점에 대한 간선을 우선순위 큐에 저장한다.
        while(!pq.isEmpty()) { //우선순위큐가 비기 전까지 반복해서 실행한다(우선순위 큐가 비었다는 것은 모든 노드들을 방문 했다는 뜻이며, MST가 만들어 졌다는 뜻이다.)
            Node currentNode = pq.poll(); //현재 우선순위 큐에서 weight가 가장 작은 간선 하나를 꺼내어 currentNode에 저장한다.
            if(visited[currentNode.b] == true) continue; //만약 이 노드가 이미 방문한 노드이면 continue를 이용해 무시한다.
            visited[currentNode.b] = true; //아직 방문하지 않은 노드이면 이제 방문했다고 표시한다.
            if(currentNode.a != currentNode.b) { //시작노드가 아니라면 현재 선택 된 간선에 대한 정보를 StringBuilder 에 저장한다.
                sb.append("(").append(currentNode.a).append(", ").append(currentNode.b).append(", ").append(currentNode.weight).append(")\n");
            }
            for(Node besideNode : graph[currentNode.b]) { //현재 선택 된 노드에 이어진 간선들이 현재 노드들이 가지고 있는 D보다 작다면 업데이트 해준다. 이후 우선순위 큐에 넣는다.
                if(visited[besideNode.b] == false && besideNode.weight < D[besideNode.b]) {
                    D[besideNode.b] = besideNode.weight;
                    pq.add(besideNode);
                }
            }
        }

        // for(boolean visit : visited) { //방문 디버깅
        //     System.out.print(visit + " ");
        // }
        //System.out.println();

    
    }
    

}
