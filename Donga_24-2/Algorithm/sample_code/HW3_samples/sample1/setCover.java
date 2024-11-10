import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

class setCover { 
    public static void greedy() { //그리디로 셋커버 문제를 해결하는 함수
        List<Set<Integer>> f = new ArrayList<>(); //집합들을 담을 리스트를 선언한다.
        Set<Integer> u = new HashSet<>(); //1-10까지 있는 집합이다.(전체 노드)

        for(int i=0; i<8; i++) {
            f.add(new HashSet<>()); //리스트에 있는 집합들을 초기화한다.
        } 
        //각 부분집합들에 원소를 저장한다.
        Collections.addAll(f.get(0), 1, 2, 3, 8);
        Collections.addAll(f.get(1), 1, 2, 3, 4, 8);
        Collections.addAll(f.get(2), 1, 2, 3, 4);
        Collections.addAll(f.get(3), 2, 3, 4, 5, 7, 8);
        Collections.addAll(f.get(4), 4, 5, 6, 7);
        Collections.addAll(f.get(5), 5, 6, 7, 9, 10);
        Collections.addAll(f.get(6), 4, 5, 6, 7);
        Collections.addAll(f.get(7), 1, 2, 4, 8);
        //1-10까지인 u집합에도 원소를 추가한다.
        Collections.addAll(u, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        while(!u.isEmpty()) { //u집합이 비기 전까지 반복문을 실행한다.
            Set<Integer> selected = new HashSet<>(); //교집합을 확인할 때 사용 할 selected 집합이다.
            int index = 0; //몇번째 집합이 선택되었는지 확인할 때 필요한 변수
            int count = 0; //몇개의 원소가 겹치는지 셀 변수
            for(Set<Integer> cur : f) { //f에 들어있는 모든 집합에 대해서 반복한다.
                Set<Integer> tmp = new HashSet<>(cur); //임시 집합에 현재 선택
                tmp.retainAll(u); //교집합 한다.
                if(tmp.size() > count) { //현재 교집합 원소의 개수가 count보다 크다면 업데이트 한다.
                    count = tmp.size();
                    selected = cur;
                }
                index++; //인덱스 증가
            }
            if(!selected.isEmpty()) { //부분집합이 선택되었다면 출력
                System.out.println("Selected Set: " + selected);
                u.removeAll(selected); //u 집합에서 선택된 집합의 원소를 제거한다.
            }
            else {
                break;
            }
        }
    }

    public static void optimal() {
        List<Set<Integer>> f = new ArrayList<>(); //부분집합들의 리스트
        Set<Integer> u = new HashSet<>(); //전체 원소 집합 (1-10)

        //부분집합들을 초기화하고 원소를 추가한다
        for (int i = 0; i < 8; i++) {
            f.add(new HashSet<>());
        }
        Collections.addAll(f.get(0), 1, 2, 3, 8);
        Collections.addAll(f.get(1), 1, 2, 3, 4, 8);
        Collections.addAll(f.get(2), 1, 2, 3, 4);
        Collections.addAll(f.get(3), 2, 3, 4, 5, 7, 8);
        Collections.addAll(f.get(4), 4, 5, 6, 7);
        Collections.addAll(f.get(5), 5, 6, 7, 9, 10);
        Collections.addAll(f.get(6), 4, 5, 6, 7);
        Collections.addAll(f.get(7), 1, 2, 4, 8);
        Collections.addAll(u, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        int n = f.size(); //부분집합의 개수
        List<Set<Integer>> bestSolution = null; //최적해를 저장할 리스트
        int minSets = Integer.MAX_VALUE; //선택된 최소 부분집합의 수를 저장

        //모든 가능한 부분집합: 2^n 개
        for (int i = 0; i < (1 << n); i++) { //i는 0부터 2^n - 1까지 반복  비트연산자를 통해 2^n을 만듬
            Set<Integer> covered = new HashSet<>(); //현재 조합으로 커버된 원소들의 집합
            List<Set<Integer>> selectedSets = new ArrayList<>(); //선택된 부분집합들의 리스트

            //현재 조합에 포함된 부분집합들을 선택
            for (int j = 0; j < n; j++) {
                if ((i & (1 << j)) != 0) { //j번째 부분집합이 선택된 경우
                    covered.addAll(f.get(j)); //covered 집합에 현재 부분집합의 원소들을 추가
                    selectedSets.add(f.get(j)); //선택된 부분집합 리스트에 추가
                }
            }

            //모든 원소가 커버되었고, 선택된 부분집합의 수가 최소인지 확인
            if (covered.containsAll(u) && selectedSets.size() < minSets) {
                minSets = selectedSets.size();
                bestSolution = new ArrayList<>(selectedSets);
            }
        }

        //최적 해 출력
        if (bestSolution != null) {
            System.out.println("Optimal Solution:");
            for (Set<Integer> set : bestSolution) {
                System.out.println("Selected Set: " + set);
            }
        } else {
            System.out.println("No solution found.");
        }
    }
    public static void main(String[] args) {
        long start = System.nanoTime();
        greedy();
        long end = System.nanoTime();
        long greedyDuration = end - start;
        start = System.nanoTime();
        optimal();
        end = System.nanoTime();
        long optimalDuration = end - start;

        System.out.print("greedy running time: " + (double)greedyDuration/1000000000 + "\n");
        System.out.print("optimal running time: " + (double)optimalDuration/1000000000 + "\n");
    }
}