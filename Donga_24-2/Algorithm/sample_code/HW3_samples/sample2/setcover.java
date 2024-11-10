import java.util.*;

public class setcover { 

// Combination 함수 (n 전체 원소개수, r 아직 선택해야할 원소 개수, depth 현재 검사중인 원소 인덱스, visited 선택 여부를 나타내는 배열, covered_arr 최적해를 저장할 곳)
static void combination(List<Set<Integer>> F, boolean[] visited, int depth, int n, int r, Set<Integer> U, List<List<Set<Integer>>> covered_arr) {
    if(r == 0) {
        // 현재까지 완성된 집합을 저장할 곳 [ ] - U와 같은지 확인하기 위해서
        Set<Integer> covered = new HashSet<>();
        // 현재 선택된 부분집합들의 집합 [ [...] ] - 최적해를 저장하기 위해서
        List<Set<Integer>> solution = new ArrayList<>();

        for (int i= 0; i < n ; i ++) { 
            // 선택한 집합을 전체 집합에 넣고, 현재 완성된 집합에 넣는다.
            if(visited[i]) {
                covered.addAll(F.get(i));
                solution.add(F.get(i));
            }
        }
        // U와 covered가 같으면 출력        
        if(covered.containsAll(U)) { 
            // System.out.println("U: " + U);
            // System.out.println("Optimal solution");
            printCombination(F, visited, n);
            covered_arr.add(solution);
        }
        return;
    }
    if(depth == n) {
        return;
    } else {
        visited[depth] = true;
        combination(F, visited, depth + 1, n, r - 1, U,covered_arr);

        visited[depth] = false;
        combination(F, visited, depth + 1, n, r, U,covered_arr);
    }
}

// 선택된 조합을 출력하는 함수
static void printCombination(List<Set<Integer>> F, boolean[] visited, int n) {
    for (int i = 0; i < n; i++) {
        if (visited[i]) {
            // System.out.print(F.get(i) + " ");
        }
    }
    // System.out.println();
}

public static void main(String[]args) { 
    /* 최적해 알고리즘 */
    
    long startTime = System.nanoTime();
    Set<Integer> U = new HashSet<>(Arrays.asList(1,2,3,4,5,6,7,8,9,10));
    List<Set<Integer>> F = new ArrayList<>(Arrays.asList(new HashSet<>(Arrays.asList(1,2,3,8)), new HashSet<>(Arrays.asList(1,2,3,4,8)), new HashSet<>(Arrays.asList(1,2,3,4)), new HashSet<>(Arrays.asList(2,3,4,5,7,8)), new HashSet<>(Arrays.asList(4,5,6,7)), new HashSet<>(Arrays.asList(5,6,7,9,10)), new HashSet<>(Arrays.asList(4,5,6,7)), new HashSet<>(Arrays.asList(1,2,4,8)), new HashSet<>(Arrays.asList(6,9)), new HashSet<>(Arrays.asList(6,10))));
    List<Set<Integer>> C = new ArrayList<>();

    // 부분집합으로 이뤄진 집합
    List<List<Set<Integer>>> covered_arr = new ArrayList<>();
    // 조합으로 부분집합을 구한다.
    for(int i=1; i<=U.size(); i++) { 
        combination(F, new boolean[F.size()], 0, F.size(), i, U, covered_arr);
    }

    // covered_arr 에서 최소 부분집합으로 이뤄진 U를 찾자.
    int min_sub_set_size =99999;
    List<Set<Integer>> min_arr = new ArrayList<>();
    for(List<Set<Integer>> solution : covered_arr) { 
        if(solution.size() < min_sub_set_size) { 
            min_sub_set_size = solution.size();
            min_arr = solution;
        }
    }
    System.out.println("최소최적 커버링 집합의 크기:" + min_sub_set_size);
    System.out.println("최소최적 커버링 집합:" + min_arr);
    long endTime = System.nanoTime();
    System.out.println("최소최적 커버링 집합 걸린 시간: " + (endTime - startTime) + "ns");

    /* 근사 해 알고리즘 */ 

    startTime = System.nanoTime();
    while (!U.isEmpty()) { 
        // U의 원소를 가장 많이 가진 집합 Si 찾기;
        Set<Integer> Si = null;
        int max_count = 0;
        int max_index = -1;
        for(int i =0; i<F.size(); i++) { 
            int count = 0;
            // F.get(i) 집합의 원소가 U에 있는지 확인
            for(Integer f_element : F.get(i)) { 
                if(U.contains(f_element)) { 
                    count++;
                }
            }
            if(count > max_count) { 
                max_count = count;
                Si = F.get(i);
                max_index = i;
            }
        }
        if(Si != null) { 
            // U 에서 Si 삭제하기
            U.removeAll(Si);
            // F 에서 si 삭제하기
            F.remove(max_index);
            // C = Si
            C.add(Si);
        }
        else {
            break; // 더 이상 선택할 집합이 없음
        }
    }
    System.out.println("근사 알고리즘 커버링 집합");
    System.out.println(C);
    endTime = System.nanoTime();
    System.out.println("근사 알고리즘 걸린 시간: " + (endTime - startTime) + "ns");

} 

}
