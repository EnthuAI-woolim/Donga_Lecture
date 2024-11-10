import java.util.*;

public class SetCover {

    // sub-otimal 탐욕 알고리즘
    public static List<String> subOptimalSetCover(Set<Integer> U, List<Map.Entry<String, Set<Integer>>> F) {
        List<String> C = new ArrayList<>();
        
        while (!U.isEmpty()) {
            Map.Entry<String, Set<Integer>> bestSetEntry = null;
            int maxCover = 0;
            
            // 조건에 따라 U의 원소를 가장 많이 가진 집합 선택
            for (Map.Entry<String, Set<Integer>> entry : F) {
                Set<Integer> S = entry.getValue();
                Set<Integer> intersection = new HashSet<>(S);
                intersection.retainAll(U); // U와의 교집합 계산
                if (intersection.size() > maxCover) {
                    bestSetEntry = entry;
                    maxCover = intersection.size();
                }
            }
            
            // 선택된 집합을 C에 추가하고, U와 F에서 해당 집합의 원소들을 제거
            if (bestSetEntry != null) {
                U.removeAll(bestSetEntry.getValue()); // U에서 선택된 집합의 원소들을 제거
                C.add(bestSetEntry.getKey()); // 결과 집합 C에 원래 이름으로 추가
                F.remove(bestSetEntry); // F에서도 해당 집합 제거
                
                // 디버그용 출력
                //System.out.println("선택된 집합: " + bestSetEntry.getKey() + " = " + bestSetEntry.getValue());
                //System.out.println("남은 U: " + U);
            }
        }
        return C;
    }

    // optimal 알고리즘
    public static List<String> optimalSetCover(Set<Integer> U, List<Map.Entry<String, Set<Integer>>> F) {
        int n = F.size();
        List<String> bestCover = new ArrayList<>();
        int minSetCount = Integer.MAX_VALUE;
        
        // 모든 부분 집합 탐색
        for (int i = 1; i < (1 << n); i++) {
            Set<Integer> covered = new HashSet<>();
            List<String> currentCover = new ArrayList<>();
            for (int j = 0; j < n; j++) {
                if ((i & (1 << j)) != 0) {
                    covered.addAll(F.get(j).getValue());
                    currentCover.add(F.get(j).getKey());
                }
            }
            // 커버가 U와 같고, 더 적은 집합 수로 커버되는 경우 갱신
            if (covered.containsAll(U) && currentCover.size() < minSetCount) {
                bestCover = currentCover;
                minSetCount = currentCover.size();
            }
        }
        return bestCover;
    }

    public static void main(String[] args) {
        // U 집합 정의
        Set<Integer> U = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        
        // F 집합 목록 정의
        List<Map.Entry<String, Set<Integer>>> F = new ArrayList<>();
        F.add(new AbstractMap.SimpleEntry<>("S1", new HashSet<>(Arrays.asList(1, 2, 3, 8))));
        F.add(new AbstractMap.SimpleEntry<>("S2", new HashSet<>(Arrays.asList(1, 2, 3, 4, 8))));
        F.add(new AbstractMap.SimpleEntry<>("S3", new HashSet<>(Arrays.asList(1, 2, 3, 4))));
        F.add(new AbstractMap.SimpleEntry<>("S4", new HashSet<>(Arrays.asList(2, 3, 4, 5, 7, 8))));
        F.add(new AbstractMap.SimpleEntry<>("S5", new HashSet<>(Arrays.asList(4, 5, 6, 7))));
        F.add(new AbstractMap.SimpleEntry<>("S6", new HashSet<>(Arrays.asList(5, 6, 7, 9, 10))));
        F.add(new AbstractMap.SimpleEntry<>("S7", new HashSet<>(Arrays.asList(4, 5, 6, 7))));
        F.add(new AbstractMap.SimpleEntry<>("S8", new HashSet<>(Arrays.asList(1, 2, 4, 8))));
        F.add(new AbstractMap.SimpleEntry<>("S9", new HashSet<>(Arrays.asList(6, 9))));
        F.add(new AbstractMap.SimpleEntry<>("S10", new HashSet<>(Arrays.asList(6, 10))));
        
        // sub-optimal 실행 시간 측정
        Set<Integer> U1 = new HashSet<>(U); // U를 복제
        List<Map.Entry<String, Set<Integer>>> F1 = new ArrayList<>(F); // F를 복제
        long startSubOptimal = System.nanoTime();
        List<String> subOptimalCover = subOptimalSetCover(U1, F1);
        long endSubOptimal = System.nanoTime();
        
        System.out.println("sub-optimal Set Cover: " + subOptimalCover);
        System.out.println("sub-optimal running time: " + (endSubOptimal - startSubOptimal) + " ns");

        // optimal 실행 시간 측정
        Set<Integer> U2 = new HashSet<>(U); // U를 복제
        List<Map.Entry<String, Set<Integer>>> F2 = new ArrayList<>(F); // F를 복제
        long startOptimal = System.nanoTime();
        List<String> optimalCover = optimalSetCover(U2, F2);
        long endOptimal = System.nanoTime();
        
        System.out.println("optimal Set Cover: " + optimalCover);
        System.out.println("optimal running time: " + (endOptimal - startOptimal) + " ns");
    }
}

