import java.util.*;

public class SetCover {

    interface Filter<T> {
        boolean matches(T t);
    }

    public static List<Integer> greedySetCover(Set<Integer> X, List<Set<Integer>> S) {
        Set<Integer> U = new HashSet<>(X);
        
        List<Integer> selectedSets = new ArrayList<>();

        while (!U.isEmpty()) {
            int maxIntersectionSize = 0;
            int bestSetIdx = -1;

            for (int i = 0; i < S.size(); i++) {
                int intersectionSize = 0;
                for (int elem : S.get(i)) 
                    if (U.contains(elem)) intersectionSize++;
                    
                if (intersectionSize > maxIntersectionSize) {
                    maxIntersectionSize = intersectionSize;
                    bestSetIdx = i;
                }
            }

            if (bestSetIdx == -1) break;
            
            for (int elem : S.get(bestSetIdx)) U.remove(elem);
            
            selectedSets.add(bestSetIdx);
        }
        return selectedSets;
    }

    // Optimal Set Cover function
    public static Set<Integer> optimalSetCover(Set<Integer> X, List<Set<Integer>> S) {
        return findShortestCombination(new Filter<Set<Integer>>() {
            public boolean matches(Set<Integer> selectedIndices) {
                Set<Integer> unionSet = new LinkedHashSet<>();
                for (int index : selectedIndices) {
                    unionSet.addAll(S.get(index));
                }
                return unionSet.equals(X);
            }
        }, S);
    }

    private static <T> Set<Integer> findShortestCombination(Filter<Set<Integer>> filter, List<Set<T>> sets) {
        final int size = sets.size();
        if (size > 20) {
            throw new IllegalArgumentException("Too many combinations");
        }

        int combinations = 1 << size;
        List<Set<Integer>> possibleCombinations = new ArrayList<>();

        for (int i = 0; i < combinations; i++) {
            Set<Integer> combination = new LinkedHashSet<>();
            for (int j = 0; j < size; j++) {
                if (((i >> j) & 1) != 0) {
                    combination.add(j);
                }
            }
            possibleCombinations.add(combination);
        }

        Collections.sort(possibleCombinations, Comparator.comparingInt(Set::size));

        for (Set<Integer> possibleSolution : possibleCombinations) {
            if (filter.matches(possibleSolution)) {
                return possibleSolution;
            }
        }
        return null;
    }

    public static void main(String[] args) {
        Set<Integer> X = new HashSet<>(Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10));
        List<Set<Integer>> S = new ArrayList<>();
        S.add(new HashSet<>(Arrays.asList(1, 2, 3, 8)));
        S.add(new HashSet<>(Arrays.asList(1, 2, 3, 4, 8)));
        S.add(new HashSet<>(Arrays.asList(1, 2, 3, 4)));
        S.add(new HashSet<>(Arrays.asList(2, 3, 4, 5, 7, 8)));
        S.add(new HashSet<>(Arrays.asList(4, 5, 6, 7)));
        S.add(new HashSet<>(Arrays.asList(5, 6, 7, 9, 10)));
        S.add(new HashSet<>(Arrays.asList(4, 5, 6, 7)));
        S.add(new HashSet<>(Arrays.asList(1, 2, 4, 8)));
        S.add(new HashSet<>(Arrays.asList(6, 9)));
        S.add(new HashSet<>(Arrays.asList(6, 10)));

        // Greedy approach
        long start = System.nanoTime();
        List<Integer> greedyResult = greedySetCover(X, S);
        long end = System.nanoTime();

        System.out.print("\nGreedy Selected Sets: ");
        for (int idx : greedyResult) {
            System.out.print("S" + (idx + 1) + " ");
        }
        System.out.printf("\nrunning time : %.6fms%n\n", (double)(end - start) / 1_000_000);
        System.out.println();

        // Optimal approach

        start = System.nanoTime();
        Set<Integer> optimalResult = optimalSetCover(X, S);
        end = System.nanoTime();

        if (optimalResult != null) {
            System.out.print("Optimal Selected Sets: ");
            for (int index : optimalResult) {
                System.out.print("S" + (index + 1) + " ");
            }
        } else {
            System.out.println("No combination found");
        }
        System.out.printf("\nrunning time : %.6fms%n\n", (double)(end - start) / 1_000_000);
    }
}
