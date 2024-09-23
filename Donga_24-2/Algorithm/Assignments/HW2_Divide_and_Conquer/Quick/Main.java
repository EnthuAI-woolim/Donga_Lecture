import java.util.List;

public class Main {

    public static void main(String[] args) {
        QuickSort quickSort = new QuickSort();

        List<Integer> numbers = FileUtil.readNumbersFromFile("./input_sort.txt");
        Integer[] unsorted = numbers.toArray(new Integer[0]);

        long start = System.nanoTime();
        Integer[] sorted = quickSort.sort(unsorted);
        long end = System.nanoTime();

        System.out.println("MS : " + (end - start) / 1_000_000.0);

        FileUtil.writeArrayToFile("./output_quick_sort.txt", sorted);
    }
}