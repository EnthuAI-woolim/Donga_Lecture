import java.util.List;

public class Main {

    public static void main(String[] args) {
        QuickSort quickSort = new QuickSort();

        List<Integer> numbers = FileUtil.readNumbersFromFile("./input_sort.txt");
        Integer[] unsorted = numbers.toArray(new Integer[0]);
        Integer[] sorted = quickSort.sort(unsorted);

        FileUtil.writeArrayToFile("./output_quick_sort.txt", sorted);
    }
}