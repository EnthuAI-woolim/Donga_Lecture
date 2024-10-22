import java.io.*;
import java.util.*;
import java.nio.file.*;

public class QuickSort {

    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[high];
        int i = (low - 1); 
        for (int j = low; j < high; j++) {
            if (arr[j] <= pivot) {
                i++;

                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;

        return i + 1;
    }

    private static void quickSort(int[] arr, int low, int high) {
        if (low < high) {
            int pi = partition(arr, low, high);

            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        }
    }

    public static void main(String[] args) {
        List<Integer> numbers = new ArrayList<>();

        try {
            Path filePath = Paths.get("input_sort.txt");
            List<String> lines = Files.readAllLines(filePath);
            for (String line : lines) {
                numbers.add(Integer.parseInt(line.trim()));
            }
        } catch (IOException e) {
            System.err.println("Error reading the input file.");
            e.printStackTrace();
            return;
        }

        int[] arr = numbers.stream().mapToInt(Integer::intValue).toArray();

        long startTime = System.nanoTime();

        quickSort(arr, 0, arr.length - 1);

        long endTime = System.nanoTime();

        long duration = (endTime - startTime);  //nanoseconds
        System.out.println("running time " + duration / 1_000_000.0+ " miliseconds.");
        System.out.println("running time " + duration / 1_000_000_000.0 + " seconds.");

        try (PrintWriter writer = new PrintWriter(new FileWriter("output_quick_sort.txt"))) {
            for (int num : arr) {
                writer.println(num);
            }
        } catch (IOException e) {
            System.err.println("Error writing the output file.");
            e.printStackTrace();
        }
    }
}

