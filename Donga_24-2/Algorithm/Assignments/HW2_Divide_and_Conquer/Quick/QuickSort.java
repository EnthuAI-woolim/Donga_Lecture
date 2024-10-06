import java.util.List;

public class QuickSort {

    public static void main(String[] args) {
        List<Integer> numbers = FileUtil.readNumbersFromFile("../input_sort.txt");
        Integer[] arr = numbers.toArray(new Integer[0]);

        long start = System.nanoTime();
        quickSort(arr, 0, arr.length-1);
        long end = System.nanoTime();
        System.out.printf("running time : %.6fms%n", (double)(end - start) / 1_000_000);

        FileUtil.writeArrayToFile("../output_quick_sort.txt", arr);
    }

    private static void quickSort(Integer[] arr, final int l, final int r) {
        if (l < r) {
            final int pivot = partition(arr, l, r);
            quickSort(arr, l, pivot-1);
            quickSort(arr, pivot+1, r);
        }
    }

    private static int partition(Integer[] arr, int first, int r) {
        final int mid = (first + r) >>> 1;
        final int pivot = arr[mid];
        int l = first + 1;
        
        swap(arr, first, mid);
        while (l <= r) {
            while (l <= r && arr[l] < pivot) l++;
            while (l <= r && arr[r] > pivot) r--;

            if (l < r) {
                swap(arr, l, r);
                l++;
                r--;
            }
        }
        swap(arr, first, r); 
        return r; 
    }
    
    private static void swap(Integer[] arr, int i, int j) {
        Integer temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}