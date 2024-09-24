import java.util.List;

public class QuickSort {

    public static void main(String[] args) {
        List<Integer> numbers = FileUtil.readNumbersFromFile("../input_sort.txt");
        Integer[] arr = numbers.toArray(new Integer[0]);

        long start = System.nanoTime();
        quickSort(arr, 0, arr.length-1);
        long end = System.nanoTime();

        System.out.println("MS : " + (double)(end - start) / 1_000_000);

        FileUtil.writeArrayToFile("../output_quick_sort.txt", arr);
    }

    private static void quickSort(Integer[] array, final int left, final int right) {
        if (left < right) {
            final int pivot = partition(array, left, right);
            quickSort(array, left, pivot-1);
            quickSort(array, pivot+1, right);
        }
    }

    private static int partition(Integer[] array, int left, int right) {
        final int mid = (left + right) >>> 1;
        swap(array, left, mid);
        final int pivot = array[left];
        int i = left + 1;
    
        while (i <= right) {
            while (i <= right && array[i] < pivot) { 
                i++;
            }
            while (i <= right && array[right] > pivot) { 
                right--;
            }
            if (i < right) {
                swap(array, i, right);
                i++;
                right--;
            }
        }
    
        swap(array, left, right); 
        return right; 
    }
    

    private static void swap(Integer[] array, int i, int j) {
        Integer temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}