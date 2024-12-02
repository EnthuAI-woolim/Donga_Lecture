import java.util.List;

public class QuickSort {

    public static void main(String[] args) {
        List<Integer> numbers = FileUtil.readNumbersFromFile("../input_sort.txt");
        Integer[] arr = numbers.toArray(new Integer[0]);

        long start = System.nanoTime();
        quickSort(arr, 0, arr.length-1);
        long end = System.nanoTime();
        System.out.printf("running time : %.6fms%n", (double)(end - start) / 1_000_000);

        FileUtil.writeArrayToFile("output_quick_sort.txt", arr);
    }

    private static void quickSort(Integer[] arr, final int l, final int r) {
        if (l < r) {
            final int pivot = partition(arr, l, r);
            quickSort(arr, l, pivot-1);
            quickSort(arr, pivot+1, r);
        }
    }

    private static int partition(Integer[] arr, int first, int r) {
        Integer[] idx_l = {first, (first + r) >>> 1, r};
        int p_idx = (first + r) >>> 1;
        
        // 첫번째, 가운데, 마지막 값중 중앙값을 찾기
        for (int i = 0; i < 3; i++) {
            int a = (i+1) % 3;
            int b = (i+2) % 3;
            if ((arr[idx_l[i]] - arr[idx_l[a]]) * (arr[idx_l[i]] - arr[idx_l[b]]) < 0) {
                p_idx = idx_l[i];
                break;
            }
        }

        final int pivot = arr[p_idx];
        int l = first + 1;
        
        swap(arr, first, p_idx);
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