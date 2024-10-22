import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class QuickSort {
    public static void main(String[] args) throws IOException {
        // 파일 읽기
        BufferedReader reader = new BufferedReader(new FileReader("./inupt_sort.txt"));

        String int_line;
        ArrayList<Integer> targetArray = new ArrayList<Integer>();
        while((int_line = reader.readLine()) != null) {
            targetArray.add((int)Integer.parseInt(int_line));
        }
        reader.close();

        // 수행 시간 측정 및 출력
        long startTime = System.currentTimeMillis();
        quicksort(targetArray, 0, targetArray.size()-1); // quick sort 실행
        System.out.println("Running Time: " + ((double)((System.currentTimeMillis() - startTime)) / 1000.0) + "s");

        // 수행 완료 후 파일 출력
        FileOutputStream fileOutputStream = new FileOutputStream(new File("./output_quick_sort.txt"));
        for (int i: targetArray) {
            fileOutputStream.write((Integer.toString(i) + '\n').getBytes());
        }
        fileOutputStream.close();
    }

    // 세 원소 중 중간값의 인덱스를 반환
    public static int midOfThree(ArrayList<Integer> arr, int a, int b, int c) {
        if ((Integer)arr.get(a) >= (Integer)arr.get(b)) {
            if ((Integer)arr.get(b) >= (Integer)arr.get(c)) {
                return b;
            } else if ((Integer)arr.get(a) <= (Integer)arr.get(c)) {
                return a;
            } else {
                return c;
            }
        } else if ((Integer)arr.get(a) >= (Integer)arr.get(c)) {
            return a;
        } else if ((Integer)arr.get(b) >= (Integer)arr.get(c)) {
            return c;
        } else {
            return b;
        }
    }

    // 두 원소의 위치를 바꿈
    public static void swap(ArrayList<Integer> arr, int a, int b) {
        int temp = (int) arr.get(a);
        arr.set(a, (int) arr.get(b));
        arr.set(b, temp);
    }

    public static int partition(ArrayList<Integer> arr, int left, int right) {
        int middle = (left + right) / 2;
        int pivot = midOfThree(arr, left, right, middle); // pivot 선택

        swap(arr, pivot, left); // pivot과 가장 왼쪽 원소를 바꿈
        pivot = left;

        int l = left + 1, r = right;
        while (l <= r) {
            while (l <= r && (Integer)arr.get(l) < (Integer)arr.get(pivot)) l++;
            while (l <= r && (Integer)arr.get(r) > (Integer)arr.get(pivot)) r--;

            if (l <= r) {
                swap(arr, l, r);
                l++;
                r--;
            }
        }
        
        swap(arr, pivot, r);
        return r;
    }

    public static void quicksort(ArrayList<Integer> arr, int left, int right) {
        if (left < right) {
            // pivot 선택 및 이동
            int pivot = partition(arr, left, right);

            quicksort(arr, left, pivot-1);
            quicksort(arr, pivot+1, right);
        }
    }
}
