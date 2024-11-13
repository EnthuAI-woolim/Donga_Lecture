package Sorting;

import java.io.*;
import java.util.ArrayList;

public class Shell { 

    public static int readFile(String filename, ArrayList<Integer> A) {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                try {
                    A.add(Integer.parseInt(line));
                } catch (NumberFormatException e) {
                    System.out.println("숫자가 아닌 값을 발견했습니다: " + line);
                }
            }
            return A.size();
        } catch (IOException e) {
            System.out.println("파일을 열 수 없습니다.");
            return -1;
        }
    }

    public static void writeFile(String filename, ArrayList<Integer> A) {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
            for (Integer num : A) {
                bw.write(num + "\n");
            }
            System.out.println("정렬된 숫자들이 " + filename + "에 저장되었습니다.");
        } catch (IOException e) {
            System.out.println("결과 파일을 저장할 수 없습니다.");
        }
    }


    public static void main(String[] args) {
        ArrayList<Integer> A = new ArrayList<>();
        int[] h_list = {100, 50, 10, 5, 1};
        int n = readFile("input.txt", A);
        if (n < 0) {
            System.out.println("파일을 읽는 데 문제가 발생했습니다.");
            return;
        }

        // Shell Sort
        for (int h = 0; h < h_list.length; h++) {
            int gap = h_list[h];
            for (int i = gap; i < n; i++) {
                int CurrentElement = A.get(i);
                int j = i;

                while (j >= gap && A.get(j-gap) > CurrentElement) {
                    A.set(j, A.get(j-gap));
                    j = j - gap;
                }
                A.set(j, CurrentElement);
            }
        }

        writeFile("shell_output.txt", A);
    }
}