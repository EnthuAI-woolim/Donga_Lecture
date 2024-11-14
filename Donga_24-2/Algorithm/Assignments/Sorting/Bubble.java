package Sorting;

import java.io.*;
import java.util.ArrayList;

public class Bubble { 

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
            System.out.println(filename + "을 생성하였습니다.");
        } catch (IOException e) {
            System.out.println("결과 파일을 저장할 수 없습니다.");
        }
    }


    public static void main(String[] args) {
        ArrayList<Integer> A = new ArrayList<>();
        int n = readFile("input.txt", A);
        if (n < 0) {
            System.out.println("파일을 읽는 데 문제가 발생했습니다.");
            return;
        }

        // Bubble Sort
        for (int pass = 0; pass < n-1; ++pass) {
            for (int i = 0; i < n-1-pass; ++i) {
                if (A.get(i) > A.get(i + 1)) {
                    int temp = A.get(i + 1);
                    A.set(i + 1, A.get(i));
                    A.set(i, temp);
                }
            }
        }

        writeFile("bubble_output.txt", A);
    }
}