package Sorting;

public class Bubble { 


    public static void main(String[] args) {
        int n = 0;
        int[] A = new int[n];
        for (int pass = 0; pass < n-1; ++pass) {
            for (int i = 0; i < n-1-pass; ++i) {
                if (A[i] > A[i+1]) {
                    int temp = A[i+1];
                    A[i+1] = A[i];
                    A[i] = temp;
                }
            }
        }
    }
}