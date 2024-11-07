package Dynamic_Programming;

import java.util.Scanner;

public class Edit {

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        System.out.print("Enter the Source String (S) : ");
        String S = scanner.nextLine();
        System.out.print("Enter the Target String (T) : ");
        String T = scanner.nextLine();
        scanner.close();

        int m = S.length();
        int n = T.length();

        int[][] E = new int[m + 1][n + 1];

        for (int i = 0; i <= m; i++) E[i][0] = i;
        for (int j = 0; j <= n; j++) E[0][j] = j;

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                int a = (S.charAt(i - 1) == T.charAt(j - 1)) ? 0 : 1;
                E[i][j] = Math.min(Math.min(E[i - 1][j] + 1, E[i][j - 1] + 1), E[i - 1][j - 1] + a);
            }
        }

        System.out.println("Minimal editing distance: " + E[m][n]);

        // System.out.println("\nEdit Distance Table:");
        // for (int i = 0; i <= m; i++) {
        //     for (int j = 0; j <= n; j++) {
        //         System.out.print(E[i][j] + " ");
        //     }
        //     System.out.println();
        // }
    }
}
