package Dynamic_Programming;


public class Edit {


    public static void main(String[] args) {
        String S = "strong";
        String T = "stone";
        
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

        System.out.println("\nEdit Distance Table");

        System.out.print("    T   Ɛ ");
        for (char c : T.toCharArray()) System.out.print(c + " ");
        
        System.out.print("\nS  i/j  ");
        for (int i = 0; i < n+1; i++) System.out.print(i + " ");
        System.out.println();

        for (int i = 0; i <= m; i++) {
            System.out.printf("%-4s%d   ", i == 0 ? "Ɛ" : S.charAt(i - 1), i);
            for (int j = 0; j <= n; j++) System.out.print(E[i][j] + " ");
            System.out.println();
        }

        System.out.println("\nMinimal editing distance: " + E[m][n]);
    }
}
