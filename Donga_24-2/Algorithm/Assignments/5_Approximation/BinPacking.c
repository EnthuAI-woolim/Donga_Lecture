#include <stdio.h>
#include <string.h>

int first_fit(int items[], int n, int bin_capacity);
int next_fit(int items[], int n, int bin_capacity);
int best_fit(int items[], int n, int bin_capacity);
int worst_fit(int items[], int n, int bin_capacity);

int main() {
    int items[] = {7, 5, 6, 4, 2, 3, 7, 5};
    int n = sizeof(items) / sizeof(items[0]);
    int bin_capacity = 10;

    first_fit(items, n, bin_capacity);
    next_fit(items, n, bin_capacity);
    best_fit(items, n, bin_capacity);
    worst_fit(items, n, bin_capacity);

    return 0;
}

// 통 배열 초기화 함수
void initialize_bins(int bins[][100], int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 100; j++) {
            bins[i][j] = 0;
        }
    }
}

// 특정 통의 합계를 계산하는 함수
int calculate_bin_sum(int bin[], int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += bin[i];
    }
    return sum;
}

// 통의 상태를 출력하는 함수
void print_bins(int bins[][100], int bin_count, int n) {
    for (int i = 0; i < bin_count; i++) {
        printf("bin%d = [", i + 1);
        for (int j = 0; j < n; j++) {
            if (bins[i][j] > 0) {
                printf("%d", bins[i][j]);
                if (bins[i][j + 1] > 0) printf(", ");
            }
        }
        printf("]");
        if (i+1 < bin_count) printf(", ");
    }
    printf("\n");
}

// First Fit
int first_fit(int items[], int n, int bin_capacity) {
    int bins[100][100];
    int bin_count = 0;

    initialize_bins(bins, 100);

    for (int i = 0; i < n; i++) {
        int placed = 0;
        for (int j = 0; j < bin_count; j++) {
            if (calculate_bin_sum(bins[j], n) + items[i] <= bin_capacity) {
                int k = 0;
                while (bins[j][k] > 0) k++;
                bins[j][k] = items[i];
                placed = 1;
                break;
            }
        }
        if (!placed) {
            bins[bin_count][0] = items[i];
            bin_count++;
        }
    }

    printf("Output1 (First Fit): ");
    print_bins(bins, bin_count, n);
    return bin_count;
}

// Next Fit 알고리즘
int next_fit(int items[], int n, int bin_capacity) {
    int bins[100][100];
    int bin_count = 1;

    initialize_bins(bins, 100);

    for (int i = 0; i < n; i++) {
        if (calculate_bin_sum(bins[bin_count - 1], n) + items[i] <= bin_capacity) {
            int k = 0;
            while (bins[bin_count - 1][k] > 0) k++;
            bins[bin_count - 1][k] = items[i];
        } else {
            bins[bin_count][0] = items[i];
            bin_count++;
        }
    }

    printf("Output2 (Next Fit): ");
    print_bins(bins, bin_count, n);
    return bin_count;
}

// Best Fit 알고리즘
int best_fit(int items[], int n, int bin_capacity) {
    int bins[100][100];
    int bin_count = 0;

    initialize_bins(bins, 100);

    for (int i = 0; i < n; i++) {
        int best_bin = -1;
        int min_space_left = bin_capacity + 1;

        for (int j = 0; j < bin_count; j++) {
            int current_sum = calculate_bin_sum(bins[j], n);
            if (current_sum + items[i] <= bin_capacity && bin_capacity - (current_sum + items[i]) < min_space_left) {
                best_bin = j;
                min_space_left = bin_capacity - (current_sum + items[i]);
            }
        }

        if (best_bin != -1) {
            int k = 0;
            while (bins[best_bin][k] > 0) k++;
            bins[best_bin][k] = items[i];
        } else {
            bins[bin_count][0] = items[i];
            bin_count++;
        }
    }

    printf("Output3 (Best Fit): ");
    print_bins(bins, bin_count, n);
    return bin_count;
}

// Worst Fit 알고리즘
int worst_fit(int items[], int n, int bin_capacity) {
    int bins[100][100];
    int bin_count = 0;

    initialize_bins(bins, 100);

    for (int i = 0; i < n; i++) {
        int worst_bin = -1;
        int max_space_left = -1;

        for (int j = 0; j < bin_count; j++) {
            int current_sum = calculate_bin_sum(bins[j], n);
            if (current_sum + items[i] <= bin_capacity && bin_capacity - (current_sum + items[i]) > max_space_left) {
                worst_bin = j;
                max_space_left = bin_capacity - (current_sum + items[i]);
            }
        }

        if (worst_bin != -1) {
            int k = 0;
            while (bins[worst_bin][k] > 0) k++;
            bins[worst_bin][k] = items[i];
        } else {
            bins[bin_count][0] = items[i];
            bin_count++;
        }
    }

    printf("Output4 (Worst Fit): ");
    print_bins(bins, bin_count, n);
    return bin_count;
}


