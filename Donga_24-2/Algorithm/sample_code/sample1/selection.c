#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_SIZE 100 
double time_diff(struct timespec* start, struct timespec* end) {
    return (end->tv_sec - start->tv_sec) + (end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

int SelectionProb(int arr[], int left, int right, int find_idx);
int Pivot(int arr[], int left, int right);
void Swap(int arr[], int x, int y); 
int read_input(int arr[], const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening the file.\n");
        return -1;
    }
    int i = 0;
    while (fscanf(file, "%d", &arr[i]) != EOF) {
        i++;
    }
    fclose(file);
    return i; // Return the number of elements read
}

int main(void) {
    int arr[MAX_SIZE];
    int size = read_input(arr, "input_sort.txt");
    if (size == -1) return 1; // Error in reading the file

    int result50 = 0, result70 = 0;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start); // Start time

    // Finding 50th and 70th smallest elements
    result50 = SelectionProb(arr, 0, size - 1, 50);
    result70 = SelectionProb(arr, 0, size - 1, 70);

    clock_gettime(CLOCK_MONOTONIC, &end); // End time

    double time_spent = time_diff(&start, &end);

    // Output the results
    printf("50 번째: %d\n", result50);
    printf("70 번째: %d\n", result70);
    printf("Running time: %f seconds\n", time_spent);

    return 0;
}

int SelectionProb(int arr[], int left, int right, int find_idx) {
    // Pivot partitioning
    int p_idx = Pivot(arr, left, right);

    // If the pivot index is the target index
    if (p_idx == find_idx - 1)
        return arr[p_idx];
    // If the target index is in the left subarray
    else if (p_idx > find_idx - 1)
        return SelectionProb(arr, left, p_idx - 1, find_idx);
    // If the target index is in the right subarray
    else
        return SelectionProb(arr, p_idx + 1, right, find_idx);
}

int Pivot(int arr[], int left, int right) {
    int low = left;
    int high = right;
    int pivot = arr[left]; // Choosing the leftmost element as pivot

    while (low < high) {
        while (low <= right && arr[low] <= pivot) {
            low++;
        }
        while (high >= left && arr[high] > pivot) {
            high--;
        }
        if (low < high) {
            Swap(arr, low, high);
        }
    }
    Swap(arr, left, high); 
    return high; 
}

void Swap(int arr[], int x, int y) {
    int tmp = arr[x];
    arr[x] = arr[y];
    arr[y] = tmp;
}
