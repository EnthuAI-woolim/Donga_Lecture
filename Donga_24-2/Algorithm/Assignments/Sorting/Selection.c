int main() {
    int n = 0;
    int A[n];

    for (int i = 0; i < n; ++i) {
        int min = i;
        for (int j = i+1; j < n; ++j) {
            if (A[j] < A[min]) {
                min = j;
            }
        }
        int temp = A[min];
        A[min] = A[i];
        A[i] = temp;
    }
}