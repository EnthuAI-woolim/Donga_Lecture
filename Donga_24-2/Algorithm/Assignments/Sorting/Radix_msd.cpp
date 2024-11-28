// C++ implementation of MSD Radix Sort
// of MSD Radix Sort using counting sort()
#include <iostream>
#include <fstream>
#include <math.h>
#include <unordered_map>

#define MAX_NUMBERS 10000
using namespace std;

extern int readFile(const std::string &filename, int *A, int max_numbers);
extern void writeFile(const std::string &filename, int *A, int n);

// A utility function to print an array
void print(int* arr, int n)
{
    for (int i = 0; i < n; i++) {
        cout << arr[i] << " ";
    }
    cout << endl;
}

// A utility function to get the digit
// at index d in a integer
int digit_at(int x, int d)
{
    return (int)(x / pow(10, d - 1)) % 10;
}

// The main function to sort array using
// MSD Radix Sort recursively
void MSD_sort(int* arr, int lo, int hi, int d)
{

    // recursion break condition
    if (hi <= lo) {
        return;
    }

    int count[10 + 2] = { 0 };

    // temp is created to easily swap strings in arr[]
    unordered_map<int, int> temp;

    // Store occurrences of most significant character
    // from each integer in count[]
    for (int i = lo; i <= hi; i++) {
        int c = digit_at(arr[i], d);
        count[c + 2]++;
    }

    // Change count[] so that count[] now contains actual
    //  position of this digits in temp[]
    for (int r = 0; r < 10 + 1; r++)
        count[r + 1] += count[r];

    // Build the temp
    for (int i = lo; i <= hi; i++) {
        int c = digit_at(arr[i], d);
        temp[count[c + 1]++] = arr[i];
    }

    // Copy all integers of temp to arr[], so that arr[] now
    // contains partially sorted integers
    for (int i = lo; i <= hi; i++)
        arr[i] = temp[i - lo];

    // Recursively MSD_sort() on each partially sorted
    // integers set to sort them by their next digit
    for (int r = 0; r < 10; r++)
        MSD_sort(arr, lo + count[r], lo + count[r + 1] - 1,
                 d - 1);
}

// function find the largest integer
int getMax(int arr[], int n)
{
    int mx = arr[0];
    for (int i = 1; i < n; i++)
        if (arr[i] > mx)
            mx = arr[i];
    return mx;
}

// Main function to call MSD_sort
void radixsort(int* arr, int n)
{
    // Find the maximum number to know number of digits
    int m = getMax(arr, n);

    // get the length of the largest integer
    int d = floor(log10(abs(m))) + 1;

    // function call
    MSD_sort(arr, 0, n - 1, d);
}

// Driver Code
int main()
{
    // Input array
    int arr[MAX_NUMBERS];

    // Size of the array
    int n = readFile("input.txt", arr, MAX_NUMBERS);

    // printf("Unsorted array : ");

    // Print the unsorted array
    print(arr, n);

    // Function Call
    radixsort(arr, n);

    writeFile("radix_msd_output.txt", arr, n);

    // printf("Sorted array : ");

    // Print the sorted array
    // print(arr, n);

    return 0;
}


int readFile(const std::string &filename, int *A, int max_numbers) {
    std::ifstream file(filename);
    if (!file) {
        std::cout << "파일을 열 수 없습니다.\n";
        return -1;
    }

    int n = 0;
    while (file >> A[n]) {
        n++;
        if (n >= max_numbers) {
            std::cout << "저장 가능한 최대 숫자 개수를 초과했습니다.\n";
            break;
        }
    }

    file.close();
    return n;
}

void writeFile(const std::string &filename, int *A, int n) {
    std::ofstream file(filename);
    if (!file) {
        std::cout << "결과 파일을 열 수 없습니다.\n";
        return;
    }

    for (int i = 0; i < n; ++i) {
        file << A[i] << "\n";
    }

    file.close();
    std::cout << filename << "을 생성하였습니다.\n";
}