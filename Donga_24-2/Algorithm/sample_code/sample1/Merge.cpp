#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

void merge(vector<int>& arr, int first, int mid, int last)
{
    vector<int> sorted(last - first + 1);
    int i = first, j = mid + 1, k = 0;

    while (i <= mid && j <= last)
    {
        if (arr[i] <= arr[j])
            sorted[k++] = arr[i++];
        else
            sorted[k++] = arr[j++];
    }

    while (i <= mid)
        sorted[k++] = arr[i++];

    while (j <= last)
        sorted[k++] = arr[j++];

    for (i = first, k = 0; i <= last; i++, k++)
        arr[i] = sorted[k];
}

void mergeSort(vector<int>& arr, int first, int last)
{
    if (first < last)
    {
        int mid = (first + last) / 2;

        mergeSort(arr, first, mid);   
        mergeSort(arr, mid + 1, last); 
        merge(arr, first, mid, last); 
    }
}

int main()
{
    vector<int> arr;
    ifstream inputFile("input_sort.txt");
    ofstream outputFile("output_merge_sort.txt");

    if (!inputFile.is_open())
    {
        cout << "Error opening input file." << endl;
        return 1;
    }

    int num;
    while (inputFile >> num)
    {
        arr.push_back(num);
    }
    inputFile.close();

    auto start = high_resolution_clock::now();

    mergeSort(arr, 0, arr.size() - 1);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    for (const auto& elem : arr)
    {
        outputFile << elem << endl;
    }
    outputFile.close();

    cout << "running time " << duration << " milliseconds." << endl;

    return 0;
}
