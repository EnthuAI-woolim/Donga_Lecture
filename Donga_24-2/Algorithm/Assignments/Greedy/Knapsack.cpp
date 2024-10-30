#include <iostream>
#include <algorithm>
#include <vector>
#include <ctime>
#include <iomanip>

using namespace std;

// 아이템 구조체 정의
struct Item { 
    char label;    
    int weight;    
    int value;     

    Item(char label, int weight, int value) 
        : label(label), weight(weight), value(value) {} 
};

// 가치 대비 무게 비율로 정렬하기 위한 비교 함수
bool cmp(Item a, Item b) { 
    return (double)a.value / a.weight > (double)b.value / a.weight; 
}

// 분수 배낭 문제 해결 함수
void fractionalKnapsack(Item arr[], int capacity, int size, 
                          vector<int>& addedWeights, 
                          vector<double>& addedValues) { 
    sort(arr, arr + size, cmp); // 아이템을 가치 대비 무게 비율로 정렬
    int curWeight = 0;          // 현재 무게
    double totalValue = 0.0;    // 총 가치

    for (int i = 0; i < size; i++) { 
        if (curWeight + arr[i].weight <= capacity) { 
            curWeight += arr[i].weight; // 아이템을 전부 추가
            totalValue += arr[i].value;
            addedWeights.push_back(arr[i].weight);
            addedValues.push_back(arr[i].value);
        } else { 
            int remain = capacity - curWeight; // 남은 용량
            double fractionalValue = arr[i].value * ((double)remain / arr[i].weight); // 부분 아이템의 가치 계산
            totalValue += fractionalValue;
            addedWeights.push_back(remain);
            addedValues.push_back(fractionalValue);
            break; // 용량을 초과하므로 루프 종료
        }
    }

    // 추가된 아이템이 부족한 경우 0으로 채우기
    for (int i = addedWeights.size(); i < size; i++) {
        addedWeights.push_back(0);
        addedValues.push_back(0.0);
    }
}


int main() { 
    int capacity = 40;  
    Item arr[] = { { 'A', 10, 60 }, { 'B', 15, 75 }, { 'C', 25, 10 }, { 'D', 50, 5 } };
    int size = sizeof(arr) / sizeof(arr[0]);

    vector<int> addedWeights;
    vector<double> addedValues;

    clock_t start = clock();
    fractionalKnapsack(arr, capacity, size, addedWeights, addedValues);
    clock_t end = clock();

    
    
    cout << "\n" <<	"Goods    Weight of goods in knapsack     Value of goods in knapsack\n";
    
    int totalWeight = 0;
    double totalValue = 0.0;

    for (int i = 0; i < size; i++) {
        cout << setw(3) << arr[i].label
			 << setw(20) << addedWeights[i]
			 << setw(31) << addedValues[i] << "\n";
		totalWeight += addedWeights[i];
		totalValue += addedValues[i];
    }

    cout << setw(4) << "Sum" 
		 << setw(19) << totalWeight 
         << setw(31) << totalValue << "\n\n";

	cout << "Running time: " << (double)(end - start) / CLOCKS_PER_SEC * 1000 << " ms\n\n";

    return 0; 
}