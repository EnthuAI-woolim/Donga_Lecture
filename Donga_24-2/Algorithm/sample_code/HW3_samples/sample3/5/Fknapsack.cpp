#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace std;

struct Item {
    string name;
    int weight;
    int value;
    double value_per_weight;
};

// 비교 함수: 단위 무게당 가치가 높은 순서대로 정렬
bool compareItems(const Item& a, const Item& b) {
    return a.value_per_weight > b.value_per_weight;
}

void fractionalKnapsack(int maxCapacity, vector<Item>& items) {
    // 각 물건의 단위 무게당 가치 계산
    for (auto& item : items) {
        item.value_per_weight = static_cast<double>(item.value) / item.weight;
    }

    // 단위 무게당 가치에 따라 정렬
    sort(items.begin(), items.end(), compareItems);

    double totalValue = 0.0;
    int currentWeight = 0;

    cout << "Goods\tWeight of goods in knapsack\tValue of goods in knapsack\n";
    cout << "--------------------------------------------------------------\n";

    for (auto& item : items) {
        if (currentWeight < maxCapacity) {
            if (currentWeight + item.weight <= maxCapacity) {
                // 물건을 모두 배낭에 넣을 수 있을 때
                currentWeight += item.weight;
                totalValue += item.value;
                cout << item.name << "\t\t" << item.weight << "g\t\t\t" << item.value << "만원\n";
            } else {
                // 물건을 부분적으로 배낭에 넣을 때
                int remainingCapacity = maxCapacity - currentWeight;
                double fraction = static_cast<double>(remainingCapacity) / item.weight;
                double valueAdded = item.value * fraction;

                currentWeight += remainingCapacity;
                totalValue += valueAdded;
                cout << item.name << "\t\t" << remainingCapacity << "g\t\t\t" << valueAdded << "만원\n";
            }
        } else {
            // 배낭에 들어가지 않은 물건
            cout << item.name << "\t\t0g\t\t\t0만원\n";
        }
    }

    // 최종 합계 출력
    cout << "--------------------------------------------------------------\n";
    cout << "Sum\t\t" << currentWeight << "g\t\t\t" << totalValue << "만원\n";
}

int main() {
    int maxCapacity = 40;
    vector<Item> items = {
        {"D", 50, 5},
        {"A", 10, 60},
        {"C", 25, 10},
        {"B", 15, 75}
    };

    fractionalKnapsack(maxCapacity, items);
    return 0;
}

