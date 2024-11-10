#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;
/* 
물건(Things) 클래스 만들어서 진행했음.
*/
class Things { 
public:
    char name;
    int weight; // per gram
    int value;  // per won
    int value_per_weight;
};

// 내림차순
bool cmp (const Things &a , const Things &b) {
    return a.value_per_weight > b.value_per_weight; 

}

int main() { 
    
    // 배낭 초기화
    vector<Things> sack_List = { 
        {'A', 0, 0,0},{'B', 0, 0,0},{'C', 0, 0,0},{'D', 0, 0,0}
    };

    // 인풋 넣어주기

    vector<Things> input_sack_list = {
        {'D', 50, 50000, 0},
        {'A', 10, 600000, 0},
        {'C', 25, 100000, 0},
        {'B', 15, 750000, 0}
    };


    int Value_sum_in_sack = 0 ; // 배낭속 물건의 가치 합
    int Weight_sum_in_sack = 0 ; // 배낭속 무게의 합 
    int Max_sack_capacity = 40;  // 배낭 최대 용량 40g

    // item init (value_per_weight)
    vector<Things> sorted_items;
    for(Things item : input_sack_list) { 
        item.value_per_weight = item.value / item.weight;
        sorted_items.push_back(item);
    }
    sort(sorted_items.begin(),sorted_items.end(),cmp); // S queue 내림차순 정렬 (value_per_weight 을 중심으로 내림차순 정렬 cmp 함수 작성 필요). 

    // 가장 큰 VPW 가장 큰 물건 가져옴 그것이 X  
    Things X = sorted_items.front();
    // 배낭에 넣을 수 있는 무게 초과 전까지 넣음
    while(Weight_sum_in_sack + X.weight <= Max_sack_capacity) { 
        sack_List.push_back(X);
        Weight_sum_in_sack += X.weight;
        Value_sum_in_sack  += X.value;
        sorted_items.erase(sorted_items.begin());
        X = sorted_items.front();
    }
    // 남은 무게 계산
    int rest_of_sack_weight = Max_sack_capacity - Weight_sum_in_sack; 
    if (rest_of_sack_weight > 0) { 
        for (Things item : sack_List) { 
            if(item.name == X.name) { 
                sack_List[X.name].weight += rest_of_sack_weight;
                sack_List[X.name].value += rest_of_sack_weight * X.value_per_weight; 
            }
        }
    }  

    cout << "Goods  weight of goods in knapsack       value of goods in knapsack"  << endl;
    for(Things item : sack_List) { 
        cout << item.name << "           " << item.weight  << "                          " << item.value << endl;
    } 
    //sum
    cout << "Sum" <<"        " << Weight_sum_in_sack << "                         " << Value_sum_in_sack << endl; 

}