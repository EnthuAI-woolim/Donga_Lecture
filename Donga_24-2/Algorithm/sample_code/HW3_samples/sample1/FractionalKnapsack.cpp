#include <cstdlib>
#include <vector>
#include <iostream>
#include <algorithm>
class thing { //물건들에 대한 정보를 담을 class 선언
public:
    int weight; //무게
    int value; //가치
    double unitValue; //단위무게당 가치 (이게 greedy 에서 기준이 됨)
    thing(); //생성자
    thing(int weight, int value); //생성자
};
thing::thing() {
    weight = 0;
    value = 0;
    unitValue = 0.0;
}
thing::thing(int value, int weight) {
    this->weight = weight;
    this->value = value;
    this->unitValue = value*(1.0)/ weight;
}
int main() {
    int c = 40; //가방에 담을 수 있는 총 무게
    int remainCapacity = 40; //현재 가방에 담을 수 있는 남은 무게
    int currentCapacity = 0; //현재 가방에 담긴 무게
    int totalWeight = 0;
    int totalValue = 0;
    thing things[4] = {thing(600000, 10), thing(750000, 15), thing(100000, 25), thing(50000, 50)}; //물건들을 things 배열에 넣는다.
    std::sort(things, things+4, [](const thing &a, const thing &b) { //단위무게당 가치를 기준으로 정렬한다.(내림차순)
        return a.unitValue > b.unitValue;
    });
    int index = 0;
    std::cout << "GOODS     Weight of goods in knapsack     Value of goods in knapsack" << std::endl;
    while(remainCapacity > 0 && index < 4) { //가방에 남은 공간이 아직 있으면서, 모든 물건들을 다 돌아보지 않았다면 반복해서 실행한다.
        if(things[index].weight <= remainCapacity) { //현재 선택 된 물건의 무게가 가방에 남아있는 무게보다 작거나 같다면, 그 물건은 온전히 가방에 넣을 수 있음.
            currentCapacity += things[index].weight; //업데이트
            remainCapacity -= things[index].weight;
            totalWeight += things[index].weight;
            totalValue += things[index].value;
            switch(index) {
                case 0: std::cout << "  A\t\t\t"; break;
                case 1: std::cout << "  B\t\t\t"; break;
                case 2: std::cout << "  C\t\t\t"; break;
                case 3: std::cout << "  D\t\t\t"; break;
            }
            std::cout << things[index].weight << "\t\t\t   " << things[index].value << std::endl;
        }
        else { //만약 가방에 물건을 온전히 넣을 수 없다면 물건을 최대한 넣을 수 있을 만큼 쪼개서 넣는다.
            int availableValue = remainCapacity * things[index].unitValue; //업데이트
            currentCapacity += remainCapacity;
            totalValue += availableValue;
            totalWeight += remainCapacity;
            switch(index) {
                case 0: std::cout << "  A\t\t\t"; break;
                case 1: std::cout << "  B\t\t\t"; break;
                case 2: std::cout << "  C\t\t\t"; break;
                case 3: std::cout << "  D\t\t\t"; break;
            }
            std::cout << remainCapacity << "\t\t\t   " << availableValue << std::endl;
            remainCapacity = 0;
        }
        index++;
    }
    if(index < 4) {
        switch(index) {
            case 0: std::cout << "  A\t\t\t"; break;
            case 1: std::cout << "  B\t\t\t"; break;
            case 2: std::cout << "  C\t\t\t"; break;
            case 3: std::cout << "  D\t\t\t"; break;
        }
        std::cout << "0" << "\t\t\t   " << "0" << std::endl;
    }
    std::cout << " Sum\t\t\t" << totalWeight << "\t\t\t   " << totalValue << std::endl;

}