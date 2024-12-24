#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define V 8  // 도시의 개수
#define POP_SIZE 8  // 후보해의 개수
#define MUTATION_RATE 0.01  // 돌연변이 확률
#define GENERATIONS 1000  // 최대 세대 수
#define CONVERGENCE_LIMIT 10  // 적합도 변화가 멈춘 횟수 (10번 이상 변화 없으면 종료)

// 도시 구조체 정의
typedef struct {
    char name;  // 도시 이름
    int x, y;   // 도시의 좌표
} City;

// 도시 거리 계산 함수
double distance(City a, City b) {
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// 경로의 총 거리를 계산하는 함수
double totalDistance(int tour[], City cities[]) {
    double total = 0;
    for (int i = 0; i < V - 1; i++) {
        total += distance(cities[tour[i]], cities[tour[i + 1]]);
    }
    total += distance(cities[tour[V - 1]], cities[tour[0]]);  // 마지막 도시에서 첫 도시로 돌아오는 거리
    return total;
}

// 경로의 적합도를 계산하는 함수 (적합도 = 1 / 총 거리)
double fitness(int tour[], City cities[]) {
    double totalDist = totalDistance(tour, cities);
    return 1.0 / totalDist;  // 적합도는 거리가 짧을수록 높음
}

// 도시 순서 배열을 출력하는 함수
void printTour(int tour[], City cities[]) {
    for (int i = 0; i < V; i++) {
        printf("%c ", tour[i] + 'A');
    }
    printf("%c", tour[0] + 'A');  // 마지막에 A 도시 추가
}

// 각 후보해들의 이동순서와 거리를 출력하는 함수
void printPopulation(int population[POP_SIZE][V], City cities[]) {
    for (int i = 0; i < POP_SIZE; i++) {
        printf("Tour %d: ", i + 1);
        printTour(population[i], cities);
        printf(" - Distance: %.2f\n", totalDistance(population[i], cities));
    }
}

// 임의의 후보해 생성 함수
void generatePopulation(int population[POP_SIZE][V]) {
    // 도시 배열 초기화
    for (int i = 0; i < POP_SIZE; i++) {
        // 도시 순서 배열 초기화
        for (int j = 0; j < V; j++) {
            population[i][j] = j;  // 도시 번호를 배열에 저장
        }

        // 첫 번째 도시는 항상 A 도시 (인덱스 0)로 고정
        int citiesLeft[V - 1];
        for (int j = 1; j < V; j++) {
            citiesLeft[j - 1] = j;  // 1번부터 7번까지의 도시
        }

        // 나머지 도시는 무작위로 섞기
        for (int j = V - 2; j > 0; j--) {
            int k = rand() % (j + 1);
            int temp = citiesLeft[j];
            citiesLeft[j] = citiesLeft[k];
            citiesLeft[k] = temp;
        }

        // 첫 번째 도시는 항상 A로 고정하고, 나머지 도시를 순서대로 배치
        population[i][0] = 0;  // A 도시를 첫 번째로 배치
        for (int j = 1; j < V; j++) {
            population[i][j] = citiesLeft[j - 1];
        }
    }
}

// 부모 후보해에서 주어진 도시의 인덱스를 찾는 함수
int findCityIndex(int parent[], int city) {
    for (int i = 0; i < V; i++) {
        if (parent[i] == city) {
            return i;
        }
    }
    return -1;
}

// 사이클 교차 연산 함수
void cycleCrossover(int parent1[], int parent2[], int child[]) {
    int visited[V] = {0};  // 각 도시가 자식에 이미 들어갔는지 체크
    int cycleIndex = 0;    // 사이클 시작 인덱스

    while (cycleIndex < V) {
        int start = cycleIndex;
        int current = start;
        do {
            // 현재 도시를 자식에 추가
            child[current] = parent1[current];
            visited[current] = 1;

            // 부모1의 해당 도시 위치에서 부모2의 도시를 찾아 교차
            current = findCityIndex(parent2, parent1[current]);
        } while (current != start);

        do {
            cycleIndex++;
        } while (cycleIndex < V && visited[cycleIndex] == 1);
    }

    for (int i = 0; i < V; i++) {
        if (visited[i] == 0) {
            child[i] = parent2[i];
        }
    }
}

// 선택된 부모 후보해에 교차 연산을 수행하여 새로운 자식 후보해들을 만드는 함수
void applyCrossover(int population[POP_SIZE][V], City cities[]) {
    for (int i = 0; i < POP_SIZE; i += 2) {
        int parent1 = i;
        int parent2 = i + 1;

        int child[V];
        cycleCrossover(population[parent1], population[parent2], child);

        for (int j = 0; j < V; j++) {
            population[parent1][j] = child[j];
            population[parent2][j] = child[j];
        }
    }
}

// 돌연변이 연산 함수
void mutate(int population[POP_SIZE][V]) {
    for (int i = 0; i < POP_SIZE; i++) {
        if ((rand() / (double)RAND_MAX) < MUTATION_RATE) {
            int idx1 = rand() % V;
            int idx2 = rand() % V;
            while (idx1 == idx2) {
                idx2 = rand() % V;
            }
            int temp = population[i][idx1];
            population[i][idx1] = population[i][idx2];
            population[i][idx2] = temp;
        }
    }
}

// 최적 후보해를 찾는 함수
int* getBestSolution(int population[POP_SIZE][V], City cities[]) {
    int* bestTour = population[0];
    double bestFitness = fitness(population[0], cities);

    for (int i = 1; i < POP_SIZE; i++) {
        double currentFitness = fitness(population[i], cities);
        if (currentFitness > bestFitness) {
            bestFitness = currentFitness;
            bestTour = population[i];
        }
    }
    return bestTour;
}

int main() {
    srand(time(NULL));

    int population[POP_SIZE][V];
    City cities[V] = {
        {'A', 0, 3}, {'B', 7, 5}, {'C', 6, 0}, {'D', 4, 3},
        {'E', 1, 0}, {'F', 5, 3}, {'H', 4, 1}, {'G', 2, 2}
    };

    generatePopulation(population);

    int generation = 0;
    double previousBestFitness = -1.0;
    double currentBestFitness;
    int noChangeCount = 0;  // 적합도가 변화하지 않은 횟수

    while (generation < GENERATIONS && noChangeCount < CONVERGENCE_LIMIT) {
        // 현재 세대에서 가장 적합한 후보해 찾기
        int* bestTour = getBestSolution(population, cities);
        currentBestFitness = fitness(bestTour, cities);

        // 적합도가 변하지 않으면 변화 횟수 증가
        if (currentBestFitness == previousBestFitness) {
            noChangeCount++;
        } else {
            noChangeCount = 0;  // 적합도가 변하면 카운트를 초기화
        }

        // 적합도가 변화하지 않으면 종료
        if (noChangeCount >= CONVERGENCE_LIMIT) {
            break;
        }

        // 적합도가 변했다면 교차와 돌연변이 연산 수행
        applyCrossover(population, cities);
        mutate(population);

        previousBestFitness = currentBestFitness;
        generation++;
    }

    // 마지막 세대의 후보해 출력
    printf("\nLast generation's population:\n");
    printPopulation(population, cities);

    // 최적 후보해 출력
    printf("\nBest solution after %d generations:\n", generation);
    int* bestTour = getBestSolution(population, cities);
    printf("Tour: ");
    printTour(bestTour, cities);
    printf(" - Distance: %.2f\n\n", totalDistance(bestTour, cities));

    return 0;
}
