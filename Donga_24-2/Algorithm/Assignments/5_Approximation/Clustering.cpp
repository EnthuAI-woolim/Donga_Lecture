#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <unordered_map>
#include <algorithm>

using namespace std;

struct Point {
    int x, y;

    // == 연산자 정의
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

// 유클리드 거리 계산 함수
double calculateDistance(const Point& p1, const Point& p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

// Approx_k_Clusters 함수 구현
void Approx_k_Clusters(const vector<Point>& points, int k) {
    int n = points.size();
    if (k <= 1 || n < k) {
        cout << "Invalid number of clusters." << endl;
        return;
    }

    vector<Point> centers;  // 센터 리스트
    vector<int> clusters(n, -1);  // 각 점의 클러스터 번호 저장
    vector<double> distances(n, numeric_limits<double>::max());  // 각 점의 최소 거리

    srand(static_cast<unsigned>(time(0)));

    // 1. 첫 번째 센터를 랜덤으로 선택
    int randomIndex = rand() % n;
    centers.push_back(points[randomIndex]);

    // 2. 2번째부터 k번째 센터 선택
    for (int j = 1; j < k; ++j) {
        int farthestPointIndex = -1;
        double maxDistance = -1;

        for (int i = 0; i < n; ++i) {
            // 4. 센터가 아닌 점에 대해 거리 계산
            if (find(centers.begin(), centers.end(), points[i]) == centers.end()) {
                double distanceToClosestCenter = calculateDistance(points[i], centers.back());
                distances[i] = min(distances[i], distanceToClosestCenter);

                // 가장 먼 점 찾기
                if (distances[i] > maxDistance) {
                    maxDistance = distances[i];
                    farthestPointIndex = i;
                }
            }
        }

        // 6. 가장 먼 점을 새로운 센터로 추가
        centers.push_back(points[farthestPointIndex]);
    }

    // 7. 각 점을 가장 가까운 센터에 할당
    unordered_map<int, vector<Point>> clustersMap;
    for (int i = 0; i < n; ++i) {
        int nearestCenterIndex = -1;
        double minDistance = numeric_limits<double>::max();

        for (int j = 0; j < k; ++j) {
            double distance = calculateDistance(points[i], centers[j]);
            if (distance < minDistance) {
                minDistance = distance;
                nearestCenterIndex = j;
            }
        }

        clusters[i] = nearestCenterIndex;
        clustersMap[nearestCenterIndex].push_back(points[i]);
    }

    // 결과 출력
    cout << "Clusters and their points" << endl;
    for (int j = 0; j < k; ++j) {
        cout << "Cluster " << j + 1 << " (Center: " << centers[j].x << ", " << centers[j].y << "): " << endl;
        for (const auto& point : clustersMap[j]) {
            cout << "(" << point.x << ", " << point.y << ") ";
        }
        cout << endl << endl;
    }
}

vector<Point> readDataFromFile(const string& filename) {
    vector<Point> points;
    ifstream inputFile(filename);

    if (!inputFile.is_open()) {
        cerr << "Error: Could not open the file " << filename << endl;
        return points;
    }

    int x, y;
    while (inputFile >> x >> y) {
        points.push_back({x, y}); // Point 구조체로 저장
    }

    inputFile.close();
    return points;
}

int main() {
    const string filename = "clustering_input.txt";
    vector<Point> points = readDataFromFile(filename);

    int k = 8;
    Approx_k_Clusters(points, k);

    return 0;
}
