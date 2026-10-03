#include <iostream>
#include <queue>
#include <vector>
#include <chrono>
#include <random>

using namespace std;

class Car {
public:
    int distsq;
    int idx;

    Car(int distsq, int idx) {
        this->idx = idx;
        this->distsq = distsq;
    }

    bool operator<(const Car& obj) const {
        return this->distsq > obj.distsq;
    }
};

// User's exact function (silent output for throughput timing)
int nearByCarsBench(const vector<pair<int, int>>& pos, int k) {
    vector<Car> cars;
    cars.reserve(pos.size());

    for (size_t i = 0; i < pos.size(); i++) {
        int distSq = (pos[i].first * pos[i].first) + (pos[i].second * pos[i].second);
        cars.push_back(Car(distSq, (int)i));
    }

    priority_queue<Car> pq(cars.begin(), cars.end());
    int checksum = 0;
    for (int i = 0; i < k; i++) {
        checksum ^= pq.top().idx;
        pq.pop();
    }
    return checksum;
}

int main() {
    // 1. Exact user input: 3 points, k = 2 run across 200,000 iterations to measure steady-state CPU time
    vector<pair<int, int>> p;
    p.push_back(make_pair(3, 3));
    p.push_back(make_pair(5, -1));
    p.push_back(make_pair(-2, 4));

    // Warmup
    volatile int sink = 0;
    for (int i = 0; i < 1000; i++) {
        sink += nearByCarsBench(p, 2);
    }

    // Benchmark exact user function across 100,000 runs
    const int ITERATIONS = 100000;
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        sink += nearByCarsBench(p, 2);
    }
    auto t1 = chrono::high_resolution_clock::now();
    double user_case_ms = chrono::duration<double, milli>(t1 - t0).count();

    // 2. Large Scale Stress Test: 50,000 cars, k = 100
    const int N_CARS = 50000;
    vector<pair<int, int>> large_p;
    large_p.reserve(N_CARS);
    mt19937 rng(42);
    for (int i = 0; i < N_CARS; i++) {
        large_p.push_back(make_pair((int)(rng() % 2000 - 1000), (int)(rng() % 2000 - 1000)));
    }

    auto t2 = chrono::high_resolution_clock::now();
    sink += nearByCarsBench(large_p, 100);
    auto t3 = chrono::high_resolution_clock::now();
    double large_case_ms = chrono::duration<double, milli>(t3 - t2).count();

    cout << "USER_CASE_TOTAL_MS: " << user_case_ms << endl;
    cout << "USER_CASE_PER_OP_US: " << (user_case_ms * 1000.0 / ITERATIONS) << endl;
    cout << "LARGE_SCALE_MS: " << large_case_ms << endl;
    cout << "SINK: " << sink << endl;

    return 0;
}
