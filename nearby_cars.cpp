#include <iostream>
#include <queue>
#include <vector>
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

void nearByCars(vector<pair<int, int>> pos, int k) {
    vector<Car> cars;

    for (int i = 0; i < (int)pos.size(); i++) {
        int distSq = (pos[i].first * pos[i].first) + (pos[i].second * pos[i].second);
        cars.push_back(Car(distSq, i));
    }

    priority_queue<Car> pq(cars.begin(), cars.end());
    for (int i = 0; i < k; i++) {
        cout << "CAR" << pq.top().idx << endl;
        pq.pop();
    }
}

int main() {
    vector<pair<int, int>> p;
    p.push_back(make_pair(3, 3));
    p.push_back(make_pair(5, -1));
    p.push_back(make_pair(-2, 4));
    nearByCars(p, 2);
    return 0;
}
