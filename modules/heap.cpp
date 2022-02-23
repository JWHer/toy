#include <vector>
#include <stdexcept>
using namespace std;

template <typename T>
class Heap {
    public:
    Heap(){
        // max heap
        heap = new vector<T>();
    }

    void insert(T item) {
        // 1. insert back
        heap.push_back(item);
        int index = heap.size();

        /** upheap **/
        // 2. if parent smaller than me, change
        do {
            int parent_index = Heap.parent(index);
            T parent = select(parent_index);
            if ( smaller(parent, item) ){
                swap(&parent, &item);
            } else {
                break;
            }
            index = parent_index;
        } while(index!=0)
        // 3. do while parent bigger than me or root
    }

    T select(int index) {
        return heap[index];
    }

    int search(T item) {
        return _search(0, item);
    }

    T update(int index, T item) {
        throw logic_error("Function not yet implemented") {};
    }

    T remove(int index) {
        T item = select(index);

        // int last_index = size() - 1;
        // T last = select(last_index);
        // swap(&item, &last);

        T last = heap.pop_back();
        heap[index] = last;

        /** downheap **/

        int left_index = left_child(index);
        T left = select(left_index);
        
    }

    int remove(T item) {

    }

    int size(){
        return heap.size();
    }
    
    private:
    vector<T> heap;

    static int parent(int n) {
        return (n-1)/2;
    }

    static int left_child(int n) {
        return 2*n+1;
    }

    static int right_child(int n) {
        return 2*n+2;
    }

    void heapify(int n) {

    }

    int _search(int index, T item) {
        if (index >= size()) return -1;

        T me = select(index);
        if (me == item) return index;
        // do not need to find child smaller than me
        if (smaller(item, me)) return -1;

        index = _search(Heap.left_child(index), item);
        if (index!=-1) return index;

        index = _search(Heap.right_child(index), item);
        return index;
    }

    virtual bool smaller(T a, T b) {
        return a < b;
    }
};