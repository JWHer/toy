def solution(N, number):
    solver = Solver(N, number)
    # return solver.solve(8)
    solver.answer()

class Solver:
    def __init__(self, N, number):
        self.N = N
        self.number = number
        self.map = {
            0:{
                "answer": 0,
                "operator": None,
                "before": None,
            }
        }
        self.algorithms = [
            self._plus,
            self._minus,
            self._multifly,
            self._division,
            self._concatenate
        ]
        
    def solve(self, max):
        
        for it in range(1,max):
            self._find(it)
            if self.number in self.map:
                return self.map[self.number]["answer"]
        # if you can't find number in max iteration, return -1
        return -1

    def answer(self):
        if self.number not in self.map:
            self.solve(10)
        if self.number not in self.map:
            print("can't find answer in 10 iterations")
            return

        top = self.map[self.number]
        print(top)
        while top["before"]:
            top = self.map[top["before"]]
            # TODO print fomular
            print(top)

        
    def _find(self, iteration):
        keys = list(self.map.keys())
        values = [ value['answer'] for value in self.map.values() ]
        befores = [ idx for idx, value in enumerate(values) if value==(iteration-1) ]
        
        for idx in befores:
            key = keys[idx]
            for algorithm in self.algorithms:
                algorithm(key, iteration)
                # pruning
                if self.number in self.map: return
        
    def _plus(self, key, iteration):
        value = key+self.N
        if value not in self.map:
            self.map[value]={
                "answer": iteration,
                "operator": "plus",
                "before": key,
            }
            
    def _minus(self, key, iteration):
        value = key-self.N
        if value>0 and value not in self.map:
            self.map[value]={
                "answer": iteration,
                "operator": "minus",
                "before": key,
            }
            
    def _multifly(self, key, iteration):
        value = key*self.N
        if value not in self.map:
            self.map[value]={
                "answer": iteration,
                "operator": "multifly",
                "before": key,
            }
                
    def _division(self, key, iteration):
        value = key//self.N
        if value not in self.map:
            self.map[value]={
                "answer": iteration,
                "operator": "division",
                "before": key,
            }
            
    def _concatenate(self, key, iteration):
        value = key*int('1'*iteration)
        if value not in self.map:
            self.map[value]={
                "answer": iteration,
                "operator": "concatenate",
                "before": None,
            }

if __name__ == '__main__':
    solution(8, 53)
