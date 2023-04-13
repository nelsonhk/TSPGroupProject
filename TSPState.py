import copy


class TSPState:
    array = [[]]
    cost = float("inf")
    path = [0]

    # Time complexity: O(n)
    # Space complexity: O(1)
    def printArray(self):
        for arr in self.array:
            print(arr)

    # Time complexity: O(n)
    # Space complexity: O(1)
    def getMinOfRow(self, row: int):
        return min(self.array[row])

    # Time complexity: O(n)
    # Space complexity: O(1)
    def getMinOfColumn(self, col: int):
        min = float("inf")
        for i in range(len(self.array)):
            value = self.array[i][col]
            if value < min:
                min = value
        return min

    # Time complexity: O(1)
    # Space complexity: O(1)
    # I chose to prioritize getting new BSSFs over expanding every state
    # the 700 weight should provide enough of a weight to prioritize the states
    # with more paths
    def getPriorityKey(self):
        return self.cost - len(self.path) * 700

    # Time complexity: O(n^2)
    # Space complexity: O(1)
    def reduce(self):
        for i in range(0, len(self.array)): # row
            min = self.getMinOfRow(i)
            self.cost += min
            if min == 0 or min == float("inf"):
                continue
            for j in range(0, len(self.array[i])): # column
                self.array[i][j] = self.array[i][j] - min

        for j in range(0, len(self.array[0])): # column
            min = self.getMinOfColumn(j)
            self.cost += min
            if min == 0:
                continue
            for i in range(0, len(self.array)): # row
                self.array[i][j] = self.array[i][j] - min

    # Time complexity: O(1)
    # Space complexity: O(1)
    # Returns True if this state is a leaf and therefor a potential bssf
    def isLeaf(self) -> bool:
        if len(self.path) == len(self.array) + 1 and self.path[0] == self.path[len(self.path) - 1]:
            return True
        else:
            return False

    # Time complexity: O(n^2)
    # Space complexity: O(n)
    # Expands the state, returning the possible states.
    def expand(self):
        expandedStates = []
        # iterate over row, finding ones that aren't infinity
        rowIndex = self.path[len(self.path) - 1]
        for j in range(len(self.array[rowIndex])):
            if self.array[rowIndex][j] != float("inf"):
                # create new state:
                state = TSPState()
                # ... define array,
                state.array = copy.deepcopy(self.array)
                # setting source row to inf
                state.array[rowIndex] = [float("inf") for _ in range(len(state.array[rowIndex]))]
                # setting dest col to inf
                for i in range(len(state.array)):
                    state.array[i][j] = float("inf")
                # setting mirror vector to inf
                state.array[j][rowIndex] = float("inf")
                # ... define cost,
                state.cost = self.cost + self.array[rowIndex][j]
                # ... and define path
                state.path = copy.deepcopy(self.path)
                state.path.append(j)
                expandedStates.append(state)

        return expandedStates

    # this is for queue.PriorityQueue library
    # Time complexity: O(1)
    # Space complexity: O(1)
    def __lt__(self, other):
        return self.getPriorityKey() < other.getPriorityKey()

    # this is for queue.PriorityQueue library
    # Time complexity: O(1)
    # Space complexity: O(1)
    def __gt__(self, other):
        return self.getPriorityKey() > other.getPriorityKey()