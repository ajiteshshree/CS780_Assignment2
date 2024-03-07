class MazeEnvironment:
    def __init__(self):
        self.rows = 3
        self.columns = 4
        self.states = [(i, j) for i in range(self.rows) for j in range(self.columns)]
        self.actions = {0: '↑', 1: '→', 2: '↓', 3: '←'}

    def display_maze(self, actions):
        if len(actions) != len(self.states):
            print("Invalid number of actions.")
            return

        action_mapping = {self.states[i]: actions[i] for i in range(len(actions))}
        print('\n------------------------------------------------------------------------------------------')
        print("Maze Environment:")
        for i in range(self.rows):
            print(" ", end="")
            for j in range(self.columns):
                print(" ___ ", end = '')
            print()
            print("|", end="")
            for j in range(self.columns):
                state = (i, j)
                if state == (0, 3):
                    print(" G✥|", end='')
                elif state == (1, 1):
                    print(" W✥|", end='')
                elif state == (1,3):
                    print(' H✥ |', end ='') 
                else:
                    action = action_mapping.get(state, None)
                    if action is None:
                        print("     |", end='')
                    else:
                        print(f" {self.actions[action]}  |", end='')
            print()
        print(" ", end="")
        for j in range(self.columns):
            print(" ——— ", end="")
        print()