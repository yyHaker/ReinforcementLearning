# -*- coding:utf-8 -*-
"""
realize the GridWord using DP. (synchronous backups)
"""
# declare states
states = [i for i in range(16)]
# declare values
values = [0 for _  in range(16)]
# declare actions
actions = ['n', 'e', 's', 'w']
# declare discount value
gamma = 1.00

# declare the change of action w.r.t state
ds_actions = {"n": -4, "e": 1, "s": 4, "w": -1}

# (s, a) -> s'
def nextState(s, a):
    next_state = s
    # over the board state does't change
    if (a == 'w' and s % 4 == 0) or (a == 's' and s > 11) or \
            (a == 'e' and (s + 1) % 4 == 0) or (a == 'n' and s < 4):
        pass
    else:
        ds = ds_actions[a]
        next_state = s + ds
    return next_state

# get the immediate reward of a state
def rewardOf(s):
    return 0 if s in [0, 15] else -1

def isTerminateState(s):
    return s in [0, 15]

# get all successors of s
def getSuccessors(s):
    successors = []
    if isTerminateState(s):
        return successors
    else:
        for a in actions:
            next_state = nextState(s, a)
            successors.append(next_state)
        return successors

# update the value of s according to  s's successors' value
def updateValue(s):
    succesors = getSuccessors(s)
    new_value = 0.
    reward = rewardOf(s)
    for next_state in succesors:
        new_value += 0.25 * (reward + gamma * values[next_state])
    return new_value

# iterate a epoch
def performOneIteration():
    newValues = [0 for _ in range(16)]
    for s in states:
        newValues[s] = updateValue(s)
    global values  # state values function
    values = newValues
    printValue(values)

# print values
def printValue(v):
    for i in range(16):
        print('{0: >6.2f}'.format(v[i]), end=" ")
        if (i+1) % 4 == 0:
            print(" ")
    print()

def main():
    max_iterate_times = 160
    cur_iterate_times = 0
    while cur_iterate_times <= max_iterate_times:
        print("Iterate No.{0}".format(cur_iterate_times+1))
        performOneIteration()
        cur_iterate_times += 1
    print("final state value function..........")
    printValue(values)

if __name__ == "__main__":
    main()