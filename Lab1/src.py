"""This a piece of code that solves the Aggressivecows problem in python
it's purpose is to test the functioning of a pylint workflow in this repo"""

import sys

def can_place(stalls, cows, dist):
    """
    Checking whether this minimum distance dist is ok
    """
    count = 1  # first cow at first stall
    last_position = stalls[0]
    for i in range(1, len(stalls)):
        if stalls[i] - last_position >= dist:
            count += 1
            last_position = stalls[i]
            if count == cows:
                return True
    return False

def aggressive_cows(stalls, cows):
    """
    Running the function on different possible values of distance, binary search on answer
    """
    stalls.sort()
    low, high = 1, stalls[-1] - stalls[0]
    best = 0

    while low <= high:
        mid = (low + high) // 2
        if can_place(stalls, cows, mid):
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best

data = sys.stdin.read().split()
t = int(data[0])
IDX = 1
for _ in range(t):
    n, c = int(data[IDX]), int(data[IDX+1])
    IDX += 2
    s = list(map(int, data[IDX:IDX+n]))
    IDX += n
    print(aggressive_cows(s, c))
