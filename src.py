import sys

def can_place(stalls, cows, dist):
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
idx = 1
for _ in range(t):
    n, c = int(data[idx]), int(data[idx+1])
    idx += 2
    stalls = list(map(int, data[idx:idx+n]))
    idx += n
    print(aggressive_cows(stalls, c))