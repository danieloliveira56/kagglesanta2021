import numpy as np
from helpers import get_tsp_solution, SPECIAL_CHARS, word_is_contained, get_words, cost
import subprocess

def get_acvrp_solution(words):

    num_letters = len(words[0])
    special1 = SPECIAL_CHARS[num_letters][0]
    special2 = SPECIAL_CHARS[num_letters][1]
    special_suffixes = [str(special1) + str(special2), str(special1) + "*", "*" + str(special2)]


    special_words = [w for w in words if w[:2] in special_suffixes]
    words = words + special_words + special_words

    i = 1
    capacity = 0
    for _ in special_words:
        capacity += i
        i += 1

    # CREATE DISTANCE MATRIX
    SIZE = len(words)
    M = np.zeros((SIZE, SIZE), dtype='int8')
    for j in range(SIZE):
        for k in range(SIZE):
            M[j, k] = cost(words[j], words[k])
            M[k, j] = cost(words[k], words[j])

    # WRITE PROBLEM FILE
    f = open(f'group.par', 'w')
    f.write("PROBLEM_FILE = distances.vrp\n")
    f.write("MTSP_SOLUTION_FILE = output.txt\n")

    f.write("MOVE_TYPE = 5 SPECIAL\n")
    f.write("GAIN23 = NO\n")
    f.write("KICKS = 2\n")
    f.write("MAX_SWAPS = 0\n")
    f.write("POPULATION_SIZE = 100\n")

    f.write("RUNS = 1\n")
    f.write("TIME_LIMIT = 3600\n")  # seconds
    f.write("TRACE_LEVEL = 1\n")  # seconds
    # f.write("MAX_TRIALS = 10000\n")  # seconds
    f.write("RUNS = 10\n")  # seconds
    f.close()

    # WRITE PARAMETER FILE
    f = open(f'distances.vrp', 'w')
    f.write("NAME: distances\n")
    f.write("TYPE: ACVRP\n")
    f.write("COMMENT: Asymmetric CVPR\n")
    f.write(f"DIMENSION: {SIZE}\n")
    f.write("VEHICLES: 3\n")
    f.write(f"CAPACITY: {capacity}\n")
    f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
    f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
    f.write("EDGE_WEIGHT_SECTION\n")
    for j in range(SIZE):
        for k in range(SIZE):
            f.write(f"{M[j, k]:2d} ")
        f.write("\n")

    special_costs = {}
    i = 1
    for w in special_words:
        special_costs[w] = i
        i += 1

    f.write("DEMAND_SECTION\n")
    for j in range(SIZE):
        if words[j][:2] in special_suffixes:
            f.write(f"{j+1}  {special_costs[words[j]]}\n")
        else:
            f.write(f"{j+1}  0\n")

    f.write("EOF\n")


    f.close()

    # EXECUTE TSP SOLVER\
    print("RUnninglkhg")
    print(special_suffixes)
    subprocess.run(
        [
            './LKH',
            'group.par',
        ],
    )

    # READ RESULTING ORDER
    with open('output.txt') as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if 'The tours traveled by the 3 salesmen are:' in ln: break
    perms = [x.split(" (")[0].split(" ") for x in lines[i + 1:]]
    print(perms)

    # CREATE STRING
    groups = [
        [], [], []
    ]
    print(groups)
    for k, route in enumerate(perms):
        print(k, route)
        for i in route:
            print(int(i))
            groups[k].append(words[int(i)-1])
            print(words[int(i)-1], end=" ")
        print()

    for tour in groups:
        string_k = tour[0] if len(tour[0]) > 1 else ""
        for j in range(1, len(tour)):
            word_distance = cost(tour[j - 1], tour[j])
            if "*" in tour[j]:
                # the word appended should be preserved
                string_k = string_k[:-num_letters + word_distance] + tour[j]
            else:
                string_k += tour[j][-word_distance:]
        print(string_k)

    return groups