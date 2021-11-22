import gurobipy as gp
from gurobipy import GRB
from helpers import hamming_distance, SPECIAL_CHARS
import time

##REFERENCE VALUES



def get_word_clusters(words, num_clusters=3):
    num_letters = len(words[0])

    print("Running clustering function...\n")

    print("Creating model...")
    m = gp.Model("clustering")

    # Create x variables
    print(f"\nCreating {num_clusters * len(words)} x variables...")
    x = m.addVars(range(num_clusters),
                  words,
                  vtype=GRB.BINARY,
                  name="x")


    # Create y variables
    print(f"\nCreating  {len(words) * len(words)} y variables...")
    y = m.addVars(words,
                  words,
                  vtype=GRB.BINARY,
                  name="y")


    print("Setting obj\n")
    m.setObjective(sum(sum(y[w1, w2] * hamming_distance(w1, w2) for w1 in words if w1 < w2) for w2 in words), GRB.MINIMIZE)


    print(f"Writing {len(words)} Word constraints...")
    # Each free word must appear once
    m.addConstrs((x.sum(i, '*') == 1 for i in range(num_clusters)), "Word")

    print(f"Writing {len(words) * len(words)} Arc constraints...")
    m.addConstrs((y[w1, w2] >= x[i, w1] + x[i, w2] for i in range(num_clusters)
                  for w1 in words for w2 in words if w1 < w2), "Arc")

    print("Optimizing...")
    m.optimize()

    return []