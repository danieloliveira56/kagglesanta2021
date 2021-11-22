import gurobipy as gp
from gurobipy import GRB
from helpers import get_tsp_solution, SPECIAL_CHARS, word_is_contained
import time


def get_optimal_solution(words, initial_solution=None, num_strings=3, num_wildcards=2, UB=None, LB=0,
                         linear_relaxation=False, tight_model=True, word_density=None, save_model=False,
                         letter_spacing=1, fix_suffix=0):
    opt_time = time.time()

    print(num_wildcards)
    num_letters = len(words[0])
    special1 = SPECIAL_CHARS[num_letters][0]
    special2 = SPECIAL_CHARS[num_letters][1]

    print("Running optimization function...\n")
    print(f"Wildcards: {num_wildcards}")

    if not initial_solution:
        initial_solution = get_tsp_solution(words)

    print(f"\nChecking initial solution, len={len(initial_solution)}")
    print(initial_solution)
    for w in words:
        if not word_is_contained(w, initial_solution):
            print(f"ERROR: MISSING WORD {w}")
            return

    if not UB:
        UB = len(initial_solution)

    print(f"LB:{LB}, UB:{UB}")

    print("Creating positions...")
    positions = range(UB)
    print("Creating letters...")
    letters7 = range(1, (num_letters + 1))
    if num_wildcards:
        letters8 = range(1, (num_letters + 2))
    else:
        letters8 = range(1, (num_letters + 1))
    print("Creating strings...")
    routes = range(num_strings)

    print("Creating model...")
    m = gp.Model("santa")

    # Set variable type
    var_type = GRB.BINARY
    if linear_relaxation:
        var_type = GRB.SEMICONT

    # Create x variables
    print(f"\nCreating {len(positions) * len(letters8) * len(routes)} x variables...")
    x = m.addVars(positions,
                  letters8,
                  routes,
                  vtype=var_type,
                  lb=0,
                  ub=1,
                  name="x")

    # Create z variables
    print(f"Creating {len(positions[:(-num_letters + 1)]) * len(routes) * len(words)} z variables...")
    z = m.addVars(positions[:(-num_letters + 1)],
                  routes,
                  words,
                  vtype=var_type,
                  lb=0,
                  ub=1,
                  name="z")

    # Set starting solution
    if not linear_relaxation:
        wildcards_to_place = num_wildcards
        for i, letter in enumerate(initial_solution):
            if letter == '*':
                x[i, letters8[-1], 0].start = 1
                wildcards_to_place -= 1
            else:
                x[i, int(letter), 0].start = 1
        if wildcards_to_place >= 1:
            for letter in letters7:
                x[0, letter, 0].start = 0
            x[0, letters8[-1], 0].start = 1
        if wildcards_to_place >= 2:
            for letter in letters7:
                x[num_letters, letter, 0].start = 0
            x[num_letters, letters8[-1], 0].start = 1
        for i in range(fix_suffix):
            letter = initial_solution[i]
            for fix_letter in letters8:
                x[i, fix_letter, 0].lb = 0
                x[i, fix_letter, 0].ub = 0
            if letter == '*':
                x[i, letters8[-1], 0].lb = 1
                x[i, letters8[-1], 0].ub = 1
            else:
                x[i, int(letter), 0].lb = 1
                x[i, int(letter), 0].ub = 1




    print("Setting obj\n")
    obj = m.addVar(name="obj")
    m.setObjective(obj, GRB.MINIMIZE)

    # Objective Calculation
    m.addConstrs(obj >= (i + 1) * x.sum(i, "*", k) for i in positions for k in routes)

    print(f"Writing {len([w for w in words if int(w[0]) != special1 or int(w[1]) != special2])} Free Word constraints...")
    # Each free word must appear once
    m.addConstrs((sum(sum(z[i, k, w] for k in routes) for i in positions[:(-num_letters + 1)]) >= 1
                  for w in words if int(w[0]) != special1 or int(w[1]) != special2), "Word")

    print(f"Writing {len(routes) * len([w for w in words if int(w[0]) == special1 and int(w[1]) == special2])} Special Word constraints...")
    # Each special word must appear on each string
    m.addConstrs((sum(z[i, k, w] for i in positions[:(-num_letters + 1)]) >= 1
                  for k in routes
                  for w in words if int(w[0]) == special1 and int(w[1]) == special2
                  ), "SpecialWord")

    if num_wildcards:
        print(f"Writing {len(routes)} Wildcard_Route constraints...")
        # Each string must have num_wildcards wildcards
        m.addConstrs((x.sum('*', letters8[-1], k) == num_wildcards for k in routes), "Wildcard_Route")

    if num_wildcards >= 2:
        print(f"Writing {len(routes) * len(positions[:(-num_letters + 1)])} Wildcard_Spacing constraints...")
        # Wildcards must be spaced appart
        m.addConstrs((sum(x[i + j, letters8[-1], k] for j in range(num_letters)) <= 1
                      for i in positions[:(-num_letters + 1)]
                      for k in routes), "Wildcard_Spacing")

    # There's at most one letter at a position
    print(f"Writing {len(positions) * len(routes)} num letters per position constraints...")
    m.addConstrs((sum(x[i, letter, k] for letter in letters8) <= 1
                  for k in routes
                  for i in positions), "NumLettersAt")


    if num_wildcards:
        # There's at most one word at a position, unless the wildcard is there
        print(f"Writing {len(positions) * len(routes)} WildcardWord constraints...")
        m.addConstrs(
            (z.sum(i, k, '*') <= 1 + (num_letters - 1) * sum(x[i + j, letters8[-1], k] for j in range(num_letters))
             for k in routes
             for i in positions[:(-num_letters + 1)]), "WildcardWord")
    else:
        # There's at most one word at a position
        print(f"Writing {len(positions) * len(routes)} WildcardWord constraints...")
        m.addConstrs((z.sum(i, k, '*') <= 1
                      for k in routes
                      for i in positions[:(-num_letters + 1)]), "WildcardWord")

    if tight_model:
        print(f"Writing {len(positions[:(-num_letters + 1)]) * len(routes) * len(words) * num_letters} Wordx constraints...")
        m.addConstrs((z[i, k, w] <= x[i + j, int(w[j]), k] + (x[i + j, letters8[-1], k] if num_wildcards else 0)
                      for i in positions[:(-num_letters + 1)]
                      for k in routes
                      for w in words
                      for j in range(num_letters)
                      ), "Wordx")
    else:
        print(f"Writing {len(positions[:(-num_letters + 1)]) * len(routes) * len(words)} Wordx constraints...")
        m.addConstrs((num_letters * z[i, k, w] <= sum(x[i + j, int(w[j]), k] for j in range(num_letters))
                      for i in positions[:(-num_letters + 1)]
                      for k in routes
                      for w in words
                      ), "Wordx")

    #####CUTS
    if LB:
        # There's at least one letter at a position lower than LB
        print(f"Writing {LB * len(routes)} Letter at i < LB constraints...")
        m.addConstrs((x.sum(i, '*', k) == 1
                      for k in routes
                      for i in positions[:LB + 1]), "LetterAti")

    if LB and word_density:
        # There's at least word_density[0] word at any string of length word_density[1] contained lower than LB
        print(f"Writing {(LB - word_density[1]) * len(routes)} Word at i < LB  - 7 constraints...")
        m.addConstrs((sum(z.sum(i + j, k, '*') for j in range(word_density[1])) >= word_density[0]
                      for k in routes
                      for i in positions[:LB - word_density[1]]), "WordAti")

    # Clique Cuts - Aggressive and might remove feasibility]
    if letter_spacing > 1:
        m.addConstrs(sum(x[i+j, letter, k] for j in range(letter_spacing)) <= 1
                     for i in positions[:-letter_spacing]
                     for letter in letters7
                     for k in routes)

    # # Domino Constraints
    if tight_model:
        print(f"Writing {len(positions) * len(routes)} Domino Constraints...")
        # A position can have a letter only if the previous one also has
        m.addConstrs(x.sum(i, '*', k) >= x.sum(i+1, '*', k) for i in positions[:-1] for k in routes)

    if tight_model and len(routes) > 1:
        print(f"Writing {len(positions) * len(routes)} Longest Word Symmetry Breaking Constraints...")
        # Symmetry Breaking, first strings must be longest
        m.addConstrs(
            x.sum(i, '*', k) >= x[i, letter, k + 1] for i in positions for k in routes[:-1] for letter in letters8)

    if save_model:
        print("Saving model...")
        m.write("santa.lp")

    def print_x(x_sol):
        for k in routes:
            for i in positions:
                if num_wildcards and x_sol[i, letters8[-1], k] > 0.99:
                    print("*", end="")
                else:
                    for letter in letters7:
                        if x_sol[i, letter, k] > 0.99:
                            print(letter, end="")
            print()
        print()

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            # MIP solution callback
            print(f'\nCurrent Best Solution ({model.cbGet(GRB.Callback.MIPSOL_OBJ)}):')
            x_sol = model.cbGetSolution(model._x)
            print_x(x_sol)
        elif where == GRB.Callback.MIPNODE:
            return
            # MIP node callback
            print('**** New node ****')
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for key, val in model.cbGetNodeRel(model._x).items():
                    if val > 1e-8:
                        print(*key, val)
                print()
                for key, val in model.cbGetNodeRel(model._z).items():
                    if val > 1e-8:
                        print(*key, val)

    # m.tune()
    #
    # if m.tuneResultCount > 0:
    #
    #     # Load the best tuned parameters into the model
    #     m.getTuneResult(0)
    #
    #     # Write tuned parameters to a file
    #     m.write('tune.prm')

    m._x = x
    m._z = z

    # Model Parameters
    # m.setParam(GRB.Param.Heuristics, 1.0)
    # m.setParam(GRB.Param.Symmetry, 2)
    # m.setParam(GRB.Param.PreDepRow, 1)
    if UB:
        m.setParam(GRB.Param.Cutoff, UB)

    # try:
    #     m.computeIIS()
    #     m.write("model.ilp")
    # except:
    #     pass

    print("Optimizing...")
    m.optimize(mycallback)

    print('\nTOTAL COSTS: %g' % m.objVal)

    opt_string = ""

    if not linear_relaxation:
        for k in routes:
            for i in positions:
                if num_wildcards and x[i, letters8[-1], k].x > 0.01:
                    print("*", end="")
                    opt_string += "*"
                else:
                    for letter in letters7:
                        if x[i, letter, k].x > 0.01:
                            print(letter, end="")
                            opt_string += str(letter)
            print()

    if linear_relaxation:
        print("i,letter,k,x")
        for k in routes:
            for i in positions:
                for letter in letters8:
                    if x[i, letter, k].x > 0:
                        print(f"{i},{letter},{k},{x[i, letter, k].x}")
            print()

        print("i,k,w,z")
        for k in routes:
            for i in positions[:(-num_letters + 1)]:
                for w in words:
                    if z[i, k, w].x > 0:
                        print(f"{i}, {k}, {w},{z[i, k, w].x}")
            print()

        print(f"\nChecking optimal string, len={len(opt_string)}")
        print(opt_string)
        for w in words:
            if not word_is_contained(w, opt_string):
                print(f"ERROR: MISSING WORD {w}")

    print(f"get_optimal_solution Ellapsed {time.time() - opt_time}s")
    # m.write("santa.sol")

    return opt_string