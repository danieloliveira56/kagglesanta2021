from itertools import combinations
import gurobipy as gp
from gurobipy import GRB, tupledict
from helpers import get_tsp_solution, SPECIAL_CHARS, word_is_contained, get_words, cost, get_unique_words_in_string, wild_friends
import time


def solve_atsp(special_words, free_words, word_pool, initial_solution=None, num_letters=7, num_strings=3, UB=None,
               linear_relaxation=False, save_model=False, arc_limit=6, freeword_count=None, max_length=None,
               min_max_obj=True, dfj_model=True, mtz_model=False):

    if UB == 0:
        UB = 2441

    print("\nRunning solve_atsp:")
    print(f"\tspecial_words: {special_words}")
    print(f"\tfree_words: {free_words}")
    print(f"\tword_pool: {word_pool}")
    print(f"\tinitial_solution: {initial_solution}")
    print(f"\tnum_letters: {num_letters}")
    print(f"\tnum_strings: {num_strings}")
    print(f"\tUB: {UB}")
    print(f"\tlinear_relaxation: {linear_relaxation}")
    print(f"\tsave_model: {save_model}")
    print(f"\tarc_limit: {arc_limit}")
    print(f"\tfreeword_count: {freeword_count}")
    print(f"\tmax_length: {max_length}")
    print(f"\tmin_max_obj: {min_max_obj}")
    print()

    has_super_words = any(len(w) > num_letters for w in word_pool)

    assert type(special_words) is list
    assert type(special_words[0]) is list
    assert type(free_words[0]) is str
    assert type(word_pool[0]) is str
    assert len(special_words) == num_strings


    seen = set()
    word_pool = [word for word in word_pool if not (word in seen or seen.add(word))]
    seen = set()
    free_words = [word for word in free_words if not (word in seen or seen.add(word))]

    if not num_letters:
        num_letters = len(free_words[0])

    print(f"Solving ATSP with {len(word_pool)} words...")

    print("Creating strings...")
    routes = range(num_strings)

    wildcards = [w for w in word_pool if '*' in w]

    # Dummy words for route start
    for k in routes:
        word_pool.append(str(k))
    if mtz_model:
        for k in routes:
            word_pool.append(str(k)+str(k))

    print("Creating model...")
    m = gp.Model("santa")

    # Set variable type
    var_type = GRB.BINARY
    if linear_relaxation:
        var_type = GRB.SEMICONT

    # Create x variables
    print(f"\nCreating {len(word_pool) * len(word_pool) * len(routes)} x arc variables...")
    x = m.addVars(word_pool,
                  word_pool,
                  routes,
                  vtype=var_type,
                  lb=0,
                  ub=1,
                  name="x")
    if mtz_model:
        u = m.addVars(word_pool,
                      routes,
                      lb=0,
                      vtype=GRB.INTEGER,
                      name="u")

        if UB:
            Q = UB
        else:
            Q = 2429

    # Q = len(word_pool)-num_strings
    #
    # print(f"\nCreating {len(word_pool) * len(routes)} u variables...")
    # u = m.addVars(word_pool,
    #               vtype=var_type,
    #               lb=0,
    #               ub=Q,
    #               name="U")

    # for k in routes:
    #     u[str(k)].lb = 0
    #     u[str(k)].ub = 0

    redundant_arcs = 0
    costly_arcs = 0
    for wi in word_pool:
        if len(wi) == 1:
            continue
        for wj in word_pool:
            if len(wj) == 1:
                continue
            if not has_super_words:
                dist = cost(wi, wj)
            else:
                dist = cost(wi[-7:], wj[:7])

            if wi == wj or dist == 0 or ("*" in wi and "*" in wj and not wild_friends(wi, wj)):
                for k in routes:
                    x[wi, wj, k].lb = 0
                    x[wi, wj, k].ub = 0
                continue
            if dist > arc_limit and wj[:2] != '12':

                for k in routes:
                    x[wi, wj, k].lb = 0
                    x[wi, wj, k].ub = 0
                    costly_arcs += 1
                continue

            if dist < 2:
                continue

            # search inter permutations and fix var to zero if found
            # wk = wi + wj[-dist:]
            # inter_perms = get_unique_words_in_string(wk[1:-1])
            # if inter_perms and all(perm in word_pool for perm in inter_perms):
            #     # print(wi, wj, wk, wk[1:-1], inter_perms, dist, "redundant")
            #     for k in routes:
            #         x[wi, wj, k].lb = 0
            #         x[wi, wj, k].ub = 0
            #         redundant_arcs += 1


    print(f"{redundant_arcs} redundant arcs removed")
    print(f"{costly_arcs} costly_arcs arcs removed")

    for wi in word_pool:
        for wj in word_pool:
            for k in routes:
                if wi == wj:
                    x[wi, wj, k].lb = 0
                    x[wi, wj, k].ub = 0
                if len(wj) == 1:
                    x[wi, wj, k].lb = 0
                    x[wi, wj, k].ub = 0
                if len(wi) == 2:
                    x[wi, wj, k].lb = 0
                    x[wi, wj, k].ub = 0
                if len(wi) == 1 and wi != str(k):
                    x[wi, wj, k].lb = 0
                    x[wi, wj, k].ub = 0
                if len(wj) == 2 and wj != str(k)+str(k):
                    x[wi, wj, k].lb = 0
                    x[wi, wj, k].ub = 0
                if len(wi) <= 2 and len(wj) <= 2:
                    x[wi, wj, k].lb = 0
                    x[wi, wj, k].ub = 0


    print("Setting obj\n")
    if min_max_obj and dfj_model:
        obj = m.addVar(name="obj", vtype=GRB.INTEGER)
        if max_length:
            m.setObjective((gp.quicksum(x.sum("*", wi, "*") for wi in word_pool)), GRB.MAXIMIZE)
        else:
            m.setObjective(obj, GRB.MINIMIZE)

        m.addConstrs(((obj >= gp.quicksum(x[wi, wj, k] * cost(wi, wj)
                                for wi in word_pool
                                for wj in word_pool
                                if cost(wi, wj) != 0))
                     for k in routes), "Obj")
    elif mtz_model:
        obj = m.addVar(name="obj", vtype=GRB.INTEGER)
        m.setObjective(obj, GRB.MINIMIZE)

        m.addConstrs(((obj >= u[str(k)+str(k), k]) for k in routes), "Obj")
    else:
        m.setObjective((gp.quicksum(x[wi, wj, k] * cost(wi, wj)
                                    for wi in word_pool
                                    for wj in word_pool
                                    if cost(wi, wj) != 0
                                    for k in routes)),
                       GRB.MINIMIZE)

    print(f"Creating FreeWords constraints...")
    if freeword_count or max_length:
        # Each free word must appear once
        m.addConstrs((gp.quicksum(x.sum("*", wj, "*")
                      for wj in word_pool if wi in wj) <= 1
                      for wi in free_words), "FreeWord")

        if freeword_count:
            m.addConstr((gp.quicksum(x.sum("*", wi, "*") for wi in free_words) == freeword_count), "FreeWordCount")

        if max_length:
            m.addConstr(obj <= max_length, "max_string_length")
    elif dfj_model:
        # Each free word must appear once
        m.addConstrs((gp.quicksum(x.sum("*", wj, "*")
                      for wj in word_pool if word_is_contained(wi, wj)) == 1
                      for wi in free_words), "FreeWord")
    elif mtz_model:
        # Each free word must appear once
        m.addConstrs((gp.quicksum(x.sum("*", wj, "*")
                      for wj in word_pool if word_is_contained(wi, wj)) == 1
                      for wi in free_words), "FreeWordIn")
        # m.addConstrs((gp.quicksum(x.sum(wj, "*", "*")
        #               for wj in word_pool if word_is_contained(wi, wj)) == 1
        #               for wi in free_words), "FreeWordOut")

    print(f"Creating SpecialWords constraints...")
    if dfj_model:
        m.addConstrs((gp.quicksum(x.sum("*", wj, k)
                      for wj in word_pool if word_is_contained(wi, wj)) == 1
                      for wi in special_words[k] for k in routes), "SpecialWord")
    elif mtz_model:
        m.addConstrs((gp.quicksum(x.sum("*", wj, k)
                      for wj in word_pool if word_is_contained(wi, wj)) == 1
                      for wi in special_words[k] for k in routes), "SpecialWordIn")
        # m.addConstrs((gp.quicksum(x.sum(wj, "*", k)
        #               for wj in word_pool if word_is_contained(wi, wj)) == 1
        #               for wi in special_words[k] for k in routes), "SpecialWordOut")

    if dfj_model:
        m.addConstrs((x.sum(str(k), "*", k) == 1 for k in routes), "Start")
        m.addConstrs((x.sum("*", str(k), k) == 1 for k in routes), "End")
    elif mtz_model:
        m.addConstrs((x.sum(str(k), "*", k) == 1 for k in routes), "Start")
        m.addConstrs((x.sum("*", str(k)+str(k), k) == 1 for k in routes), "End")

    if dfj_model:
        print(f"Creating FlowConservation constraints...")
        m.addConstrs((x.sum("*", wj, k) == x.sum(wj, "*", k)
                      for wj in word_pool
                      for k in routes), "FlowConservation")
    elif mtz_model:
        m.addConstrs((x.sum("*", wj, k) == x.sum(wj, "*", k)
                      for wj in word_pool if len(wj) > 2
                      for k in routes), "FlowConservation")
        m.addConstrs((u[wj, k] >= u[wi, k] + x[wi, wj, k] * cost(wi, wj) - (1 - x[wi, wj, k]) * Q
                      for wi in word_pool
                      for wj in word_pool if len(wj) > 2 and wi != wj
                      for k in routes), "Distance")

        m.addConstrs((u[str(k)+str(k), k] >= u[wi, k] - (1 - x[wi, str(k)+str(k), k]) * Q
                      for wi in word_pool if len(wi) > 2
                      for k in routes), "Distance")

    if len(wildcards) > 2:
        print(f"Creating WildCardWords constraints...")
        m.addConstrs((
            gp.quicksum(
                gp.quicksum(
                    x[wi, wj, k]
                    for wi in word_pool if wi not in wildcards
                )
                for wj in wildcards) == 2
            for k in routes),
            "WildCardWord"
        )


    # m.addConstrs((u[wj] - u[wi] >= 1 - Q * (1 - x[wi, wj, k])
    #              for wi in word_pool
    #              for wj in word_pool if len(wj) > 1 and wi != wj
    #              for k in routes), "u")

        # 2 wildcards cannot be appart by 1 word
        # m.addConstrs(gp.quicksum(x[wj, wi, k] for wj in wildcards) + gp.quicksum(x[wi, wj, k] for wj in wildcards) <= 1
        #              for wi in word_pool
        #              for k in routes)

    # m.computeIIS()
    if save_model:
        print("Saving model...")
        m.write("santa_atsp.lp")
        # m.write("santa_atsp.ilp")

    seen_tours = set()
    m._seen_tours = seen_tours


    def get_tour(tour):
        string_k = tour[0] if len(tour[0]) > 1 else ""
        for j in range(1, len(tour)):
            word_distance = cost(tour[j-1], tour[j])
            if "*" in tour[j]:
                # the word appended should be preserved
                string_k = string_k[:-len(tour[j])+word_distance] + tour[j]
            else:
                string_k += tour[j][-word_distance:]
        return string_k

    def subtour(edges, tour_words, start=None):
        is_subtour = False
        if start:
            tour_words = [start, start] + tour_words
        unvisited = [w for w in tour_words]
        cycle = [w for w in tour_words]  # Dummy - guaranteed to be replaced
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*')
                             if j in unvisited]
            if len(thiscycle) <= len(cycle):
                is_subtour = True
                cycle = thiscycle # New shortest subtour
                return cycle, is_subtour
            else:
                cycle = thiscycle
        return cycle, is_subtour

    def mycallback(model, where):
        if where == GRB.Callback.MIPSOL:
            if mtz_model:
                return
            print(f'\nCurrent Best Solution ({model.cbGet(GRB.Callback.MIPSOL_OBJ)}):')
            x_sol = model.cbGetSolution(model._x)

            # for key, val in x_sol.items():
            #     if val > 1e-8:
            #         print(*key, val)

            iteration_seen_tours = set()

            for k in routes:
                selected = gp.tuplelist((wi, wj) for wi in word_pool for wj in word_pool
                                        if x_sol[wi, wj, k] > 0.5)
                # print(selected)
                tour_words = gp.tuplelist(wi for wi in word_pool for wj in word_pool
                                          if x_sol[wi, wj, k] > 0.5 if len(wi) > 1)
                tour, is_subtour = subtour(selected, tour_words, str(k))

                print(f"String {k+1} Arc numbers: {len(tour_words)}")

                tour_bases = set()
                for w in tour_words:
                    tour_bases.add(w[:2])
                tour_bases_ct = {
                    base: 0
                    for base in tour_bases
                }
                for w in tour_words:
                    tour_bases_ct[w[:2]] += 1
                print(tour_bases_ct)

                if tour[0] == str(k):
                    tour.pop(0)
                if tour[-1] == str(k):
                    tour.pop(-1)
                if len(tour) < len(tour_words):
                    while tour:
                        # add subtour elimination constr. for every pair of cities in subtour
                        if tuple(sorted(tour)) not in iteration_seen_tours:
                            for kk in routes:
                                model.cbLazy(gp.quicksum(model._x[wi, wj, kk]+model._x[wj, wi, kk] for wi, wj in combinations(tour, 2))
                                             <= len(tour)-1)

                                model.cbLazy(gp.quicksum(model._x[wi, wj, kk]+model._x[wj, wi, kk] for wi, wj in combinations(tour+[str(kk)], 2))
                                             <= len(tour))

                        # print(sum(x_sol[wi, wj, k] for wi, wj in combinations(tour, 2)))
                        # print(gp.quicksum(model._x[wi, wj, k] for wi, wj in combinations(tour+[str(k)], 2))
                        #              <= len(tour))
                        tour_str = get_tour(tour)

                        if tuple(sorted(tour)) in model._seen_tours:
                            print("***************** ERROR")
                            print(tour_str, " seen before")
                            print(tour, " seen before")

                        print(f"subtour{k+1}: {tour_str} ({len(tour_str)}) {tour}")
                        # print(f"tour_words : {tour_words}")
                        # print(f"tour : {tour}")
                        # print(list(combinations(tour, 2)))

                        iteration_seen_tours.add(tuple(sorted(tour)))
                        # print()

                        tour_words = [word for word in tour_words if word not in tour]
                        tour, is_subtour = subtour(selected, tour_words)
                else:
                    tour_str = get_tour(tour)
                    print(f"tour{k+1}: {tour_str} ({len(tour_str)}) {tour}")

            model._seen_tours =model._seen_tours.union(iteration_seen_tours)

        elif where == GRB.Callback.MIPNODE:
            return
            # MIP node callback
            print('**** New node ****')
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                for key, val in model.cbGetNodeRel(model._x).items():
                    if val > 1e-8:
                        print(*key, val)
                print()

    m._x = x



    # Model Parameters
    m.Params.lazyConstraints = 1


    # m.setParam(GRB.Param.Heuristics, 0.001)
    # m.setParam(GRB.Param.Aggregate, 0)

    # m.setParam(GRB.Param.NoRelHeurTime, 36_000)
    # m.setParam(GRB.Param.Symmetry, 2)
    # m.setParam(GRB.Param.PreDepRow, 1)
    # m.setParam(GRB.Param., 1)
    m.setParam(GRB.Param.MIPFocus, 1)
    if UB and not max_length:
        m.setParam(GRB.Param.Cutoff, UB)

    # m.tune()
    # if m.tuneResultCount > 0:
    #     # Load the best tuned parameters into the model
    #     m.getTuneResult(0)
    #     # Write tuned parameters to a file
    #     m.write('tune.prm')


    print("Optimizing...")
    m.optimize(mycallback)

    print(m.Status)
    if m.Status == GRB.INFEASIBLE or m.Status == GRB.CUTOFF:
        return initial_solution

    print('\nTOTAL COSTS: %g' % m.objVal)

    tours = []
    for k in routes:
        selected = gp.tuplelist((wi, wj) for wi in word_pool for wj in word_pool
                                if x[wi, wj, k].x > 0.5)
        tour_words = gp.tuplelist(wi for wi in word_pool for wj in word_pool
                                  if x[wi, wj, k].x > 0.5)
        for wi in word_pool:
            for wj in word_pool:
                if x[wi, wj, k].x > 0.5:
                    print(f"x{wi},{wj},{k}={x[wi, wj, k].x}")

        for wi in word_pool:
            if u[wi, k].x > 0.5:
                print(f"u{wi},{k}={u[wi, k].x}")
        print()
        tour, _ = subtour(selected, tour_words, str(k))
        # tour.pop(-1)
        tours.append(get_tour(tour))

    print(tours)
    return tours