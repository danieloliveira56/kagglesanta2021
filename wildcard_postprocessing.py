import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

"""
Permutations Rebalancing (https://www.kaggle.com/kostyaatarik/permutations-rebalancing/notebook)

This leads a relaxation of constraints and sometimes you can find a better solution of wildcard positions.
"""
perms = list(map(lambda p: "".join(p), itertools.permutations("1234567")))

def find_strings_perms(strings, verbose=False):
    global perms
    found_perms = []
    for s in strings:
        found_perms.append([])
        for i in range(len(s)-6):
            p = s[i:i+7]
            if p in perms:
                found_perms[-1].append(p)
    if verbose:
        lens = [len(_) for _ in  found_perms]
        print(f'There are {lens} permutations in strings, {sum(lens)} in total.')
        lens = [len(set(_)) for _ in  found_perms]
        print(f'There are {lens} unique permutations in strings, {sum(lens)} in total.')
    return found_perms

def rebalance_perms(strings_perms, verbose=False):
    # convert to dicts for fast lookup and to keep permutations order
    strings_perms = [dict.fromkeys(_) for _ in strings_perms]
    for p in strings_perms[0].copy():  # iterate over the copy to allow modification during iteration
        if p[:2] != "12" and (p in strings_perms[1] or p in strings_perms[2]):
            strings_perms[0].pop(p)
    for p in strings_perms[1].copy():
        if p[:2] != "12" and p in strings_perms[2]:
            strings_perms[1].pop(p)
    if verbose:
        lens = [len(_) for _ in  strings_perms]
        print(f'There are {lens} permutations left in strings after rebalancing, {sum(lens)} in total.')
    return [list(_) for _ in strings_perms]

def improve_submission_with_wildcards():

    perm2id = {p: i for i, p in enumerate(perms)}

    perms_arr = np.array([list(map(int, p)) for p in perms])
    perms_arr.shape

    perms_onehot = np.eye(7)[perms_arr-1, :].transpose(0, 2, 1)
    assert np.allclose(perms_onehot[:,0,:].astype(np.int64), (perms_arr == 1).astype(np.int64))

    print("onehot 1234567:")
    print(perms_onehot[perm2id["1234567"]])

    print("onehot 5671234:")
    print(perms_onehot[perm2id["5671234"]])

    print("correlate between 1234567 and 5671234")
    left = perms_onehot[perm2id["1234567"]]
    right = perms_onehot[perm2id["5671234"]]
    matches = F.conv2d(
        F.pad(torch.Tensor(left[None, None, :, :]), (7, 7)),
        torch.Tensor(right[None, None, :, :]),
        padding="valid"
    ).numpy().reshape(-1)
    print(matches)
    must_match_left2right = np.array([-1, -1, -1, -1, -1, -1, -1, 7, 6, 5, 4, 3, 2, 1, 0])
    must_match_right2left = np.array([0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1])
    cost_ifmatch = np.array([7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7])
    print("cost of 1234567 -> 5671234:", min(cost_ifmatch[np.equal(must_match_left2right, matches)]))
    print("cost of 5671234 -> 1234567:", min(cost_ifmatch[np.equal(must_match_right2left, matches)]))

    M = F.conv2d(
        F.pad(torch.Tensor(perms_onehot[:, None, :, :]), (7, 7)),
        torch.Tensor(perms_onehot[:, None, :, :]),
        padding="valid"
    ).squeeze().numpy()

    must_match_left2right = np.array([-1, -1, -1, -1, -1, -1, -1, 7, 6, 5, 4, 3, 2, 1, 0])
    must_match_left2right_wild = np.array([-1, -1, -1, -1, -1, -1, -1, 6, 5, 4, 3, 2, 1, 0, 0])

    cost_ifmatch = np.array([7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7])

    costMat = np.where(M == must_match_left2right, cost_ifmatch, np.inf).min(axis=-1).astype(np.int8)
    costMatWild = np.minimum(costMat, np.where(M == must_match_left2right_wild, cost_ifmatch, np.inf).min(axis=-1)).astype(np.int8)

    symbols = "🎅🤶🦌🧝🎄🎁🎀"
    schedule = pd.read_csv("submission.csv").schedule.tolist()
    words = [s.translate(str.maketrans(symbols, "1234567")) for s in schedule]

    found_perms = find_strings_perms(words, verbose=True)
    balanced_perms = rebalance_perms(found_perms, verbose=True)

    nodes_list = []
    table_list = []
    for i in range(3):
        word = words[i]
        nodes = [perm2id[p] for p in balanced_perms[i]]

        table = np.zeros((len(nodes), 10), np.int64)
        table[0, :] = 7
        for i in range(1, len(nodes)):
            e = costMat[nodes[i - 1], nodes[i]]
            ew = costMatWild[nodes[i - 1], nodes[i]]
            table[i, 0] = table[i - 1, 0] + e
            table[i, 1] = min(table[i - 1, 1] + e, table[i - 1, 0] + ew)
            table[i, 2] = min(table[i - 1, 2], table[i - 1, 1]) + e  # TODO: better transition
            table[i, 3] = min(table[i - 1, 3], table[i - 1, 2]) + e
            table[i, 4] = min(table[i - 1, 4], table[i - 1, 3]) + e
            table[i, 5] = min(table[i - 1, 5], table[i - 1, 4]) + e
            table[i, 6] = min(table[i - 1, 6], table[i - 1, 5]) + e
            table[i, 7] = min(table[i - 1, 7], table[i - 1, 6]) + e
            table[i, 8] = min(table[i - 1, 8], table[i - 1, 7]) + e
            table[i, 9] = min(table[i - 1, 9] + e, table[i - 1, 8] + ew)
        print(table[-1].min(), table[-1])
        nodes_list.append(nodes)
        table_list.append(table)

    # backtrack
    new_words = []
    wilds = []
    for nodes, table in zip(nodes_list, table_list):
        ns = [perms[nodes[-1]]]
        track = np.argmin(table[-1])
        wild = []
        for i in range(len(nodes) - 2, -1, -1):
            e = costMat[nodes[i], nodes[i + 1]]
            ew = costMatWild[nodes[i], nodes[i + 1]]
            if track == 0:
                ns.append(perms[nodes[i]][:e])
            elif track == 1:
                if table[i, 1] + e < table[i, 0] + ew:
                    ns.append(perms[nodes[i]][:e])
                else:
                    left = np.array(list(map(int, perms[nodes[i]][ew:])))
                    right = np.array(list(map(int, perms[nodes[i + 1]][:-ew])))
                    mis = np.where(left != right)[0][0]
                    wild.append(table[i, track - 1] - 7 + ew + mis)
                    ns.append(perms[nodes[i]][:ew])
                    track = track - 1
            elif 2 <= track <= 8:
                if table[i, track] >= table[i, track - 1]:
                    track = track - 1
                ns.append(perms[nodes[i]][:e])
            elif track == 9:
                if table[i, 9] + e < table[i, 8] + ew:
                    ns.append(perms[nodes[i]][:e])
                else:
                    ns.append(perms[nodes[i]][:ew])
                    left = np.array(list(map(int, perms[nodes[i]][ew:])))
                    right = np.array(list(map(int, perms[nodes[i + 1]][:-ew])))
                    mis = np.where(left != right)[0][0]
                    wild.append(table[i, track - 1] - 7 + ew + mis)
                    track = track - 1
            else:
                assert False
        assert track == 0
        wilds.append(wild)
        nsw = list("".join(ns[::-1]))
        for w in wild:
            nsw[w] = "*"
        new_words.append("".join(nsw))

        print("score: ", max(map(len, words)), "->", max(map(len, new_words)))


    submission = pd.Series([a.translate(str.maketrans("1234567*", symbols+"🌟")) for a in new_words], name='schedule')
    submission.to_csv('submission_wild.csv', index=False)