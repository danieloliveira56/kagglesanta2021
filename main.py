import pandas as pd
import time
import subprocess
import numpy as np
import math
from helpers import REPLACE_DICT, SPECIAL_CHARS, get_words_in_string, offset, is_perm, get_tsp_solution, get_words, map_solution, check_word_count
from optimization import get_optimal_solution
from clustering import get_word_clusters

##CONFIGURATIOMNS
relax = False
single_string = True
num_letters = 7
num_strings = 120
num_wildcards = 2
partition_idx = 1


# (num of words, per num of positions)
DENSITY = {
    4: (2, 4),
    5: (3, 4),
    6: (4, 6),
    7: (5, 7),
}
REAL_NUM_PARTITIONS = 3


def partition_and_solve(num_letters, num_partitions=3, partition_idx=0, num_wildcards=2, disperse_specials=False,
                        concentrate_specials=True):
    '''

    :param num_letters:
    :param num_partitions:
    :param partition_idx:
    :param num_wildcards:
    :param disperse_specials:
    :param concentrate_specials:

    If both are disperse_specials and concentrate_specials are False, we exclude the specials from the word list.
    :return:
    '''

    words = get_words(num_letters)

    (special1, special2) = SPECIAL_CHARS[7]
    free_words = [w for w in words if int(w[0]) != special1 or int(w[1]) != special2]
    special_words = [w for w in words if int(w[0]) == special1 and int(w[1]) == special2]

    print(f"Originally have {len(words)} words: {len(free_words)} free_words and {len(special_words)} special_words")

    word_partitions = []
    start = end = 0
    remaining_words = len(free_words)
    while end < len(free_words):
        n = math.ceil(remaining_words / (num_partitions - len(word_partitions)))
        end += n
        remaining_words -= n
        word_partitions.append(free_words[start:end])
        start = end

    if disperse_specials:
        num_special_words_partitions = num_partitions // REAL_NUM_PARTITIONS

        special_word_partitions = []
        start = end = 0
        remaining_words = len(special_words)
        while end < len(special_words):
            n = remaining_words // (num_special_words_partitions - len(special_word_partitions))
            end += n
            remaining_words -= n
            special_word_partitions.append(special_words[start:end])
            start = end
        print(f"Partition sizes: {[len(p) for p in word_partitions]}")
        print(sum([len(p) for p in word_partitions]))
        print(special_word_partitions)
        for i in range(REAL_NUM_PARTITIONS):
            print(len(word_partitions[num_special_words_partitions * i:num_special_words_partitions * (i+1)]))
            print(num_special_words_partitions * i, num_special_words_partitions * (i+1))
            for j, p in enumerate(word_partitions[num_special_words_partitions * i:num_special_words_partitions * (i+1)]):
                for w in special_word_partitions[j]:
                    p.append(w)
    elif concentrate_specials:
        for w in special_words:
            word_partitions[partition_idx].append(w)

    print(f"Partition sizes: {[len(p) for p in word_partitions]}")
    print(sum([len(p) for p in word_partitions]))

    print(f"Optimizing partition {partition_idx} with {len(word_partitions[partition_idx])} words")
    words = word_partitions[partition_idx]

    opt_solution = get_optimal_solution(words,
                                        num_strings=1,
                                        num_letters=7,
                                        word_density=DENSITY[num_letters],
                                        num_wildcards=num_wildcards,
                                        )

    return opt_solution


def split_and_search(num_letters=7):
    words = get_words(num_letters)

    (special1, special2) = SPECIAL_CHARS[7]
    free_words = [w for w in words if int(w[0]) != special1 or int(w[1]) != special2]
    special_words = [w for w in words if int(w[0]) == special1 and int(w[1]) == special2]

    adjust1 = 18
    adjust2 = 12

    group1 = free_words[:1640-adjust1] + special_words
    group2 = free_words[1640-adjust1:3280+adjust2] + special_words
    group3 = free_words[3280+adjust2:] + special_words

    print("Finding initial TSP solutions...")
    string1 = get_tsp_solution(group1)
    string2 = get_tsp_solution(group2)
    string3 = get_tsp_solution(group3)

    print("Improving string 1...")
    better_string1 = local_search(num_letters=num_letters, search_length=30, initial_solution=string1)
    print("Improving string 2...")
    better_string2 = local_search(num_letters=num_letters, search_length=30, initial_solution=string2)
    print("Improving string 3...")
    better_string3 = local_search(num_letters=num_letters, search_length=30, initial_solution=string3)

    for k, v in REPLACE_DICT.items():
        better_string1 = better_string1.replace(k, v)
        better_string2 = better_string2.replace(k, v)
        better_string3 = better_string3.replace(k, v)

    sub = pd.DataFrame()
    sub['schedule'] = [string1, string2, string3]
    sub.to_csv('submission.csv', index=False)
    sub.head()


def local_search(num_letters=7, search_length=30, initial_solution=None, other_strings=None):

    other_strings_words = set()
    for s in other_strings:
        for w in get_words_in_string(num_letters, s):
            other_strings_words.add(w)

    if not initial_solution:
        print("Getting initial solution...")
        words = get_words(num_letters)
        initial_solution = get_tsp_solution(words)
        initial_solution = list(initial_solution)
        initial_solution[num_letters] = "*"
        initial_solution[2*num_letters] = "*"
        initial_solution = "".join(initial_solution)

    words = get_words_in_string(num_letters, initial_solution)
    special_words = [w for w in words if int(w[0]) == 5 and int(w[1]) == 4]

    essential_words = [w for w in words if w not in other_strings_words] + special_words
    print(f"Words in string: {len(words)}")
    current_solution = initial_solution

    print("Mapping words to positions and vice-versa...")
    # Positions where each w can be found
    word_map, position_map = map_solution(words, current_solution)

    i = 0
    while i < len(current_solution)-num_letters and i+search_length < len(current_solution):
        print(f"\nChecking position {i} to {i+search_length}: {current_solution[i:i+search_length]}")
        print("Word List:")
        search_words = set()
        other_words = set()

        for j in range(len(current_solution) - num_letters):
            if j in range(i, i+search_length-num_letters+1):
                for w in position_map[j]:
                    search_words.add(w)
                print(j, position_map[j])

            else:
                for w in position_map[j]:
                    other_words.add(w)

        num_wildcards = 0
        for j in range(i, i + search_length):
            if current_solution[j] == '*':
                num_wildcards += 1

        # Words that must continue existing in the substring
        unique_search_words = [w for w in search_words
                               if (w in special_words and w not in other_words)
                               or (w not in special_words and w not in other_words and w not in other_strings_words)]
        print(f"Substring has {len(search_words)} words, of which {len(search_words)-len(unique_search_words)} can be found somewhere else.", )

        if len(unique_search_words) < len(search_words):
            print("Optimizing substring...")
            optimal_order = get_optimal_solution(
                list(search_words),
                initial_solution=current_solution[i:i+search_length],
                num_strings=1,
                num_wildcards=num_wildcards,
                save_model=False,
                LB=max(num_letters, num_letters-1+len(search_words)-num_letters*num_wildcards),
                fix_suffix=num_letters,
            )
            if len(optimal_order) < search_length:
                print(f"Shortening the initial solution by {search_length - len(optimal_order)} letters...")

                print(current_solution)
                new_solution = current_solution[:i] + optimal_order + current_solution[i+search_length:]
                print(new_solution)
                new_words = get_words_in_string(num_letters, new_solution)

                if any(w not in new_words for w in essential_words):
                    print("\n\nERROR: Words disappeared ###############################\n\n")
                else:
                    current_solution = new_solution
                    print("Remapping words...")
                    word_map, position_map = map_solution(words, current_solution)
                    # Reset loop
                    i = -1
        i += 1

# get_optimal_solution(get_words(4), num_strings=2, LB=10, letter_spacing=2, tight_model=True, linear_relaxation=True)

# get_optimal_solution(get_words(7), UB=2430, LB=2430)

# split_and_search(7)

# local_search(7)

# words = get_words(7)

# words = [w for w in words if int(w[0]) != 5 and int(w[1]) != 4]

# get_word_clusters(words[:1000])

# # partition_and_solve(num_letters=7,
# #                     num_partitions=200,
# #                     partition_idx=0,
#                     num_wildcards=2,
#                     disperse_specials=True)