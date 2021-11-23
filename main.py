import pandas as pd
import time
import subprocess
import numpy as np
import math
from helpers import REPLACE_DICT, SPECIAL_CHARS, get_words_in_string, offset, is_perm, get_tsp_solution, get_words, map_solution, check_word_count
from optimization import get_optimal_solution
from optimization_120 import maximize_specials
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


def local_search(num_letters=7, search_length=30, initial_solution=None, other_strings=None, assume_shortening=0):

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
    while i < len(current_solution)-num_letters and i+search_length <= len(current_solution):
        print(f"\nChecking position {i} to {i+search_length-1}: {current_solution[i:i+search_length]}")
        print("Word List:")
        search_words = set()
        other_words = set()

        for j in range(i-num_letters, i):
            if j >= 0:
                print(j, position_map[j], *[word_map[position_map[j]], position_map[j] in essential_words] if position_map[j] else "")
        print("----")
        for j in range(len(current_solution) - num_letters):
            if j in range(i, i+search_length-num_letters+1):
                if position_map[j]:
                    search_words.add(position_map[j])
                print(j, position_map[j], *[word_map[position_map[j]], position_map[j] in essential_words] if position_map[j] else "")
            else:
                if position_map[j]:
                    other_words.add(position_map[j])
        print("----")
        for j in range(i+search_length-num_letters+1, i+search_length):
            if j < len(current_solution):
                print(j, position_map[j], *[word_map[position_map[j]], position_map[j] in essential_words] if position_map[j] else "")

        num_wildcards = 0
        for j in range(i, i + search_length):
            if current_solution[j] == '*':
                num_wildcards += 1

        # Words that must continue existing in the substring
        unique_search_words = [w for w in search_words
                               if (w in special_words and w not in other_words)
                               or (w not in special_words and w not in other_words and w not in other_strings_words)]
        print(f"Substring has {len(search_words)} words, of which {len(search_words)-len(unique_search_words)} can be found somewhere else.", )

        if i == 0:
            fix_suffix = 0
        else:
            fix_suffix = num_letters - 1
        j = i - 1
        while j > 0 and not position_map[j] and fix_suffix > 0:
            fix_suffix -= 1
            j -= 1

        loose_following = 0
        j = i + search_length - num_letters + 1
        while (not position_map.get(j) or len(word_map[position_map[j]]) > 1 or position_map[j] not in essential_words) and loose_following < num_letters:
            loose_following += 1
            j += 1

        print(f"Loose_following: {loose_following}")
        print(f"Fix_suffix: {fix_suffix}")

        if unique_search_words and (loose_following >= num_letters - 2): #  or fix_suffix < num_letters / 2
            print("Optimizing substring...")
            optimal_order = get_optimal_solution(
                list(unique_search_words),
                initial_solution=current_solution[i:i+search_length],
                num_strings=1,
                num_wildcards=num_wildcards,
                save_model=True,
                UB=search_length,
                LB=max(num_letters, num_letters+len(unique_search_words)-1-num_letters*num_wildcards),
                fix_suffix=fix_suffix,
                assume_shortening=assume_shortening,
            )
            if len(optimal_order) < search_length:
                print(f"Shortening the initial solution by {search_length - len(optimal_order)} letters...")

                print(current_solution)
                new_solution = current_solution[:i] + optimal_order + current_solution[i+search_length:]
                print(new_solution)
                new_words = get_words_in_string(num_letters, new_solution)

                if any(w not in new_words for w in essential_words):
                    print("\n\nERROR: Words disappeared ###############################")
                    for w in essential_words:
                        if w not in new_words:
                            print(w)
                            print("essential_words", essential_words)
                    print()
                elif len(new_solution) < len(current_solution):
                    print("Found Improved Solution ####################################")
                    current_solution = new_solution
                    print("Remapping words...")
                    word_map, position_map = map_solution(words, current_solution)
                    # Reset loop
                    i = -1
        i += 1
    return current_solution

# get_optimal_solution(get_words(5), num_strings=2, UB=100)
get_optimal_solution(get_words(7), num_strings=3, LB=2100, UB=2440)

# string1 = maximize_specials(get_words(7), num_strings=1)
#
# print(string1)
#
# string1 = "1234567123465712346751234671253467123546321754362175432617543216754321765471362546213754736215437621543167245316742531674523167453216745312674531627453167243651274365124736512437651243675124365712436517243654612375421637542671354126735421673524163752416357241635274163524716352417635241673521467351246375124635712463517246351274635124763512467351426735146273514672351467325146735216473521674352167345216374521634752163457216345271634521763452167354261375417326145736214573612457361425736145273614572361457623145762134576214357621453762145463172543617254317265461327546327154362715412376451237645471326457132645173264513726453172645371264537216453726145372641537264513276453127645321764532716453276145327641532764513267451326475132*462731541672345167234547213615742361574326157436215743612574316257431265741326574123657412635741265374126573412657431256741325674123567412536741256374125673412567431257643125746312574361527436152473615243761524367152436175243615724361547236154732617453621745361274536172453617425361745236174532617435261743256174326517423651742635174265317426513742651734261573426175342167534217653421756342175364217534621753426173452617342561734265174326751432675413726547261354271635427631542736154273167524316752341675231467532146753124657312465371246531724653127465312476531246753142675314627531467253146752316475321647531264753162475316427531647253164752316541632754127635421763542713654217365241736521473652174365217346521736452173654213675413276541736254673125476312547361254673215476321457632147563214765321476352147632514763215476231543726154372163547263154731625471263547216357421635*7612345761234756132475613427561347256134752613475621347561234754176234517623475162347546271354732165461723456172346517234654612734561273454126375421376452137645417263451276345126734512637451263475126345712634517263412576341254162734516273465127346513274651324765132467153246713524671354621735426317543126754317625471623457162345416732546713254672315463127543612754316271543267154327615432716542637154236715423761542371654613724561374256137452613745621374561237456132745613724651372465467213541623745162374541376254172364517236454136275413267543127654716325462317542361754231675423176452317643521764354276135417632547231654617325476132574613257641325761432576134257613245761325476213542617354267315416372546371254637215436721543716254123674512367454712364571236475123647547312654623712543671254376125347612537461253764125376142537612453761254371265413672541273645127364125736412357641235467123456"
# string2 = "6145732614235761423541672354162735462173546172354617325413267541236754362715463271456327146532714635271463257416325*4271634257163425176342516734251637425163472516342751634*4326175463217546371265473126547231654237165423176542361754263175426173546271354672135467312546713245671324657132467513246157324617532461735246173541623751462375142637514236751423765142736514276351427653142765134276514327651423756142375421637542137654327165432671546137254316725431267541263715426137541276354172635417362547316254732615472631547261354136275427136543217654321675421763543621734562173454367213456721346572136457213654721365742136572413657214365413726543172654361275463721542173654712365471326547136254371625347162537416253714625371642537162453716254376215421367541237654231675427316543276154276315427613541327654312765436172356417235614732561473526147356214735612473561427356147235617423561724356172546317254612735416372541367251436725134672541376254317625341726534712653472156347215364721534672153476215347216534726153472651437265147326514723651472635147265314726513472653417256341725364172534617253416725341762531476253174625317642531762453176254367126354712635431627546312754612371546237154126735462137542736154176325471632547236147523614753261475362147536124753614275361472536147623514762354176235471623547216354236715426371542167315462731546723154671235412736541723654173265437126543721654372615473621547361254732165243716524317652431567243156274351624735162437516243571642351764235167423516472351642735164237516423571643251764325167432516473251643725614372564137256431725643712564731256471325647123564712536471256347125643721564372516432756143275641327564312756432175643271564327516432571643527164357216435712643517264351276435126743512647351264375126435716243517624351672435162743561274356217435627143567214356712435671423567143256714352671435627413562471356241735624137562413576241356724135627431562473156243715624317562341576234156723415627341562374156234715623417562314756231745623175462317564231576421356742135647213564271356421735642137564213576421537462153742615374216537421563742153674215376421573642157634215764321576423156742315647231564273156423715642317562431576243154762315476321745632174653217463521746315247631524673152463715246317524631572463152746315742631574623517462357146235741623574612357462135746231574632157463251746325714632751463271546732154267315426713547621354761235476132541673254163274516324751632457163245176324516732451637245163274543761243576124375612437547631254613275423761****"
# string3 = "7214365721346752134672513647251367425136724513672547216354761325436172542637145263714256371425423176543216754213675423716452371645432617543271654273615476213547263145726314752631472563147254371625471326547231657423165724316572341657231465723164572316547236154732615436271453627145467231456723145473621546731245673124543267145326714543762154637214563721465372146357214637521436752143765214375621437526143752164375214637251463721546371245637124543127654632713456271345471263754126375432761543726154217365412736541327654132674153267413526471352641735264137524613752413675241376524137526413572614357261345726135742613572461357264157326415723641527364152376415236741523647152364175236415726341527634152673415263741526347152634175263415726431527643152674315264731526437152643175264315726413527641352674136257413624571362451736245*631274563127454763215467132547136254716327541632754613275463217547631245763124756312476153247615471236754123675421763541276354136725416372541673254176325417263547312645731264543217654213765423671452367142536714275631427564123756412735641275346127534162753412675341276534127563412753641275431672546137254617324561732465173246153724615327461532647153264175326417325641732546172354276135247613527461352761435276134527613542763145276314547612376541237654216735412673541627354167235436712453671245431726541372654173265417236542713652471365274136527143652713465271364527136542376145237614546712354216375426137542631745263174256317425476231546732156473215674321567342156732415637241563274156324715632417563241576324156732145673215462173546271354623175437216547321657432165734216573241653724165327416532471653241765324167532416573214657321645732165473162457316241537624153672415362741536247153624175362415736241543621754236175436127354612735463172456317245437126547162375416237546123754621375462371456237146523741652374615234765123476521347652314765234176523471652347615234671523461752346157234615273461523746512374652137465231746523714654672135462731456273145267314521367452136475213642571364251736425*312675431627543176254137625143627513462751364275136247513627451362754136275143625714362517436251473625143762513476251374625137642513762451376254173625417623547261354721365427316524731652743165273416527314652731645273165427163542617354267134526713425671342657143265714236571426357142653714265731425673142576314257316425731462573416257346125734621573462517346257134625731426571342675134267153427615342716534271563427153642715346271534267135426731547361254231675437612543672145367214***"
#
# better_string1 = local_search(num_letters=7, search_length=60, assume_shortening=0, initial_solution=string1, other_strings=[string2, string3])

# print(len(string1))
# print(string1)


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

# words = get_words(7)
# special_words = [w for w in words if w[:2] == '54']
# # other_words = [w for w in words if w[:2] != '54']
#
# string1 = maximize_specials(words, num_strings=1)
# print(string1)
# other54_words = [w for w in words if ('54' in w) and w not in special_words]
# words_45 = [w for w in words if '45' in w]
#
# print(f"words: {len(words)}")
# print(f"special_words: {len(special_words)}")
# print(f"other54_words: {len(other54_words)}")
# print(f"words_45: {len(words_45)}")
#
# words54 = {}
# words54[0] = [w for w in words
#               if (w[0] == '5' and w[2] == '4')
#               or (w[0] == '5' and w[3] == '4')
#               or (w[0] == '5' and w[4] == '4')
#               or (w[0] == '5' and w[5] == '4')
#               or (w[0] == '5' and w[6] == '4')]
#
# words54[1] = [w for w in words
#               if (w[1] == '5' and w[3] == '4')
#               or (w[1] == '5' and w[4] == '4')
#               or (w[1] == '5' and w[5] == '4')
#               or (w[1] == '5' and w[6] == '4')]
#
# words54[2] = [w for w in words
#               if (w[2] == '5' and w[4] == '4')
#               or (w[2] == '5' and w[5] == '4')
#               or (w[2] == '5' and w[6] == '4')]
#
# words54[3] = [w for w in words
#               if (w[3] == '5' and w[5] == '4')
#               or (w[3] == '5' and w[6] == '4')]
#
# words54[4] = [w for w in words
#               if (w[4] == '5' and w[6] == '4')]
#
# words45 = {}
# words45[0] = [w for w in words
#               if (w[0] == '4' and w[2] == '5')
#               or (w[0] == '4' and w[3] == '5')
#               or (w[0] == '4' and w[4] == '5')
#               or (w[0] == '4' and w[5] == '5')
#               or (w[0] == '4' and w[6] == '5')]
#
# words45[1] = [w for w in words
#               if (w[1] == '4' and w[3] == '5')
#               or (w[1] == '4' and w[4] == '5')
#               or (w[1] == '4' and w[5] == '5')
#               or (w[1] == '4' and w[6] == '5')]
#
# words45[2] = [w for w in words
#               if (w[2] == '4' and w[4] == '5')
#               or (w[2] == '4' and w[5] == '5')
#               or (w[2] == '4' and w[6] == '5')]
#
# words45[3] = [w for w in words
#               if (w[3] == '4' and w[5] == '5')
#               or (w[3] == '4' and w[6] == '5')]
#
# words45[4] = [w for w in words
#               if (w[4] == '4' and w[6] == '5')]
#
# total_words = len(special_words)
# total_words += len(other54_words)
# total_words += len(words_45)
# for i in range(5):
#     print(len(words54[i]))
#     print(len(words45[i]))
#     total_words += len(words54[i])
#     total_words += len(words45[i])
# print()
# print(total_words)
# print()
#
# groups = [[], [], []]
#
# groups[0] += special_words
# groups[1] += special_words
# groups[2] += special_words
#
# groups[0] += other54_words[:200]
# groups[1] += other54_words[200:400]
# groups[2] += other54_words[400:]
#
# for i in range(3):
#     print(len(groups[i]))
# print()
#
# groups[0] += words_45[:240]
# groups[1] += words_45[240:480]
# groups[2] += words_45[480:]
#
# for i in range(3):
#     print(len(groups[i]))
# print()
#
# for i in range(5):
#     s = len(words54[i]) // 3
#     print(s, len(words54[i]))
#     print(len(words54[i][:s]))
#     print(len(words54[i][s:2 * s]))
#     print(len(words54[i][2 * s:]))
#
#     groups[0] += words54[i][:s]
#     groups[1] += words54[i][s:2 * s]
#     groups[2] += words54[i][2 * s:]
#
#     s = len(words45[i]) // 3
#     print(s, len(words45[i]))
#     print(len(words45[i][:s]))
#     print(len(words45[i][s:2 * s]))
#     print(len(words45[i][2 * s:]))
#     groups[0] += words45[i][:s]
#     groups[1] += words45[i][s:2 * s]
#     groups[2] += words45[i][2 * s:]
#     for i in range(3):
#         print(len(groups[i]))
#
# for i in range(3):
#     print(len(groups[i]))
#
# for g in groups:
#     string1 = get_optimal_solution(g, num_strings=1, num_wildcards=2)
#     print(len(string1), string1)