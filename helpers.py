import subprocess
import itertools
import numpy as np
from functools import lru_cache

SUPERPERMUTATIONS = {
    3: "3212312132",
    4: "123412314231243121342132413214321",
    5: "123451234152341253412354123145231425314235142315423124531243512431524312543121345213425134215342135421324513241532413524132541321453214352143251432154321",
    6: "12345612345162345126345123645132645136245136425136452136451234651234156234152634152364152346152341652341256341253641253461253416253412653412356412354612354162354126354123654132654312645316243516243156243165243162543162453164253146253142653142563142536142531645231465231456231452631452361452316453216453126435126431526431256432156423154623154263154236154231654231564213564215362415362145362154362153462135462134562134652134625134621536421563421653421635421634521634251634215643251643256143256413256431265432165432615342613542613452613425613426513426153246513246531246351246315246312546321546325146325416325461325463124563214563241563245163245613245631246532146532416532461532641532614532615432651436251436521435621435261435216435214635214365124361524361254361245361243561243651423561423516423514623514263514236514326541362541365241356241352641352461352416352413654213654123",
    7: "12345671234561723456127345612374561327456137245613742561374526137456213745612347561324756134275613472561347526134756213475612345761234516723451627345162374516234751623457162345176234512673451263745126347512634571263451726345127634512367451236475123645712364517236451273645123764512346751234657123465172346512734651243765124367512436571243651724365127436512473651246375124635712463517246351274635124763512467351426735146273514672351467325146735216473521674352167345216374521634752163457216345271634521764352176453271645327614532764153276451326745132647513264571326451732645137264531726453712645372164537261453726415372645132764531276453217645231764521376452173654217365241736521473652174365217346521736452176345216735421637542163574216354721635427163542176354216735241637524163572416352741635247163524176352416735214673512465371246531724653127465312476531246753142675314627531467253146752316475321647531264753162475316427531647253164752316745321674531267453162745316724531674253167452316754231675243167523416752314675321467531246573124651372465132746513247651324671532467135246713254671235467125346712543671524367154236715432675143267541326754312675432167543261745362174536127453617245361742536174523617453261743526174325617432651742365174263517426531742651374265173426157342617534216753421765342175634217536421753462175342617354261734526173425617342651743261574362157436125743162574312657413265741236574126357412653741265734126574312567413256741235674125367412563741256734125674312576413257614325761342576132457613254761325746132576412357641253761425376124537612543761524376154237615432761543726154376215437612534761253746125376412573641257634125764312574631257436152743615724361574236157432617543621754361275436172543617524361754236175432671543627154367215436712546371254673125476312547361524736154273615472361547326145736214576321475632147653214763521476325147632154763214576231457621345762143576214537621457361245736142573614527361457236145732614753621475361247536142753614725361475236147532614735261473256147326514723651472635147265314726513472651437265147326154736215473612547316254731265471326547123654712635471265347126543716253471625374162537146253716425371624537162543716524371654237165432716543721654371265473125647132564712356471253647125634712564372156437251643275614327564132756431275643217564327156432751643257163425176342516734251637425163472516342751634257163245176324516732451637245163274516324751632457163254716325741632571463275146327154632714563271465327146352714632571643527164357216435712643517264351276435126743512647351264375126435716243517624351672435162743516247351624375162435716423517642351674235164723516427351642375146237514263751423675142376514273651427635142765314276513427651432765142375614235761423567143256714352671435627143567214356712435617243561274356124735612437561243576124356714235617423561472356142735614237516423571643251764325167432516473251643725614372564137256431725643712564731254671324567132465713246751324615732461753246173524617325416723541762354716235476123547621354762315467231546273154623715462317564231576421356742135647213564271356421735624137562413576241356724135627413562471356241735621473562174356217345621735462173564213756421357642153746215374261537421653742156374215367421537642157364215763421576432157642315674231564723156427315642371564231756243157624315672431562743156247315624371562431756234157623415672341562734156237415623471562341756231475623174562317546321745632174653217463521746325174632157463217546312754631725463175246315724631527463152476315246731524637152463175426315742631547263154276315426731542637154263175462315746235174623571462357416235746123574621357462315476235147623541726354172365417235641723546172354167253417625314762531746253176425317624531762543176524317654231765432176543127654317265431762534172653417256341725364172534617253416725431672541367251436725134672153476215347261534721653472156347215364721534672135467213456721346572136457213654721365742136572413657214365721346752136475213674521367542136752413675214376521437562143752614375216437521463725146372154637214563721465372146357214637521436752134672513647251367425136724513672541637254167325417632541736251473625174362517346257136425713624571362547136257413625714362571346275136427513624751362745136275416327541623754126375412367541237654132765413726541376251437625134762513746251376425137624513762541376524137654213765412375641237546132754613725461375246137542613754621375461237541627354126735412763541273654127356412735461273541627534126753412765341275634127536412753461275341627543162754136275143627513462715342671354267134526713425671342657143265714236571426357142653714265731426571342675134267153427615342716534271563427153642715346271354627134562713465271364527136542713652471365274136527143652713462573146257341625734612573462157346251736425173624517362541732654173256417325461732456173246517324615372461532746153247615324167532416573214657321645731264573162457316425731645273165427316524731652743165273416527314652731645723165472316574231657243165723416572314657231645732165473216574321657342165732416537241653274165324716532417653241567321456731245637124563172456312745631247563124576312456731425637142563174256314725631427563142576314256731452637145236714532671453627145367214536712453671425367145237614523716452371465237416523746152347651234765213476523147652341765234716523476152346715234617523461572346152734615237465123746521374652317465237145623714526317452631475263145726314527631452673145627314567231456732154673215647321567432156734215673241563724156327415632471563241756324157632415367241536274153624715362417536241573624153762415326741532647153264175326415732641523764152367415236471523641752364157236415273641526374152634715263417526341572634152763415267341526437152643175264315726431527643152674315264731526413752641357261435726134572613547261357426135724613572641352761435276134527613542761352476135274613527641352674135264713526417352641",
}
SPECIAL_CHARS = {
    3: (1, 0),
    4: (2, 1),
    5: (3, 2),
    6: (4, 5),
    7: (1, 2),
}
# REPLACE_DICT = {
#  '5': '🎅',
#  '4': '🤶',
#  '8': '🌟',
#  '1': '🦌',
#  '2': '🧝',
#  '3': '🎄',
#  '6': '🎁',
#  '7': '🎀'}


def word_is_contained(word, string_container):
    missing = word not in string_container
    if missing and "*" in string_container:
        for i in range(len(word) - 1):
            if word[:i] + "*" + word[i + 1:] in string_container:
                missing = False
                break
        if word[:-1] + "*" in string_container:
            missing = False
    return not missing

def get_words(num_letters=7):
    words = []
    for k in range(len(SUPERPERMUTATIONS[num_letters]) - (num_letters-1)):
        s = SUPERPERMUTATIONS[num_letters][k:k + num_letters]
        if (is_perm(s)) & (s not in words):
            words.append(s)
    return words

def wild_friends(wi, wj):
    if "*" not in wi:
        return False
    if "*" not in wj:
        return False

    assert "*" not in wi.replace("*", "", 1)
    assert "*" not in wj.replace("*", "", 1)

    if cost(wi, wj) == 0:
        return True

    if wi[0] == "*" or wj[-1] == "*":
        return False

    return wi[1:] == wj[:-1]

@lru_cache(maxsize=2 ** 25)
def cost(str1, str2):
    if len(str1) == 1:
        return len(str2)
    if len(str2) == 1:
        return 0
    if "*" in str1 and "*" in str2:
        return offset(str1, str2)
    if "*" in str1:
        return min([offset(str1.replace("*", str(i)), str2) for i in range(1, 8)])
    if "*" in str2:
        return min([offset(str1, str2.replace("*", str(i))) for i in range(1, 8)])

    return offset(str1, str2)

def hamming_distance(str1, str2):
    return sum((c1!=c2) for c1, c2 in zip(str1, str2))

def real_offset(s1, s2):
    assert(len(s1) == len(s2))
    ln = len(s1)
    j = ln
    for k in range(0, ln):
        if hamming_distance(s1[k:], s2[:len(s1)-k]) == 0:
            j = k
            break
    return j

def offset(word1, word2):
    subword1 = word1
    subword2 = word2

    if len(word1) < len(word2):
        subword2 = word2[:len(word1)]

    if len(word2) < len(word1):
        subword1 = word1[-len(word2):]

    dist = real_offset(subword1, subword2)
    if len(word1) < len(word2):
        dist += len(word2) - len(word1)

    return dist


def get_tsp_solution(group, use_initial_solution=True, start='0', end='0'):
    # CREATE DISTANCE MATRIX
    SIZE = len(group)
    M = np.zeros((SIZE + 1, SIZE + 1), dtype='int8')
    for j in range(SIZE):

        if start != '0':
            M[0, j+1] = offset(start, group[j])
        else:
            M[0, j+1] = len(group[j])

        if end != '0':
            M[j+1, 0] = offset(group[j], end)

        for k in range(SIZE):
            M[j+1, k+1] = offset(group[j], group[k])

    # WRITE PROBLEM FILE
    f = open(f'group.par', 'w')
    f.write("PROBLEM_FILE = distances.atsp\n")
    f.write("TOUR_FILE = output.txt\n")
    f.write(f"OPTIMUM = {SIZE}\n")
    if use_initial_solution:
        f.write("INITIAL_TOUR_FILE = initial.tour\n")

    f.write("MOVE_TYPE = 3\n")
    f.write("PATCHING_C = 4\n")
    f.write("PATCHING_A = 5\n")
    # # f.write("BACKTRACKING = YES\n")
    f.write("MAX_CANDIDATES = 6\n")
    f.write("MAX_TRIALS = 100000\n")
    f.write("PRECISION = 1\n")
    f.write("TRACE_LEVEL = 1\n")


    # f.write("MOVE_TYPE = 5 SPECIAL\n")
    f.write("GAIN23 = YES\n")
    f.write("KICKS = 2\n")
    # f.write("MAX_SWAPS = 0\n")
    f.write("POPULATION_SIZE = 100\n")
    f.write("RUNS = 10\n")

    f.write("TIME_LIMIT = 300\n")  # seconds
    f.close()

    # WRITE PARAMETER FILE
    f = open(f'distances.atsp', 'w')
    f.write("NAME: distances\n")
    f.write("TYPE: ATSP\n")
    f.write("COMMENT: Asymmetric TSP\n")
    f.write(f"DIMENSION: {SIZE + 1}\n")
    f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
    f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
    f.write("EDGE_WEIGHT_SECTION\n")
    for j in range(SIZE + 1):
        # if j%25==0: print(j,', ',end='')
        for k in range(SIZE + 1):
            f.write(f"{M[j, k]:2d} ")
        f.write("\n")
    f.close()

    f = open(f'initial.tour', 'w')
    f.write("TOUR_SECTION\n")
    for i in range(SIZE + 1):
        f.write(f"{i+1}\n")
    f.write("-1\n")
    f.close()

    # EXECUTE TSP SOLVER
    subprocess.run(
        [
            './LKH',
            'group.par',
        ],
    )
        # stdout=subprocess.DEVNULL

    # READ RESULTING ORDER
    with open('output.txt') as f:
        lines = f.readlines()
    for i, ln in enumerate(lines):
        if 'TOUR_SECTION' in ln: break
    perms = [int(x[:-1])-1 for x in lines[i + 1:-2]]

    i = perms.index(0)

    perms = perms[i+1:] + perms[:i]

    # CREATE STRING
    result = group[perms[0] - 1]
    for k in range(1, len(perms)):
        s1 = group[perms[k - 1] - 1]
        s2 = group[perms[k] - 1]
        d = offset(s1, s2)
        assert (d != 0)
        result += s2[-d:]

    return result


def is_perm(s):
    num_letters = len(s)
    if '*' not in s:
        y = True
        for k in range(1, num_letters+1):
            y = y & (str(k) in s)
            if not y:
                break
        return y
    else:
        letter_count = 0
        for k in range(1, num_letters+1):
            letter_count += (str(k) in s)

    return letter_count == num_letters - 1



def fact(n):
    if n == 1:
        return 1
    else:
        return n * fact(n-1)


def map_solution(words, solution):
    num_letters = len(words[0])
    word_map = {
        w: [] for w in words
    }
    # Words that can be found at each position
    position_map = {
        i: None for i in range(len(solution))
    }

    for i in range(len(solution) - num_letters+1):
        words = get_words_in_string(num_letters, solution[i:i + num_letters])
        for w in words:
            if w not in word_map:
                word_map[w] = []
            word_map[w].append(i)
            position_map[i] = w
    return word_map, position_map


def get_words_in_string(num_letters, string1):

    word_list = set()
    for i in range(len(string1) - num_letters + 1):
        w = string1[i:i + num_letters]

        if "*" in w:
            for p in range(num_letters-1, 0, -1):
                if "*" * p in w:
                    existing_letters = [int(c) for c in w if c != '*']
                    letters = [letter for letter in list(range(1, num_letters + 1)) if letter not in existing_letters]
                    for x in itertools.permutations(letters):
                        new_w = w.replace("*" * p, "".join(map(str, list(x))))
                        if not is_perm(new_w):
                            continue
                        word_list.add(new_w)
                    break
        elif is_perm(w):
            word_list.add(w)

    return list(word_list)


def check_word_count(num_letters, string1, string2):

    return len(get_words_in_string(num_letters, string1)) == len(get_words_in_string(num_letters, string2))

def export_string_map(num_letters, string_solution, other_strings, filename='string_map.txt'):

    words = get_words_in_string(num_letters, string_solution)

    word_map, position_map = map_solution(words, string_solution)

    other_strings_words = set()
    for s in other_strings:
        for w in get_words_in_string(num_letters, s):
            other_strings_words.add(w)

    special_words = [w for w in words if w[:2] == '12']

    essential_words = [w for w in words if w not in other_strings_words] + special_words

    loose_following = 0
    with open(filename, 'w') as f:
        for j in range(len(string_solution)):
            if (not position_map.get(j) or len(word_map[position_map[j]]) > 1
                    or position_map[j] not in essential_words):
                loose_following += 1
            else:
                loose_following = 0
            if loose_following > 5:
                m = f"{loose_following}#######"
            else:
                m = ""
            f.write(f"{j+1}\t{string_solution[j]}\t{position_map[j]}\t"
                    f"{[[i+1 for i in word_map[position_map[j]]], position_map[j] in essential_words] if position_map[j] else ''}\t{m}\n")


def get_unique_words_in_string(string1, num_letters=7):
    words = []
    for i in range(len(string1) - num_letters + 1):
        if '*' in string1[i:i + num_letters]:
            for j in range(1, 8):
                new_word = string1[i:i + num_letters]
                new_word = new_word.replace("*", str(j))
                if is_perm(new_word):
                    words.append(new_word)
        else:
            if is_perm(string1[i:i + num_letters]):
                words.append(string1[i:i + num_letters])

    seen = set()
    return [word for word in words if not (word in seen or seen.add(word))]


def get_superwords(words):
    i = 0
    super_words = []
    while i < len(words):
        new_word = words[i]
        i += 1
        while i < len(words) and offset(words[i - 1], words[i]) <= 2:
            new_word += words[i][-offset(words[i - 1], words[i]):]
            i += 1
        super_words.append(new_word)

    return super_words


def get_wild_variants(word):

    # assert len(word) == 7

    word_list = list(word)
    wild_variants = []
    for i in range(len(word)):
        if 7 < i < len(word) - 7:
            continue
        w = [l for l in word_list]
        w[i] = '*'
        w = ''.join(w)
        if i < 7 and is_perm(w[:7]):
            wild_variants.append(w)
        elif i > len(word) - 7 and is_perm(w[-7:]):
            wild_variants.append(w)

    return wild_variants


def get_wild_super_words(super_words):

    wild_super_words = super_words

    for i in range(0, len(super_words)):
        wild_super_words += get_wild_variants(super_words[i])

    return wild_super_words


def words_match(word, wild_word):

    assert len(wild_word) == len(word)
    assert "*" in wild_word

    if word == wild_word:
        return True

    return (cost(word, wild_word) == 0)


def get_5913_sol():
    solution = "121"

    for N in range(3, 8):
        new_solution = []
        for i in range(len(solution) - N + 2):
            substring = solution[i:i + (N - 1)]
            if all(str(x) in substring for x in range(1, N)):
                new_solution.append(substring + str(N) + substring)
        solution = new_solution[:1]
        for i in range(len(new_solution) - 1):
            x1 = new_solution[i]
            x2 = new_solution[i + 1]
            for j in range(1, len(x1) + 1):
                if x2[:j] == x1[-1 * j:]:
                    break
            solution.append(x2[j:])
        solution = "".join(solution)
        print(N, len(solution))

    solution = solution.replace('1', 't')
    solution = solution.replace('7', '1')
    solution = solution.replace('t', '7')
    solution = solution.replace('2', 't')
    solution = solution.replace('6', '2')
    solution = solution.replace('t', '6')

    return solution
    # solutions = {}
    # sub_perms = {}
    # ordered_solutions = []
    # start_idx = 0
    # sol = []
    # for i in range(len(solution) - 6):
    #     perm = solution[i:i + 7]
    #     if ''.join((sorted(perm))) == '1234567':
    #         if start_idx == -1:
    #             start_idx = i
    #             sol = []
    #         sol.append(perm)
    #         if perm.startswith('12'):
    #             end_idx = i + 7
    #
    #             solutions[perm] = solution[start_idx:end_idx]
    #             sub_perms[perm] = sol
    #             print(len(solutions[perm]), end=" ")
    #             ordered_solutions.append(solutions[perm])
    #             start_idx = -1