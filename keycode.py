import string
from itertools import product
from collections import defaultdict

keycode = defaultdict(dict)

keycode['default'][0] = 'unknown'
keycode['default'][8] = 'backspace'
keycode['default'][9] = 'tab'
keycode['default'][13] = 'enter'
keycode['default'][16] = 'shift'
keycode['default'][17] = 'ctrl'
keycode['default'][18] = 'alt'
keycode['default'][19] = 'pause'
keycode['default'][20] = 'caps_lock'
keycode['default'][27] = 'escape'
keycode['default'][32] = 'space'
keycode['default'][33] = 'page_up'
keycode['default'][34] = 'page_down'
keycode['default'][35] = 'end'
keycode['default'][36] = 'home'
keycode['default'][37] = 'left'
keycode['default'][38] = 'up'
keycode['default'][39] = 'right'
keycode['default'][40] = 'down'
keycode['default'][45] = 'insert'
keycode['default'][46] = 'delete'
keycode['default'][48] = '0'
keycode['default'][49] = '1'
keycode['default'][50] = '2'
keycode['default'][51] = '3'
keycode['default'][52] = '4'
keycode['default'][53] = '5'
keycode['default'][54] = '6'
keycode['default'][55] = '7'
keycode['default'][56] = '8'
keycode['default'][57] = '9'
keycode['default'][59] = 'semicolon'
keycode['default'][61] = 'equals'
keycode['default'][65] = 'a'
keycode['default'][66] = 'b'
keycode['default'][67] = 'c'
keycode['default'][68] = 'd'
keycode['default'][69] = 'e'
keycode['default'][70] = 'f'
keycode['default'][71] = 'g'
keycode['default'][72] = 'h'
keycode['default'][73] = 'i'
keycode['default'][74] = 'j'
keycode['default'][75] = 'k'
keycode['default'][76] = 'l'
keycode['default'][77] = 'm'
keycode['default'][78] = 'n'
keycode['default'][79] = 'o'
keycode['default'][80] = 'p'
keycode['default'][81] = 'q'
keycode['default'][82] = 'r'
keycode['default'][83] = 's'
keycode['default'][84] = 't'
keycode['default'][85] = 'u'
keycode['default'][86] = 'v'
keycode['default'][87] = 'w'
keycode['default'][88] = 'x'
keycode['default'][89] = 'y'
keycode['default'][90] = 'z'
keycode['default'][91] = 'left_windows'
keycode['default'][92] = 'right_windows'
keycode['default'][96] = 'numpad_0'
keycode['default'][97] = 'numpad_1'
keycode['default'][98] = 'numpad_2'
keycode['default'][99] = 'numpad_3'
keycode['default'][100] = 'numpad_4'
keycode['default'][101] = 'numpad_5'
keycode['default'][102] = 'numpad_6'
keycode['default'][103] = 'numpad_7'
keycode['default'][104] = 'numpad_8'
keycode['default'][105] = 'numpad_9'
keycode['default'][106] = 'numpad_multiply'
keycode['default'][107] = 'numpad_add'
keycode['default'][109] = 'numpad_subtract'
keycode['default'][110] = 'numpad_decimal'
keycode['default'][111] = 'numpad_divide'
keycode['default'][112] = 'f1'
keycode['default'][113] = 'f2'
keycode['default'][114] = 'f3'
keycode['default'][115] = 'f4'
keycode['default'][116] = 'f5'
keycode['default'][117] = 'f6'
keycode['default'][118] = 'f7'
keycode['default'][119] = 'f8'
keycode['default'][120] = 'f9'
keycode['default'][121] = 'f10'
keycode['default'][122] = 'f11'
keycode['default'][123] = 'f12'
keycode['default'][144] = 'num_lock'
keycode['default'][145] = 'scroll_lock'
keycode['default'][188] = 'comma'
keycode['default'][189] = 'dash'
keycode['default'][190] = 'period'
keycode['default'][191] = 'slash'
keycode['default'][192] = 'back_quote'
keycode['default'][219] = 'open_bracket'
keycode['default'][220] = 'back_slash'
keycode['default'][221] = 'close_bracket'
keycode['default'][222] = 'quote'
keycode['default'][224] = 'left_apple'

keycode['native'][10] = 'enter'
keycode['native'][24] = 'quote'
keycode['native'][44] = 'comma'
keycode['native'][45] = 'dash'
keycode['native'][46] = 'period'
keycode['native'][47] = 'slash'
keycode['native'][91] = 'open_bracket'
keycode['native'][92] = 'back_slash'
keycode['native'][93] = 'close_bracket'
keycode['native'][127] = 'delete'
keycode['native'][155] = 'insert'
keycode['native'][524] = 'windows'

keycode['native'][129] = 'back_quote'
keycode['native'][151] = '8'
keycode['native'][152] = 'quote'
keycode['native'][154] = 'print_screen'
keycode['native'][156] = 'help'
keycode['native'][157] = 'alt'
keycode['native'][160] = 'comma'
keycode['native'][161] = 'open_bracket'
keycode['native'][162] = 'close_bracket'
keycode['native'][512] = '2'
keycode['native'][513] = 'semicolon'
keycode['native'][515] = '4'
keycode['native'][517] = '1'
keycode['native'][519] = '9'
keycode['native'][520] = '3'
keycode['native'][521] = 'equals'
keycode['native'][522] = '0'
keycode['native'][523] = 'dash'
keycode['native'][525] = 'menu'

keycode['opera'][59] = 'semicolon'
keycode['opera'][61] = 'equals'
keycode['opera'][109] = 'dash'
keycode['opera'][219] = 'windows'
keycode['opera'][0] = 'menu'

keycode['msie'][186] = 'semicolon'
keycode['msie'][187] = 'equals'
keycode['msie'][189] = 'dash'
keycode['msie'][91] = 'windows'
keycode['msie'][93] = 'menu'

keycode['firefox'][59] = 'semicolon'
keycode['firefox'][107] = 'equals'
keycode['firefox'][109] = 'dash'
keycode['firefox'][91] = 'windows'
keycode['firefox'][93] = 'menu'

keycode['safari'][186] = 'semicolon'
keycode['safari'][187] = 'equals'
keycode['safari'][189] = 'dash'
keycode['safari'][91] = 'windows'
keycode['safari'][93] = 'menu'

keycode['chrome'][186] = 'semicolon'
keycode['chrome'][187] = 'equals'
keycode['chrome'][189] = 'dash'
keycode['chrome'][91] = 'windows'
keycode['chrome'][93] = 'menu'


def detect_agent(user_agent):
    user_agent = user_agent.lower()
    if 'firefox' in user_agent:
        return 'firefox'
    elif 'chrome' in user_agent:
        return 'chrome'
    elif 'safari' in user_agent:
        return 'safari'
    elif 'msie' in user_agent:
        return 'msie'
    elif 'opera' in user_agent:
        return 'opera'
    else:
        return 'default'


def lookup_key(kc, agent='default'):
    if kc in keycode[agent]:
        return keycode[agent][kc]
    elif kc in keycode['default']:
        return keycode['default'][kc]
    else:
        print('Warning: unknown keycode', kc, 'with agent', agent)
        return None


def transitions(set_a, set_b):
    return set(['%s__%s' % (k1, k2) for k1, k2 in product(set_a, set_b)])


# key sets
vowels = set('aeiou')
consonants = set('bcdfghjklmnpqrstvwxyz')
freq_cons = set('tnsrh')
next_freq_cons = set('ldcpf')
least_freq_cons = set('mwybg')
other_cons = set('jkqvxz')
left_letters = set('qwertasdfgzxcvb')
right_letters = set('yuiophjklnm')

left_hand = set('qwertasdfgzxcvb12345')
right_hand = set('yuiophjklnm67890')

digits = set(string.digits)
letters = set(string.ascii_lowercase)
all_keys = set([k for agent in keycode.values() for k in agent.values()])

non_letters = set([c for c in all_keys if c not in set(letters)])
punctuation = set(['period', 'comma', 'quote', 'back_quote', 'slash', 'semicolon', 'open_bracket', 'close_bracket'])
punctuation_other = set(['back_quote', 'slash', 'semicolon', 'open_bracket', 'close_bracket'])
other_non_letters = set(
        [c for c in all_keys if c not in set().union(letters, punctuation, digits, set(['space', 'shift']))])

# sets of transitions
left_hand__left_hand = transitions(left_hand, left_hand)
left_hand__right_hand = transitions(left_hand, right_hand)
right_hand__left_hand = transitions(right_hand, left_hand)
right_hand__right_hand = transitions(right_hand, right_hand)

all_keys__all_keys = transitions(all_keys, all_keys)
letters__letters = transitions(letters, letters)
letters__non_letters = transitions(letters, non_letters)
non_letters__letters = transitions(non_letters, letters)
non_letters__non_letters = transitions(non_letters, non_letters)
consonants__consonants = transitions(consonants, consonants)
vowels__consonants = transitions(vowels, consonants)
consonants__vowels = transitions(consonants, vowels)
vowels__vowels = transitions(vowels, vowels)

double__letters = set(['%s__%s' % (k1, k2) for k1, k2 in zip(letters, letters)])

letters__space = transitions(letters, ['space'])
letters__punctuation = transitions(letters, punctuation)
shift__letters = transitions(['shift'], letters)
space__letters = transitions(['space'], letters)
punctuation__space = transitions(punctuation, ['space'])

left_index = set('rtfgvb')
left_middle = set('dec')
left_ring = set('wsx')
left_pinky = set('qaz')

right_index = set('yuhjvm')
right_middle = {'i', 'k'}  # , 'comma'}
right_ring = {'o', 'l'}  # , 'period'}
right_pinky = {'p', }  # , 'semicolon', 'slash'}

left_index__left_index = transitions(left_index, left_index)
left_index__left_middle = transitions(left_index, left_middle)
left_index__left_ring = transitions(left_index, left_ring)
left_index__left_pinky = transitions(left_index, left_pinky)
left_index__right_index = transitions(left_index, right_index)
left_index__right_middle = transitions(left_index, right_middle)
left_index__right_ring = transitions(left_index, right_ring)
left_index__right_pinky = transitions(left_index, right_pinky)

left_middle__left_index = transitions(left_middle, left_index)
left_middle__left_middle = transitions(left_middle, left_middle)
left_middle__left_ring = transitions(left_middle, left_ring)
left_middle__left_pinky = transitions(left_middle, left_pinky)
left_middle__right_index = transitions(left_middle, right_index)
left_middle__right_middle = transitions(left_middle, right_middle)
left_middle__right_ring = transitions(left_middle, right_ring)
left_middle__right_pinky = transitions(left_middle, right_pinky)

left_ring__left_index = transitions(left_ring, left_index)
left_ring__left_middle = transitions(left_ring, left_middle)
left_ring__left_ring = transitions(left_ring, left_ring)
left_ring__left_pinky = transitions(left_ring, left_pinky)
left_ring__right_index = transitions(left_ring, right_index)
left_ring__right_middle = transitions(left_ring, right_middle)
left_ring__right_ring = transitions(left_ring, right_ring)
left_ring__right_pinky = transitions(left_ring, right_pinky)

left_pinky__left_index = transitions(left_pinky, left_index)
left_pinky__left_middle = transitions(left_pinky, left_middle)
left_pinky__left_ring = transitions(left_pinky, left_ring)
left_pinky__left_pinky = transitions(left_pinky, left_pinky)
left_pinky__right_index = transitions(left_pinky, right_index)
left_pinky__right_middle = transitions(left_pinky, right_middle)
left_pinky__right_ring = transitions(left_pinky, right_ring)
left_pinky__right_pinky = transitions(left_pinky, right_pinky)

right_index__left_index = transitions(right_index, left_index)
right_index__left_middle = transitions(right_index, left_middle)
right_index__left_ring = transitions(right_index, left_ring)
right_index__left_pinky = transitions(right_index, left_pinky)
right_index__right_index = transitions(right_index, right_index)
right_index__right_middle = transitions(right_index, right_middle)
right_index__right_ring = transitions(right_index, right_ring)
right_index__right_pinky = transitions(right_index, right_pinky)

right_middle__left_index = transitions(right_middle, left_index)
right_middle__left_middle = transitions(right_middle, left_middle)
right_middle__left_ring = transitions(right_middle, left_ring)
right_middle__left_pinky = transitions(right_middle, left_pinky)
right_middle__right_index = transitions(right_middle, right_index)
right_middle__right_middle = transitions(right_middle, right_middle)
right_middle__right_ring = transitions(right_middle, right_ring)
right_middle__right_pinky = transitions(right_middle, right_pinky)

right_ring__left_index = transitions(right_ring, left_index)
right_ring__left_middle = transitions(right_ring, left_middle)
right_ring__left_ring = transitions(right_ring, left_ring)
right_ring__left_pinky = transitions(right_ring, left_pinky)
right_ring__right_index = transitions(right_ring, right_index)
right_ring__right_middle = transitions(right_ring, right_middle)
right_ring__right_ring = transitions(right_ring, right_ring)
right_ring__right_pinky = transitions(right_ring, right_pinky)

right_pinky__left_index = transitions(right_pinky, left_index)
right_pinky__left_middle = transitions(right_pinky, left_middle)
right_pinky__left_ring = transitions(right_pinky, left_ring)
right_pinky__left_pinky = transitions(right_pinky, left_pinky)
right_pinky__right_index = transitions(right_pinky, right_index)
right_pinky__right_middle = transitions(right_pinky, right_middle)
right_pinky__right_ring = transitions(right_pinky, right_ring)
right_pinky__right_pinky = transitions(right_pinky, right_pinky)

hand_features = {
    'LL': left_hand__left_hand,
    'LR': left_hand__right_hand,
    'RL': right_hand__left_hand,
    'RR': right_hand__right_hand,
    'space__letters': space__letters,
    'letters__space': letters__space,
}

from functools import reduce

hand_features['OTHER'] = all_keys__all_keys - reduce(lambda x, y: x.union(y), hand_features.values())

###### Key and finger locations

key_locations = {
    'a': (3.5, 2.25),
    'b': (2.5, 6.75),
    'c': (2.5, 4.75),
    'd': (3.5, 4.25),
    'e': (4.5, 4),
    'f': (3.5, 5.25),
    'g': (3.5, 6.25),
    'h': (3.5, 7.25),
    'i': (4.5, 9),
    'j': (3.5, 8.25),
    'k': (3.5, 9.25),
    'l': (3.5, 10.25),
    'm': (2.5, 8.75),
    'n': (2.5, 7.75),
    'o': (4.5, 10),
    'p': (4.5, 11),
    'q': (4.5, 2),
    'r': (4.5, 5),
    's': (3.5, 3.25),
    't': (4.5, 6),
    'u': (4.5, 8),
    'v': (2.5, 5.75),
    'w': (4.5, 3),
    'x': (2.5, 3.75),
    'y': (4.5, 7),
    'z': (2.5, 2.75),

}

finger_locations = {
    'a': (1, 1),
    'b': (1, 4),
    'c': (1, 3),
    'd': (1, 3),
    'e': (1, 3),
    'f': (1, 4),
    'g': (1, 4),
    'h': (2, 5),
    'i': (2, 6),
    'j': (2, 5),
    'k': (2, 6),
    'l': (2, 7),
    'm': (1, 5),
    'n': (1, 5),
    'o': (2, 7),
    'p': (2, 8),
    'q': (1, 1),
    'r': (1, 4),
    's': (1, 2),
    't': (1, 4),
    'u': (2, 5),
    'v': (1, 4),
    'w': (1, 2),
    'x': (1, 2),
    'y': (2, 5),
    'z': (1, 1),
}

complex_key_distances = {'%s__%s' % (k1, k2): abs(complex(*key_locations[k1]) - complex(*key_locations[k2])) for k1, k2
                         in product(key_locations.keys(), key_locations.keys())}
complex_finger_distances = {'%s__%s' % (k1, k2): abs(complex(*finger_locations[k1]) - complex(*finger_locations[k2]))
                            for k1, k2 in product(finger_locations.keys(), finger_locations.keys())}

LINGUISTIC_FEATURES = {
    # duration features
    'all_keys': all_keys,
    'letters': letters,
    'vowels': vowels,
    'freq_cons': freq_cons,
    'next_freq_cons': next_freq_cons,
    'least_freq_cons': least_freq_cons,
    'other_cons': other_cons,
    'left_letters': left_letters,
    'right_letters': right_letters,
    'non_letters': non_letters,
    'punctuation': punctuation,
    'punctuation_other': punctuation_other,
    'digits': digits,
    'other_non_letters': other_non_letters,

    'a': set(['a']),
    'e': set(['e']),
    'i': set(['i']),
    'o': set(['o']),
    'u': set(['u']),
    't': set(['t']),
    'n': set(['n']),
    's': set(['s']),
    'r': set(['r']),
    'h': set(['h']),
    'l': set(['l']),
    'd': set(['d']),
    'c': set(['c']),
    'p': set(['p']),
    'f': set(['f']),
    'm': set(['m']),
    'w': set(['w']),
    'y': set(['y']),
    'b': set(['b']),
    'g': set(['g']),
    'space': set(['space']),
    'shift': set(['shift']),
    'period': set(['period']),
    'comma': set(['comma']),
    'quote': set(['quote']),

    # transition features, key sets separated by '__'
    'all_keys__all_keys': all_keys__all_keys,
    'letters__letters': letters__letters,
    'consonants__consonants': consonants__consonants,
    't__h': set(['t__h']),
    's__t': set(['s__t']),
    'n__d': set(['n__d']),
    'vowels__consonants': vowels__consonants,
    'a__n': set(['a__n']),
    'i__n': set(['i__n']),
    'e__r': set(['e__r']),
    'e__s': set(['e__s']),
    'o__n': set(['o__n']),
    'a__t': set(['a__t']),
    'e__n': set(['e__n']),
    'o__r': set(['o__r']),
    'consonants__vowels': consonants__vowels,
    'h__e': set(['h__e']),
    'r__e': set(['r__e']),
    't__i': set(['t__i']),
    'vowels__vowels': vowels__vowels,
    'e__a': set(['e__a']),
    'double__letters': double__letters,
    'left_hand__left_hand': left_hand__left_hand,
    'left_hand__right_hand': left_hand__right_hand,
    'right_hand__left_hand': right_hand__left_hand,
    'right_hand__right_hand': right_hand__right_hand,
    'letters__non_letters': letters__non_letters,
    'letters__space': letters__space,
    'letters__punctuation': letters__punctuation,
    'non_letters__letters': non_letters__letters,
    'shift__letters': shift__letters,
    'space__letters': space__letters,
    'non_letters__non_letters': non_letters__non_letters,
    'space__shift': set(['space__shift']),
    'punctuation__space': punctuation__space
}

LINGUISTIC_FALLBACK = {
    # duration fallback
    'all_keys': ['letters', 'non_letters'],
    'letters': ['vowels', 'freq_cons', 'next_freq_cons', 'least_freq_cons', 'left_letters', 'right_letters'],
    'vowels': vowels,
    'freq_cons': freq_cons,
    'next_freq_cons': next_freq_cons,
    'least_freq_cons': least_freq_cons.union(set(['other_cons'])),  # not a typo, 'other_cons' is a leaf node

    'non_letters': ['digits', 'other_non_letters', 'punctuation', 'space', 'shift'],
    'punctuation': ['punctuation_other', 'period', 'comma', 'quote'],

    # transition fallback
    'all_keys__all_keys': ['letters__letters', 'letters__non_letters', 'non_letters__letters',
                           'non_letters__non_letters'],
    'letters__letters': ['consonants__consonants', 'vowels__consonants', 'consonants__vowels', 'vowels__vowels',
                         'double__letters', 'left_hand__left_hand', 'left_hand__right_hand', 'right_hand__left_hand',
                         'right_hand__right_hand'],

    'consonants__consonants': ['t__h', 's__t', 'n__d'],
    'vowels__consonants': ['a__n', 'i__n', 'e__r', 'e__s', 'o__n', 'a__t', 'e__n', 'o__r'],
    'consonants__vowels': ['h__e', 'r__e', 't__i'],
    'vowels__vowels': ['e__a', ],

    'letters__non_letters': ['letters__space', 'letters__punctuation'],
    'non_letters__letters': ['shift__letters', 'space__letters'],
    'non_letters__non_letters': ['space__shift', 'punctuation__space']
}
