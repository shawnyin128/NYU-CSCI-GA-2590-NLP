import random
from nltk.corpus import wordnet
from nltk.metrics.distance import edit_distance

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    tokens = text.split()
    max_changes = max(1, int(0.5 * len(tokens)))
    num_changes = 0

    keyboard_neighbors = {
        "a": ["q", "w", "s", "z"], "b": ["v", "g", "h", "n"], "c": ["x", "d", "f", "v"],
        "d": ["s", "e", "r", "f", "c", "x"], "e": ["w", "r", "s", "d"], "f": ["d", "r", "t", "g", "c", "v"],
        "g": ["f", "t", "y", "h", "v", "b"], "h": ["g", "y", "u", "j", "b", "n"], "i": ["u", "o", "j", "k"],
        "j": ["h", "u", "i", "k", "n", "m"], "k": ["j", "i", "o", "l", "m"], "l": ["k", "o", "p"],
        "m": ["n", "j", "k"], "n": ["b", "h", "j", "m"], "o": ["i", "p", "k", "l"],
        "p": ["o", "l"], "q": ["w", "a"], "r": ["e", "t", "d", "f"], "s": ["a", "w", "e", "d", "x", "z"],
        "t": ["r", "y", "f", "g"], "u": ["y", "i", "h", "j"], "v": ["c", "f", "g", "b"],
        "w": ["q", "e", "a", "s"], "x": ["z", "s", "d", "c"], "y": ["t", "u", "g", "h"], "z": ["a", "s", "x"],
    }
    protected_words = {"not", "no", "never", "n't", "nor", "none", "nothing", "nowhere", "hardly", "barely", "scarcely"}

    for i, token in enumerate(tokens):
        if num_changes >= max_changes:
            break
        if not token.isalpha():
            continue

        lower_token = token.lower()

        changed = False

        if lower_token not in protected_words and len(token) >= 4 and random.random() < 0.5:
            candidates = set()
            for synset in wordnet.synsets(lower_token):
                for lemma in synset.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    candidate_lower = candidate.lower()
                    if (
                        candidate_lower != lower_token
                        and " " not in candidate
                        and candidate.isalpha()
                        and candidate_lower not in protected_words
                    ):
                        candidates.add(candidate)

            if candidates:
                filtered_candidates = [c for c in candidates if edit_distance(lower_token, c.lower()) <= max(len(lower_token) // 2, 2)]
                if not filtered_candidates:
                    filtered_candidates = list(candidates)

                replacement = random.choice(filtered_candidates)
                if token[0].isupper():
                    replacement = replacement.capitalize()
                token = replacement
                lower_token = token.lower()
                changed = True

        if len(token) >= 4 and random.random() < 0.5:
            positions = [j for j, ch in enumerate(lower_token) if ch in keyboard_neighbors]
            if positions:
                pos = random.choice(positions)
                replacement = random.choice(keyboard_neighbors[lower_token[pos]])
                typo_word = token[:pos] + (replacement.upper() if token[pos].isupper() else replacement) + token[pos + 1:]
                if edit_distance(lower_token, typo_word.lower()) == 1:
                    token = typo_word
                    changed = True

        if changed:
            tokens[i] = token
            num_changes += 1

    example["text"] = " ".join(tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
