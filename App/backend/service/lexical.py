"""Conservative lexical ranking for short French and Arabic questions."""
import math

from search import ordered, terms

STOP_WORDS = {
    "fr": set(terms(
        "à au aux avec c ça ce ces cela comment d dans de des du elle elles en est et il ils l la le les leur leurs "
        "moi nous on ou par pendant pour qu quand que quel quelle quelles quels qui quoi s sa se ses si son sont "
        "sur t tu un une vos votre vous y dit dire veut signifie exactement fait servent sert fonctionne combien "
        "pourquoi après avant"
    )),
    "ar": set(terms(
        "ما ماذا كيف هل من في إلى على عن مع و أو هذا هذه هو هي هم ذلك التي الذي يعني معنى المقصود عدد كم "
        "لماذا عند خلال كل نفس تم يتم فائدة"
    )),
}
SHORT_QUERY_MAX_TERMS = 5
MIN_TERM_COVERAGE = 0.4
MIN_RELATIVE_LEAD = 0.05


class ShortLexicalSearch:
    def __init__(self, index, cards):
        self.index = index
        self.titles = {card.id: set(terms(card.title)) - STOP_WORDS[card.lang] for card in cards}
        self.passages = {card.id: set(terms(card.passage)) for card in cards}

    def keywords(self, question, lang):
        vocabulary = self.index.vocabulary(lang)
        keywords = set()
        for word in set(terms(question)) - STOP_WORDS[lang]:
            if word not in vocabulary:
                if lang == "fr":
                    alternate = word[:-1] if word.endswith("s") else word + "s"
                    if alternate in vocabulary:
                        word = alternate
                elif len(word) > 3 and word[0] in "بولف" and word[1:] in vocabulary:
                    word = word[1:]
            keywords.add(word)
        return keywords

    def rank(self, keywords, lang, k=3):
        if not keywords:
            return []
        weights = self.index.term_weights(keywords, lang)
        matches = self.index.rank(" ".join(sorted(keywords)), lang, k=len(self.passages))
        scores = []
        for card, score in matches:
            title = self.titles[card.id]
            if title:
                title_weight = sum(weights[word] for word in sorted(keywords & title))
                score += 2 * title_weight / math.sqrt(len(title))
            scores.append((card, score))
        return ordered(scores, k)

    def accepts(self, keywords, lang, hits):
        if not hits:
            return False
        weights = self.index.term_weights(keywords, lang)
        card, score = hits[0]
        coverage = sum(weights[word] for word in sorted(keywords & self.passages[card.id])) / sum(weights.values())
        runner_up = hits[1][1] if len(hits) > 1 else 0.0
        return coverage >= MIN_TERM_COVERAGE and (score - runner_up) / score >= MIN_RELATIVE_LEAD

    def trusted_hits(self, keywords, lang):
        hits = self.rank(keywords, lang, k=2)
        return hits if self.accepts(keywords, lang, hits) else []
