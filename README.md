# Natural-Language-Processing
CSC733
Lecturer: Yuji Zhang

---

## 1. Why "word" is not a good unit

A computer needs a fixed list of units to work with. Using whole words breaks down:

- **Morphemes.** The smallest pieces that carry meaning. "unhappiness" = *un* + *happy* + *ness*. Treating it as one opaque word throws that structure away.
- **Not universal.** Languages like Turkish or Finnish glue many morphemes into one word; Chinese writes without spaces at all. "Split on spaces" is an English habit, not a rule.
- **OOV (out of vocabulary).** Any word not seen in training has no entry at all. New names, typos, slang — the model is blind to them.

### Byte-level tokenization

Go to the other extreme: treat text as raw bytes. There are only 256 possible byte values, so nothing is ever out of vocabulary. ASCII covers English in 1 byte per character; UTF-8 uses 1–4 bytes, so non-English text costs more bytes per character. The downside: sequences get very long and each unit carries almost no meaning.

### Subword tokenization

The compromise that won. Common words stay whole, rare words break into pieces. "lowest" → `low` + `est`. No OOV, sequences stay short, and the pieces still carry meaning.

### Tokenizer bias

Tokenizers are trained on a corpus. If that corpus is mostly English, English gets efficient whole-word tokens and other languages get chopped into many small pieces. That means the same sentence costs more tokens — and more money, and more of the context window — in some languages than others.

---

## 2. Byte Pair Encoding (BPE)

**Idea:** start from characters, repeatedly merge the most frequent *adjacent* pair, and save the list of merges.

Adjacency matters. In `l o w`, the candidate pairs are `l+o` and `o+w`. Never `l+w` — they aren't neighbours.

### Corpus vs vocabulary

- **Corpus** — the actual training text, with repetition. "the cat sat on the mat" is 6 tokens.
- **Vocabulary** — the set of distinct units. That same sentence has 5 types, because "the" appears twice.

Tokens = total count. Types = distinct count.

### Worked example

Training corpus (word : frequency):

| word | freq |
|---|---|
| low | 5 |
| lower | 2 |
| newest | 6 |
| widest | 3 |

Every word starts split into characters with an end marker: `l o w </w>`.

**Merge 1.** Count every adjacent pair. Max frequency is 9, with a three-way tie between `e·s`, `s·t`, and `t·</w>` (all from newest ×6 + widest ×3). Tie rule picks **`e+s` → `es`**.

Result: `newest` → `n e w es t </w>`, `widest` → `w i d es t </w>`. The `low`/`lower` entries are untouched — they contain no `es`.

**Merge 2.** Now `es·t` and `t·</w>` are tied at 9. Picks **`es+t` → `est`**.

### Encoding new text

To tokenize an unseen word, you **replay the saved merge list in order**. You do *not* recount anything — frequencies were frozen at training time.

`lowest` → `l o w e s t </w>` → apply `e+s` → `l o w es t </w>` → apply `es+t` → **`l o w est </w>`**

This learn-once / replay-later structure shows up again and again in this field.

---

## 3. N-gram language models

An **n-gram** is a sequence of n adjacent words.

- **Unigram** — ignores context entirely, just overall word frequency. P(cats) = 0.10
- **Bigram** — one word of context. "I like cats" gives the bigrams "I like" and "like cats".
- **Trigram** — two words of context.

### How a bigram is computed

P(cats | like) = count("like cats") / count("like") = 6 / 10 = **0.60**

Counts are tabled at training time and simply looked up at prediction time. Same pattern as BPE's merge list: learn once, replay later.

### Disadvantages of the bigram

1. **Short memory.** One word back only. In "the cats that live next door **are**", the verb agrees with "cats", but the bigram only sees "door".
2. **Zeros.** Any pair never seen in training gets probability exactly 0.
3. **No understanding of meaning.** It cannot know "cat" and "kitten" are related, so learning about one never helps the other.

Smoothing patches #2. Nothing here patches #1 or #3 — that takes embeddings and, eventually, transformers.

---

## 4. Markov chains

A **Markov chain** is a system that jumps between states where the next state depends only on the *current* state, not the full history. That is the **Markov property** (sometimes called memoryless).

A bigram model *is* a Markov chain whose states are words.

### Markov's original experiment (1913)

Andrey Markov hand-counted the first 20,000 letters of Pushkin's *Eugene Onegin*, classifying each as vowel or consonant. He found:

- after a vowel → ~13% chance of another vowel
- after a consonant → ~66% chance of a vowel

That was the first demonstration that the next symbol depends on the previous one.

### Why it mattered

Before Markov, probability math largely assumed **independence** — coin flips, dice. He proved the math still works under *dependence*, which is how the real world actually behaves. That opened the door to PageRank, speech recognition, weather and finance models, and early language models.

### Casino vs Markov — the key distinction

- **Roulette is independent.** After red, red is still ~47%. The past never shifts the odds.
- **A Markov chain has one step of memory.** After a consonant, vowel is 66%; after a vowel, only 13%.
- **Blackjack is the exception** — cards leave the deck, creating dependence. That's why card counting works.

### Generating text

Current state in → prediction out → that prediction becomes the new current state → repeat.

Important detail: the model **samples using the odds** rather than always taking the most likely option. With 66% vowel / 34% consonant, it does sometimes pick the consonant. Always taking the maximum would produce a boringly perfect alternation.

### Historical note

Claude Shannon used n-grams in 1948 to generate English-like text. N-grams went on to power speech recognition, phone keyboard suggestions, spell checkers, and early machine translation until neural nets took over in the 2010s.

Phone keyboards genuinely worked this way; Apple moved to an on-device small neural network around iOS 17. The output concept is unchanged — rank the next words, show the top three — and it personalizes from your own typing history stored on-device, which is why it learns your names and slang.

---

## 5. Smoothing

**Smoothing** means taking a little probability from pairs you've seen and giving it to pairs you haven't, so nothing lands on exactly zero.

### Add-one (Laplace) smoothing

Pretend every possible item showed up one extra time.

P(cats | like) = (count + 1) / (count of "like" + V)

With count("like cats") = 6, count("like") = 10, vocabulary V = 1,000:

- unsmoothed: 6/10 = 0.60
- smoothed: 7/1,010 ≈ 0.007

### Where "Laplace" comes from — the sunrise problem

Pierre-Simon Laplace asked: if the sun has risen every day you've ever seen, how sure can you be about tomorrow?

His answer: add two imaginary days *before* you start counting — one where the sun rose, one where it didn't.

> **successes + 1** over **trials + 2**

After 5 real sunrises: 6/7 ≈ 86%.
After 100: 101/102 ≈ 99%.
After 10,000: 10,001/10,002 — extremely close to certain, never exactly certain.

**Common confusion:** the +1 on top is *not* tomorrow. Tomorrow is the thing being predicted, so it appears nowhere in the fraction. The +1 is the imaginary rise, which is also one of the two on the bottom. The bottom counts *every* day (real and imaginary); the top counts only the *rises*. That's why the bottom is larger — it's questions-right over questions-total.

**Marble version:** a jar where white = rose, black = didn't. Laplace drops in one white and one black before you see anything. Each real sunrise adds a white. After 5 days: 6 white, 7 total. Tomorrow is the draw.

### Why add-one is too strong on large vocabularies

You add one for *every* word in the vocabulary, not just the missing one. Those pile up.

With count("like") = 10 and V = 50,000, you've just invented 50,000 imaginary counts to sit against 10 real ones. "like cats", genuinely seen 6 times, becomes 7/50,010 — essentially nothing. Meanwhile every pair you never saw collectively takes ~99.9% of the probability mass.

**Pizza analogy.** You ask 10 friends their favourite topping; 6 say pepperoni → 60%. Now, to be "fair", you add one imaginary vote for each of 1,000 toppings in the world. Pepperoni drops to 7/1,010, under 1%, and pineapple-anchovy-marshmallow gets a vote too. Your 10 real answers are buried under 1,000 fake ones.

(Note on the denominator: it counts *votes*, not toppings. The 1,000 toppings only matter as the list of options — most got zero votes. The problem begins when add-one hands each of them a fake vote, growing the vote count from 10 to 1,010.)

**Fixes:** add-k with a tiny k (0.01), or interpolation.

### Add-one in spam filters

Naive Bayes learns how often each word appears in spam vs normal mail, then multiplies the probabilities of all words in a new email. If one word never appeared in spam training, that probability is 0 — and anything × 0 = 0. A single unseen word would mark the email "definitely not spam" no matter how spammy the rest is.

Add-one gives every word a small nonzero count:

P(zebra | spam) = (0 + 1) / (10,000 + 5,000) = 1/15,000

Tiny, but not zero — and that's exactly the point. It isn't supposed to carry weight. An unseen word gets a tiny probability on *both* the spam side and the normal side, so it barely tips the balance; the decision comes from the words the filter actually knows. Smoothing turns an unknown word from a **veto** into a **shrug**.

Also note the filter computes *two* scores — how likely the email is under spam, and under normal — and the bigger wins. A word like "meeting" pulls toward normal, not just away from spam.

Paul Graham's 2002 essay "A Plan for Spam" popularized Bayesian filtering; Thunderbird and SpamAssassin adopted it. (Graham's own version used custom fudge factors rather than strict add-one, but the goal was the same.) Gmail today uses neural networks plus sender reputation and link analysis.

---

## 6. Interpolation

Blend several models instead of trusting one. Weights must be non-negative and sum to 1.

**Two-way (bigram + unigram):**

0.7 × P(cats|like) + 0.3 × P(cats)
= 0.7 × 0.60 + 0.3 × 0.10 = **0.45**

**Three-way (trigram + bigram + unigram):**

0.5 × 0.80 + 0.3 × 0.60 + 0.2 × 0.10 = **0.60**

More context is more accurate *when the data exists*, but rarer — so weights descend as context length grows.

The advantage over add-one: **no fake counts are invented**. A missing bigram just falls back on real unigram data, so evidence you actually collected isn't drowned out. The name is borrowed from math, where interpolating means estimating a value between two known points — here you land between what the bigram says and what the unigram says.

---

## 7. Perplexity and cross-entropy

**Perplexity** measures how surprised a model is by text it hasn't seen. Lower is better. Intuitively: *how many words does the model feel it's choosing between at each step?*

### The pipeline

1. **Train** the bigram on training text by counting pairs.
2. **Test** on *different* text: for each position, look up the model's probability for the word that actually came next.
3. **Score**: take −log₂ of each probability, average them, then raise 2 to that power.

### Why the minus sign

Probabilities are between 0 and 1, so their logs are always negative:

- log₂(0.1) ≈ −3.32
- log₂(0.8) ≈ −0.32
- log₂(1) = 0 ← perfect certainty

The minus flips them positive. It also flips the direction: a good model gives *high* probability but should score *low* perplexity.

### Why base 2 — bits

With log base 2, the result is in **bits** — yes/no questions. If the model gives the right word a 1/8 chance, −log₂(1/8) = 3: three yes/no questions to pin it down (8 → 4 → 2 → 1).

Averaged over every token, that's **bits per token**, which is **cross-entropy**. Perplexity is just 2 raised to it. 3 bits per token → perplexity 8. Same measurement, two notations.

### Worked example

Four words → three predictions, with probabilities 1/2, 2/3, 1:

1. Surprises: −log₂(1/2) = 1, −log₂(2/3) ≈ 0.585, −log₂(1) = 0. Sum ≈ 1.585
2. Divide by N = 3 → **0.53 bits per token** (cross-entropy)
3. 2^0.53 ≈ **1.44** (perplexity)

**Why divide by N?** So texts of different lengths compare fairly. A 1,000-word text piles up far more total surprise than a 4-word one purely by being longer. The denominator is the **number of predictions**, not a sum of probabilities.

### Reading the numbers

- Cross-entropy 0.53 → on average about half a yes/no question per word. Very little surprise.
- Perplexity 1.44 → the model behaved as if choosing between ~1.4 words.

It isn't literally 1.4 words — you can't pick 1.4 of anything. It's a comparison scale, like dice:

| perplexity | as unsure as |
|---|---|
| 1 | knowing the answer |
| 2 | a coin flip |
| 6 | a fair die |
| 100 | a fair 100-sided die |

A perplexity of 100 doesn't mean the model examines 100 words — it always assigns a probability to every word in its vocabulary. It means its total uncertainty *equals* that of a fair choice among 100.

### What counts as good

No universal number. Minimum is 1. With smoothing, a random model scores about the vocabulary size; without smoothing it can be infinite (any zero probability → infinite perplexity, so you can't even evaluate the model — another reason smoothing exists).

Classic Wall Street Journal figures from the Jurafsky textbook: unigram ≈ 962, bigram ≈ 170, trigram ≈ 109. Modern neural models reach low double digits.

Perplexity is only meaningful as a **comparison** — same test text, same tokenizer, change one thing at a time:

- *Same text, two models* → the lower one is better.
- *Same model, two texts* → the higher one just means that text was harder.

Caveat: lower perplexity doesn't always mean better performance on a real task like translation. It's a cheap check, not the final word.

### Cross-entropy, more precisely

The model's average surprise per token on real text. The "cross" is because two distributions are involved: the text comes from how people actually write, but it's scored using the model's guesses. Plain entropy is how unpredictable the language itself is even for a perfect model; cross-entropy is always at least that high, and the gap is your model's error.

It's also the **loss function** used to train neural language models. Training is, essentially, pushing cross-entropy down.

### Origin

Introduced in 1977 by Frederick Jelinek's speech recognition group at IBM, with Robert Mercer, Lalit Bahl and James Baker — at the Thomas J. Watson Research Center in Yorktown Heights, NY. Built on Shannon's 1948 entropy.

---

## 8. Representing words as vectors

### One-hot vectors

A list as long as the vocabulary: all zeros, a single 1 in this word's slot.

```
cat   1, 0, 0
dog   0, 1, 0
fish  0, 0, 1
```

Two problems: with a 10,000-word vocabulary each vector is 10,000 numbers and almost all zeros; and it carries **no meaning** — "cat" is exactly as different from "dog" as from "refrigerator".

That's the same "no understanding of meaning" weakness the bigram has — though they're different kinds of thing. A bigram is a *model that predicts*; a one-hot vector is just a *notation for writing a word down*.

### Learned word embeddings

A short, dense list — maybe 300 numbers instead of 10,000 — that the model works out during training. Nobody sets them by hand.

The individual numbers mean nothing on their own. There is no "furriness" dimension. Only the relationships between vectors matter.

The underlying principle: **you can tell what a word means by the words around it.**

> **One-hot = an ID badge.** Says only which word it is; every word is equally unrelated to every other.
> **Embedding = a short profile.** Learned from context, so "cat" and "kitten" end up with similar numbers.

In practice "embedding" almost always means a *learned* one; the word "learned" distinguishes it from one-hot vectors and older count-built vectors.

---

## 9. Cosine similarity

How you check whether two embeddings are close in meaning. Picture each word as an arrow from a common origin — cosine similarity measures the **angle**, ignoring length.

| angle | score | meaning |
|---|---|---|
| 0° | 1 | same direction — very similar |
| 90° | 0 | unrelated |
| 180° | −1 | opposite |

It returns the *cosine* of the angle, not degrees, which is why the range is −1 to 1. It stops at 180° because that's as far apart as two arrows can get.

Important distinction: **0 means unrelated** ("cat" and "refrigerator" — nothing in common), while **−1 means actively opposite** — a stronger relationship than zero. In real embeddings a true −1 is almost unheard of; even "hot" and "cold" usually score positive, because they appear in the same kinds of sentences. Most real scores sit between 0 and 1.

And: any two different one-hot vectors score exactly 0. That's the "no meaning" problem expressed numerically.

### The formula

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \, \|B\|}$$

**Numerator — the dot product.** Multiply matching positions and add. For A = (1, 5), B = (2, 3): (1×2) + (5×3) = 2 + 15 = **17**.

The dot product is *not* a length — it combines two different vectors into one number. (Related, though: the dot product of a vector with itself, square-rooted, *is* its length.)

**Denominator — the two lengths, multiplied.** Each length is its own square root of squares (Pythagoras — the vector is the hypotenuse):

- ‖A‖ = √(1 + 25) ≈ 5.1
- ‖B‖ = √(4 + 9) ≈ 3.6
- product ≈ **18.4** (multiplied, not added — 8.7 would be wrong)

**Result:** 17 / 18.4 ≈ **0.92** — very similar.

### Why divide

The denominator is the **largest the numerator could possibly be** — what you'd get if the vectors pointed in exactly the same direction. So dividing asks: *what fraction of maximum possible agreement did we get?* 17 out of a possible 18.4 → 92%. That's also why the result can never escape −1 to 1.

### What "the top measures agreement" means

The dot product grows when both vectors are large in the *same positions*. Pretend position 1 = "animal-ness", position 2 = "kitchen-ness":

- cat (5, 0) · dog (4, 0) = 20 + 0 = **20** — both strong in the same place
- cat (5, 0) · fridge (0, 5) = 0 + 0 = **0** — never strong in the same place

And if one is positive where the other is negative, that product is negative and drags the total down — disagreement. (Real embedding dimensions have no tidy labels like this; it's just an easy way to see the mechanism.)

### Why cosine beats a raw dot product for count vectors

When vectors hold raw co-occurrence counts, common words have huge counts everywhere. "the" would have a large dot product with nearly everything — not because it's similar, but because its vector is *long*.

**Diet analogy.** One person eats 10 apples and 10 bananas; another eats 1 apple and 1 banana. Same taste, different appetite. The dot product says they're very different; cosine says they're identical, because the *proportions* match.

Dividing by the lengths strips out "how much" and leaves "what pattern" — so a word seen a million times and one seen fifty times can still score as highly similar.

### Origin

The math is centuries-old geometry. Applying it to text comes from **Gerard Salton** at Cornell in the 1960s–70s; his SMART system turned documents into vectors and used the cosine to rank them against a query — the ancestor of search engines.

---

## 10. word2vec and skip-gram

**word2vec** = "word to vector". A toolkit released by **Tomas Mikolov**'s team at Google in 2013 (co-authors: Kai Chen, Greg Corrado, Jeff Dean, Ilya Sutskever), based at Google's Mountain View headquarters. Two papers that year: the first introduced the methods, the second added speed tricks. The code was released free, which is a large part of why it spread so fast.

It contains two training methods:

- **Skip-gram** — use the target word to predict its neighbours.
- **CBOW** — use the neighbours to predict the target word.

The famous result: **king − man + woman ≈ queen**.

### What came before

- **Count-based vectors** — a huge table of how often each word appears near every other; each row is a vector. Long and mostly zeros (which is precisely where the cosine-vs-raw-counts issue comes from).
- **Latent Semantic Analysis (1990)** — squeezed that table down with SVD so similar words landed close together.
- **Bengio (2003)** — a neural language model that learned embeddings, but far too slow to train at scale.

word2vec's breakthrough wasn't inventing embeddings — it was making the learned approach fast enough for *billions* of words.

### How skip-gram actually runs

**Setup, once:** build the vocabulary (the distinct words) and give every word a vector of **random** numbers — maybe 100 or 300.

**Then, sliding across the entire corpus:**

1. **Target + window.** "Centre word" is misleading — it doesn't mean the middle of the sentence. It means *whichever word is currently being looked at*, with the window centred on it. The window **slides**, so every word takes a turn.

   With window = 3 on a 10-word sentence: word 1 has nothing to its left, so it makes 3 guesses; word 2 makes 4; from word 4 onward it's the full 6, until the end of the sentence shrinks it again. Nothing is skipped. (Some implementations don't stop at sentence boundaries at all and treat the text as one continuous stream.)

2. **Pairs.** Each target + each neighbour becomes a training pair: ("fluffy", "like"), ("fluffy", "cats").

3. **Guess.** From the target's vector, produce a probability for every vocabulary word being that neighbour.

4. **Check.** Compare against the real neighbour — which is sitting right there in the text, so no human labelling is needed. The error is measured with **cross-entropy**, exactly the −log of the probability given to the correct answer.

5. **Nudge.** A gradient says which direction to push each number so the correct neighbour gets higher probability next time. It nudges *every* time — by a little or a lot depending on how wrong it was — touching only the vectors involved.

6. **Slide** to the next word, then the next sentence. After the whole corpus, start over. Each full pass is an **epoch**.

Because no human labels anything, this is **self-supervised learning**.

### Two things that trip people up

**There is no separate prediction phase later.** All the guessing and nudging happens *during* training. When training ends, you throw the guessing machinery away and keep **only the vectors**. The prediction task was never the goal — it was a workout to force the numbers into a useful shape.

**It never becomes perfect, and shouldn't.** After "fluffy" you might legitimately see cat, dog, pillow, or cloud. The best possible behaviour is to spread probability sensibly across them — and that spread *is* the meaning.

Also, 100,000 words isn't many. Common words like "the" get thousands of turns, but "kitten" might appear five times — five nudges is nothing. Hence billions of words.

### Why similar words get similar vectors

This is the part worth getting exactly right — **the causation runs backwards from intuition.**

The model does **not** know "cat" and "kitten" are similar. "kitten" starts with random numbers completely unconnected to "cat".

But "kitten" keeps appearing beside the same neighbours "cat" appears beside — fluffy, purred, pet. So training nudges it toward predicting those same words, independently, with no link to "cat" at all.

> Words that appear in similar contexts are trained to predict the same neighbouring words, so their vectors get pushed in the same direction during training.

The similar vectors are the **result**, not the reason. Nobody told the model anything.

### Window size

- Small window (≈2) → grammar-like similarity (words of the same part of speech cluster).
- Larger window (≈5) → topic-like similarity.

### The big limitation

Skip-gram gives "bank" **one single vector** — a blur of the money sense and the river sense, weighted by whichever appeared more in training. This is the limitation transformers eventually solve.

### Where skip-gram is still used

It doesn't care whether the "words" are really words — anything appearing in sequences can be embedded:

- **Recommendations** — a listening session is a "sentence", each song a "word"; songs played together get similar vectors.
- **E-commerce** — products browsed in one visit → "people also viewed".
- **Graphs** — DeepWalk and node2vec take random walks through a network and treat each walk as a sentence. (Random walks — Markov chains again.)
- **Small or specialized domains** — medical/legal text, or low-power devices where a transformer is overkill.

Modern large models don't run a separate skip-gram step; they learn embeddings as part of training the whole model.

---

## 11. Part-of-speech tagging

Labelling every word with its grammatical category: noun, verb, adjective, adverb, pronoun, preposition, determiner.

The difficulty is ambiguity — one word, several categories:

- "I **book** a flight" → verb
- "read a **book**" → noun

Only context resolves it.

### The Penn Treebank tag set (~45 tags)

Finer-grained than the basic eight:

| tag | meaning | example |
|---|---|---|
| NN | singular noun | cat |
| NNS | plural noun | cats |
| NNP | proper noun | London |
| VB | verb, base form | run |
| VBD | verb, past tense | ran |
| VBG | verb, -ing | running |
| JJ | adjective | big |
| JJR | comparative | bigger |
| JJS | superlative | biggest |

The extra granularity captures number, tense and comparison.

### Why the detail is useful

- **Tense** — VBD vs VB distinguishes "the company acquired" from "will acquire". Crucial when extracting facts from news.
- **Number** — NN vs NNS catches "the cats is"; also helps match "cat" and "cats" in search.
- **Proper nouns** — NNP is a strong signal a word is part of a name, the first step toward NER.
- **Benchmarking** — the tagged Treebank became the standard test set for comparing taggers and parsers.

### How it's done

1. **Rule-based** — hand-written rules ("a word after 'the' is probably a noun").
2. **Hidden Markov Models** — the classic statistical method. Tags are Markov states (after a determiner, a noun is very likely), and each tag has a probability of emitting each word. Markov's vowels and consonants, with grammar categories as the states.
3. **Neural** — BERT-class models reach ~97–98% by using the whole sentence.

### What it's used for

- **Text-to-speech** — "a **record**" (noun) and "to re**cord**" (verb) are pronounced differently.
- **Information extraction** — find the nouns first.
- **Search** — "book a flight" is a request, not a request for books.
- **Grammar checking**.
- **Parsing** — tags are the first step toward sentence structure.

### Origin

Parts of speech are ancient — Dionysius Thrax described eight over 2,000 years ago. For NLP the turning point was the **Brown Corpus** (Brown University, 1960s; Henry Kučera and W. Nelson Francis) — a million words of English, eventually tagged, giving computers their first large labelled dataset. In 1988 Kenneth Church (Bell Labs) and Steven DeRose published statistical taggers trained on it. In the early 1990s Mitch Marcus's team at Penn built the **Penn Treebank**.

### Connection to skip-gram

Standard skip-gram sees only raw words — no tags. But it *picks up* part-of-speech information on its own: past-tense verbs cluster with past-tense verbs, adjectives with adjectives, purely from context. Research has tried feeding tags *into* word2vec (so noun-"book" and verb-"book" get separate vectors, which helps the bank problem a little), but the usual direction is the opposite: embeddings are fed into a tagger to improve it.

---

## 12. Named entity recognition (NER)

Finding **named entities** — specific, real-world things — in text and labelling their type.

> "Tim Cook visited Apple's office in New York on Monday."

| entity | type |
|---|---|
| Tim Cook | PERSON |
| Apple | ORGANIZATION |
| New York | LOCATION |
| Monday | DATE |

Two difficulties:

- **Ambiguity** — "Apple" (company or fruit), "Jordan" (person or country).
- **Span** — "New York" is one entity, not two words.

"Names" here is broader than people: organizations, places, dates, money amounts, events ("DivHacks 2026"). The test is *specific vs general* — "city" is not an entity, "New York" is. Most are proper nouns, which is why NNP tags help, but the technical term is **named entities**.

**Used for:** news analysis, search, customer-support ticket routing, resume screening.

### POS vs NER

| | question | coverage |
|---|---|---|
| **POS** | what grammatical role does each word play? | **every** word |
| **NER** | which spans are names, and of what type? | **only** names, possibly multi-word |

POS is grammar; NER is meaning.

---

## 13. Dependency parsing

Describing a sentence as links between words. Each link has a **head** (the boss) and a **dependent** (the word that attaches to it).

> "The fluffy cat purred loudly."

- **purred** — the main verb, so it's the **root**
- **cat** → head is *purred* (it's doing the purring)
- **fluffy** → head is *cat*
- **the** → head is *cat*
- **loudly** → head is *purred*

A quick test for which is head: the head is the word you can't drop. "The cat purred" works without "fluffy"; "the fluffy purred" doesn't work without "cat".

### The dependency tree

Draw all those links and you get a tree — a family tree, upside down. The root is at the top; every other word hangs below its head. Each word has **exactly one** head; a head may have many dependents.

### Relation labels

The head tells you *what* a word attaches to; the **label** tells you *how*.

> "The cat chased the mouse."

Both "cat" and "mouse" have *chased* as their head — but "cat" is the **subject** and "mouse" is the **object**. The head alone can't tell you who chased whom. The label can.

Common labels: subject, object, adjective modifier, determiner, adverb modifier.

### Why no cycles

Three rules: exactly one root; every other word has exactly one head; following heads upward from any word always reaches the root.

A cycle breaks the third. If "cat" depends on "purred" and "purred" depends on "cat", neither is in charge and you loop forever without reaching the top.

> In plain terms: every word needs a boss, and following the bosses has to end at the top. If two words were each other's boss, you'd go in circles. You can't be your own grandparent.

### The parser

The **parser** is the program; the **tree** is its output. (Printer and printed page.)

- **Transition-based** — reads left to right making quick attach decisions. Fast; similar in spirit to shift-reduce in compilers. (Joakim Nivre, early 2000s.)
- **Graph-based** — scores every possible link and picks the best overall tree. Slower, global view. (Ryan McDonald, ~2005.)

Danqi Chen and Christopher Manning made it neural in 2014. Tools: spaCy, Stanford Stanza.

The underlying grammar theory comes from **Lucien Tesnière**, a French linguist whose major work was published posthumously in 1959 — the verb at the centre, everything hanging off it. The **Universal Dependencies** project (Nivre, Manning and others) now applies one label set across 100+ languages.

### Scoring: UAS and LAS

- **UAS (unlabelled attachment score)** — percentage of words assigned the **correct head**. Labels ignored.
- **LAS (labelled attachment score)** — a word counts only if **head *and* label** are correct. Always ≤ UAS.

The words themselves are given; you aren't predicting words. For each word you ask one question: *did the parser attach it to the same head a human did?* 8 of 10 correct → UAS 80%.

**Why both:** in "the dog bit the man", if the parser swaps subject and object, every head is still right — UAS says 100% — but now the man bit the dog. LAS catches it: 3/5 = 60%.

> **Seating-chart analogy.** UAS asks whether each kid is at the right table. LAS asks whether they're at the right table *and* wearing the right name tag. Two kids at the right table with swapped tags pass UAS and fail LAS.

So a **high UAS with a low LAS** means the parser finds the right structure but confuses the roles. Good modern parsers reach ~95% UAS on English news text.

This is the same reason perplexity exists: one number, computed identically by everyone, that lets you compare systems fairly on the same test data. At 80% UAS, roughly one link in five is wrong, so anything built on top — question answering, fact extraction — inherits that error rate.

### What parse trees are for

Word order alone doesn't tell you who did what.

- **Question answering** — "Who founded Google?" needs the *subject* of "founded".
- **Information extraction** — company A acquired company B, without reversing the direction.
- **Sentiment** — in "the food was not good", the parser links "not" to "good".
- **Grammar checking** — in "the list of items are long", the subject is "list", not "items".
- **Translation** — languages order words differently; the tree guides the rearrangement.

---

## 14. Semantic role labelling (SRL)

Who did what to whom, with what, when and where.

> "Maria cut the bread with a knife yesterday."

| role | filler |
|---|---|
| Action (predicate) | cut |
| Agent (doer) | Maria |
| Patient (affected) | the bread |
| Instrument | a knife |
| Time | yesterday |

### SRL vs dependency parsing

Parsing is **grammar**; SRL is **meaning**.

- "Maria broke the window." → grammatical subject = Maria
- "The window broke." → grammatical subject = the window

The subject changed, but in both sentences the window is the thing that got broken. SRL labels it **patient** both times, because the meaning didn't change. That's precisely what grammar alone can't give you.

---

## 15. Where it all converges: Transformers

Everything above is a separate tool, and classic NLP systems chained them into a **pipeline**: tokenize → tag → parse → extract. Each stage's errors fed the next.

| tool | job | limitation |
|---|---|---|
| Bigram | predict next word | one word of memory; zeros |
| Smoothing / interpolation | remove zeros | doesn't fix memory or meaning |
| Skip-gram | learn word meaning | one fixed vector per word |
| POS tagging | grammar roles | a separate stage |
| NER | find names | a separate stage |
| Dependency parsing | who did what to whom | a separate stage |

The **Transformer** collapses the whole pipeline into one network. Introduced in the 2017 Google paper **"Attention Is All You Need"** — a different team and a different idea from word2vec, four years later. It's the **T** in GPT and in BERT.

### The problem it solves

Recall the bigram's short memory and skip-gram's single blurred vector for "bank". The transformer fixes both with one mechanism.

### Attention, in plain terms

For each word, the model asks: **which other words in this sentence should I pay attention to in order to understand this one?** Then it builds that word's representation as a weighted blend of the others.

In "the cats that live next door **are** loud", the word "are" attends strongly to "cats" — even though six words separate them. No fixed window, no one-step memory. That's the bigram's biggest weakness gone.

### Contextual embeddings

Because each word's representation is built from the words actually around it:

- "river **bank**" and "**bank** account" get **different vectors**

This is the direct fix to skip-gram's one-vector-per-word limitation. Word2vec embeddings are **static**; transformer embeddings are **contextual**.

### Query, key, value

Attention weights are computed from three vectors derived from each word's embedding:

- **Query** — what this word is looking for
- **Key** — what each other word offers
- **Value** — what each other word actually contributes

The match between a query and a key is scored with a **dot product** — the same operation from the cosine similarity formula in §9. Higher dot product → more attention. Those scores are normalized into weights that sum to 1 (a softmax), and the output is the weighted sum of the values.

Note the pattern: it's a weighted blend of multiple sources, weights summing to 1 — structurally the same shape as **interpolation** in §6, except the weights are learned and computed per word rather than fixed.

### Multi-head attention

Run several attention mechanisms in parallel, each learning to look for different things — one head may track subject–verb agreement, another may track what a pronoun refers to. These end up resembling the links a **dependency parser** produces, except nobody specified them; they emerge from training.

This is the recurring theme of the whole document: structure that used to be hand-built becomes something the model discovers from context on its own.

### Positional encoding

Attention looks at all words simultaneously, so by itself it has no notion of order — "dog bites man" and "man bites dog" would look identical. Position information is therefore added to each token's embedding explicitly.

### Two families

- **BERT** (encoder) — sees the whole sentence in both directions. Trained by masking random words and predicting them — the same fill-in-the-blank self-supervision as skip-gram, scaled up and made contextual. Good for tagging, NER, classification.
- **GPT** (decoder) — sees only words to the left, predicting the next token. That is *exactly the bigram's task*, with the entire preceding context instead of one word.

### What hasn't changed

- Training still minimizes **cross-entropy** (§7) — average surprise per token.
- Evaluation still reports **perplexity** (§7).
- Input is still **subword tokens**, usually BPE or a close relative (§2).
- Meaning is still learned from **context** (§10).
- Similarity is still measured with the **dot product** and **cosine** (§9).

The transformer didn't replace these ideas. It scaled them up and stopped doing POS tagging, NER, and parsing as separate steps — the model learns that structure internally.

### Where the old tools still live

Explicit taggers and parsers remain in use where you need **exact, inspectable, explainable** output rather than a black box — and n-grams and skip-gram remain in use where cheap and fast matters more than best.

---

## Quick reference

| Concept | One line |
|---|---|
| Morpheme | Smallest meaning-carrying piece of a word |
| BPE | Merge the most frequent adjacent pair; save the merge list; replay it |
| Tokens vs types | Total count vs distinct count |
| Bigram | P(w₂\|w₁) = count(w₁w₂) / count(w₁) |
| Markov property | Next state depends only on the current state |
| Add-one smoothing | (count + 1) / (total + V) — never zero, but too strong on large V |
| Interpolation | Weighted blend of n-gram orders; weights sum to 1 |
| Cross-entropy | Average −log₂(probability of the true word) per token |
| Perplexity | 2^(cross-entropy) — "as unsure as choosing among N words" |
| One-hot | All zeros, one 1 — no meaning |
| Embedding | Short dense learned vector — similar contexts → similar vectors |
| Dot product | Σ(aᵢ × bᵢ) — agreement, but length-sensitive |
| Cosine similarity | Dot product ÷ (both lengths) — range −1 to 1 |
| Skip-gram | Target word predicts its neighbours; keep the vectors, discard the rest |
| CBOW | Neighbours predict the target word |
| POS tagging | Grammatical category for every word (~45 Penn Treebank tags) |
| NER | Find named entities and their types |
| Head / dependent | Which word a word attaches to, and the attaching word |
| UAS / LAS | % correct heads / % correct heads **and** labels |
| SRL | Agent, patient, instrument, time — meaning roles |
| Attention | Each word attends to the words that help explain it |
| Transformer | Contextual embeddings + attention; replaces the classic pipeline |

---

## Timeline

| year | event |
|---|---|
| 1913 | Markov counts vowels/consonants in *Eugene Onegin* |
| 1948 | Shannon: entropy, and n-gram text generation |
| 1959 | Tesnière's dependency grammar (posthumous) |
| 1960s–70s | Salton's SMART system — cosine similarity for documents |
| 1960s | Brown Corpus |
| 1977 | Jelinek's IBM group introduces perplexity |
| 1988 | Church and DeRose: statistical POS taggers |
| 1990 | Latent Semantic Analysis |
| early 1990s | Penn Treebank |
| 2002 | Graham's "A Plan for Spam" — Bayesian filtering goes mainstream |
| 2003 | Bengio's neural language model |
| early 2000s | Nivre's transition-based parser |
| 2005 | McDonald's graph-based parser |
| 2013 | word2vec — skip-gram and CBOW |
| 2014 | Chen and Manning: neural dependency parsing |
| 2017 | "Attention Is All You Need" — the Transformer |
| 2018+ | BERT, GPT and successors |
