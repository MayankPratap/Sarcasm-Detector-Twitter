# Sarcasm-Detector-Twitter
Made for MLWARE 1 2017 by IIT BHU

Training Set contains over 90000 tweets with label "sarcastic" or "not-sarcastic".

accuracy_score = 0.875962559745

f1_score = 0.875851300621

Features used :-

1. Lexical Density Of Tweet :- Ratio Of count of Nouns,Verbs,Adjectives and Adverbs to all words.
2. Intensifiers :- A binary feature which tells whether a tweet contains a word in list of intensifier words. (https://en.wikipedia.org/wiki/Intensifier)
3. Sentiment Of Tweet
4. Maximum word sentiment score, minimum word sentiment score and diff between maximum word sentiment score and minimum word sentiment score.
5. Number of Words with initial Caps and number of words with all caps.


