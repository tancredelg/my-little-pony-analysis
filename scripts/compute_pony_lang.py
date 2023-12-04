import argparse
import json
import math
from collections import Counter


# tf-idf(word, pony, script) = tf(word, pony) * idf(word, script)
#   tf(word, pony) = the number of times pony uses `word`
#   idf(word, script) = log[(total number of ponies) / (number of ponies that use `word`)]
def tfidf(word: str, pony: str, script: dict[str, dict[str, int]]) -> float:
    tf = script[pony].get(word, 0)
    
    # Collect all the ponies that used `word` using their word frequency dictionaries.
    ponies_that_used_word = []
    for p, p_word_freq in script.items():
        if word in p_word_freq.keys():
            ponies_that_used_word.append(p)
    
    idf = math.log(len(script) / len(ponies_that_used_word))
    return tf * idf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--pony-counts", required=True,
                        help="The path to the json file containing the word frequency for each pony.")
    parser.add_argument("-n", "--num-words", required=True, type=int,
                        help="The number of words by highest TF-IDF score to output for each pony.")
    args = parser.parse_args()
    
    # Load word frequency json file back into a nested dictionary. 
    with open(args.pony_counts, 'r') as file:
        word_freq_by_pony: dict[str, dict[str, int]] = json.load(file)     
        
    # Compute tfidf scores for all words and then store top N words in output dictionary.
    top_tfidf_words_by_pony: dict[str, list[str]] = {}
    for pony, pony_word_freq in word_freq_by_pony.items():
        tfidf_scores: dict[str, float] = {}
        for word in pony_word_freq.keys():
            tfidf_scores[word] = tfidf(word, pony, word_freq_by_pony)
        
        top_tfidf_words = Counter(tfidf_scores).most_common(args.num_words)
        top_tfidf_words_by_pony[pony] = [w for w, score in top_tfidf_words]
    
    print(json.dumps(top_tfidf_words_by_pony, indent=4))


if __name__ == '__main__':
    main()
