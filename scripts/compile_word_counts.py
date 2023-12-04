import argparse
import json
import re
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

PONY_NAMES = ['twilight sparkle', 'applejack', 'rarity', 'pinkie pie', 'rainbow dash', 'fluttershy']
STOPWORDS_FILEPATH = Path(__file__).parent.parent / "data" / "stopwords.txt"
THRESHOLD_FREQUENCY = 5


def load_stopwords() -> set:
    with open(STOPWORDS_FILEPATH) as file:
        stopwords = [line.rstrip() for line in file]
    
    return set(stopwords)


def remove_punctuation(dialog: str) -> str:
    # Remove stage queues like "[sigh]", and contractions like "I've".
    dialog = re.sub("(\[.*?\])|(\w*'\w*)", '', dialog)
    # Replace punctuation with a space.
    chars_to_replace = '()[],-.?!:;#&'
    for c in chars_to_replace:
        if c in dialog:
            dialog = dialog.replace(c, ' ')
    
    return dialog


def count_word_freq_per_pony(csv_filepath) -> dict[str, dict[str, int]]:
    df = pd.read_csv(csv_filepath, encoding='utf-8')
    stopwords = load_stopwords()
    word_freq_by_pony = {}

    df.reset_index()
    for index, row in df.iterrows():
        pony = row['pony'].lower()
        if pony not in PONY_NAMES:
            continue
        
        if pony not in word_freq_by_pony:
            word_freq_by_pony[pony] = defaultdict(int)

        dialog = remove_punctuation(row['dialog'])
        words = dialog.lower().split()
        dialog_word_freq = Counter(words).most_common()
        
        for word, freq in dialog_word_freq:
            if word not in stopwords:
                word_freq_by_pony[pony][word] += freq

    # Remove words with frequencies below the threshold.
    for pony_word_freq in word_freq_by_pony.values():
        for word, freq in dict(pony_word_freq).items():
            if freq < THRESHOLD_FREQUENCY:
                del pony_word_freq[word]
    
    return word_freq_by_pony


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True, help="The path to the output file.")
    parser.add_argument("-d", "--dialog", required=True, help="The path to the dialog csv file.")
    args = parser.parse_args()
    
    word_frequencies = count_word_freq_per_pony(args.dialog)
    
    output_file = Path(args.output)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(word_frequencies, file, indent=4)


if __name__ == '__main__':
    main()
