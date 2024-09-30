"""
Performs preprocessing on the full text of books included in BookSORT. Mainly consists of identifying chapter
boundaries and chapter titles.

This preprocessing script takes the full text files of the books, provided in `../data/pg/full_text/`, and outputs
two numpy files: (1) {book_id}_words.npy and (2) {book_id}_chapter_info.npy. These depend on the user to define an
initial `ch_dict` metadata dictionary in the script. More details on this dictionary are given below. Here we have
fully defined the metadata dictionary for all the books in BookSORT.

Outputs of this script
(1) {book_id}_words.npy: a cleaned numpy array of the words in the full text, including chapter titles but excluding
                         front and back matter in the book such as the table of contents, author's note, etc.
(2) {book_id}_chapter_info.npy: a metadata dictionary containing information about the chapters in the book. this
                                allows us to generate the BookSORT samples excluding the chapter titles.
"""
import argparse
import numpy as np
import os
import re


""" Below is the user-defined `ch_dict`, a chapter metadata dictionary for all books in BookSORT.
Each book is indexed by its Project Gutenberg ID.
To add a new book, you will need to provide this information.

Below is a description of each key.

n_chapters: number of chapters in the book
split_str: the string that should be used to split the full text into chapters. Due to slight differences in PG book 
           formatting, this may not always be the same. However, it's typically 4 newlines.
inds: after splitting on the `split_str`, the output is a list. these values are indices to that list that represent 
      the first and last chapter of the full text to keep. i.e. split_list[inds[0]] yields the first chapter, 
      and split_list[inds[1]] yields the last chapter. This allows you to remove the extraneous front and back matter in 
      the book, such as the table of contents, author's notes, etc.
parse_end_str: after splitting the book into chapters, the last chapter may still have text at the end that needs to be 
               removed (e.g. publisher information). This string and everything beyond it will be stripped from the 
               last chapter.
chapter_titles: the titles for each chapter. These are stripped from the full book text in BookSORT.
"""
ch_dict = {
           '72963': {'inds': [4, 22], 'n_chapters': 16, 'split_str': "\n\n\n\n\n", 'parse_end_str': None},
           '72869': {'inds': [2, 22], 'n_chapters': 20, 'split_str': "\n\n\n\n", 'parse_end_str': 'The End'},
           '72578': {'inds': [5, 30], 'n_chapters': 25, 'split_str': "\n\n\n\n", 'parse_end_str': 'THE END'},
           '72600': {'inds': [13, 31], 'n_chapters': 18, 'split_str': "\n\n\n", 'parse_end_str': None},
           '72958': {'inds': [5, 29], 'n_chapters': 24, 'split_str': "\n\n\n\n", 'parse_end_str': 'THE END'},
           '73042': {'inds': [1, 57], 'n_chapters': 56, 'split_str': "CHAPTER ", 'parse_end_str': 'THE END'},
           '72972': {'inds': [3, 18], 'n_chapters': 15, 'split_str': "\n\n\n\n", 'parse_end_str': 'THE END'},
           '73017': {'inds': [5, 6], 'n_chapters': 1, 'split_str': "\n\n\n\n", 'parse_end_str': '\*       \*       \*       \*       \*'},
           '69087': {'inds': [2, 30], 'n_chapters': 27, 'split_str': "CHAPTER", 'parse_end_str': 'THE END'},
          }
ch_dict['72963']['chapter_titles'] = ['The Downfall of Classical Physics', 'Relativity', 'Time', 'The Running-Down of the Universe', '“Becoming”', 'Gravitation—the Law', 'Gravitation—the Explanation', 'Man’s Place in the Universe', 'The Quantum Theory', 'The New Quantum Theory', 'World Building', 'Pointer Readings', 'Reality', 'Causation', 'Science and Mysticism', 'Conclusion']
ch_dict['72578']['chapter_titles'] = ['I BLASTING FIRE', 'II NED DISAPPEARS', 'III SUSPICIONS', 'IV A STRANGE MESSAGE', 'V ON A MYSTERIOUS TRAIL', 'VI TOO LATE', 'VII A WILD CHASE', 'VIII TWO CAPTIVES', 'IX ON THE ISLAND', 'X THE ESCAPE', 'XI RESCUED', 'XII GREENBAUM THREATENS', 'XIII MR DAMON DANCES', 'XIV KOKU IS DRUGGED', 'XV A SINISTER WARNING', 'XVI A STARTLING DISCOVERY', 'XVII USELESS PLEADINGS', 'XVIII AN ANONYMOUS ADVERTISEMENT', 'XIX THE MEETING', 'XX MASKED MEN', 'XXI A TEMPTING OFFER', 'XXII FLASHING LIGHTS', 'XXIII TOM ACCEPTS', 'XXIV A FINAL TEST', 'XXV A BRIGHT FUTURE']
ch_dict['72600']['chapter_titles'] = ['The Broken Note', 'I The Man Who Wouldn’t Sell His Pumpkin', 'II Krakow', 'III The Alchemist', 'IV The Good Jan Kanty', 'V In the Street of the Pigeons', 'VI The Tower of the Trumpeter', 'VII In the Alchemist’s Loft', 'VIII Peter of the Button Face', 'IX Button-Face Peter Attacks', 'X The Evil One Takes a Hand', 'XI The Attack on the Church', 'XII Elzbietka Misses the Broken Note', 'XIII The Great Tarnov Crystal', 'XIV A Great Fire Rages', 'XV King Kazimir Jagiello', 'XVI The Last of the Great Tarnov Crystal', 'Epilogue The Broken Note']
ch_dict['73042']['chapter_titles'] = None
ch_dict['72972']['chapter_titles'] = None
ch_dict['73017']['chapter_titles'] = None
ch_dict['72958']['chapter_titles'] = ['I IN THE STORM', 'II A CALL FOR HELP', 'III JADBURY WILSON', 'IV A TALE OF THE WEST', 'V CON RILEY UNDER FIRE', 'VI A MESSAGE FROM MONTANA', 'VII IN THE WINDY CITY', 'VIII THE SECOND STRANGER', 'IX THE ESCAPE', 'X ON GUARD', "XI FENTON HARDY'S STORY", 'XII THE CAVE-IN', 'XIII IN THE DEPTHS OF THE EARTH', 'XIV ATTACKED BY THE OUTLAWS', 'XV THE TRAP', 'XVI INFORMATION', "XVII THE OUTLAW'S NOTEBOOK", 'XVIII THE BLIZZARD', 'XIX THE LONE TREE', 'XX DOWN THE SHAFT', 'XXI UNDERGROUND', 'XXII BLACK PEPPER', 'XXIII THE CAPTURE', 'XXIV BART DAWSON EXPLAINS']
ch_dict['72869']['chapter_titles'] = ['I The Pill Box', 'II The Naturalist', 'III A Little Melodrama', 'IV A Social Evening', 'V Aunt Agatha is Upset', 'VI The Kindness of the Tiger', 'VII The Fun Continues', 'VIII The Saint is Dense', 'IX Patricia Perseveres', 'X The Old House', 'XI Carn Listens In', 'XII Tea with Lapping', 'XIII The Brand', 'XIV Captain Patricia', 'XV Spurs for Algy', 'XVI In the Swim', 'XVII Piracy', 'XVIII The Saint Returns', 'XIX The Tiger', 'XX The Last Laugh']
ch_dict['69087']['chapter_titles'] = ['Dr. Sheppard at the Breakfast Table', 'Who’s Who in King’s Abbot', 'The Man Who Grew Vegetable Marrows', 'Dinner at Fernly', 'Murder', 'The Tunisian Dagger', 'I Learn My Neighbour’s Profession', 'Inspector Raglan is Confident', 'The Goldfish Pond', 'The Parlormaid', 'Poirot Pays a Call', 'Round the Table', 'The Goose Quill', 'Mrs. Ackroyd', 'Geoffrey Raymond', 'An Evening at Mah Jong', 'Parker', 'Charles Kent', 'Flora Ackroyd', 'Miss Russell', 'The Paragraph in the Paper', 'Ursula’s Story', 'Poirot’s Little Reunion', 'Ralph Paton’s Story', 'The Whole Truth', 'And Nothing But The Truth', 'Apologia']


# 5 new line split, 2:22
def split_book_into_chapters(book_text, ch_info, book_id):
    split_text = book_text.split(ch_info['split_str'])
    chapter_text = split_text[ch_info['inds'][0]:ch_info['inds'][1]]
    chapter_text = [ch.strip() for ch in chapter_text]
    if ch_info['chapter_titles'] and (len(chapter_text) > ch_info['n_chapters']):
        new_chapter_text = []
        if book_id == '72963':
            # This book has a formatting irregularity between Chapter I/II, so split those
            ch0 = chapter_text[0]
            ch_list = ch0.split('\n\n\n')
            ch0 = ch_list[0] + '\n' + ch_list[1]
            ch1 = ch_list[2][1:] + '\n' + ch_list[3] + ch_list[4] + ch_list[5]
            chapter_text[0] = ch0
            chapter_text.insert(1, ch1)
        j = 0
        # May need to merge some chapters together
        for i, title in enumerate(ch_info['chapter_titles']):
            this_chapter = chapter_text[j]
            m = re.search(f'{title}', chapter_text[j], re.IGNORECASE)
            while m is None:
                j += 1
                m = re.search(f'{title}\n', chapter_text[j], re.IGNORECASE)
                new_chapter_text[-1] += this_chapter
                this_chapter = chapter_text[j]
            new_chapter_text.append(this_chapter)
            j += 1
        chapter_text = new_chapter_text
        assert len(new_chapter_text) == ch_info['n_chapters'], "Chapter parsing does not yield expected number of chapters"
    if ch_info['parse_end_str']:
        # Cut off the meta-text, notes, etc. at the end of the book
        last_chapter = chapter_text[-1]
        m = re.search(f"{ch_info['parse_end_str']}", last_chapter)
        chapter_text[-1] = last_chapter[:m.span()[0]].strip()
    assert len(chapter_text) == ch_info['n_chapters'], "Chapter parsing does not yield expect number of chapters"
    return chapter_text


def save_book_data(book_path, text_path, ch_dict=ch_dict):
    for book_id, bdict in ch_dict.items():
        print(f"Parsing {book_id}")
        # Read in the full text of the book
        path = f'{book_path}/{book_id}-0.txt'
        with open(path, 'r') as fp:
            tmp = fp.read()
        # Strip out the title from the book and save
        search_obj = re.search('EBOOK(?P<title>.*)\*\*\*', tmp)
        title = search_obj.group('title').strip()
        ch_dict[book_id]['book_title'] = title
        if bdict['chapter_titles']:
            # Sanity check of the chapter metadata
            assert len(bdict['chapter_titles']) == bdict['n_chapters'], \
                f"Chapter metadata is wrong, titles gave {len(bdict['chapter_titles'])} but expected {bdict['n_chapters']}"
        # Split the book into chapters and remove extraneous information
        clean_book_text = split_book_into_chapters(tmp, bdict, book_id)
        chapters = [np.array(x.split()) for x in clean_book_text]
        print("printing beginning of first and last chapter for sanity check:\n", chapters[0][:50], chapters[-1][-50:])
        # Get the length of each chapter
        ch_len = [ch.shape[0] for ch in chapters]
        ch_dict[book_id]['chapter_inds'] = ch_len
        clean_book_text = "\n".join(clean_book_text)
        # Compute the number of words by splitting on whitespaces. Includes chapter titles in word count.
        tmp = clean_book_text.split()
        ch_dict[book_id]['num_words'] = len(tmp)
        assert sum(ch_len) == len(tmp), "Chapter lengths don't sum to book length!"
        book_words = np.array(tmp)
        # Save the outputs of preprocessing
        os.makedirs(text_path, exist_ok=True)
        np.save(f'{text_path}/{book_id}_words.npy', book_words, allow_pickle=True)
        np.save(f'{text_path}/{book_id}_chapter_info.npy', ch_dict[book_id], allow_pickle=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_text_path', type=str, default='../../data/pg/full_text')
    parser.add_argument('--output_path', type=str, default='../../data/pg/text_arrays')
    args = parser.parse_args()

    save_book_data(book_path=args.full_text_path, text_path=args.output_path)
