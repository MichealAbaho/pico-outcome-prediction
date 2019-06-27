# -*- coding: utf-8 -*-
# @Author: micheala
# @Created: 21/06/19 
# @Contact: michealabaho265@gmail.com
import os
from glob import glob
import pprint
import requests as req
from bs4 import BeautifulSoup
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import customize_spacy
import time
import numpy as np
from multiprocessing import Pool, Process, cpu_count

#creating a tag map dictionary that specifies tbe morphological features for the tags, from the universal scheme PENNBANK TREE
TAG_MAP = {'II': {'pos': 'ADP'}, 'NN': {'pos': 'NOUN'}, ',': {'pos': 'PUNCT'},
           'NNS': {'pos': 'NOUN'}, 'VBD': {'pos': 'VERB'}, 'JJ': {'pos': 'ADJ'},
           'CC': {'pos': 'CONJ'}, 'DD': {'pos': 'DET'}, 'VVD': {'pos': 'VERB'},
           'RR': {'pos': 'ADV'}, '.': {'pos': 'PUNCT'}, 'VVB': {'pos': 'VERB'},
           'CST': {'pos': 'ADP'}, 'VBZ': {'pos': 'VERB'}, 'MC': {'pos': 'NUM'},
           '(': {'pos': 'PUNCT'}, ')': {'pos': 'PUNCT'}, 'VVN': {'pos': 'VERB'}, 'VM': {'pos': 'VERB'},
           'VVNJ': {'pos': 'VERB'}, 'SYM': {'pos': 'SYM'}, 'JJR': {'pos': 'ADJ'},
           'CSN': {'pos': 'ADP'}, 'CS': {'pos': 'ADP'}, 'PN': {'pos': 'PRON'}, 'VVGJ': {'pos': 'VERB'},
           'TO': {'pos': 'PART'}, 'VVI': {'pos': 'NOUN'}, 'VDD': {'pos': 'VERB'}, 'RR+': {'pos': 'ADP'},
           'VVGN': {'pos': 'VERB'}, ':': {'pos': 'PUNCT'}, 'PNG': {'pos': 'ADJ'}, 'VVG': {'pos': 'VERB'},
           'VBI': {'pos': 'VERB'}, 'VHD': {'pos': 'VERB'}, 'PNR': {'pos': 'ADP'}, 'PND': {'pos': 'DET'},
           'II+': {'pos': 'ADJ'}, 'VVZ': {'pos': 'NOUN'}, 'NNP': {'pos': 'PROPN'}, '``': {'pos': 'PUNCT'},
           "''": {'pos': 'PUNCT'}, 'VBB': {'pos': 'verb'}, 'GE': {'pos': 'VERB'}, 'VHZ': {'pos': 'VERB'},
           'VBN': {'pos': 'VERB'}, 'VDB': {'pos': 'VERB'}, 'RRR': {'pos': 'ADJ'}, 'VDN': {'pos': 'VERB'},
           'VHB': {'pos': 'VERB'}, 'VDZ': {'pos': 'VERB'}, 'VBG': {'pos': 'VERB'}, 'RRT': {'pos': 'ADJ'},
           'EX': {'pos': 'ADV'}, 'JJT': {'pos': 'ADV'}, 'JJ+': {'pos': 'ADP'}, 'CC+': {'pos': 'VERB'},
           'DB': {'pos': 'DET'}, 'CS+': {'pos': 'ADJ'}, 'NN+': {'pos': 'VERB'}, 'VHI': {'pos': 'VERB'}, 'VHG': {'pos': 'VERB'}}

# """Create a new model, set up the pipeline and train the tagger. Train the tagger with a custom tag map defined above,
# i.e. creating a new language creating a new Language
#   """

def run_update(train_data, lang="en", output_dir=None, n_iter=25):
    nlp = spacy.blank(lang)
    #customize spacy tokenizer
    nlp.tokenizer = customize_spacy.customize_spacy(nlp.vocab)
    # add the tagger to the pipeline, nlp.create_pipe works for built-ins that are registered with spaCy
    tagger = nlp.create_pipe("tagger")
    # Add the tags. This needs to be done before you start training.
    for tag, values in TAG_MAP.items():
        tagger.add_label(tag, values)
    nlp.add_pipe(tagger)

    nlp.vocab.vectors.name = 'spacy_pretrained_vectors'
    optimizer = nlp.begin_training()

    for i in range(n_iter):
        shuffle_indices = np.random.permutation(len(train_data))
        train_data = [train_data[i] for i in shuffle_indices]
        losses = {}
        #batch up the examples using spaCy's minibatch
        batches = get_batches(train_data, 'tagger')
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, losses=losses)
        print("Losses", losses)

    #test the trained model
    test_text = "In addition , liver biopsy specimens"
    doc = nlp(test_text)
    print("Tags", [(t.text, t.tag_, t.pos_) for t in doc])

    # save the trained model
    outputdir = os.path.join(os.path.abspath(os.path.curdir), 'trained_tagger')
    if output_dir is None:
        if not outputdir.exists():
            os.makedirs(outputdir)
        nlp.to_disk(output_dir)


#batch generator using spacy mini batch
def get_batches(train_data, model_type):
    max_batch_sizes = {"tagger": 32, "parser": 16, "ner": 16, "textcat": 64}
    max_batch_size = max_batch_sizes[model_type]
    if len(train_data) < 1000:
        max_batch_size /= 2
    if len(train_data) < 500:
        max_batch_size /= 2
    batch_size = compounding(1, max_batch_size, 1.001)
    batches = minibatch(train_data, size=batch_size)
    return batches


def fetch_medpost_corpus(medpost_path):
    Med_tag_map = {}
    Med_train_data = []
    tag_list = []

    data_dir = glob(os.path.join(medpost_path, '*.ioc'))
    data_dir_files = [os.path.abspath(i) for i in data_dir]

    for i in data_dir_files:
        with open(i, 'r') as f:
            sentences = f.readlines()
            for i,sentence in enumerate(sentences):
                if i % 2 != 0:
                    word_list = sentence.split()
                    words_split = create_sentence_tags_tuple(word_list)
                    for k,v in words_split[1].items():
                        for value in v:
                            if value not in tag_list:
                                tag_list.append(value)

                    Med_train_data.append(words_split)

    universal_pos_tags = scrap_spacy_tags(scrap_source=['https://spacy.io/api/annotation#pos-tagging'])
    #y = [i for i in list(universal_pos_tags.keys()) if i not in tag_list]

    for tag in tag_list:
        for m,n in universal_pos_tags.items():
            pos = {}
            if tag.strip() == m.strip():
                pos['pos'] = n.strip()
                Med_tag_map[m] = pos
                break
            else:
                pass
        p = {}
        if tag not in Med_tag_map:
            p['pos'] = ''
            Med_tag_map[tag] = p

    return Med_train_data

#create a tuple to match the default input required by spacy TAG MAP
def create_sentence_tags_tuple(words_list):
    words = []
    tags = {}
    tags['tags'] = []
    for item in words_list:
        w,t = item.rsplit('_', 1)
        words.append(w.strip())
        tags['tags'].append(t)

    sent = ' '.join(i for i in words)
    return sent, tags

#fetch spacy morphological terms from the site directly
def scrap_spacy_tags(scrap_source = []):
    spacy_universal_tags = {}
    scrapped_terms = []
    for url in scrap_source:
        content = req.get(url, stream=True)
        if content.status_code:
            if content.headers['Content-Type'].lower().find('html'):
                needed_content = BeautifulSoup(content.content, 'lxml')
                english_table = needed_content.find_all('table')[1]
                for i,j in enumerate(english_table.find_all('tr')):
                    g = j.find_all('td')
                    u = 0
                    for b in range(len(g)):
                        if b == u:
                            spacy_universal_tags[g[u].text] = g[u+1].text
                            u += 4
                        else:
                            pass
    return spacy_universal_tags


if __name__ == "__main__":
    pool = Pool(cpu_count())
    start = time.time()
    #path to the medpost corpus
    locate_medpost_corpus = '../Medpost/medtag/medpost'
    TRAIN_DATA = fetch_medpost_corpus(locate_medpost_corpus)
    d = Process(target=run_update, args=([TRAIN_DATA]))
    d.start()
    d.join()
    print('Time spent {}/s'.format(time.time() - start))

