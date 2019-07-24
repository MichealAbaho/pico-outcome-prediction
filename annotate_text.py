import os
import pandas as pd
from nltk.tokenize import word_tokenize
import ebm_nlp_demo as e
import re
from glob import glob
import correcting_spans
import matplotlib.pyplot as plt
import pprint
import string
import timeit
import time
from multiprocessing import Process, Pool
from pprint import pprint

import numpy as np
from numba import jit
import numba
import spacy

from geniatagger import GeniaTagger

pd.options.display.max_rows = 500000

x = e.PHASES
y = e.LABEL_DECODERS
core_outcome = y[x[1]]['outcomes']
print(core_outcome)


def annotate_text(tager=''):
    genia = GeniaTagger('../genia-tagger/geniatagger-3.0.2/geniatagger')
    medpost = spacy.load(os.path.abspath('trained_tagger'))
    main_dir = 'corrected_outcomes'
    data_dir = os.path.abspath(os.path.join(main_dir, 'aggregated'))
    if not os.path.exists(os.path.dirname(data_dir)):
        os.makedirs(os.path.dirname(data_dir))

    sub_dir = os.path.abspath(os.path.join(data_dir, 'test'))
    if not os.path.exists(os.path.dirname(sub_dir)):
        os.makedirs(os.path.dirname(sub_dir))

    turker, ebm_extract = e.read_anns('hierarchical_labels', 'outcomes', \
                                      ann_type='aggregated', model_phase='train')
    t1 = time.time()

    seq_dir = os.path.abspath(os.path.join(os.path.curdir, 'corrected_outcomes', 'BIO-data'))
    if not os.path.exists(seq_dir):
        os.makedirs(seq_dir)

    with open(os.path.join(seq_dir, 'train.bmes'), 'w') as f:
        for pmid, doc in ebm_extract.items():
            abstract = ' '.join(i for i in doc.tokens)
            pprint(abstract)
            u = doc.anns['AGGREGATED']
            v = doc.tokens
            o = []
            corr_outcomes = []
            temp, temp_2 = [], []
            t = 0
            m = 0
            o_come = e.print_labeled_spans_2(doc)[0] #extract outcomes from the abstract being examined, [(Outcome-type, Outcome), (Outcome-type, Outcome2)]

            #store the annotations and the index of the annotations for each abstract
            for x in range(len(u)):
                if x == t:
                    if u[x] != 0:
                        for ff in o_come:
                            for j in range(len(u)):
                                if j < len(ff[1].split()):
                                    o.append((t, u[x]))
                                    t += 1
                            break
                        o_come.pop(0)

                        txt_toks = [v[i[0]] for i in o]
                        text_wrds = ' '.join(i for i in txt_toks)
                        if tager.lower() == 'genia':
                            tagged = genia.parse(text_wrds)
                            pos = [i[2] for i in tagged]
                        elif tager.lower() == 'medpost':
                            tagged = medpost(text_wrds)
                            pos = [i.tag_ for i in tagged]

                        text_pos = ' '.join(i for i in pos)

                        label = core_outcome[u[x]]

                        corr = correcting_spans.correct_text()
                        corrected_spans = corr.pos_co_occurrence_cleaning(text_wrds, text_pos, label)

                        if len(corrected_spans) == 0:
                            v[o[0][0]:(o[-1][0] + 1)] = txt_toks
                            u[o[0][0]:(o[-1][0] + 1)] = [0 for i in range(len(txt_toks))]
                        elif len(corrected_spans) < 2:
                            span = corrected_spans[0]
                            s = [i for i in span[1].split()]
                            ll = [o[0][1] if i in s else 0 for i in txt_toks]
                            v[o[0][0]:(o[-1][0] + 1)] = txt_toks
                            u[o[0][0]:(o[-1][0] + 1)] = ll
                        else:
                            s = [i for j in corrected_spans for i in j[1].split()]
                            ll = [o[0][1] if i in s else 0 for i in txt_toks]
                            v[o[0][0]:(o[-1][0] + 1)] = txt_toks
                            u[o[0][0]:(o[-1][0] + 1)] = ll

                        p = [i for i in corrected_spans]
                        if len(p) > 0:
                            for i in p:
                                corr_outcomes.append(i)
                        o.clear()

                    else:
                        t += 1
            if corr_outcomes:
                #print(corr_outcomes)
                temp_2 = build_sequence_model(v, u, core_outcome, corr_outcomes)
                for i in temp_2:
                    f.write('{}\n'.format(i))
                f.write('\n')
        f.close()
    t2 = time.time()

#BIO tagging function
def build_sequence_model(tokens, anns, cos, corr_outcomes):
    b = 0
    temp = []
    temp_2 = []
    for i, j in zip(range(len(tokens)), range(len(anns))):
        if i == b:
            if anns[j] != 0:
                tokens_2, anns_2 = tokens[i:], anns[j:]
                for n in corr_outcomes:
                    for k, v in zip(range(len(tokens_2)), range(len(anns_2))):
                        if k < len(n[1].split()):
                            temp.append((tokens_2[k], anns_2[v]))
                            b += 1
                    break

                if temp:
                    t_temp = [i for i in temp]
                    if len(t_temp) == 1:
                        temp_2.append('{} B-{}'.format(t_temp[0][0], label_tag(core_outcome, t_temp[0][1])))
                    elif len(t_temp) == 2:
                        temp_2.append('{} B-{}'.format(t_temp[0][0], label_tag(core_outcome, t_temp[0][1])))
                        temp_2.append('{} I-{}'.format(t_temp[1][0], label_tag(core_outcome, t_temp[1][1])))
                    else:
                        for h in range(len(t_temp)):
                            if h == 0:
                                temp_2.append('{} B-{}'.format(t_temp[h][0], label_tag(core_outcome, t_temp[h][1])))
                            else:
                                temp_2.append('{} I-{}'.format(t_temp[h][0], label_tag(core_outcome, t_temp[h][1])))

                temp.clear()
                if corr_outcomes:
                    del corr_outcomes[0]

            else:
                temp_2.append('{} {}'.format(tokens[i], "O"))
                b += 1
    return temp_2

def label_tag(d, x):
    w = "O"
    if x != 0:
        w = d[x].upper()
    return w


if __name__=='__main__':
    annotate_text(tager='genia')
    #print(timeit.timeit("xml_wrapper()", setup="from __main__ import xml_wrapper"))
