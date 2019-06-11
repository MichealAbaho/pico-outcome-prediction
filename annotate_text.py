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

import numpy as np
from numba import jit
import numba

from geniatagger import GeniaTagger

pd.options.display.max_rows = 500000

x = e.PHASES
y = e.LABEL_DECODERS
core_outcome = y[x[1]]['outcomes']
print(core_outcome)


def xml_wrapper():
    tagger = GeniaTagger('../genia-tagger/geniatagger-3.0.2/geniatagger')
    main_dir = 'corrected_outcomes'
    data_dir = os.path.abspath(os.path.join(main_dir, 'aggregated'))
    if not os.path.exists(os.path.dirname(data_dir)):
        os.makedirs(os.path.dirname(data_dir))

    sub_dir = os.path.abspath(os.path.join(data_dir, 'test'))
    if not os.path.exists(os.path.dirname(sub_dir)):
        os.makedirs(os.path.dirname(sub_dir))

    turker, ebm_extract = e.read_anns('hierarchical_labels', 'outcomes', \
                                      ann_type='aggregated', model_phase='test/gold')
    t1 = time.time()
    qwe = open('x.txt', 'w')
    re = 1
    for pmid, doc in ebm_extract.items():
        abstract = ' '.join(i for i in doc.tokens)
        u = doc.anns['AGGREGATED']
        v = doc.tokens
        q = []
        o = []
        #print(u)
        #print(abstract)
        t = 0
        for x in range(len(u)):
            if x == t:
                if u[x] != 0:
                    o.append((x, u[x]))
                    t = x+1
                    u2 = u[t:]
                    for j in range(len(u2)):
                        if u2[j] == u[x]:
                            o.append((t, u[x]))
                            t += 1
                        else:
                            break
                    p = [i for i in o]
                    if len(p) > 0:
                        q.append(p)
                    o.clear()
                else:
                    t += 1

        for y in q:
            txt_toks = [v[i[0]] for i in y]
            text_wrds = ' '.join(i for i in txt_toks)
            tagged = tagger.parse(text_wrds)

            pos = [i[2] for i in tagged]
            text_pos = ' '.join(i for i in pos)

            label = core_outcome[y[0][1]]
            corr = correcting_spans.correct_text()
            corrected_spans = corr.pos_co_occurrence_cleaning(text_wrds, text_pos, label)

            if len(corrected_spans) == 0:
                v[y[0][0]:(y[-1][0] + 1)] = txt_toks
                u[y[0][0]:(y[-1][0] + 1)] = [0 for i in range(len(txt_toks))]
            elif len(corrected_spans) < 2:
                span = corrected_spans[0]
                s = [i for i in span[1].split()]
                ll = [y[0][1] if i in s else 0 for i in txt_toks]
                v[y[0][0]:(y[-1][0]+1)] = txt_toks
                u[y[0][0]:(y[-1][0]+1)] = ll
            else:
                s = [i for j in corrected_spans for i in j[1].split()]
                ll = [y[0][1] if i in s else 0 for i in txt_toks]
                v[y[0][0]:(y[-1][0] + 1)] = txt_toks
                u[y[0][0]:(y[-1][0] + 1)] = ll

        qwe.write('{}:{}\n'.format(re, abstract))
        for i,j in zip(v,u):
            qwe.write('{}:__:{}\n'.format(i,j))

        re += 1
    t2 = time.time()
    print('Time with Numba',(t2-t1),'secs')


def build_anns(tokens, labels):
    u = []
    for i in tokens:
        if not i.__contains__('___'):
            u.append(0)
        else:
            q = i.split('___')
            t = [a for a,b in labels.items() if b == q[0]]
            u.append(t[0])
    #u_str = ','.join(str(i) for i in u)
    return tokens, u

if __name__=='__main__':
    xml_wrapper()
