# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:03:45 2019

@author: user
"""
import pandas as pd
from Bio import Entrez, Medline
Entrez.email = "michealabaho265@gmail.com"
Entrez.api_key = "f71be69d0dd7b2da28787942f337abae8108"
import pprint
import json
import os
import time
from urllib.error import HTTPError

class fetch_data:
    def __init__(self, query_term, key_search_wrds={}, key_fetch_terms={}):
        self.query = query_term
        self.search_terms = key_search_wrds
        self.fetch_terms = key_fetch_terms
    
    #checking which databases exist in the pubmed
    def db_info(self):
        handle = Entrez.einfo()
        db_list = Entrez.read(handle)
        return db_list['DbList']
     
    #checking which datatabases have got the most uneful     
    def db_with_most_records_for_query(self):
        handle = Entrez.egquery(term = self.query)
        record = Entrez.read(handle)
        db_precendence = dict()
        for i,j in enumerate(record['eGQueryResult']):
            db_precendence[j['DbName']] = int(j['Count'])
            
        db_precendence = dict(sorted(db_precendence.items(), key=lambda x: x[1], reverse=True)[:5])
        return db_precendence
        
    def fetch_data(self, db_dict):
        attempts = 1
        for k,v in db_dict.items():
            
            if k == 'pubmed':
                handle = Entrez.esearch(db=k, **self.search_terms)
                search_result = Entrez.read(handle)
                relevant_id_list = search_result['IdList']
                if relevant_id_list:
                    
                    try:
                        fetch_query = Entrez.efetch(db=k, id=relevant_id_list, webenv=search_result['WebEnv'], query_key=search_result['QueryKey'], **self.fetch_terms)
                        fetch_result = Medline.parse(fetch_query)
                        with open('all_abstracts_pubdate', 'w') as abs_pbmd:
                            json.dump(list(fetch_result), abs_pbmd, indent=2, sort_keys=True)
                           
                                #json.dump(record, abs_pbmd, indent=2)
                                
                    except HTTPError as e:
                        if 500 <= e.code <= 599:
                            print("Received error from server %s"%e)
                            print("Attempt %i of 5"%attempts)
                            attempts += 1
                            time.sleep(15)
                        else:
                            raise
    
def sort_retrieved_abstracts(json_file, json_file_2):
    main_dir = '..\\pico-outcome-prediction'
    abs_dirs = os.path.abspath(os.path.join(main_dir, 'rct_abstracts', 'relevance'))
    
    if not os.path.exists(abs_dirs):
        os.makedirs(abs_dirs)
    
    d,n = [],[]
    with open(json_file, 'r') as tg_json:
        ab_bs = json.load(tg_json)
        for entry in ab_bs:
            d.append(entry['PMID'])
            
    with open(json_file, 'r') as tg_json:
        ab_bs = json.load(tg_json)
        for entry in ab_bs:
            n.append(entry['PMID'])
    
    x = [i for i in d if i not in n]
    print(x)

            
            
if __name__ == "__main__":
    query = 'randomized controlled trial [FILT]' #[FILT] limits the search inpubmed to
    key_wrds_search = {'term':query,
                       'retmax':300,
                       'retmode':'xml',
                       'usehistory':'y',
                       'sort':'pub+date'}
    key_wrds_fetch = {'retmode':'json',
                      'rettype':'medline'}

    ftc_data = fetch_data(query_term=query, key_search_wrds=key_wrds_search, key_fetch_terms=key_wrds_fetch)
    db_dict = ftc_data.db_with_most_records_for_query()
    ftc_data.fetch_data(db_dict=db_dict)
    sort_retrieved_abstracts('all_abstracts', 'all_abstracts_pubdate')


