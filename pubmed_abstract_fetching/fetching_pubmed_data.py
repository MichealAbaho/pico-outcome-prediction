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
from shutil import make_archive

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
            try:
                db_precendence[j['DbName']] = int(j['Count'])
            except Exception as e:
                print(e)

        db_precendence = dict(sorted(db_precendence.items(), key=lambda x: x[1], reverse=True)[:5])
        return db_precendence
        
    def fetch_data(self, db_dict):
        attempts = 1
        file_name = 'all_abstracts_pubdate'
        for k,v in db_dict.items():
            if k == 'pubmed':
                handle = Entrez.esearch(db=k, **self.search_terms)
                search_result = Entrez.read(handle)
                relevant_id_list = search_result['IdList']
                if relevant_id_list:
                    try:
                        fetch_query = Entrez.efetch(db=k, id=relevant_id_list, webenv=search_result['WebEnv'], query_key=search_result['QueryKey'], **self.fetch_terms)
                        fetch_result = Medline.parse(fetch_query)
                        with open(file_name, 'w') as abs_pbmd:
                            json.dump(list(fetch_result), abs_pbmd, indent=2, sort_keys=True)
                        abs_pbmd.close()
                                #json.dump(record, abs_pbmd, indent=2)
                    except HTTPError as e:
                        if 500 <= e.code <= 599:
                            print("Received error from server %s"%e)
                            print("Attempt %i of 5"%attempts)
                            attempts += 1
                            time.sleep(15)
                        else:
                            raise
        return file_name

def sort_retrieved_abstracts(json_file):
    d,n = [],[]
    dir_ = os.path.join(os.path.abspath(os.path.curdir), 'new_batch')
    if not os.path.exists(dir_):
        os.makedirs(dir_)

    current_dir_files = os.listdir('.')
    if json_file in current_dir_files:
        with open(json_file, 'r') as tg_json:
            ab_bs = json.load(tg_json)
            for entry in ab_bs:
                file = open(os.path.join(dir_, entry['PMID']+'_PD.txt'), 'w')
                file.writelines('PMID: {}\nTitle: {}\nPublication Type: {}\nJournal-Name: {}\nJournal ID: {}\nPublication date: {}\n\n{}'.format(
                                entry['PMID'],
                                entry['TI'],
                                ' '.join([i+',' for i in entry['PT']]),
                                entry['JT'],
                                entry['JID'],
                                entry["CRDT"],
                                entry['AB']

                ))
                file.close()
            tg_json.close()

    #send folder to archive
    make_archive('extra_abstracts', 'zip', dir_)


            
            
if __name__ == "__main__":
    query = 'randomized controlled trial [FILT]' #[FILT] limits the search inpubmed to
    key_wrds_search = {'term':query,
                       'retmax':20,
                       'retmode':'xml',
                       'usehistory':'y',
                       'sort':'pub+date'}
    key_wrds_fetch = {'retmode':'json',
                      'rettype':'medline'}

    ftc_data = fetch_data(query_term=query, key_search_wrds=key_wrds_search, key_fetch_terms=key_wrds_fetch)
    db_dict = ftc_data.db_with_most_records_for_query()
    file_name = ftc_data.fetch_data(db_dict=db_dict)
    sort_retrieved_abstracts(file_name)


