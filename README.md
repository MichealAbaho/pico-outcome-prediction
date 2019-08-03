# pico-outcome-prediction
Detecting outcomes from clinical records for evidence based practice.

The goal of the project is to identify and extract outcome phrases from medical text i.e. spans of text designated to be health outcomes as described by the Comet and Cochrnae Institutions. Outcomes are essentially a diagnosis measured by clinical researchers during Clinical Trials. In Evidence based care or practice, clinicians and biomedics assess the impact new interventions have on these outcomes and therefore establish a new intervention if results are determined to be significant.

#Phase 1: Reducing noise in crowdsourced outcome annotations - EBM-NLP corpus
- Extract crowdsourced outcomes from EBM-NLP courpus - ebm_nlp_demo.py

This repository contains source code used in publications:
- Abaho, Micheal, et al. "Correcting Crowdsourced annotations to Improve Detection of Outcome types in Evidence Based Medicine."http://danushka.net/papers/abaho_kdh_2019.pdf

Reproducing the work or for purposes of running experiments on different datasets, Follow the instructions below,

Requirements
- Python 3
- Any of the following NLP libraries (Stanford or Spacy)
- EBM-NLP corpus https://github.com/bepnye/EBM-NLP/blob/master/ebm_nlp_2_00.tar.gz

Noise Filtration
Run this script to correct any flaws noise  the annotations within
- python annotate_text.py medpost
NB: The second argument can as well be genia or stanford

Train a classifier
- python main.py lstm
NB: The second argument can as well be cnn or svm



