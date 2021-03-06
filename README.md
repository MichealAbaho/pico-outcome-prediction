# pico-outcome-prediction
Detecting outcomes from clinical records for evidence based practice.

The goal of the project is to identify and extract outcome phrases from medical text i.e. spans of text designated to be health outcomes as described by the Comet and Cochrnae Institutions. Outcomes are essentially a diagnosis measured by clinical researchers during Clinical Trials. In Evidence based care or practice, clinicians and biomedics assess the impact new interventions have on these outcomes and therefore establish a new intervention if results are determined to be significant.

#Phase 1: Reducing noise in crowdsourced outcome annotations - EBM-NLP corpus
- Extract crowdsourced outcomes from EBM-NLP courpus - ebm_nlp_demo.py

This repository contains source code used in publications:
- Abaho, Micheal, et al. "Correcting Crowdsourced annotations to Improve Detection of Outcome types in Evidence Based Medicine."http://ceur-ws.org/Vol-2429/paper1.pdf

Follow the instructions below to reproduce the results for purposes of running experiments on different datasets, otherwise the ebm_nlp_1_00 or ebm_nlp_2_00 datasets used can be found here https://github.com/bepnye/EBM-NLP 

Requirements
```
- Python 3, tensorflow and keras
- Install and setup any one of the following NLP libraries (Spacy or Stanford), Spacy recommended.
- EBM-NLP corpus https://github.com/bepnye/EBM-NLP/blob/master/ebm_nlp_2_00.tar.gz, add the annotations and documents directories to your setup project directory
- Install geniatagger-python via pip,
Other relevant packages are matplotlib
```
Noise Filtration
Run this script to correct any flaws noise  the annotations within

```
- python annotate_text.py medpost
```
NB: The second argument can as well be genia or stanford

Train a classifier
```
- python main.py lstm
```
NB: The second argument can as well be cnn or svm

