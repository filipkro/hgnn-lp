# BioKG
Download the BioKG dataset using the ogb api with `dataset` as root, i.e. a directory called `ogbl_biokg` should be created in `dataset`.

All hierarchy files (except `MRREL.RRF` needed for `disease` and `sideeffect` hierarchies), including temporary files generated during creation process, are found in `datafiles/`.

MRREL.RRF can be downloaded from (license needed):
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

Scripts to generate datasets for training are found in `data_prep`.

# CoDEx
For CoDEx, download the dataset and put the `codex-l` directory in `datafiles`. Also for wikidata5m (e.g. included in the codex download) we ran following commands to generate files containing only hierarchy information:

```
cat full.txt | grep P279 >> full_P279.txt
```
and
```
cat full.txt | grep P31 >> full_P31.txt
```

These files can be found in `datafiles/wikidata5m`.

# AIFB
Both the dataset and hierarchy are included in this repo. They can also be created by running `data_prep/aifb_dataset.py`