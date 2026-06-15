All datafiles (except `MRREL.RRF` needed for `disease` and `sideeffect` hierarchies), including temporary files generated during creation process, are found in `datafiles/`.

MRREL.RRF can be downloaded from (license needed):
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

Scripts to generate datasets for training are found in `data_prep`.

For CoDEx, download the dataset and put the `codex-l` directory in `datafiles`. Also for wikidata5m (e.g. included in the codex download) we ran following commands to generate files containing only hierarchy information:

```
cat full.txt | grep P279 >> full_P279.txt
```
and
```
cat full.txt | grep P31 >> full_P31.txt
```

These files can be found in `datafiles/wikidata5m`.