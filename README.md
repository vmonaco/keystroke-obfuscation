# ICB 2016

**Obfuscating Keystroke Time Intervals to Avoid Identification and Impersonation**

* John V. Monaco
* Charles C. Tappert

## Reproducing experiments

Run the main script to reproduce all experiments.

`python main.py [seed]`

This will download and preprocess all datasets, followed by obfuscation, feature extraction, and classification. If any of the datasets fails to download, they must be manually placed in the data/download folder. In particular, the Villani dataset requires approved access.