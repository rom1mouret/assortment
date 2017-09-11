### Motivation And Approach

It is notoriously difficult to ensemble arbitrary outlier detectors, e.g. Isolation Forest and ABOD. A common approach is to harmonize anomaly scores and to combine them with a function such as PROD, AVG or MAX. In my experience, however, underperforming models drag other models down and,  as a result, the ensemble disappoints, with performances landing somewhere near the average between its constituents.

This project follows another popular approach, known as gating. The problem is solved by learning to pick the best detector, based on the score distributions and the nature of the dataset.

Since the purpose of this experiment is to demonstrate the capability of our model selection method, little attention has been paid to the performance of the base detectors. In fact, both KNN and ABOD implementations are simplifications of their original counterpart. They rely heavily on random sampling to speed up the scoring.
For the same reason, base detectors are taken out-of-the-box, with no tuning whatsoever. 

### Results

  
| model | average recall |
| --- | --- |
| this project | 30.81% |
| KNN |  26.32% |
| IForest |  26.22%|
| Quasi-ABOD | 23.33%|
| LOF | 21.26% |
| One-Class SVM |  25.80% |
| AutoEncoder (Neural Net)|  24.15% |
| Perfect Selection | 39.75% |


This table reports the average recall-to-5% over 30 datasets, i.e. the number of outliers found in the 5% top ranked rows, out of the number that should be found. It has been prefered over [precision@n](http://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/) as it is more stable for datasets with few outliers, especially considering that base detectors are not that accurate. More conventional evaluation metrics can be found in the source code.

The default strategy is to pick the detector with the largest predicted recall. To do so, we train separately one regressor for each detector, using only features that matter for the detector at hand. 

<div align="right">
<img src="https://github.com/rom1mouret/assortment/blob/master/iforest_prediction.png">
<img src="https://github.com/rom1mouret/assortment/blob/master/autoencoder_prediction.png">
</div>

Other strategies are available, like learning to predict the difference between recalls, using the group of features that quantifies the agreement between the rankings.

### Installing and Running

After installing Python 3, install every Python library of [requirements.txt](requirements.txt), preferably with pip:

```sh
pip3 install -r requirements.txt
```

To reproduce the results, run:

```sh
./kcross_validate_mdl_selection.py mdl_selection_dataset.csv -n 6 --features 0 2 3 8 9
```
## Literature-datasets
| filename | source |
| --- | --- |
| ALOI.arff | (1) |
| Annthyroid_02_v01.arff | (1) |
| Arrhythmia_withoutdupl_02_v03.arff | (1) |
| Cardiotocography_02_v01.arff | (1) |
| cover.csv | (2) |
| Glass_withoutdupl_norm.arff | (1) |
| HeartDisease_withoutdupl_02_v01.arff | (1) |
| Hepatitis_withoutdupl_05_v01.arff | (1) |
| Ionosphere_withoutdupl_norm.arff | (1) |
| KDDCup99_original.arff | (1) |
| letter.csv | (2) |
| mnist.csv | (2) |
| musk.csv | (2) |
| PageBlocks_norm_02_v01.arff | (1) |
| Parkinson_withoutdupl_05_v01.arff | (1) |
| PenDigits_withoutdupl_norm_v01.arff | (1) |
| Pima_withoutdupl_02_v03.arff | (1) |
| satimage-2.csv | (2) |
| seismic-bumps_v1.arff (formatted like (1)) | (2) |
| Shuttle_withoutdupl_norm_v01.arff | (1) |
| SpamBase_02_v01.arff | (1) |
| speech.csv | (2) |
| Stamps_withoutdupl_02_v02.arff | (1) |
| vertebral.csv | (2) |
| Waveform_withoutdupl_norm_v01.arff | (1) |
| WBC_norm_v01.arff | (1) |
| WDBC_withoutdupl_v01.arff | (1) |
| Wilt_02_v01.arff | (1) |
| wine.csv | (2) |
| WPBC_withoutdupl_norm.arff | (1) |


* (1) http://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/
* (2) http://odds.cs.stonybrook.edu/

### Interpreting and Generating the Gating-Dataset

Each row of the gating-dataset [mdl_selection_dataset.csv](mdl_selection_dataset.csv) represents one scoring round on one litterature-dataset. During each round, we gather the scores of each outlier detector, generate the features and compute the evaluation metrics. 

Here is the description of one row:

| Indices | Description |
| --- | --- |
| 0 | litterature-dataset filename |
| 1-n | names of the outlier detectors |
| n+1-2n+1 | precision or recall for each detector |
| 2n+2 - m| agreement between rankings, indexed as follows: for i in (0..n) for j in (i+1..n) { index_agreement[(i,j)] = k++ } |
| m+1 - m+10 | flat features for the 1st detector |
| m+11 - m+20 | flat features for the 2nd detector |
And it goes on…

Here is the description for the flat features of the 1st detector

| Indices | Description |
| --- | --- |
| m+1 | Shapiro–Wilk normality test on each feature, even though it is the same value for every detector |
| m+2 | 70% percentile of the score distribution |
| m+3 | 90% percentile of the score distribution |
| m+4 | 95% percentile of the score distribution |
| m+5 | how much the score explains the data with Ridge regression and linear features |
| m+6 | 1st bin of the histogram of the score distribution, in log scale |
| m+7 | 2nd bin of the histogram of the score distribution, in log scale |
| m+8 | 3rd bin of the histogram of the score distribution, in log scale |
| m+9 | 4th bin of the histogram of the score distribution, in log scale |
| m+10 | 5th bin of the histogram of the score distribution, in log scale |

In order to provide more data to the model selection module, we have run 27 rounds for each literature-dataset. The kcross validation should not separate rows from the same dataset.

To rebuild a similar dataset, run the following command in the [gen_dataset](gen_dataset) directory:
```sh
./generate_dataset.py datasets/*
```

It should take about one hour on a modern computer. It will output a gating-dataset with 3 rounds for each literature-dataset. Concatenate the outputs to get a higher number of rounds.

Note: generate_dataset.py cannot parse every kind of ARFF or CSV file. If you intend to use your own datasets, format them as in source (1) or source (2)
