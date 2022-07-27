# Dynamics of social media behavior before and after COVID-19 infection

## Methods 
The first task is to identify Twitter users who reported that they tested positive to Covid-19. This step is achieved with [`positive_filter.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/positive_filter.py)(which depends on [filters.py](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/filters.py)).
The so-called test-positive tweets are stored under `data/positive` in daily Parquet files and are then grouped in a single file (`data/df_positive.pkl`). 
The Twitter timelines of the selected users are then retrieved with `download_timelines.py` and stored (Pickle files in `data/timelines/raw`). 
The script [`parse_timelines.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/parse_timelines.py) is then used to parse the raw timelines (in JSON Line files) and store the output data in Parquet files.
The following analyses are applied to the parsed timelines:
- Tagging of tweets containing symptoms ([`timeline_medcat.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/timeline_medcat.py)). Tweets are tagged with MedCAT (the comparison of this method with the lexicon-based approach developed by [Sarker et al.](https://doi.org/10.1093/jamia/ocaa116) is carried out with [`sampling_for_comparison.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/sampling_for_comparison.py))
- Temporal assessment of the self-reports of symptoms ([`time_extract.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/time_extract.py))
- Filtering self-reports of symptoms (*cf.* [`reporting_classification` folder](https://github.com/digitalepidemiologylab/content_changes_paper/tree/main/reporting_classification))
- Domain analysis of shared URLs ([`timeline_url.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/timeline_url.py))
- Multi-label classification of the tweets into different general topics (*cf.* [`topic_classification` folder](https://github.com/digitalepidemiologylab/content_changes_paper/tree/main/topic_classification))
- Multi-label classification of tweets according to the expressed emotions (*cf.* `SpanEmo` folder)

The results of these various analyses are collected and concatenated with [`timeline_combine_all.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/timeline_combine_all.py), which enables to generate a user-specific files in `data/language/all_timelines`.

## Pre/post comparisons
Files in `data/language/all_timelines` contain the information required for the individual-level pre/post comparisons (*cf.* [`statistical_analysis.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/time_extract.py)) after the tweets of the selected users were processed with the various ML-based methods described above.
The collective analyses consist of Wilcoxon signed-rank tests, as detailed in [`wilcoxon_features.R`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/wilcoxon_features.R) and [`adjusted_pvalues.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/adjusted_pvalues.py).

## Figures
The figures shown in the article can be generated as follows:
- Figure 1: [`analysis_positive_tweets.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/analysis_positive_tweets.py)
- Figure 2: [RankFlow visualization tool](https://labs.polsys.net/tools/rankflow/)
- Figures 3, 4, and 5: [`plot_median_differences.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/plot_median_differences.py)
- Supplementary Figure 1: [`generate_causal_impact_figure.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/generate_causal_impact_figure.py)
- Supplementary Figure 2: [`analysis_positive_tweets.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/analysis_positive_tweets.py)
- Supplementary Figure 3: [`timeline_symptoms.py`](https://github.com/digitalepidemiologylab/content_changes_paper/blob/main/timeline_symptoms.py)

