# 🎬 Movies Box Office Analysis

Exploratory analysis and predictive modelling of 3,800+ films (1980–2016) to understand what actually drives box office gross.

## Key findings

- **Horror has the best ROI** — median 2.2× return on budget, beating Action despite far lower production costs
- **Budget explains less than expected** — the strongest predictor is audience engagement (`num_voted_users`), not spend
- **Gradient Boosting** achieves CV-R² = 0.68, MAE = $23M, outperforming linear models by a wide margin

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit altair

python 01_data_cleaning.py       # clean raw data
python 02_modelling.py           # train & evaluate models
jupyter notebook movies_analysis.ipynb  # full narrative analysis
streamlit run MDapp.py           # interactive dashboard
```

## Data sources
[IMDB Movie Metadata](https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset) · [TMDB 5000 Movies](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
