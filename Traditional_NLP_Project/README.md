# Traditional NLP Project

This project was developed as part of the NLP track in the NTI AI Scholarship. It focuses on traditional Natural Language Processing techniques to build a smart movie information and recommendation system based on the IMDb Top 100 Movies list.

## Objective

To design an interactive system that:
- Scrapes data from IMDb's Top 100 movies
- Allows movie lookup using fuzzy matching
- Recommends similar movies based on plot description similarity

##  Tasks Overview

### 1. IMDb Top 100 Dataset Builder
- Scrape key details for each movie:
  - Title
  - Year
  - Duration
  - IMDb Rating
  - Plot Description
- Store results in `top_100_movies.csv`

### 2. Movie Lookup System
- Accepts user input for a movie title
- Handles typos using fuzzy string matching (`fuzzywuzzy`)
- Displays the full movie information

### 3. Movie Recommendation Engine
- Uses the movie description to compute similarity with others
- Returns top 5 most similar movies using content-based filtering

## 🛠️ Tools & Libraries

- Python
- BeautifulSoup (for scraping)
- pandas, NumPy
- fuzzywuzzy
- scikit-learn (TF-IDF, similarity measures)

## Files

- `Traditional_NLP_Project.ipynb`: Full notebook implementation
- `top_100_movies.csv`: Generated movie dataset
- `README.md`: Project documentation

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/HossamCSE/NTI_Projects.git
