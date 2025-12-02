# DSA4213-Final-Project-Movie-Recommmendation-System
This repository uses NLP applications to implements a Movie Recommendation System from the Kaggle Movies Dataset.
Our goal is to use vector embeddings to compare the similarity between queries and movies to recommand the most right movies to users.

# Repository Layout
project/
│
├── README.md                  
├── requirements.txt           # Python dependencies
│
├── data/                      # Datasets
│   ├── movies_metadata.csv
│   ├── ratings.csv
│   ├── keywords.csv
│   ├── links_small.csv
│   ├── ratings_small.csv
│   └── links.csv
│
├── src/                      
│   ├── movie_chatbot.py
│   ├── retrieval_demo.py
│   └── retrieval.ipynb

# Typical Workflow
Create environment python -m venv venv
source venv/bin/activate
venv\Scripts\activate (Windows)

Install dependencies
pip install -r requirements.txt

Run py files in order:
retrieval_demo → movie_chatbot.

Run notebook file:
retrieval.ipynb

Ensure working directory is project root for correct relative paths.

# Notes
Kaggle raw data is large; keep out of version control.
