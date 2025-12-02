# DSA4213-Final-Project-Movie-Recommmendation-System
This repository uses NLP applications to implements a Movie Recommendation System from the Kaggle Movies Dataset.
Our goal is to use vector embeddings to compare the similarity between queries and movies to recommand the most right movies to users.

## Repository Layout

```
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
```
---

## Typical Workflow

1. Create environment 
   python -m venv venv  
   source venv/bin/activate  
   venv\Scripts\activate (Windows)

2. Install dependencies  
   pip install -r requirements.txt

3. Download dataset from kaggle

4. Run py files in order: 
   retrieval_demo → movie_chatbot.
   
5. Run notebook file:
    retrieval.ipynb
   
6. Ensure working directory is project root for correct relative paths.

---

## Notes
- Kaggle raw data is large; keep out of version control.
