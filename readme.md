# AI for Music Intelligence: Lyrical Evolution and Thematic Profiling

## Team Members: Tanya Gupta tanya5

## Project Coordinator: Tanya Gupta

## Project Description

This project develops an AI-powered tool designed to analyze song lyrics, providing deep insights into how lyrical themes, sentiment, and vocabulary evolve over an artist's career or across music genres. By leveraging advanced Natural Language Processing (NLP) and AI techniques, the tool aims to uncover patterns that might hint at future music trends, offering valuable information for commercial artists, music researchers, and industry experts.

Traditional lyrical analysis is often subjective, time-consuming, and struggles with large datasets. This tool addresses these pain points by offering a data-driven, scalable, and objective approach. It uniquely integrates multiple AI methods—sentiment analysis, topic modeling, and named entity recognition—to track and visualize lyrical changes over time, going beyond basic keyword searches or mood classifications.

The project delivers a working prototype that processes lyrical data for selected artists (Eminem and Taylor Swift), generates comprehensive analytical visualizations, and presents them through an interactive Streamlit dashboard.

## Features

*   **Lyrical Data Collection & Preprocessing:** Handles raw lyrical data, including robust cleaning, normalization, and removal of artist-specific self-references or common locations to ensure unbiased analysis.
*   **Sentiment Analysis:** Utilizes a pre-trained transformer model (DistilBERT) to determine the emotional tone (positive, negative, neutral) of lyrics and track its evolution over time, including a basic next-year prediction.
*   **Topic Modeling:** Employs TF-IDF vectorization and Non-negative Matrix Factorization (NMF) to identify and categorize dominant lyrical themes, revealing shifts in an artist's thematic focus.
*   **Named Entity Recognition (NER):** Leverages SpaCy to extract and count mentions of key entities such as people, organizations, and geographical locations, highlighting significant references in lyrics.
*   **Data Aggregation:** Consolidates analytical results by artist and release year to provide a structured overview of trends.
*   **Interactive Visualizations (Plotly):** Generates dynamic and interactive charts (line graphs, stacked bar charts, bar charts) for sentiment evolution, topic dominance, and top entities, allowing users to explore data with zoom, pan, and hover functionalities.
*   **Streamlit Dashboard:** Provides an intuitive, user-friendly web interface to display all interactive visualizations, making the complex analytical insights easily accessible.

## How it Works and is implemented (High-Level Architecture)

1.  **Data Ingestion:** Lyrics for target artists (Eminem, Taylor Swift) are loaded from CSV files.
2.  **Text Preprocessing:**
    *   Initial cleaning: Removes non-lyrical markers, punctuation, numbers, converts to lowercase, and normalizes whitespace.
    *   Custom filtering: Removes artist-specific names and common locations to refine entity and topic analysis.
    *   Stopword removal: Utilizes NLTK's stopwords, with an extended set for sentiment analysis.
3.  **AI/NLP Analysis:**
    *   **Sentiment Analysis:** HuggingFace `transformers` pipeline (DistilBERT) processes lyrics in chunks, aggregating sentiment scores.
    *   **Topic Modeling:** `TfidfVectorizer` creates document-term matrices, followed by `NMF` to extract latent topics.
    *   **Named Entity Recognition:** SpaCy's `en_core_web_sm` model identifies and categorizes entities (PERSON, ORG, GPE, LOC).
4.  **Data Aggregation:** Results from AI analyses are grouped by artist and release year to calculate average sentiment, topic proportions, and entity frequencies.
5.  **Visualization Generation:** Plotly is used to create interactive HTML charts from the aggregated data, saved to an `output_plots` directory.
6.  **Interactive Dashboard:** A Streamlit application loads and displays these interactive HTML files, providing a navigable interface for users.

## Getting Started (How to use)
To set up and run the AI Music Intelligence tool, follow these steps:

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   Internet connection (for downloading NLTK data and SpaCy model)

### Installation Steps

1.  **Clone the Repository (or create project directory):**
    Create a directory and place the .py files there
    mkdir LyricalEvolutionProject
    cd LyricalEvolutionProject
    # Place music_intelligence_analysis.py and streamlit_app.py here
    

2.  **Create a Virtual Environment (Recommended):**
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate

3.  **Install Dependencies:**
    Use the `requirements.txt` file with the following content:
    ```
    pandas
    nltk
    spacy
    transformers
    scikit-learn
    matplotlib
    seaborn
    plotly
    streamlit
    ```
    install them using:
    pip install -r requirements.txt
    

4.  **Download SpaCy Model:**
    You need to manually download the `en_core_web_sm` SpaCy model:
    python -m spacy download en_core_web_sm

5.  **Place Data Files:**
    Ensure you have the lyrical data files (`Eminem.csv` and `TaylorSwift.csv`) in the root directory of your project (the same directory as `music_intelligence_analysis.py`). They are apartof the github folder. These can typically be obtained from Kaggle datasets. (https://www.kaggle.com/datasets/deepshah16/song-lyrics-dataset/data)

### Running the Analysis

First, run the main analysis script to generate the plots:
python music_intelligence_analysis.py

### Running Streamlit
After the analysis script has successfully generated the output plots, launch the Streamlit dashboard:

streamlit run streamlit_app.py

This will open a new tab in your web browser displaying the interactive dashboard. You can use the sidebar to navigate between Sentiment Analysis, Topic Modeling, and Named Entity Recognition sections.


### Evaluation

I will evaluate my tool by demonstrating a fully functional Python program that generates lyrical profiles and visualizations. I will assess the tool's accuracy by observing how well the generated visuals accurately reflect known lyrical changes and themes for the chosen artists, providing a data-driven view that goes beyond traditional, subjective analysis. Furthermore, my project aims to reveal new connections between music, culture, and society, and potentially identify emerging trends that could hint at future album themes.


### Future Plans and Outlook

This project lays a strong foundation for AI-driven music intelligence, and I have several exciting avenues for future development:


Expanded Data Coverage: I plan to broaden the analysis to include a wider array of artists, explore different music genres, and facilitate cross-genre comparisons. This could involve integrating with larger lyrical databases or music APIs.

Advanced NLP Models: I intend to incorporate more sophisticated contextual embedding models (e.g., BERT, GPT variants) for deeper semantic understanding in sentiment analysis and explore dynamic topic modeling techniques to track topic evolution more fluidly.

Enhanced Temporal Granularity: I will allow users to analyze lyrical trends at finer time scales, such as by album, quarter, or even month, to capture more nuanced shifts in an artist's career.

Improved User Interaction and Customization: I will develop features within the Streamlit dashboard that enable users to upload their own lyrical datasets, select specific time ranges for analysis, or customize parameters like the number of topics for modeling.

Refined Predictive Capabilities: I will implement more robust time-series forecasting models to generate more accurate and insightful predictions for future lyrical sentiment and thematic trends.

Vocabulary and Stylistic Analysis: I will add modules to analyze vocabulary richness, lyrical complexity, sentence structure, and specific stylistic elements (e.g., use of metaphors, slang) over time.

Integration with Music Metadata: I will explore combining lyrical insights with audio features (e.g., tempo, key, mood from audio analysis) and other music metadata for a holistic understanding of how music evolves.

Collaborative Features: I will consider adding functionalities that would allow multiple users to interact with the dashboard, save analyses, and share insights.

Cloud Deployment: I aim to deploy the Streamlit application to a public cloud platform (e.g., Streamlit Cloud, AWS, Azure) to make it easily accessible to a wider audience without requiring local setup.

These enhancements will transform the tool into an even more powerful, flexible, and comprehensive resource for understanding the rich cultural narratives embedded within music.