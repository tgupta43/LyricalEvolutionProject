import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from nltk.tokenize import sent_tokenize
import plotly.express as px
import plotly.graph_objects as go

# --- NLTK Data Setup ---
# This section makes sure NLTK data (like stopwords and word definitions) is downloaded.
# It creates a special folder for NLTK data in your project to keep things organized.
print("Checking NLTK data...")
project_nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.environ['NLTK_DATA'] = project_nltk_data_path # Tells NLTK where to find its data
if not os.path.exists(project_nltk_data_path):
    os.makedirs(project_nltk_data_path) # Create the folder if it doesn't exist
    print(f"Created custom NLTK data directory: {project_nltk_data_path}")
else:
    print(f"Using custom NLTK data directory: {project_nltk_data_path}")
nltk_resources = ['stopwords', 'wordnet', 'punkt'] # List of NLTK resources needed
for resource in nltk_resources:
    print(f"Ensuring NLTK '{resource}' is available...")
    try:
        # Check if the resource is already downloaded
        if resource == 'punkt':
            nltk.data.find(f'tokenizers/{resource}')
        else:
            nltk.data.find(f'corpora/{resource}')
        print(f"Resource '{resource}' found or already up-to-date.")
    except LookupError:
        # If not found, download it
        print(f"Downloading NLTK '{resource}' to {project_nltk_data_path}...")
        try:
            nltk.download(resource, download_dir=project_nltk_data_path) # Download to the custom folder
            print(f"Successfully downloaded '{resource}'.")
        except Exception as e:
            print(f"Error downloading '{resource}': {e}")
            print("Please check your internet connection or try running 'python -c \"import nltk; nltk.download('all', download_dir=\\'" + project_nltk_data_path.replace('\\', '\\\\') + "\\')\"' manually.")
            exit()
print("NLTK data check complete.")

# --- SpaCy Model Loading ---
# This loads the SpaCy language model used for finding names and places (Named Entity Recognition) in text.
print("Loading SpaCy model 'en_core_web_sm'...")
try:
    nlp = spacy.load("en_core_web_sm") # Load the small English SpaCy model
    print("SpaCy model loaded.")
except OSError:
    print("\nSpaCy model 'en_core_web_sm' not found.")
    print("Please run 'python -m spacy download en_core_web_sm' in your terminal.")
    print("Exiting script.")
    exit()

# --- Configuration ---
# Defines which artist data files to use and where to save the output plots.
ARTIST_DATA_FILES = {
    'Eminem': 'Eminem.csv', # File for Eminem's lyrics
    'Taylor Swift': 'TaylorSwift.csv' # File for Taylor Swift's lyrics
}
ARTISTS_TO_ANALYZE = list(ARTIST_DATA_FILES.keys()) # Get a list of artist names
OUTPUT_DIR = 'output_plots' # Folder name to save all generated plots
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR) # Create the output folder if it doesn't exist
    print(f"Created custom NLTK data directory: {OUTPUT_DIR}")
else:
    print(f"Output directory '{OUTPUT_DIR}' already exists.")

# --- Data Loading & Initial Cleaning ---
# Loads song lyrics from CSV files, renames columns, and converts release years to numbers.
# It combines data from all artists into one main dataset.
all_dfs = [] # List to hold dataframes for each artist
print("\nLoading data for specified artists...")
for artist_name, file_name in ARTIST_DATA_FILES.items():
    print(f"Loading {artist_name} data from {file_name}...")
    try:
        df_artist = pd.read_csv(file_name) # Read the CSV file into a dataframe
        # Rename columns to a standard format for easier use
        df_artist.rename(columns={'Lyric': 'lyrics', 'Year': 'release_year'}, inplace=True)
        df_artist['artist_name'] = artist_name # Add artist name as a new column
        # Select and keep only the necessary columns
        df_artist = df_artist[['artist_name', 'Title', 'Album', 'release_year', 'lyrics']].copy()
        # Convert 'release_year' to a number, turning any errors (like 'Unknown') into 'Not a Number' (NaN)
        df_artist['release_year'] = pd.to_numeric(df_artist['release_year'], errors='coerce')
        df_artist.dropna(subset=['release_year'], inplace=True) # Remove rows where release_year is NaN
        df_artist['release_year'] = df_artist['release_year'].astype(int) # Convert year to whole numbers
        all_dfs.append(df_artist) # Add processed artist dataframe to the list
        print(f"  Loaded {len(df_artist)} songs for {artist_name}.")
    except FileNotFoundError:
        print(f"Error: {file_name} not found. Make sure it's in the correct directory.")
        exit()
    except KeyError as e:
        print(f"Error: Missing expected column in {file_name}. Details: {e}")
        print("Expected columns: 'Artist', 'Title', 'Album', 'Year', 'Date', 'Lyric'")
        exit()
if not all_dfs:
    print("No artist data loaded. Exiting.")
    exit()
df_filtered = pd.concat(all_dfs, ignore_index=True) # Combine all artist dataframes into one
print(f"\nCombined dataset contains {len(df_filtered)} songs for {len(ARTIST_DATA_FILES)} artists.")
df_filtered['lyrics'] = df_filtered['lyrics'].astype(str).fillna('') # Ensure lyrics are strings and fill empty ones
df_filtered.dropna(subset=['release_year'], inplace=True) # Drop any remaining rows with missing years
df_filtered['release_year'] = df_filtered['release_year'].astype(int) # Convert years to integers again
df_filtered.sort_values(by=['artist_name', 'release_year'], inplace=True) # Sort data by artist and year
print("\nFirst 5 rows of prepared data:")
print(df_filtered.head())
print("\nColumn information:")
df_filtered.info()

# --- Text Cleaning Function ---
# This function removes unwanted parts from lyrics like brackets, punctuation, and makes text lowercase.
stop_words = set(stopwords.words('english')) # Get a list of common English stopwords
lemmatizer = WordNetLemmatizer() # Tool to reduce words to their base form (e.g., "running" to "run")
def clean_lyrics(text):
    if not isinstance(text, str):
        text = str(text) # Make sure the input is a string
    text = re.sub(r'\[.*?\]', '', text) # Remove text inside square brackets like [Chorus]
    text = re.sub(r'\(.*?\)', '', text) # Remove text inside parentheses like (feat. Artist)
    text = re.sub(r'\{.*?\}', '', text) # Remove text inside curly braces
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove anything that's not a letter or a space
    text = text.lower() # Convert all text to lowercase
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with one, and remove spaces from ends
    return text
print("\nCleaning lyrics...")
df_filtered['cleaned_lyrics'] = df_filtered['lyrics'].apply(clean_lyrics) # Apply cleaning function to all lyrics
df_filtered = df_filtered[df_filtered['cleaned_lyrics'].str.strip() != ''] # Remove songs where lyrics became empty after cleaning
print(f"Dataset after cleaning and removing empty lyrics: {len(df_filtered)} songs.")
print("\nSample of cleaned lyrics:")
print(df_filtered[['artist_name', 'Title', 'cleaned_lyrics']].head())

# --- Specific Celebrity/Location Removal ---
# This section removes names of the artists themselves or common locations they mention.
# This helps focus the analysis on other topics and entities.
celebrity_names_to_remove = {
    'Taylor Swift': [
        r'\btaylorswift\b', r'\btaylor\b', r'\bswift\b', # Remove Taylor Swift's own name variants
        r'\bjustin\b', r'\bjustin bieber\b', # Example of other celebrity names
        r'\blondon\b', # Example of a location
        r'\bbrenda lee\b', r'\bbrenda\b', r'\blee\b',
        r'\bhayley williams\b', r'\bhayley\b', r'\bwilliams\b',
        r'\bsanta clara\b',
        r'\bed sheeran\b', r'\bed\b', r'\bsheeran\b',
        r'\btim\b',
        r'\bcolbie caillat\b', r'\bcolbie\b', r'\bcaillat\b',
        r'\bbrendon urie\b', r'\bbrendon\b', r'\burie\b',
        r'\blana del rey\b', r'\blana\b', r'\bdel rey\b',
        r'\bharry styles\b', r'\bharry\b', r'\bstyles\b',
        r'\bjake gyllenhaal\b', r'\bjake\b', r'\bgyllenhaal\b',
        r'\bjohn mayer\b', r'\bjohn\b', r'\bmayer\b',
        r'\bjoe alwyn\b', r'\bjoe\b', r'\balwyn\b',
        r'\btom hiddleston\b', r'\btom\b', r'\bhiddleston\b',
        r'\bcalvin harris\b', r'\bcalvin\b', r'\bharris\b',
        r'\baustralia\b',
        r'\bkendrick lamar\b', r'\bkendrick\b', r'\blamar\b',
        r'\blos angeles\b', r'\blosangeles\b', r'\bla\b',
        r'\bye\b',
        r'\bjames dean\b', r'\bjames\b', r'\bdean\b',
        r'\bmad mad\b', r'\bmadmad\b',
        r'\bmmm mmm\b', r'\bmmmmmm\b',
        r'\bbette davis\b', r'\bbette\b', r'\bdavis\b', r'\beminem\b', r'\barlington\b',
        r'\bminneapolis\b',r'\bcanada\b',r'\bdublin ireland\b',
        r'\btoronto\b',r'\bphiladelphia\b',r'\bmm\b',
        r'\bnorth america\b',r'\bga\b',r'\bya\b', r'\babigail\b'
    ],
    'Eminem': [
        r'\btaylor\b', r'\bswift\b', # Remove if Eminem mentions Taylor
        r'\bdr dre\b', r'\bdre\b', # Remove Dr. Dre
        r'\bsnoop dogg\b', r'\bsnoop\b', r'\bdogg\b', # Remove Snoop Dogg
        r'\b50 cent\b', r'\b50\b', r'\bcent\b', # Remove 50 Cent
        r'\bkim\b', r'\bkim mathers\b', r'\bmathers\b', # Remove Kim Mathers
        r'\bchuck\b', r'\bboom\b', r'\bmarshall\b', # Other common names/terms associated with Eminem
        r'\bshady\b', r'\beminem\b',r'\bya\b' # Remove Eminem's own name variants
    ]
}
def remove_specific_celebrity_names(text, artist_name, celeb_dict):
    if artist_name in celeb_dict: # Check if we have specific names to remove for this artist
        names_to_remove = celeb_dict[artist_name]
        for name_pattern in names_to_remove:
            text = re.sub(name_pattern, '', text) # Replace the name with nothing
        text = re.sub(r'\s+', ' ', text).strip() # Clean up extra spaces left after removal
    return text
print("\nRemoving specific celebrity names and locations from lyrics...")
df_filtered['further_cleaned_lyrics'] = df_filtered.apply(
    lambda row: remove_specific_celebrity_names(row['cleaned_lyrics'], row['artist_name'], celebrity_names_to_remove),
    axis=1
)
df_filtered['further_cleaned_lyrics'].replace('', np.nan, inplace=True) # Mark empty strings as 'Not a Number' (NaN)
df_filtered.dropna(subset=['further_cleaned_lyrics'], inplace=True) # Remove songs that became empty after this cleaning step
print(f"Dataset after removing specific celebrity names and locations: {len(df_filtered)} songs.")
print("\nSample of further cleaned lyrics:")
print(df_filtered[['artist_name', 'Title', 'cleaned_lyrics', 'further_cleaned_lyrics']].head())

# --- Extended Stopwords for Sentiment ---
# Prepares lyrics for sentiment analysis by removing common words that don't carry much emotion.
extended_sentiment_stopwords = set(stopwords.words('english')) # Start with standard English stopwords
extended_sentiment_stopwords.add('im') # Add 'im' (from "I'm") as a custom stopword, as it often remains after cleaning
def remove_custom_stopwords_for_sentiment(text, stopwords_set):
    # Splits text into words and keeps only those not in the stopword set
    return ' '.join([word for word in text.split() if word not in stopwords_set])
print("\nPreparing lyrics for sentiment analysis with extended stopword removal...")
lyrics_for_sentiment_processing = [
    remove_custom_stopwords_for_sentiment(lyric, extended_sentiment_stopwords)
    for lyric in df_filtered['further_cleaned_lyrics'].tolist() # Apply to the further cleaned lyrics
]
print("Lyrics prepared for sentiment analysis.")

# --- Sentiment Analysis ---
# Uses a special AI model to figure out if lyrics are positive, negative, or neutral.
# It breaks long lyrics into smaller parts so the model can handle them.
print("\nPerforming Sentiment Analysis (chunking long lyrics)...")
sentiment_pipeline_instance = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english", # Specify the pre-trained model to use
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english", # The tokenizer prepares text for this model
    truncation=True, # Automatically shorten texts that are too long for the model
    max_length=512 # The maximum number of words/tokens the model can process at once
)
sentiment_tokenizer = sentiment_pipeline_instance.tokenizer # Get the tokenizer from the pipeline
def chunk_text(text, tokenizer, max_len=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False) # Convert text to numerical tokens
    chunks = []
    if not tokens:
        return [""] # Return an empty string if there are no tokens
    # Create overlapping chunks of text
    for i in range(0, len(tokens), max_len - overlap):
        chunk_tokens = tokens[i : i + max_len]
        chunks.append(tokenizer.decode(chunk_tokens)) # Convert tokens back to text for the model
    return chunks
batch_size = 100 # Process lyrics in groups of 100 to speed things up
all_song_sentiments = [] # List to store sentiment results for all songs
lyrics_to_process = lyrics_for_sentiment_processing # Use the lyrics with extended stopwords removed
for i in range(0, len(lyrics_to_process), batch_size):
    batch_lyrics = lyrics_to_process[i:i+batch_size] # Get a batch of lyrics
    batch_results = []
    for lyric in batch_lyrics:
        if not lyric.strip():
            all_song_sentiments.append({'label': 'NEUTRAL', 'score': 0.0}) # Assign neutral to empty lyrics
            continue
        chunks = chunk_text(lyric, sentiment_tokenizer, max_len=512, overlap=50) # Break lyric into smaller chunks
        chunk_sentiments = sentiment_pipeline_instance(chunks) # Analyze sentiment for each chunk
        positive_scores = [res['score'] for res in chunk_sentiments if res['label'] == 'POSITIVE'] # Get scores for positive chunks
        negative_scores = [res['score'] for res in chunk_sentiments if res['label'] == 'NEGATIVE'] # Get scores for negative chunks
        avg_positive_score = np.mean(positive_scores) if positive_scores else 0 # Average positive scores
        avg_negative_score = np.mean(negative_scores) if negative_scores else 0 # Average negative scores
        # Determine overall sentiment based on average scores
        if avg_positive_score > avg_negative_score and len(positive_scores) > 0:
            overall_label = 'POSITIVE'
            overall_score = avg_positive_score
        elif avg_negative_score > avg_positive_score and len(negative_scores) > 0:
            overall_label = 'NEGATIVE'
            overall_score = avg_negative_score
        else: # If scores are similar or no strong sentiment, mark as neutral
            overall_label = 'NEUTRAL'
            overall_score = 0.0
        all_song_sentiments.append({'label': overall_label, 'score': overall_score})
df_filtered['sentiment_label'] = [res['label'] for res in all_song_sentiments] # Add sentiment label to dataframe
# Convert sentiment scores to a -1 (very negative) to 1 (very positive) range for easier comparison
df_filtered['normalized_sentiment'] = [
    score if label == 'POSITIVE' else (-score if label == 'NEGATIVE' else 0)
    for score, label in zip(
        [res['score'] for res in all_song_sentiments],
        [res['label'] for res in all_song_sentiments]
    )
]
print("Sentiment Analysis complete.")

# --- Topic Modeling ---
# Finds common themes or "topics" within the lyrics.
# It groups words that often appear together to define these topics.
print("\nPerforming Topic Modeling...")
# Filter out very short lyrics which might not have enough content for topic modeling
df_topic_model = df_filtered[df_filtered['further_cleaned_lyrics'].apply(lambda x: len(x.split())) > 10].copy()
# TF-IDF Vectorization: Converts text into numerical values, giving more importance to rare but significant words
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = vectorizer.fit_transform(df_topic_model['further_cleaned_lyrics']) # Apply TF-IDF to lyrics
num_topics = 5 # We choose to find 5 main topics
nmf_model = NMF(n_components=num_topics, random_state=1, max_iter=500) # Non-negative Matrix Factorization model
nmf_model.fit(tfidf) # Train the NMF model to find topics
topic_results = nmf_model.transform(tfidf) # Assigns each song a score for each topic
df_topic_model['dominant_topic_idx'] = topic_results.argmax(axis=1) # Find the topic with the highest score for each song
feature_names = vectorizer.get_feature_names_out() # Get all unique words used by the vectorizer
def get_topic_words(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        # For each topic, get the top words that define it
        topics[f"Topic {topic_idx}"] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    return topics
n_top_words = 10 # Show the top 10 words for each topic
topic_words = get_topic_words(nmf_model, feature_names, n_top_words)
print("Topics identified:")
for topic_idx, words in topic_words.items():
    print(f"{topic_idx}: {', '.join(words)}") # Print the top words for each topic
# Merge topic results back to the main dataframe
topic_mapping = df_topic_model[['dominant_topic_idx']].set_index(df_topic_model.index)
df_filtered['dominant_topic_idx'] = topic_mapping['dominant_topic_idx']
# Create a readable name for each dominant topic (e.g., "Topic 0: im, like, shit")
df_filtered['dominant_topic_name'] = df_filtered['dominant_topic_idx'].apply(
    lambda x: f"Topic {int(x)}: {', '.join(topic_words[f'Topic {int(x)}'][:3])}" if pd.notna(x) else 'N/A'
)
print("Topic Modeling complete.")

# --- Named Entity Recognition (NER) ---
# Identifies and labels important things in the lyrics like people, organizations, and locations.
print("\nPerforming Named Entity Recognition...")
all_entities = [] # List to store all identified entities
batch_size_ner = 50 # Process lyrics in smaller groups for NER to manage memory
lyrics_for_ner = df_filtered['further_cleaned_lyrics'].tolist() # Get the cleaned lyrics
for i in range(0, len(lyrics_for_ner), batch_size_ner):
    batch_lyrics = lyrics_for_ner[i:i+batch_size_ner] # Get a batch of lyrics
    batch_lyrics_non_empty = [l for l in batch_lyrics if l.strip()] # Filter out any empty strings
    current_batch_entities = []
    if batch_lyrics_non_empty:
        # Process the batch of lyrics using SpaCy's NER pipeline
        for doc in nlp.pipe(batch_lyrics_non_empty, disable=["parser", "tagger", "textcat"]):
            entities = [(ent.text, ent.label_) for ent in doc.ents] # Extract entities (text and label)
            current_batch_entities.append(entities)
    res_idx = 0
    # Map the found entities back to the original lyrics, handling empty ones
    for original_lyric in batch_lyrics:
        if original_lyric.strip():
            all_entities.append(current_batch_entities[res_idx])
            res_idx += 1
        else:
            all_entities.append([]) # No entities for empty lyrics
df_filtered['entities'] = all_entities # Add the list of entities to the dataframe
print("\n--- DEBUG: Sample of Raw NER Results (first 5 songs) ---")
for idx, row in df_filtered.head(5).iterrows():
    print(f"  Song: {row['Title']} by {row['artist_name']} (Year: {row['release_year']})")
    print(f"  Raw Entities: {row['entities']}")
    print("-" * 20)
print("NER complete.")

# --- Data Aggregation ---
# Summarizes the results of sentiment, topics, and entities by artist and year.
print("\nAggregating data by artist and year...")
# Calculate average sentiment per artist and year
df_agg_sentiment = df_filtered.groupby(['artist_name', 'release_year'])['normalized_sentiment'].mean().reset_index()
# Calculate topic distribution per artist and year
df_topic_expanded = df_filtered.dropna(subset=['dominant_topic_idx']).copy()
df_topic_expanded['dominant_topic_idx'] = df_topic_expanded['dominant_topic_idx'].astype(int)
# Count how many songs fall into each topic for each artist and year
df_agg_topics = df_topic_expanded.groupby(['artist_name', 'release_year', 'dominant_topic_idx']).size().unstack(fill_value=0)
df_agg_topics = df_agg_topics.div(df_agg_topics.sum(axis=1), axis=0) # Convert counts to proportions (percentages)
# Rename topic columns to be more descriptive (e.g., "Topic 0: im, like, shit")
df_agg_topics = df_agg_topics.rename(columns={i: f"Topic {i}: {', '.join(topic_words[f'Topic {i}'][:3])}" for i in range(num_topics)}).reset_index()
# Calculate entity frequencies per artist and year
entity_counts_list = []
for (artist, year), group in df_filtered.groupby(['artist_name', 'release_year']):
    # Filter for common entity types like PERSON, ORGANIZATION, GEOPOLITICAL ENTITY, LOCATION
    all_ents = [ent[0] for sublist in group['entities'] for ent in sublist if ent[1] in ['PERSON', 'ORG', 'GPE', 'LOC']]
    if all_ents:
        entity_counts = pd.Series(all_ents).value_counts().head(5).to_dict() # Get the top 5 most frequent entities
        entity_counts_list.append({'artist_name': artist, 'release_year': year, 'top_entities': entity_counts})
    else:
        entity_counts_list.append({'artist_name': artist, 'release_year': year, 'top_entities': {}}) # No entities found for this group
df_agg_entities = pd.DataFrame(entity_counts_list) # Create a dataframe from the aggregated entities
print("\n--- DEBUG: Aggregated Entities DataFrame ---")
print(df_agg_entities.head(10)) # Show first 10 rows of aggregated entities
print(f"\nTotal rows in df_agg_entities: {len(df_agg_entities)}")
num_non_empty_entities = df_agg_entities['top_entities'].apply(lambda x: bool(x)).sum()
print(f"Number of artist-years with non-empty 'top_entities': {num_non_empty_entities}")
if num_non_empty_entities == 0:
    print("WARNING: No entities of types PERSON, ORG, GPE, LOC were found in any aggregated group.")
    print("Consider broadening the entity types or checking NER performance.")
print("Aggregation complete.")

# --- Visualization (Matplotlib) ---
# Generates static (non-interactive) plots for sentiment, topics, and entities.
print("\nGenerating visualizations...")
sns.set_theme(style="whitegrid") # Set a nice visual theme for plots
plt.figure(figsize=(14, 7)) # Create a figure for the plot
# Plot sentiment evolution over time for both artists
sns.lineplot(data=df_agg_sentiment, x='release_year', y='normalized_sentiment', hue='artist_name', marker='o', palette='deep')
# Add a simple prediction for the next year's sentiment
for artist in ARTISTS_TO_ANALYZE:
    last_data_point = df_agg_sentiment[df_agg_sentiment['artist_name'] == artist].sort_values(by='release_year', ascending=False).iloc[0]
    last_year = last_data_point['release_year']
    last_sentiment = last_data_point['normalized_sentiment']
    predicted_year = last_year + 1
    predicted_sentiment = last_sentiment # Naive prediction: assume next year is same as last
    plt.plot(
        [last_year, predicted_year], # X-coordinates for line segment
        [last_sentiment, predicted_sentiment], # Y-coordinates for line segment
        linestyle='--', # Dashed line for prediction
        marker='X', # 'X' marker for prediction point
        markersize=10,
        color=sns.color_palette("deep")[ARTISTS_TO_ANALYZE.index(artist)], # Match artist color
        label=f'{artist} (Predicted)'
    )
    # Add text label for the predicted sentiment value
    plt.text(predicted_year + 0.5, predicted_sentiment, f'{predicted_sentiment:.2f}',
             horizontalalignment='left', verticalalignment='center', fontsize=9, color=sns.color_palette("deep")[ARTISTS_TO_ANALYZE.index(artist)])
plt.title('Lyrical Sentiment Evolution Over Time (Eminem vs. Taylor Swift) with Next Year Prediction')
plt.xlabel('Release Year')
plt.ylabel('Average Normalized Sentiment (-1 to 1)')
plt.xticks(rotation=45) # Rotate year labels for readability
plt.legend(title='Artist')
plt.grid(True, linestyle='--', alpha=0.6) # Add a grid
plt.tight_layout() # Adjust plot to prevent labels overlapping
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_evolution_comparative.png')) # Save the plot as an image
plt.close() # Close the plot to free up memory

# --- Visualization (Plotly Interactive Sentiment) ---
# Creates an interactive plot for sentiment evolution that can be viewed in a web browser.
print("\nGenerating interactive Plotly sentiment visualization...")
fig_sentiment_plotly = go.Figure() # Create a new Plotly figure
plotly_colors = px.colors.qualitative.Plotly # Get a color palette for Plotly
for i, artist in enumerate(ARTISTS_TO_ANALYZE):
    artist_data = df_agg_sentiment[df_agg_sentiment['artist_name'] == artist]
    # Add the main sentiment line for the artist
    fig_sentiment_plotly.add_trace(go.Scatter(
        x=artist_data['release_year'],
        y=artist_data['normalized_sentiment'],
        mode='lines+markers', # Show lines and markers
        name=artist,
        line=dict(color=plotly_colors[i % len(plotly_colors)]), # Assign color
        marker=dict(symbol='circle'),
        hovertemplate= # Customize what shows up when you hover over a point
            '<b>Artist</b>: %{customdata[0]}<br>' +
            '<b>Year</b>: %{x}<br>' +
            '<b>Sentiment</b>: %{y:.2f}<extra></extra>',
        customdata=artist_data[['artist_name']].values
    ))
    # Add the prediction point for the next year
    last_data_point = artist_data.sort_values(by='release_year', ascending=False).iloc[0]
    last_year = last_data_point['release_year']
    last_sentiment = last_data_point['normalized_sentiment']
    predicted_year = last_year + 1
    predicted_sentiment = last_sentiment # Naive prediction
    fig_sentiment_plotly.add_trace(go.Scatter(
        x=[last_year, predicted_year],
        y=[last_sentiment, predicted_sentiment],
        mode='lines+markers',
        name=f'{artist} (Predicted)',
        line=dict(dash='dash', color=plotly_colors[i % len(plotly_colors)]), # Dashed line for prediction
        marker=dict(symbol='x', size=10), # 'X' marker for prediction
        hovertemplate= # Customize hover for prediction point
            '<b>Artist</b>: %{customdata[0]}<br>' +
            '<b>Year</b>: %{x}<br>' +
            '<b>Predicted Year</b>: ' + str(predicted_year) + '<br>' +
            '<b>Predicted Sentiment</b>: %{y:.2f}<extra></extra>',
        customdata=[[artist], [artist]]
    ))
fig_sentiment_plotly.update_layout(
    title='Interactive: Lyrical Sentiment Evolution Over Time with Next Year Prediction',
    xaxis_title='Release Year',
    yaxis_title='Average Normalized Sentiment (-1 to 1)',
    hovermode='x unified' # Show unified hover info across traces
)
fig_sentiment_plotly.write_html(os.path.join(OUTPUT_DIR, 'sentiment_evolution_interactive.html')) # Save as HTML
print("Interactive sentiment plot saved.")

# --- Visualization (Plotly Interactive Topic) ---
# Creates interactive stacked bar charts showing how topics change over time for each artist.
print("\nGenerating interactive Plotly topic dominance visualizations...")
for artist in ARTISTS_TO_ANALYZE:
    df_artist_topics = df_agg_topics[df_agg_topics['artist_name'] == artist].set_index('release_year').drop(columns='artist_name')
    if not df_artist_topics.empty:
        # Sort topic columns by their index for consistent display
        topic_cols_sorted = sorted(df_artist_topics.columns, key=lambda x: int(x.split(':')[0].replace('Topic ', '')))
        df_artist_topics_plot = df_artist_topics[topic_cols_sorted].reset_index() # Prepare data for Plotly bar chart
        fig_topic_plotly = px.bar(
            df_artist_topics_plot,
            x='release_year',
            y=topic_cols_sorted, # Each topic column becomes a stack in the bar
            title=f'Interactive: {artist}: Lyrical Topic Dominance Over Time',
            labels={'value': 'Proportion of Lyrics', 'release_year': 'Release Year', 'variable': 'Topic'},
            hover_data={'release_year': True, 'value': ':.2f', 'variable': True}, # Customize hover info
            color_discrete_sequence=px.colors.qualitative.Vivid, # Use a vivid color palette
        )
        fig_topic_plotly.update_layout(barmode='stack') # Ensure bars are stacked
        fig_topic_plotly.write_html(os.path.join(OUTPUT_DIR, f'{artist}_topic_dominance_interactive.html')) # Save as HTML
        print(f"Interactive topic dominance plot for {artist} saved.")

# --- Visualization (Plotly Interactive Entities) ---
# Creates interactive bar charts to show the most frequently mentioned entities for each artist.
print("\nGenerating Top Entities plots (overall per artist)...")
for artist in ARTISTS_TO_ANALYZE:
    df_artist_all_entities = df_filtered[df_filtered['artist_name'] == artist]
    # Flatten all entities for this artist and filter for specific types
    all_ents_for_artist = [
        ent[0] for sublist in df_artist_all_entities['entities'] for ent in sublist
        if ent[1] in ['PERSON', 'ORG', 'GPE', 'LOC']
    ]
    if all_ents_for_artist:
        # Get the top 10 most frequent entities for the artist
        overall_top_entities = pd.Series(all_ents_for_artist).value_counts().head(10).reset_index()
        overall_top_entities.columns = ['entity', 'count']
        # Create a static Matplotlib bar plot for top entities
        plt.figure(figsize=(12, 8))
        sns.barplot(data=overall_top_entities, x='count', y='entity', palette='viridis')
        plt.title(f'{artist}: Overall Top 10 Lyrical Entities')
        plt.xlabel('Total Mentions Across Discography')
        plt.ylabel('Entity')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{artist}_overall_top_entities.png'))
        plt.close()
print("\nGenerating interactive Plotly top entities visualizations...")
for artist in ARTISTS_TO_ANALYZE:
    df_artist_all_entities = df_filtered[df_filtered['artist_name'] == artist]
    all_ents_for_artist = [
        ent[0] for sublist in df_artist_all_entities['entities'] for ent in sublist
        if ent[1] in ['PERSON', 'ORG', 'GPE', 'LOC']
    ]
    if all_ents_for_artist:
        overall_top_entities = pd.Series(all_ents_for_artist).value_counts().head(10).reset_index()
        overall_top_entities.columns = ['entity', 'count']
        # Create an interactive Plotly bar chart for top entities
        fig_ner_plotly = px.bar(
            overall_top_entities,
            x='count',
            y='entity',
            orientation='h', # Horizontal bars
            title=f'Interactive: {artist}: Overall Top 10 Lyrical Entities',
            labels={'count': 'Total Mentions Across Discography', 'entity': 'Entity'},
            hover_data={'entity': True, 'count': True}, # Customize hover info
            color='count', # Color bars based on their count
            color_continuous_scale=px.colors.sequential.Viridis # Use a Viridis color scale
        )
        fig_ner_plotly.update_layout(yaxis={'categoryorder':'total ascending'}) # Sort entities by count
        fig_ner_plotly.write_html(os.path.join(OUTPUT_DIR, f'{artist}_overall_top_entities_interactive.html')) # Save as HTML
        print(f"Interactive top entities plot for {artist} saved.")
    else:
        print(f"No significant entities found for {artist} to plot interactive overall top entities.")
print(f"\nVisualizations saved to '{OUTPUT_DIR}' directory.")
print("Analysis complete.")




