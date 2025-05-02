import pandas as pd
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split

#IMPORTS
# bs4 clean_text() function to strip out any HTML tags so you’re left with plain text.


# Load Dataset 
df = pd.read_csv("./data/Kaggle.csv")

# Inspect missing values in key columns
missing_summary = df[['sender','receiver','subject','body','label']].isnull().sum()

# Drop rows with missing 'body' or 'label'
df_clean = df.dropna(subset=['body', 'label']).copy()

# Define cleaning function for body text
def clean_text(text):
    # Remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning to body
df_clean['clean_body'] = df_clean['body'].apply(clean_text)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    df_clean[['clean_body']],
    df_clean['label'],
    test_size=0.2,
    random_state=42,
    stratify=df_clean['label']
)

# Print summaries and sample data
print("Missing values before drop:\n", missing_summary)
print(f"\nOriginal rows: {df.shape[0]}")
print(f"Rows after drop: {df_clean.shape[0]}\n")

print("Sample cleaned training emails:")
print(pd.concat([X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1).head(10))
