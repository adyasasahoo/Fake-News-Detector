print("Script is starting...")

import pandas as pd

# Load CSV files
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Add labels
fake['label'] = 0
real['label'] = 1

# Keep only needed columns
fake = fake[['title', 'label']]
real = real[['title', 'label']]

# Combine and shuffle
combined = pd.concat([fake, real]).sample(frac=1).reset_index(drop=True)

# Save combined dataset
combined.to_csv('fake_news_headlines.csv', index=False)

print("Dataset created successfully!")