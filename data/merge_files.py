import pandas as pd
import glob

csv_files = glob.glob('data/shopee_scincare_reviews*.csv')
dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

data = pd.concat(dataframes, ignore_index=True)

data.to_csv('shopee_scincare_reviews_final.csv', index=False)

print("Merged data saved as 'shopee_scincare_reviews_final.csv'.")


