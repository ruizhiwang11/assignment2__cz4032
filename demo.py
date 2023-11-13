import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from collections import defaultdict

# Function to encode a chunk of transactions
def encode_transactions(transactions, te):
    te_ary = te.transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

# Initialize TransactionEncoder
te = TransactionEncoder()

# Parameters
chunk_size = 50000  # Adjust chunk size to your system's memory limit
min_support = 0.01  # Adjust the minimum support value as needed
total_transactions = 0  # Keep track of total number of transactions
all_items_set = set()  # This will hold all unique items
selected_columns = ['Primary Type', 'Arrest', 'Domestic']

# First pass: create the set of all unique items
for chunk in pd.read_csv("Crimes_-_2001_to_Present.csv", chunksize=chunk_size):
    chunk['Arrest'] = chunk['Arrest'].apply(lambda x: 'Arrest' if x else 'No Arrest')
    chunk['Domestic'] = chunk['Domestic'].apply(lambda x: 'Domestic' if x else 'No Domestic')
    chunk = chunk.astype(str)
    transactions = chunk[selected_columns].values.tolist()
    all_items_set.update([item for sublist in transactions for item in sublist])

# Fit the TransactionEncoder to all items
te.fit([list(all_items_set)])

# Initialize dictionary to keep track of itemset support across all chunks
itemset_support_dict = defaultdict(int)

# Second pass: read the CSV file in chunks and process each chunk
for chunk in pd.read_csv("Crimes_-_2001_to_Present.csv", chunksize=chunk_size):
    chunk['Arrest'] = chunk['Arrest'].apply(lambda x: 'Arrest' if x else 'No Arrest')
    chunk['Domestic'] = chunk['Domestic'].apply(lambda x: 'Domestic' if x else 'No Domestic')
    chunk = chunk.astype(str)
    transactions = chunk[selected_columns].values.tolist()
    df_encoded = encode_transactions(transactions, te)
    chunk_frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    for index, row in chunk_frequent_itemsets.iterrows():
        itemset = frozenset(row['itemsets'])
        support = row['support']
        itemset_support_dict[itemset] += support * len(chunk)
    total_transactions += len(chunk)

# Convert support counts to relative support values
for itemset in itemset_support_dict.keys():
    itemset_support_dict[itemset] /= total_transactions

# Filter itemsets by global support to get the final frequent itemsets
global_frequent_itemsets = {itemset for itemset, support in itemset_support_dict.items() if support >= min_support}

# Display the global frequent itemsets
print(global_frequent_itemsets)