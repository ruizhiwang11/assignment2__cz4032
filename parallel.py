import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from collections import defaultdict
from tqdm import tqdm

total_rows = sum(1 for _ in pd.read_csv("trimmed_file.csv", usecols=['Trip ID']))


chunk_size = 1000  # Adjust chunk size to your system's memory limit
min_support = 0.01  # Minimum support threshold
all_items_set = set()  # This will hold all unique items

# Pass 1: Iterate over the CSV file in chunks to collect all unique items
for chunk in pd.read_csv("trimmed_file.csv", chunksize=chunk_size):
    # We are interested in these columns for the Apriori algorithm
    chunk['Pickup Community Area'] = 'Pickup CA ' + chunk['Pickup Community Area'].astype(str)
    chunk['Dropoff Community Area'] = 'Dropoff CA ' + chunk['Dropoff Community Area'].astype(str)
    chunk['Payment Type'] = 'Payment ' + chunk['Payment Type'].astype(str)
    
    # Update the set of all unique items
    transactions = chunk[['Pickup Community Area', 'Dropoff Community Area', 'Payment Type']].values.tolist()
    all_items_set.update(item for sublist in transactions for item in sublist if str(item) != 'nan')

# Initialize TransactionEncoder
te = TransactionEncoder()
te.fit([list(all_items_set)])  # Fit the encoder with all unique items

# Initialize dictionary to keep track of itemset support across all chunks
itemset_support_dict = defaultdict(int)

# Pass 2: Re-iterate over the CSV file in chunks and process each chunk
for chunk in tqdm(pd.read_csv("trimmed_file.csv", chunksize=chunk_size), total=total_rows/chunk_size, desc='Processing Chunks'):
    # We are interested in these columns for the Apriori algorithm
    chunk['Pickup Community Area'] = 'Pickup CA ' + chunk['Pickup Community Area'].astype(str)
    chunk['Dropoff Community Area'] = 'Dropoff CA ' + chunk['Dropoff Community Area'].astype(str)
    chunk['Payment Type'] = 'Payment ' + chunk['Payment Type'].astype(str)
    
    # Prepare the transactions
    transactions = chunk[['Pickup Community Area', 'Dropoff Community Area', 'Payment Type']].values.tolist()
    transactions = [[item for item in transaction if str(item) != 'nan'] for transaction in transactions]
    
    # Encode the transactions
    df_encoded = te.transform(transactions)
    df_encoded = pd.DataFrame(df_encoded, columns=te.columns_)
    
    # Apply the Apriori algorithm to find frequent itemsets in the chunk
    chunk_frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    # Update global itemset support counts
    for index, row in chunk_frequent_itemsets.iterrows():
        itemset = row['itemsets']
        support = row['support']
        itemset_support_dict[frozenset(itemset)] += support * len(chunk)

# Calculate the relative support values
for itemset in itemset_support_dict.keys():
    itemset_support_dict[itemset] /= total_rows

# Filter itemsets by global support to get the final frequent itemsets
global_frequent_itemsets = {itemset for itemset, support in itemset_support_dict.items() if support >= min_support}

# Display the global frequent itemsets
print(global_frequent_itemsets)