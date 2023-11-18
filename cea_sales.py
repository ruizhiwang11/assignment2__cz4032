import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from collections import defaultdict
from tqdm import tqdm

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MultiLabelBinarizer


# Function to encode a chunk of transactions
def encode_transactions(transactions, te):
    te_ary = te.transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

# Initialize TransactionEncoder
te = TransactionEncoder()

# Parameters
chunk_size = 600000  # Adjust chunk size to your system's memory limit
min_support = 0.08  # Adjust the minimum support value as needed
total_transactions = 0  # Keep track of total number of transactions
all_items_set = set()  # This will hold all unique items
selected_columns = ['salesperson_name', 'property_type', 'transaction_type', "represented"]  # Example columns from your dataset

# First pass: create the set of all unique items
for chunk in tqdm(pd.read_csv("CEASalespersonsPropertyTransactionRecordsresidential.csv", chunksize=chunk_size, usecols=selected_columns)):
    # Convert columns to string to create unique items
    chunk = chunk.astype(str)
    transactions = chunk[selected_columns].values.tolist()
    all_items_set.update([item for sublist in transactions for item in sublist])

# Fit the TransactionEncoder to all items
te.fit([list(all_items_set)])

# Initialize dictionary to keep track of itemset support across all chunks
itemset_support_dict = defaultdict(int)

# Second pass: read the CSV file in chunks and process each chunk
for chunk in tqdm(pd.read_csv("CEASalespersonsPropertyTransactionRecordsresidential.csv", chunksize=chunk_size, usecols=selected_columns)):
    # Convert columns to string to create unique items
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


df = pd.read_csv("CEASalespersonsPropertyTransactionRecordsresidential.csv")

# Define the columns that you consider as items for the Apriori algorithm
selected_columns = ['salesperson_name', 'property_type', 'transaction_type', "represented"]

# Create transactions: Each transaction is a list of items
all_transactions = df.apply(lambda x: x.tolist(), axis=1).tolist()


unique_items = {item for itemset in global_frequent_itemsets for item in itemset}

# Initialize MultiLabelBinarizer with the unique items obtained from global_frequent_itemsets
mlb = MultiLabelBinarizer(classes=sorted(unique_items))

# Prepare the binary matrix for the given transactions
binary_matrix = mlb.fit_transform(all_transactions)

# Now, use the binary_matrix for clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans_labels = kmeans.fit_predict(binary_matrix)

agglo = AgglomerativeClustering(n_clusters=2)
agglo_labels = agglo.fit_predict(binary_matrix)

# Create a DataFrame to hold the cluster labels
clustered_df = pd.DataFrame({
    'KMeans_Cluster': kmeans_labels,
    'Agglomerative_Cluster': agglo_labels
})

# Display the DataFrame with cluster labels
print(clustered_df.head())