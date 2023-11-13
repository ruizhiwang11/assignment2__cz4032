import pandas as pd
from itertools import combinations
from collections import defaultdict

# Define the Apriori function for processing a single chunk
def apriori_on_chunk(chunk, min_support):
    # Here we assume chunk is a list of lists (transactions)
    # Each inner list is a transaction and contains items

    # Step 1: Count how many times each item appears
    item_count = defaultdict(int)
    for transaction in chunk:
        for item in transaction:
            item_count[item] += 1

    # Step 2: Keep only items that meet the min_support
    num_transactions = len(chunk)
    frequent_items = {item for item, count in item_count.items() if count / num_transactions >= min_support}

    # Step 3: Create combinations of these items to form itemsets
    # Note: This is a very basic and not optimized way of doing it
    # In practice, you would use more efficient data structures and algorithms
    itemsets = list(frequent_items)
    frequent_itemsets = []
    for k in range(2, len(itemsets) + 1):
        for combo in combinations(itemsets, k):
            itemset = frozenset(combo)
            itemset_count = sum(1 for transaction in chunk if itemset.issubset(transaction))
            if itemset_count / num_transactions >= min_support:
                frequent_itemsets.append((itemset, itemset_count))

    return frequent_itemsets

# Define the function to read the CSV in chunks
def get_chunks(filename, chunk_size):
    return pd.read_csv(filename, chunksize=chunk_size)

# Define the main function to run the disk-based Apriori
def disk_based_apriori(filename, min_support):
    global_itemsets_counts = defaultdict(int)

    # Process each chunk
    for chunk in get_chunks(filename, chunk_size=80):
        # Pre-process the chunk to fit the expected format for the apriori_on_chunk function
        # Assuming each row is a transaction and we select a subset of columns to consider
        transactions = chunk[['Primary Type', 'Location Description', 'Arrest', 'Domestic']].values.tolist()
        # Find frequent itemsets in this chunk

        local_frequent_itemsets = apriori_on_chunk(transactions, min_support)
        print(local_frequent_itemsets)
        # Merge with global itemsets counts
        for itemset, count in local_frequent_itemsets:
            global_itemsets_counts[frozenset(itemset)] += count
        print(global_itemsets_counts)

    # Filter global itemsets by min_support (considering all chunks)
    total_transactions = sum(1 for chunk in get_chunks(filename, chunk_size=80) for _ in chunk.index)
    global_frequent_itemsets = {itemset: count for itemset, count in global_itemsets_counts.items()
                                if count / total_transactions >= min_support}

    return global_frequent_itemsets

# Run the disk-based Apriori
filename = 'Crimes_-_2001_to_Present.csv'
min_support = 0.01  # e.g., 1%
frequent_itemsets = disk_based_apriori(filename, min_support)
print(frequent_itemsets)
