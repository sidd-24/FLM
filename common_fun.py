from sklearn.preprocessing import OneHotEncoder
from flwr_datasets.visualization import plot_label_distributions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def showDataDist(partition_name, column):
    _ = plot_label_distributions(
        partitioner=partition_name.partitioners["train"],
        label_name=column,
        legend=True
        # colors=colors
    )

def show_num_samples_per_client(partitions, number_of_clients):
    for client_id in range(number_of_clients):
        print(f"Client {client_id} → {partitions.load_partition(client_id).num_rows} samples")
        
        
def preprocess_data(pdf):

    pdf.replace("?", np.nan, inplace=True)
    pdf = pdf.dropna()
    
    new_col_name = []

    for col in pdf.columns:
        new_col_name.append(col.replace(".", "_"))
        
    pdf = pdf.rename(columns=dict(zip(pdf.columns, new_col_name)))
        
    columns_to_encode = ["workclass", "occupation", "income"]
# , "native_country"
    encoder = OneHotEncoder(drop = 'first', handle_unknown="error", dtype=np.int8)
    encoded_data = encoder.fit_transform(pdf[columns_to_encode]).toarray()

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(), index=pdf.index)
    
    pdf_final = pd.concat([pdf, encoded_df], axis=1)

    pdf_final = pdf_final.drop(columns=["workclass", "occupation", "native_country", "income", "race", "relationship", "marital_status", "fnlwgt", "education", "sex", "fnlwgt"])
    
    return pdf_final

def plot_number_of_features(dict):
    df = pd.DataFrame(dict.items(), columns=['Partition', 'Number of Features'])
    # plt.figure(figsize=(10, 6))
    plt.bar(df['Partition'], df['Number of Features'])
    plt.xlabel('Partitions', fontsize=12)
    plt.ylabel('Number of Features', fontsize=12)
    plt.title('Number of Features per Partition', fontsize=14)
    plt.tight_layout()
    plt.savefig('features_per_partition.png')
    plt.show()