from sklearn.preprocessing import OneHotEncoder

def show_num_samples_per_client(partitions, number_of_clients):
    for client_id in range(number_of_clients):
        print(f"Client {client_id} → {partitions.load_partition(client_id).num_rows} samples")
        
        
def preprocess_data(pdf):
    """
    Preprocesses the input pandas DataFrame by:
    1. Replacing '?' with NaN and dropping rows with NaN values.
    2. Renaming columns to replace '.' with '_'.
    3. One-hot encoding specified categorical columns ('workclass', 'occupation', 'native_country', 'income').
    4. Concatenating the original DataFrame with the one-hot encoded DataFrame.
    5. Dropping the original categorical columns and other specified columns ('race', 'relationship', 'marital_status', 'fnlwgt', 'education', 'sex').

    Args:
        pdf (pd.DataFrame): The input pandas DataFrame.

    Returns:
        pd.DataFrame: The preprocessed pandas DataFrame.
    """    
    pdf.replace("?", np.nan, inplace=True)
    pdf = pdf.dropna()
    
    new_col_name = []

    for col in pdf.columns:
        new_col_name.append(col.replace(".", "_"))
        
    pdf = pdf.rename(columns=dict(zip(pdf.columns, new_col_name)))
        
    columns_to_encode = ["workclass", "occupation", "native_country", "income"]

    encoder = OneHotEncoder(drop = 'first', handle_unknown="error", dtype=np.int8)
    encoded_data = encoder.fit_transform(pdf[columns_to_encode]).toarray()

    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(), index=pdf.index)
    
    pdf_final = pd.concat([pdf, encoded_df], axis=1)

    pdf_final = pdf_final.drop(columns=["workclass", "occupation", "native_country", "income", "race", "relationship", "marital_status", "fnlwgt", "education", "sex"])
    
    return pdf_final