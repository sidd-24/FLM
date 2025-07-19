def show_num_samples_per_client(partitions, number_of_clients):
    for client_id in range(number_of_clients):
        print(f"Client {client_id} → {partitions.load_partition(client_id).num_rows} samples")