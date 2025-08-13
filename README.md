# Federated Learning Management: Influence of Design Choices on Model Performance

This project explores the impact of different federated learning (FL) strategies on model performance, specifically in a setting with non-identically distributed (non-IID) data. It compares a centralized PyTorch model with synchronous (FedProx) and asynchronous (FedAsync) aggregation methods to analyze their stability and effectiveness.

---

## 📊 Dataset: Adult Census Income

The project uses the **Adult Census Income** dataset to predict whether an individual's income exceeds $50,000 per year.

* **Partitioning**: The data is distributed among 6 clients using a **Dirichlet distribution** based on the `income` column. This creates a challenging non-IID scenario, simulating real-world data heterogeneity.
* **Preprocessing**:
    * Missing values (denoted by "?") are removed.
    * Categorical features (`workclass`, `occupation`, `income`) are one-hot encoded.
    * Irrelevant features are dropped to simplify the model.

---

## 🤖 Model Architecture

A Multi-Layer Perceptron (MLP) model is implemented in PyTorch for the binary classification task.

* **Input Layer**: Accepts 24 features from the preprocessed dataset.
* **Hidden Layers**: Two hidden layers with 128 and 64 neurons, respectively, using ReLU activation.
* **Regularization**: A dropout layer with a rate of 0.2 is included to prevent overfitting.
* **Output Layer**: A single neuron for the final binary prediction.
* **Loss Function**: `BCEWithLogitsLoss` with a positive class weight to handle class imbalance.
* **Optimizer**: Adam optimizer with a learning rate of `0.001`.

---

## 🚀 Federated Learning Strategies

The core of the project is the comparison of different FL aggregation strategies.

* **Centralized Baseline**: A standard MLP model trained on the entire dataset to establish a performance benchmark.
* **FedProx (Synchronous)**: All 6 clients participate in each training round, providing a stable and consistent aggregation method.
* **FedAsync (2 Clients)**: An asynchronous approach where a minimum of 2 clients (33%) are required for a training round, simulating lower client availability.
* **FedAsync (4 Clients)**: An asynchronous approach requiring a minimum of 4 clients (66%), representing higher client availability.

---

## 📈 Results and Key Findings

The experiments highlight the critical role of the aggregation strategy in non-IID environments.

* **Stability**: Synchronous strategies (**FedProx**) provide consistent and stable performance, while asynchronous strategies are more volatile, with performance fluctuating based on which clients participate in a round.
* **Client Participation**: The stability of asynchronous models improves with a higher number of participating clients. **FedAsync_4** was more stable than **FedAsync_2**.
* **Performance**: The synchronous FedProx model achieved a final accuracy of **~80-82%**, which is comparable to the centralized baseline, proving the effectiveness of FL without centralizing data.
* **Non-IID Challenge**: The instability observed in asynchronous models is a direct result of the non-IID data distribution, as all models performed stably on IID data.

---

## ⚙️ How to Run

To replicate the experiments, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sidd-24/FLM.git
    ```

2.  **Set up the environment:**
    It is recommended to use a virtual environment. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the experiments:**
    * For the **centralized baseline**, run the `experiment_pytorch.ipynb` notebook.
    * For the **federated learning experiments**, run the `experiment_non-iid_pytorch.ipynb` notebook. You can modify the `IID_SENARIO` and strategy selection within the notebook to test different configurations.

---

## 🙌 Acknowledgments


This project was developed as part of the Federated Learning Management course at **Friedrich-Alexander-Universität Erlangen-Nürnberg**, under the supervision of Prof. Kristina Müller.

---

Authors: Prem Prakash Singh, Siddhant Chindhe, Pratiksha Katap

