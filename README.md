# Linear Regression ML Service

This project is a **Python-based Machine Learning service** built with **FastAPI**. It allows users to create, train, and use linear regression models via REST API endpoints.

---

##  Features

1. **Create Model** – Create a new linear regression model with configurable options:  
   - `fit_intercept` (bool): Include the intercept in the model  
   - `positive` (bool): Constrain coefficients to be positive  

2. **Train Model** – Train the model on a CSV dataset:  
   - Supports dropping columns (`drop_columns`), converting Yes/No columns (`binary_columns`), and one-hot encoding the `degree` column (`encode_degree`)  
   - Returns model metrics: MSE, R² and number of samples  

3. **Predict** – Make predictions on new data:  
   - Validates input data against feature columns used for training  
   - Returns a list of predicted values  

4. **Get Model Columns** – Retrieve the feature columns expected by the model, so users know what to send for predictions.

---


---

##  Installation & Running the Service

1. **Clone the repository** and enter the project folder:

```bash
git clone <repo-url>
cd project_root

```

2. **Install dependencies**
```bash
pip install -r requirements.txt

```

3. **Run the FastAPI service from your root folder**
```bash
uvicorn app.app:app --reload

```

##  Using the API (Swagger / OpenAPI)

This service provides an **interactive API documentation** powered by **Swagger UI**, which allows you to test endpoints directly from your browser.

1. **Access Swagger UI**  
   After running the service, open your browser and navigate to: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).  
   Here you'll see a user-friendly interface listing all available endpoints, their parameters, and request/response examples.  
   This provides a more detailed, structured view of the API, including data schemas.

2. **Endpoint Usage Tips**

- **Model Matching by Name**:  
  Each model is identified and tracked by its `model_name`.  
  Make sure to **create a model first** using `create_model`, then use the **exact same `model_name`** when training, predicting, or getting columns.  
  Swagger will only execute actions correctly if the model exists and the names match.

- **“Try it out” Button**:  
  In Swagger UI, click **Try it out** for any endpoint before filling parameters and executing the request.  
  This enables editing the input fields (JSON body or query parameters) directly in the browser.

3. **Testing Endpoints via Swagger**  
- **Create Model**: Click on the endpoint, fill in the optional parameters (`fit_intercept`, `positive`), and click **Execute**.  
- **Train Model**: Upload a path to CSV dataset, configure preprocessing options (`drop_columns`, `binary_columns`, `encode_degree`) and click **Execute** to train the model and view metrics.  
- **Predict**: Provide new input data in JSON format according to the model’s feature columns and click **Execute** to receive predictions.  
- **Get Model Columns**: Click **Execute** to retrieve the list of feature columns expected by the model.



> Note: Always check Get Model Columns before making predictions to ensure your input data matches the trained model's expected features.



