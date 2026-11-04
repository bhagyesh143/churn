\# Lab 10 - ML Model Serving and API Engineering using FastAPI



\## 1. Objective



The objective of Lab 10 is to deploy the trained machine learning model as a FastAPI inference service and evaluate its API functionality, validation behavior, robustness, performance, and reliability.



\---



\## 2. Implementation Completed



The following components were implemented:



\- FastAPI application

\- Health endpoint

\- ModelLoader

\- Model and preprocessor loading

\- Metadata endpoint

\- Single prediction endpoint

\- Pydantic input/output schemas

\- Request validation

\- Standardized error responses

\- Batch prediction endpoint

\- API testing

\- Invalid and corrupted input testing

\- Performance testing

\- Reliability testing

\- Throughput measurement



\---



\## 3. API Endpoints



\### GET `/`



Returns basic API service information.



\### GET `/health`



Checks the API health endpoint.



\### GET `/metadata`



Returns model and service metadata.



\### POST `/predict`



Accepts one customer record and returns:



\- Prediction

\- Churn probability



\### POST `/predict/batch`



Accepts multiple customer records and returns predictions for all customers.



\---



\## 4. Input Validation



Pydantic schemas are used to validate API requests.



Validation includes:



\- Required fields

\- Categorical values

\- Numeric datatypes

\- Non-negative tenure

\- Non-negative monthly charges

\- Batch size validation

\- Null-value rejection

\- Invalid datatype detection



\---



\## 5. Standardized Error Handling



Validation failures return HTTP 422 responses.



The standardized error response contains:



\- `status`

\- `message`

\- `details`



Invalid input was tested by sending an unsupported gender value. The API returned HTTP 422 with a descriptive validation message.



\---



\## 6. Testing Results



The following tests passed during local testing:



| Test category | Passed | Failed |

|---|---:|---:|

| API tests | 23 | 0 |

| Performance tests | 6 | 0 |

| \*\*Total\*\* | \*\*29\*\* | \*\*0\*\* |



\### Performance and Reliability



The performance tests covered:



\- Single-request latency

\- Repeated-request latency

\- Prediction consistency

\- API reliability

\- Batch prediction performance

\- Throughput



All 29 tests passed during the recorded local test run.



\---



\## 7. Deployment Artifacts



The deployment directory contains:



\- `deploy\_production.py`

\- `model.pkl`

\- `preprocessor.pkl`

\- `model\_metadata.json`



The model and preprocessor were successfully loaded using Joblib.



\---



\## 8. API Execution



The FastAPI server started successfully using Uvicorn.



The following endpoints were checked:



\- `/`

\- `/health`

\- `/metadata`

\- `/predict`

\- `/predict/batch`



The Swagger UI at `/docs` was used to test API requests, including invalid input validation and batch prediction.



\---



\## 9. Conclusion



Lab 10 implemented a FastAPI-based machine learning inference service with model loading, input validation, single and batch prediction, and inference testing.



All 29 recorded local automated tests passed. The API also started successfully and responded to requests through Swagger UI.





