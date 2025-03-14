Here’s an improved version of your README.md with the correct run script:

# exolab-agents

Exolab-Agents is a Python microservice designed to guide the AI agents used inside the **ExoLab Product**.

## Installation

Ensure you have **Python 3** installed, then install the required dependencies:

```bash
pip install -r requirements.txt

Running the Service

To start the FastAPI server, use the following command:

python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

	•	--host 0.0.0.0 → Allows access from any network.
	•	--port 8000 → Runs the server on port 8000 (change if needed).
	•	--workers 4 → Runs with 4 worker processes for better performance.

API Documentation

Once the server is running, access the API documentation:
	•	Swagger UI: http://127.0.0.1:8000/docs
	•	ReDoc UI: http://127.0.0.1:8000/redoc

License

This project is licensed under the MIT License.

⸻

Let me know if you want any modifications! 🚀

