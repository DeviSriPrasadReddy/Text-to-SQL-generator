Genie – Known Limitations:
Token Capacity: Maximum support of 14,000 tokens.
Throughput Constraints: Can handle no more than 10 users or 5 queries per minute at peak.
Query Complexity: Limited to the capabilities defined within the Genie environment.

Visualizations:
When datasets exceed 100 rows, generating visualizations becomes challenging due to limitations in both the frontend and the LLM.

Integration of Emily’s Code
Emily’s TypeScript implementation is incompatible with our current Python-based backend. Converting the backend to TypeScript would be required to adopt her approach.
The existing Databricks (Flask) application lacks clean, easily readable function structures.

Custom Solution:
A custom-built solution could be completed within 2–3 sprints (including MLflow integration), but initial accuracy would be low.
