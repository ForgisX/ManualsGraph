# Setup and Installation

This project uses `uv` for lightning-fast Python package and environment management.

## Prerequisites

-   **Python 3.12**: Recommended. The project is compatible with 3.10 to 3.13.
-   **Azure OpenAI API Access**: Required for AI-driven classification and extraction.
-   **uv**: [Install uv](https://github.com/astral-sh/uv) if you haven't already.

> [!WARNING]
> **Python 3.14 is currently NOT supported** due to compatibility issues with critical dependencies like `pandas`.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd ManualsGraph
    ```

2.  **Create a virtual environment**:
    ```bash
    uv venv --python 3.12
    ```

3.  **Activate the environment**:
    -   Windows: `.venv\Scripts\activate`
    -   macOS/Linux: `source .venv/bin/activate`

4.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **Configure Environment Variables**:
    Create a `.env` file in the root directory and add your Azure OpenAI credentials:
    ```env
    AZURE_OPENAI_API_KEY="your-key"
    AZURE_OPENAI_ENDPOINT="your-endpoint"
    AZURE_OPENAI_DEPLOYMENT_NAME="your-model-deployment-name"
    AZURE_OPENAI_API_VERSION="2024-02-15-preview"
    ```

## Development Tools

We use `pre-commit` to ensure code quality.

1.  **Install pre-commit hooks**:
    ```bash
    pre-commit install
    ```
2.  **Run manually**:
    ```bash
    pre-commit run --all-files
    ```
