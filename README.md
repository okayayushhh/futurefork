# FutureFork

FutureFork is an AI-powered metabolic health assistant that helps you understand the glycemic impact of your meals. By analyzing food images, it predicts glucose curves and provides personalized insights to support healthier eating decisions.

## Features

- **AI-Powered Food Analysis**: Uses Gemini to identify foods in images and estimate nutritional content.
- **Glycemic Response Prediction**: Generates predicted glucose curves (low, medium, high GI) based on meal composition.
- **What-If Analysis**: Compare the metabolic impact of swapping ingredients (e.g., white rice vs. brown rice).
- **Clinical Design**: A clean, professional interface inspired by medical documentation.
- **Privacy-Conscious**: Local `.env` file for API keys and a `.gitignore` to prevent accidental commits.

## Getting Started

### Prerequisites

- Python 3.8+
- An API key from Google AI Studio for the Gemini API

### Installation

1.  **Clone the repository** (or download the source code).

2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**:
    Create a `.env` file in the root directory with your API key:
    ```env
    GEMINI_API_KEY="your_gemini_api_key_here"
    ```

### Running the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1.  **Upload or Capture**: Use the sidebar to upload a photo of your meal or take a live photo using your webcam.
2.  **Analyze**: The app will process the image and display the predicted glucose curve.
3.  **Explore**: View detailed metrics and use the "What If?" feature to experiment with different food choices.

## Development

### Testing

To run the unit tests for the `nutrition_math` module:

```bash
pytest
```

### Code Style

This project uses `ruff` for linting and formatting.

- **Check formatting**: `ruff check`
- **Fix formatting**: `ruff check --fix`

## License

MIT