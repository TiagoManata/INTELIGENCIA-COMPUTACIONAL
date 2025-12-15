# NeuroNet PSO - Surgical Instrument Classifier

## Overview
A Streamlit web application for classifying surgical instruments using a pre-trained Keras/TensorFlow neural network model. Users can upload a .h5 or SavedModel model file and images to classify surgical instruments into 5 categories.

## Current State
- Fully functional classification app with batch processing
- Supports .h5 and SavedModel (.zip) model uploads
- Supports JPG, PNG, JPEG image formats
- Displays predictions with confidence scores and probability bar charts
- Export results to CSV/JSON
- Model metrics and architecture visualization
- Image preprocessing pipeline visualization

## Project Architecture

### Files
- `app.py` - Main Streamlit application with model loading, image preprocessing, and classification
- `.streamlit/config.toml` - Streamlit server configuration
- `pyproject.toml` - Python dependencies

### Key Features
1. **Model Loader** - Loads Keras .h5 models or SavedModel (.zip) with caching for performance
2. **Image Preprocessor** - Resizes images to 224x224, normalizes to 0-1 range, adds batch dimension
3. **Batch Classification** - Process multiple images at once with progress tracking
4. **Classification Display** - Shows predicted class, confidence percentage, and probability bar chart
5. **Export Results** - Download classification results as CSV or JSON
6. **Model Metrics** - View layer count, parameters, and architecture table
7. **Preprocessing Preview** - Visualize image transformation steps

### Class Labels (Portuguese)
- Bisturi (Scalpel)
- Hemostática (Hemostatic)
- Tesoura Curva (Curved Scissors)
- Tesoura Reta (Straight Scissors)
- Pinça (Forceps)

### Model Format Support
- **H5 (.h5)**: Standard Keras model format
- **SavedModel (.zip)**: TensorFlow SavedModel format compressed in a zip file
  - Note: When zipping a SavedModel, the zip should contain the SavedModel directory at the root level

## Dependencies
- streamlit
- tensorflow-cpu
- numpy
- pandas
- Pillow

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Recent Changes
- 2025-12-08: Added batch image classification with progress tracking
- 2025-12-08: Added image preprocessing visualization tab
- 2025-12-08: Added model metrics and architecture display tab
- 2025-12-08: Added CSV/JSON export functionality
- 2025-12-08: Added SavedModel format support
- 2025-12-08: Initial implementation with model upload, image classification, and probability visualization
