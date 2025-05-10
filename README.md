# AI for Ayurveda

## demo






https://github.com/user-attachments/assets/d8f33788-c5b9-4d75-978a-01e92dec832b





## Overview
"AI for Ayurveda" is a web app combining AI with Ayurveda for personalized health solutions via:
- **Prakriti Analysis**: Facial image-based dosha classification (Vata, Pitta, Kapha).
- <div style="display: flex; justify-content: center; gap: 10px;">
  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/e4bfa649-65bc-4063-810a-ae313ec1b1a3"  width="50" height="50">
    <p><em>vata</em></p>
  </div>

  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/311c7fac-9489-4106-abce-0b86bc17c085"  width="50" height="50">
    <p><em>pitta</em></p>
  </div>

  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/8ae44dec-0f61-4092-b5bd-3072bd2dd037"  width="50" height="50">
    <p><em>kapha</em></p>
  </div>  
</div>
- **Chatbot**: Real-time Ayurvedic health advice tailored to Prakriti.

<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/9228dfc8-ea42-4204-8136-fd52bbb70d5d" alt="Architecture" width="200" height="200">
    <p><em>System Architecture</em></p>
  </div>
</div>

## Key Features

- **Prakriti Analysis**: Uses CNNs (MobileNetV2, ResNet50) and Random Forest for dosha reports.
- **Chatbot**: LangChain and Groq LLM for personalized advice.
- **Interface**: Responsive HTML/CSS/JS with image upload and chat.

## Technical Stack
- **Backend**: FastAPI, Python
- **ML**: TensorFlow/Keras, Scikit-learn, MediaPipe
- **NLP**: LangChain, HuggingFace, Chroma, Groq LLM
- **Frontend**: HTML, CSS, JavaScript
- **Other**: Jinja2, ngrok

## Installation
1. Clone: `git clone https://github.com/dharmthummar/ai-ayurveda-assistant.git`
2. Install: `pip install -r requirements.txt` (Python 3.8+)
3. Set `.env`: `GROQ_API_KEY=your_api_key`
4. Run: `uvicorn main:app --reload`

## Usage
- Visit `http://localhost:8000`.
- Analyze Prakriti via webcam/image.
- Chat for Ayurvedic advice.
- Explore demo pages.

## Repository Structure
- `README.md`: Project guide
- `requirements.txt`: Dependencies
- `src/`: Python files
- `templates/`: HTML
- `static/`: JS/CSS
- `data/`: (Optional) Ayurvedic texts

## Notes
- Use ngrok for dev; deploy on Heroku/AWS for production.
- Requires Groq API key
