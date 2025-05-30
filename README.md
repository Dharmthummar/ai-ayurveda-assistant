# AI for Ayurveda
[![LINK](https://img.shields.io/badge/Visit%20Website-8D6F64)](https://dharmthummar.github.io/ai-ayurveda-assistant/)
### Demonstation video





https://github.com/user-attachments/assets/d8f33788-c5b9-4d75-978a-01e92dec832b





## Overview
"AI for Ayurveda" is a web app combining AI with Ayurveda for personalized health solutions via:
- **Prakriti Analysis**: Facial image-based dosha classification (Vata, Pitta, Kapha).
 <table style="margin: 0 auto; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/e4bfa649-65bc-4063-810a-ae313ec1b1a3" alt="Vata" width="50" height="50">
      <p><em>Vata</em></p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/311c7fac-9489-4106-abce-0b86bc17c085" alt="Pitta" width="50" height="50">
      <p><em>Pitta</em></p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/8ae44dec-0f61-4092-b5bd-3072bd2dd037" alt="Kapha" width="50" height="50">
      <p><em>Kapha</em></p>
    </td>
  </tr>
</table>


- **Chatbot**: A system providing real-time Ayurvedic health advice, personalized according to individual Prakriti, powered by a Mistral-7B multilingual large language model specifically fine-tuned for healthcare applications using Groq API.

<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <img src="https://github.com/user-attachments/assets/9228dfc8-ea42-4204-8136-fd52bbb70d5d" alt="Architecture" width="500" height="500">
    <p><em>System Architecture</em></p>
  </div>
</div>

## Key Features

- **Prakriti Analysis**: Apply MediaPipe to detect landmarks and compute geometric ratios, such as the face width-to-height ratio.
<table border="1">
  <tr>
    <th>Prakriti</th>
    <th>Mouth, Ear, Nose</th>
  </tr>
  <tr>
    <td>Vata</td>
    <td>Width &lt; Height</td>
  </tr>
  <tr>
    <td>Pitta</td>
    <td>Width = Height</td>
  </tr>
  <tr>
    <td>Kapha</td>
    <td>Width &gt; Height</td>
  </tr>
</table>

<table style="margin: 0 auto; text-align: center;">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/88656fbb-9467-479c-8f22-3c7347990c62"  width="200" height="200">
      <p><em>Face landmark</em></p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/e0e91a16-70b1-4221-ba86-be5f075365fd"  width="200" height="200">
      <p><em>Calculate all features</em></p>
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/ff0ddba7-7ba0-4739-8365-19918eb08648" width="200" height="200">
      <p><em>Analysis</em></p>
    </td>
  </tr>
</table>


  


hear is the table gives more features for classification which also helps to parkriti identification  


<table border="1">
  <tr>
    <th>Feature</th>
    <th>Vata</th>
    <th>Pitta</th>
    <th>Kapha</th>
  </tr>
  <tr>
    <td>Cheeks</td>
    <td>Sunken</td>
    <td>Flat</td>
    <td>Plump</td>
  </tr>
  <tr>
    <td>Face Shape</td>
    <td>Angular</td>
    <td>Triangular</td>
    <td>Rounded</td>
  </tr>
  <tr>
    <td>Eyes</td>
    <td>Small, dry</td>
    <td>Bright, sensitive</td>
    <td>Large, calm</td>
  </tr>
  <tr>
    <td>Nose</td>
    <td>Uneven</td>
    <td>Pointed, red tip</td>
    <td>Rounded</td>
  </tr>
  <tr>
    <td>Lips</td>
    <td>Dry, cracked</td>
    <td>Red, inflamed</td>
    <td>Smooth, pale</td>
  </tr>
  <tr>
    <td>Teeth</td>
    <td>Big, thin gums</td>
    <td>Medium, soft gums</td>
    <td>White, strong</td>
  </tr>
  <tr>
    <td>Skin Tone</td>
    <td>Dark, rough</td>
    <td>Rosy, oily</td>
    <td>Pale, thick</td>
  </tr>
  <tr>
    <td>Hair</td>
    <td>Dry, brittle</td>
    <td>Straight, prone to balding</td>
    <td>Thick, curly</td>
  </tr>
  <tr>
    <td>Acne</td>
    <td>Dry patches</td>
    <td>Inflamed, red</td>
    <td>Mild, oily</td>
  </tr>
</table>



  
- **Chatbot**: LangChain and Groq LLM mistral-7B for personalized advice.
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
<!---
![over](https://github.com/user-attachments/assets/88c93995-d6c5-4c24-8c47-a25472f50017)
![future](https://github.com/user-attachments/assets/e3d6aa6b-d369-4011-a9f7-c979395a2274)
![background](https://github.com/user-attachments/assets/bd5c9b73-80b7-4267-884f-7c0d5b8194d5)
--->
