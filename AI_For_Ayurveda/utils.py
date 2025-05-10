import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io

# Initialize MediaPipe Face Mesh globally
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Dosha characteristics from Ayurvedic texts
DOSHA_CHARACTERISTICS = {
    "vata": {
        "face_shape": "long, thin, or angular",
        "skin": "dry, rough, thin, cool to touch",
        "eyes": "small, active, nervous, dry",
        "lips": "thin, dry, darker in color",
        "teeth": "irregular, protruding, large, gums recede easily",
        "hair": "dry, coarse, curly, brittle",
        "features": "asymmetrical, prominent, sharp cheekbones"
    },
    "pitta": {
        "face_shape": "medium, heart-shaped, pointed chin",
        "skin": "warm, reddish, sensitive, prone to rashes",
        "eyes": "medium-sized, intense, penetrating, sensitive to light",
        "lips": "medium, soft, reddish, prone to inflammation",
        "teeth": "medium-sized, yellowish, sensitive to hot/cold",
        "hair": "fine, straight, early graying, tendency for baldness",
        "features": "symmetrical, sharp, defined nose"
    },
    "kapha": {
        "face_shape": "round, full, moon-shaped",
        "skin": "thick, oily, smooth, pale, cool and moist",
        "eyes": "large, attractive, thick lashes, whites very white",
        "lips": "full, moist, well-defined",
        "teeth": "strong, white, well-aligned, healthy gums",
        "hair": "thick, wavy, lustrous, oily",
        "features": "symmetrical, well-proportioned, soft"
    }
}

# Body Outline SVG for visualization
body_outline_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 150">
  <ellipse cx="50" cy="30" rx="25" ry="30" fill="none" stroke="#666" stroke-width="1"/>
  <path d="M35,60 C35,60 25,90 25,110 C25,130 35,150 50,150 C65,150 75,130 75,110 C75,90 65,60 65,60 Z" fill="none" stroke="#666" stroke-width="1"/>
  <path d="M40,60 C40,50 60,50 60,60" fill="none" stroke="#666" stroke-width="1"/>
  <ellipse cx="40" cy="25" rx="5" ry="3" fill="none" stroke="#666" stroke-width="1"/>
  <ellipse cx="60" cy="25" rx="5" ry="3" fill="none" stroke="#666" stroke-width="1"/>
  <path d="M40,40 C45,45 55,45 60,40" fill="none" stroke="#666" stroke-width="1"/>
  <path d="M50,25 L50,35 C50,38 53,38 53,35" fill="none" stroke="#666" stroke-width="1"/>
</svg>"""

# Constants for facial analysis
DOSHA_THRESHOLDS = {
    "vata": {
        "NAR": {"max": 0.8},
        "EAR": {"max": 0.1},
        "MAR": {"max": 0.5}
    },
    "pitta": {
        "NAR": {"min": 0.8, "max": 1.0},
        "EAR": {"min": 0.1, "max": 0.2},
        "MAR": {"min": 0.5, "max": 0.6}
    },
    "kapha": {
        "NAR": {"min": 1.0},
        "EAR": {"min": 0.2},
        "MAR": {"min": 0.6}
    }
}

# Age-specific average ratios
AGE_SPECIFIC_RATIOS = {
    "children": {  # 6-15 years
        "vata": {"NAR": 0.84, "EAR": 0.14, "MAR": 0.52},
        "pitta": {"NAR": 0.97, "EAR": 0.17, "MAR": 0.55},
        "kapha": {"NAR": 0.89, "EAR": 0.15, "MAR": 0.53}
    },
    "adults": {  # 16-53 years
        "vata": {"NAR": 0.87, "EAR": 0.14, "MAR": 0.50},
        "pitta": {"NAR": 0.97, "EAR": 0.17, "MAR": 0.52},
        "kapha": {"NAR": 0.90, "EAR": 0.17, "MAR": 0.51}
    }
}

def get_distance(point1, point2):
    """Calculate distance between two points"""
    return ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5

def analyze_with_mediapipe(image_data):
    """Analyze face using MediaPipe face mesh"""
    try:
        # Convert image to numpy array
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        # Process with face mesh
        results = face_mesh.process(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Calculate key measurements
        face_width = get_distance(landmarks[234], landmarks[454])
        face_height = get_distance(landmarks[10], landmarks[152])
        left_eye_width = get_distance(landmarks[33], landmarks[133])
        right_eye_width = get_distance(landmarks[362], landmarks[263])
        left_eye_height = get_distance(landmarks[159], landmarks[145])
        right_eye_height = get_distance(landmarks[386], landmarks[374])
        mouth_width = get_distance(landmarks[61], landmarks[291])
        mouth_height = get_distance(landmarks[0], landmarks[17])
        nose_width = get_distance(landmarks[129], landmarks[358])
        nose_height = get_distance(landmarks[168], landmarks[2])
        
        return {
            "face_width": face_width,
            "face_height": face_height,
            "face_ratio": face_width / face_height,
            "eye_width": (left_eye_width + right_eye_width) / 2,
            "eye_height": (left_eye_height + right_eye_height) / 2,
            "mouth_width": mouth_width,
            "mouth_height": mouth_height,
            "nose_width": nose_width,
            "nose_height": nose_height
        }
    except Exception as e:
        print(f"MediaPipe analysis error: {e}")
        return None

def analyze_skin_texture(skin_img):
    """Analyze skin texture for dosha indicators"""
    try:
        # Convert image to numpy array
        img = Image.open(io.BytesIO(skin_img)).convert('RGB')
        img_array = np.array(img)
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Calculate texture features
        variance = np.var(gray)
        avg_color = np.mean(img_array, axis=(0, 1))
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # Determine dosha factors
        vata_factor = 10 if variance > 500 and edge_density > 0.1 else 0
        pitta_factor = 10 if avg_color[0] > 150 and avg_color[1] < 120 else 0
        kapha_factor = 10 if variance < 300 and edge_density < 0.05 else 0
        
        return {
            "texture_variance": float(variance),
            "edge_density": float(edge_density),
            "avg_color": [float(c) for c in avg_color],
            "vata_factor": vata_factor,
            "pitta_factor": pitta_factor,
            "kapha_factor": kapha_factor
        }
    except Exception as e:
        print(f"Skin texture analysis error: {e}")
        return {
            "vata_factor": 0,
            "pitta_factor": 0,
            "kapha_factor": 0
        }

def analyze_facial_features(face_img, eyes_img, mouth_img, skin_img, profile_img=None):
    """Comprehensive facial analysis for dosha determination"""
    try:
        # Basic analysis with mediapipe
        face_results = analyze_with_mediapipe(face_img)
        if not face_results:
            return {"error": "No face detected in the image"}
            
        # Default values
        age_group = "adults"
        
        # Optional DeepFace analysis if available
        try:
            from deepface import DeepFace
            face_analysis = DeepFace.analyze(
                img_path=io.BytesIO(face_img),
                actions=['age', 'gender'],
                enforce_detection=False
            )
            
            if isinstance(face_analysis, list) and len(face_analysis) > 0:
                face_analysis = face_analysis[0]
                age = face_analysis.get('age', 25)
                gender = face_analysis.get('gender', None)
                age_group = "children" if age < 16 else "adults"
        except Exception:
            gender = None
        
        # Analyze skin texture
        skin_features = analyze_skin_texture(skin_img) if skin_img else {}
        
        # Calculate dosha scores
        vata_score = calculate_vata_score(face_results, age_group)
        pitta_score = calculate_pitta_score(face_results, age_group)
        kapha_score = calculate_kapha_score(face_results, age_group)
        
        # Adjust scores based on skin analysis
        vata_score += skin_features.get("vata_factor", 0)
        pitta_score += skin_features.get("pitta_factor", 0)
        kapha_score += skin_features.get("kapha_factor", 0)
        
        # Calculate percentages
        total = max(1, vata_score + pitta_score + kapha_score)
        vata_percentage = (vata_score / total) * 100
        pitta_percentage = (pitta_score / total) * 100
        kapha_percentage = (kapha_score / total) * 100
        
        doshas = {
            "vata": vata_percentage,
            "pitta": pitta_percentage,
            "kapha": kapha_percentage
        }
        
        # Determine dominant dosha
        dominant_dosha = max(doshas, key=doshas.get)
        
        # Create feature descriptions
        feature_description = describe_features(face_results)
        dosha_description = get_dosha_description(dominant_dosha)
        recommendations = get_dosha_recommendations(dominant_dosha)
        
        return {
            "doshas": doshas,
            "dominant_dosha": dominant_dosha,
            "feature_description": feature_description,
            "dosha_description": dosha_description,
            "recommendations": recommendations,
            "age_group": age_group,
            "gender": gender
        }
    except Exception as e:
        return {"error": str(e)}

def calculate_vata_score(face_results, age_group="adults"):
    """Calculate Vata score based on facial features"""
    score = 0
    ref_values = AGE_SPECIFIC_RATIOS[age_group]["vata"]
    
    # Face ratio (long face is Vata)
    if "face_ratio" in face_results:
        face_ratio = face_results["face_ratio"]
        if face_ratio < 0.85:  # Long face
            score += 15
        
    # Thin face
    if "face_width" in face_results and "face_height" in face_results:
        face_ratio = face_results["face_height"] / face_results["face_width"]
        if face_ratio > 1.5:
            score += 10
    
    # Small eyes
    if "eye_width" in face_results and "face_width" in face_results:
        eye_ratio = face_results["eye_width"] / face_results["face_width"]
        if eye_ratio < 0.1:
            score += 10
    
    # Thin lips
    if "mouth_width" in face_results and "mouth_height" in face_results:
        mouth_ratio = face_results["mouth_height"] / face_results["mouth_width"]
        if mouth_ratio < 0.3:
            score += 10
            
    return max(1, score)

def calculate_pitta_score(face_results, age_group="adults"):
    """Calculate Pitta score based on facial features"""
    score = 0
    ref_values = AGE_SPECIFIC_RATIOS[age_group]["pitta"]
    
    # Medium face ratio is Pitta
    if "face_ratio" in face_results:
        face_ratio = face_results["face_ratio"]
        if 0.85 <= face_ratio <= 0.95:  # Medium face
            score += 15
    
    # Medium-sized eyes
    if "eye_width" in face_results and "face_width" in face_results:
        eye_ratio = face_results["eye_width"] / face_results["face_width"]
        if 0.1 <= eye_ratio <= 0.15:
            score += 10
    
    # Medium lips
    if "mouth_width" in face_results and "mouth_height" in face_results:
        mouth_ratio = face_results["mouth_height"] / face_results["mouth_width"]
        if 0.3 <= mouth_ratio <= 0.4:
            score += 10
            
    return max(1, score)

def calculate_kapha_score(face_results, age_group="adults"):
    """Calculate Kapha score based on facial features"""
    score = 0
    ref_values = AGE_SPECIFIC_RATIOS[age_group]["kapha"]
    
    # Round face is Kapha
    if "face_ratio" in face_results:
        face_ratio = face_results["face_ratio"]
        if face_ratio > 0.95:  # Round face
            score += 15
    
    # Large eyes
    if "eye_width" in face_results and "face_width" in face_results:
        eye_ratio = face_results["eye_width"] / face_results["face_width"]
        if eye_ratio > 0.15:
            score += 10
    
    # Full lips
    if "mouth_width" in face_results and "mouth_height" in face_results:
        mouth_ratio = face_results["mouth_height"] / face_results["mouth_width"]
        if mouth_ratio > 0.4:
            score += 10
            
    return max(1, score)
def describe_features(face_results):
    """Generate descriptions of facial features based on measurements"""
    descriptions = {
        "face_shape": "",
        "eyes": "",
        "mouth": "",
        "overall_impression": ""
    }

    # Face shape analysis
    if "face_ratio" in face_results:
        face_ratio = face_results["face_ratio"]
        if face_ratio < 0.85:
            descriptions["face_shape"] = "long and angular with prominent features (Vata characteristic)"
        elif 0.85 <= face_ratio <= 0.95:
            descriptions["face_shape"] = "medium and heart-shaped with defined features (Pitta characteristic)"
        else:
            descriptions["face_shape"] = "round and full with soft features (Kapha characteristic)"

    # Eye analysis
    if "eye_width" in face_results and "face_width" in face_results:
        eye_ratio = face_results["eye_width"] / face_results["face_width"]
        if eye_ratio < 0.1:
            descriptions["eyes"] = "small, alert, and quick-moving (Vata characteristic)"
        elif 0.1 <= eye_ratio <= 0.15:
            descriptions["eyes"] = "medium-sized, intense, and penetrating (Pitta characteristic)"
        else:
            descriptions["eyes"] = "large, calm, and attractive with thick lashes (Kapha characteristic)"

    # Mouth analysis
    if "mouth_width" in face_results and "mouth_height" in face_results:
        mouth_ratio = face_results["mouth_height"] / face_results["mouth_width"]
        if mouth_ratio < 0.3:
            descriptions["mouth"] = "thin lips that tend to be dry (Vata characteristic)"
        elif 0.3 <= mouth_ratio <= 0.4:
            descriptions["mouth"] = "medium lips with a reddish tint (Pitta characteristic)"
        else:
            descriptions["mouth"] = "full, well-defined lips (Kapha characteristic)"
            
    # Overall impression
    vata_count = sum(1 for desc in descriptions.values() if "Vata" in desc)
    pitta_count = sum(1 for desc in descriptions.values() if "Pitta" in desc)
    kapha_count = sum(1 for desc in descriptions.values() if "Kapha" in desc)
    
    if vata_count > pitta_count and vata_count > kapha_count:
        descriptions["overall_impression"] = "Your facial features predominantly show Vata characteristics."
    elif pitta_count > vata_count and pitta_count > kapha_count:
        descriptions["overall_impression"] = "Your facial features predominantly show Pitta characteristics."
    elif kapha_count > vata_count and kapha_count > pitta_count:
        descriptions["overall_impression"] = "Your facial features predominantly show Kapha characteristics."
    else:
        descriptions["overall_impression"] = "Your facial features show a balanced mix of dosha characteristics."

    return descriptions

def get_dosha_description(dosha):
    """Get description for a specific dosha"""
    descriptions = {
        "vata": """Vata represents the elements of air and space (Vayu and Akasha). 
People with Vata-dominant constitutions tend to be creative, quick-thinking, and adaptable when in balance. 
They typically have a light frame, dry skin, and irregular appetite and digestion.
When out of balance, they may experience anxiety, insomnia, constipation, or joint pain.
According to Ayurvedic texts like Charaka Samhita, Vata governs all movement in the body and mind.""",

        "pitta": """Pitta represents the elements of fire and water (Agni and Jala).
Pitta-dominant individuals are often intelligent, focused, and have strong digestion and metabolism.
They typically have a medium build, warm skin tone, and sharp hunger.
When out of balance, they may become irritable, experience inflammation, acid reflux, or skin rashes.
According to Ayurvedic texts like Sushruta Samhita, Pitta governs all transformation processes in the body.""",

        "kapha": """Kapha represents the elements of earth and water (Prithvi and Jala).
Those with Kapha dominance are typically calm, strong, and nurturing with excellent stamina.
They have a solid build, smooth skin, and steady energy levels.
When out of balance, they may experience weight gain, lethargy, congestion, or attachment issues.
According to Ayurvedic texts like Ashtanga Hridayam, Kapha provides structure and lubrication to the body."""
    }

    return descriptions.get(dosha, "Your constitution shows a balance of multiple doshas.")

def get_dosha_recommendations(dosha):
    """Get recommendations based on dosha analysis from authentic Ayurvedic sources"""
    recommendations = {
        "vata": [
            "Follow a regular daily routine (dinacharya) with consistent meal and sleep times",
            "Favor warm, cooked, moist foods with sweet, sour, and salty tastes",
            "Include healthy oils like ghee and sesame oil in your diet",
            "Engage in gentle exercise like yoga, tai chi, or walking",
            "Practice calming meditation and deep breathing exercises",
            "Keep warm and avoid cold, windy environments",
            "Use warming spices like ginger, cinnamon, and cumin in cooking",
            "Massage your body with warm sesame oil (abhyanga)",
            "Stay hydrated with warm or room temperature beverages",
            "Avoid excessive travel, multitasking, and overstimulation"
        ],
        "pitta": [
            "Avoid excessive heat, sun exposure, and spicy foods",
            "Favor cooling foods like sweet fruits, vegetables, and grains",
            "Include coconut, mint, and cilantro in your diet",
            "Engage in moderate exercise like swimming or walking in nature",
            "Practice cooling breath exercises (sheetali pranayama)",
            "Avoid working during the pitta time of day (10am-2pm)",
            "Use cooling spices like coriander, fennel, and cardamom",
            "Massage with cooling oils like coconut or sunflower",
            "Create time for play and relaxation in your schedule",
            "Practice compassion and patience in challenging situations"
        ],
        "kapha": [
            "Engage in regular vigorous exercise, preferably in the morning",
            "Favor light, warm, and dry foods with pungent, bitter, and astringent tastes",
            "Limit dairy, sweets, and heavy foods",
            "Wake up before 6am and avoid daytime napping",
            "Use stimulating spices like black pepper, ginger, and cayenne",
            "Practice energizing breath exercises (kapalabhati pranayama)",
            "Seek new experiences and mental stimulation",
            "Massage with stimulating oils like mustard or eucalyptus",
            "Maintain a varied and stimulating environment",
            "Practice regular detoxification practices"
        ]
    }
    
    return recommendations.get(dosha, [
        "Maintain a balanced diet with all six tastes (sweet, sour, salty, bitter, pungent, astringent)",
        "Follow a regular daily routine (dinacharya)",
        "Engage in moderate exercise suitable for your energy levels",
        "Practice meditation and mindfulness for mental balance",
        "Adjust your lifestyle according to the seasons",
        "Get adequate sleep based on your constitution",
        "Stay hydrated with appropriate beverages for your type",
        "Practice self-massage (abhyanga) with suitable oils",
        "Maintain a positive outlook and healthy relationships",
        "Consult with an Ayurvedic practitioner for personalized guidance"
    ])