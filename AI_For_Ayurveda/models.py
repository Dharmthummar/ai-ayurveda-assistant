import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from PIL import Image
import io
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrakritiModel:
    """Model for analyzing Prakriti (Ayurvedic constitution) from images"""
    
    def __init__(self):
        """Initialize the model"""
        self.model = self._build_model()
        self._initialize_chroma_db()

    def _initialize_chroma_db(self):
        """Initialize Chroma database with Ayurvedic texts"""
        try:
            # Initialize embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Create or load Chroma database
            self.db = Chroma(
                persist_directory="chroma_db",
                embedding_function=embeddings
            )
            
            # Load documents if database is empty
            if len(self.db.get()['ids']) == 0:
                self._load_documents()
                
        except Exception as e:
            logger.error(f"Error initializing Chroma database: {str(e)}")
            raise

    def _load_documents(self):
        """Load and process Ayurvedic documents"""
        try:
            # Check if directory exists
            if not os.path.exists("ayurvedic_texts"):
                logger.warning("ayurvedic_texts directory not found. Creating empty database.")
                return
                
            # Load documents from directory
            loader = DirectoryLoader(
                "ayurvedic_texts",
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                silent_errors=True  # Skip files that can't be loaded
            )
            
            try:
                documents = loader.load()
                if not documents:
                    logger.warning("No documents found in ayurvedic_texts directory")
                    return
                    
                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                texts = text_splitter.split_documents(documents)
                
                # Add documents to database
                self.db.add_documents(texts)
                self.db.persist()
                logger.info(f"Successfully loaded {len(texts)} document chunks into database")
                
            except Exception as e:
                logger.error(f"Error loading documents: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Error in document loading process: {str(e)}")
            raise

    def _build_model(self):
        """Build the underlying neural network model"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
            
        # Add classification layers
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        predictions = Dense(3, activation='softmax')(x)
        
        # Create and compile model
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def analyze_features(self, face_img, eyes_img, mouth_img, skin_img):
        """
        Analyze facial features to determine dosha composition
        
        Args:
            face_img: Image data for full face
            eyes_img: Image data for eyes
            mouth_img: Image data for mouth
            skin_img: Image data for skin
            
        Returns:
            Dict with dosha percentages
        """
        # Extract features from each image
        face_features = self._extract_features(face_img)
        eyes_features = self._extract_features(eyes_img)
        mouth_features = self._extract_features(mouth_img)
        skin_features = self._extract_features(skin_img)
        
        # Average the features
        combined_features = (face_features + eyes_features + mouth_features + skin_features) / 4
        
        # Calculate dosha percentages
        doshas = {
            "vata": float(combined_features[0]),
            "pitta": float(combined_features[1]),
            "kapha": float(combined_features[2])
        }
        
        # Normalize to percentages
        total = sum(doshas.values())
        for key in doshas:
            doshas[key] = (doshas[key] / total) * 100
        return doshas

    def _extract_features(self, img):
        """
        Extract features from an image using the model
        
        Args:
            img: Raw image binary data
            
        Returns:
            Array of feature values
        """
        # Process image
        img = Image.open(io.BytesIO(img)).convert('RGB')
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Get predictions from model
        return self.model.predict(x)[0]

if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'