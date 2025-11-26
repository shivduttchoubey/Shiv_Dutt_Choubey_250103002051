# Trains model, do not show evaluation metrics, provides an interface to test the model.

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Flask
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

# ============================================================================
# PART 1: DATA GENERATION - Generate synthetic dataset with various techniques
# ============================================================================

class ImageProcessingDataGenerator:
    """Generate training data by applying various image processing techniques"""

    def __init__(self):
        # Techniques aligned with the GUI + new additions
        self.techniques = {
            0: 'original',
            1: 'gaussian_blur',
            2: 'median_blur',
            3: 'sharpen',
            4: 'edge_detection',
            5: 'brightness_increase',
            6: 'brightness_decrease',
            7: 'contrast_increase',
            8: 'contrast_decrease',
            9: 'histogram_equalization',
            10: 'gamma_correction_bright',
            11: 'gamma_correction_dark',
            12: 'bilateral_filter',
            13: 'rotation',
            14: 'gaussian_noise',
            15: 'salt_pepper_noise',
            16: 'box_blur',
            17: 'motion_blur'
        }
        self.class_count = len(self.techniques)

    def apply_technique(self, img, technique_id):
        """Apply specific image processing technique"""
        try:
            if technique_id == 0:  # original
                return img

            elif technique_id == 1:  # gaussian blur
                return cv2.GaussianBlur(img, (9, 9), 0)

            elif technique_id == 2:  # median blur
                return cv2.medianBlur(img, 7)

            elif technique_id == 3:  # sharpen
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                return cv2.filter2D(img, -1, kernel)

            elif technique_id == 4:  # edge detection
                return cv2.Canny(img, 100, 200)

            elif technique_id == 5:  # brightness increase
                return cv2.convertScaleAbs(img, alpha=1.0, beta=50)

            elif technique_id == 6:  # brightness decrease
                return cv2.convertScaleAbs(img, alpha=1.0, beta=-50)

            elif technique_id == 7:  # contrast increase
                return cv2.convertScaleAbs(img, alpha=1.5, beta=0)

            elif technique_id == 8:  # contrast decrease
                return cv2.convertScaleAbs(img, alpha=0.5, beta=0)

            elif technique_id == 9:  # histogram equalization
                if len(img.shape) == 3:
                    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                else:
                    return cv2.equalizeHist(img)

            elif technique_id == 10:
                inv_gamma = 1.0 / 1.5
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(img, table)

            elif technique_id == 11:
                inv_gamma = 1.0 / 0.5
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(img, table)

            elif technique_id == 12:  # bilateral filter
                return cv2.bilateralFilter(img, 9, 75, 75)

            elif technique_id == 13:  # rotation
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w/2, h/2), 15, 1.0)
                return cv2.warpAffine(img, M, (w, h))

            elif technique_id == 14:  # gaussian noise
                noise = np.random.normal(0, 25, img.shape).astype(np.int16)
                noisy_img = img.astype(np.int16) + noise
                return np.clip(noisy_img, 0, 255).astype(np.uint8)

            elif technique_id == 15:  # salt and pepper noise
                noisy = img.copy()
                prob = 0.02
                rnd = np.random.rand(img.shape[0], img.shape[1])
                noisy[rnd < prob/2] = 0
                noisy[rnd > 1 - prob/2] = 255
                return noisy

            elif technique_id == 16:
                return cv2.blur(img, (7, 7))

            elif technique_id == 17:
                size = 15
                kernel_motion_blur = np.zeros((size, size))
                kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
                kernel_motion_blur = kernel_motion_blur / size
                return cv2.filter2D(img, -1, kernel_motion_blur)

            return img
        except cv2.error as e:
            print(f"OpenCV error applying technique {technique_id}: {e}", file=sys.stderr)
            return None
        except Exception as e:
            print(f"Error applying technique {technique_id}: {e}", file=sys.stderr)
            return None

# ============================================================================
# PART 2: FEATURE EXTRACTION
# ============================================================================

class ImageFeatureExtractor:
    """Extract features from single images and image pairs"""

    EXPECTED_SINGLE_FEATURES = 48

    def extract_single_image_features(self, img):
        """Extract features from a single image (without original)"""
        features = []

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if gray.size == 0 or gray.shape[0] < 2 or gray.shape[1] < 2:
            return np.array([0] * self.EXPECTED_SINGLE_FEATURES)

        try:
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.median(gray),
                np.min(gray),
                np.max(gray),
                np.percentile(gray, 25),
                np.percentile(gray, 75)
            ])

            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features.extend(hist.tolist())

            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-6)
            features.append(edge_density)

            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(lap_var)

            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            features.extend([
                np.mean(magnitude),
                np.std(magnitude),
                np.median(magnitude)
            ])

            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)
            features.extend([
                np.mean(np.abs(dx)),
                np.std(np.abs(dx)),
                np.mean(np.abs(dy)),
                np.std(np.abs(dy))
            ])

            if len(features) != self.EXPECTED_SINGLE_FEATURES:
                return (np.array(features + [0] * self.EXPECTED_SINGLE_FEATURES))[:self.EXPECTED_SINGLE_FEATURES]

            return np.array(features)

        except Exception as e:
            print(f"Error in extract_single_image_features: {e}", file=sys.stderr)
            return np.array([0] * self.EXPECTED_SINGLE_FEATURES)

    def extract_paired_features(self, original, processed):
        """Extract features from image pair (original + processed)"""
        features = []

        try:
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original

            if len(processed.shape) == 3:
                proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                proc_gray = processed

            if orig_gray.size == 0 or proc_gray.size == 0:
                return np.array([])

            if orig_gray.shape != proc_gray.shape:
                proc_gray = cv2.resize(proc_gray, (orig_gray.shape[1], orig_gray.shape[0]))

            diff = cv2.absdiff(orig_gray, proc_gray)
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.median(diff),
                np.max(diff)
            ])

            features.extend([
                np.mean(proc_gray) - np.mean(orig_gray),
                np.std(proc_gray) - np.std(orig_gray),
                np.median(proc_gray) - np.median(orig_gray)
            ])

            hist_orig = cv2.calcHist([orig_gray], [0], None, [32], [0, 256])
            hist_proc = cv2.calcHist([proc_gray], [0], None, [32], [0, 256])
            cv2.normalize(hist_orig, hist_orig, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_proc, hist_proc, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_corr = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CORREL)
            hist_chisq = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CHISQR)
            features.extend([hist_corr, hist_chisq])

            edges_orig = cv2.Canny(orig_gray, 50, 150)
            edges_proc = cv2.Canny(proc_gray, 50, 150)
            edge_diff = np.sum(cv2.absdiff(edges_orig, edges_proc)) / (edges_orig.size + 1e-6)
            features.append(edge_diff)

            mu1 = np.mean(orig_gray)
            mu2 = np.mean(proc_gray)
            sigma1 = np.std(orig_gray)
            sigma2 = np.std(proc_gray)
            features.extend([mu1, mu2, sigma1, sigma2])

            single_features = self.extract_single_image_features(processed)
            features.extend(single_features.tolist())

            return np.array(features)

        except Exception as e:
            print(f"Error in extract_paired_features: {e}", file=sys.stderr)
            return np.array([])

# ============================================================================
# PART 3-7: DATA LOADING, TRAINING, EVALUATION (Same as before)
# ============================================================================

def create_synthetic_dataset(n_samples_per_class=100, img_size=(128, 128)):
    """Create synthetic dataset for training"""
    print("Generating synthetic dataset...")

    generator = ImageProcessingDataGenerator()
    extractor = ImageFeatureExtractor()

    X_single = []
    X_paired = []
    y = []
    technique_names = generator.techniques

    expected_paired_features = -1

    for tech_id in technique_names.keys():
        print(f"Generating samples for: {technique_names[tech_id]}")
        samples_generated = 0

        for i in range(n_samples_per_class):
            if samples_generated >= n_samples_per_class:
                break

            base = np.random.randint(0, 256, (*img_size, 3), dtype=np.uint8)
            processed = generator.apply_technique(base, tech_id)

            if processed is not None and len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

            if processed is None or processed.size == 0:
                continue

            single_feat = extractor.extract_single_image_features(processed)
            paired_feat = extractor.extract_paired_features(base, processed)

            if single_feat.size != extractor.EXPECTED_SINGLE_FEATURES:
                continue

            if paired_feat.size == 0:
                continue

            if expected_paired_features == -1:
                expected_paired_features = paired_feat.size
            elif paired_feat.size != expected_paired_features:
                continue

            X_single.append(single_feat)
            X_paired.append(paired_feat)
            y.append(tech_id)
            samples_generated += 1

    return np.array(X_single), np.array(X_paired), np.array(y), technique_names

def train_models(X_single, X_paired, y, test_size=0.2):
    """Train both single-image and paired-image models"""
    print("\nTraining models...")

    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
        X_single, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler_single = StandardScaler()
    X_s_train_scaled = scaler_single.fit_transform(X_s_train)
    X_s_test_scaled = scaler_single.transform(X_s_test)

    model_single = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_iter_no_change=10
    )
    model_single.fit(X_s_train_scaled, y_s_train)

    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(
        X_paired, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler_paired = StandardScaler()
    X_p_train_scaled = scaler_paired.fit_transform(X_p_train)
    X_p_test_scaled = scaler_paired.transform(X_p_test)

    model_paired = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_iter_no_change=10
    )
    model_paired.fit(X_p_train_scaled, y_p_train)

    return model_single, scaler_single, model_paired, scaler_paired

def save_models(model_single, scaler_single, model_paired, scaler_paired,
                technique_names, save_dir='models'):
    """Save trained models and metadata"""
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(model_single, f'{save_dir}/model_single_image.pkl')
    joblib.dump(scaler_single, f'{save_dir}/scaler_single_image.pkl')
    joblib.dump(model_paired, f'{save_dir}/model_paired_image.pkl')
    joblib.dump(scaler_paired, f'{save_dir}/scaler_paired_image.pkl')
    joblib.dump(technique_names, f'{save_dir}/technique_names.pkl')

    print(f"\n✓ Models saved to '{save_dir}/' directory")

def load_models(save_dir='models'):
    """Load trained models"""
    model_single = joblib.load(f'{save_dir}/model_single_image.pkl')
    scaler_single = joblib.load(f'{save_dir}/scaler_single_image.pkl')
    model_paired = joblib.load(f'{save_dir}/model_paired_image.pkl')
    scaler_paired = joblib.load(f'{save_dir}/scaler_paired_image.pkl')
    technique_names = joblib.load(f'{save_dir}/technique_names.pkl')

    return model_single, scaler_single, model_paired, scaler_paired, technique_names

def train_and_save_models():
    """Train and save models"""
    X_single, X_paired, y, technique_names = create_synthetic_dataset(n_samples_per_class=100)
    
    if len(y) == 0:
        print("ERROR: No data was generated.")
        return False
    
    model_single, scaler_single, model_paired, scaler_paired = train_models(X_single, X_paired, y)
    save_models(model_single, scaler_single, model_paired, scaler_paired, technique_names)
    
    return True

# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for models
MODEL_PAIRED = None
SCALER_PAIRED = None
TECHNIQUE_NAMES = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(img):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global MODEL_PAIRED, SCALER_PAIRED, TECHNIQUE_NAMES
    
    # Load models if not loaded
    if MODEL_PAIRED is None:
        try:
            _, _, MODEL_PAIRED, SCALER_PAIRED, TECHNIQUE_NAMES = load_models()
        except:
            return jsonify({'error': 'Models not found. Please train the models first.'}), 500
    
    # Check if files are present
    if 'original' not in request.files or 'processed' not in request.files:
        return jsonify({'error': 'Both original and processed images are required'}), 400
    
    original_file = request.files['original']
    processed_file = request.files['processed']
    
    if original_file.filename == '' or processed_file.filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    if not (allowed_file(original_file.filename) and allowed_file(processed_file.filename)):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, tiff'}), 400
    
    try:
        # Save uploaded files
        original_filename = secure_filename(original_file.filename)
        processed_filename = secure_filename(processed_file.filename)
        
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_original_' + original_filename)
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_processed_' + processed_filename)
        
        original_file.save(original_path)
        processed_file.save(processed_path)
        
        # Load images
        original_img = cv2.imread(original_path)
        processed_img = cv2.imread(processed_path)
        
        if original_img is None or processed_img is None:
            return jsonify({'error': 'Could not read image files'}), 400
        
        # Extract features
        extractor = ImageFeatureExtractor()
        features = extractor.extract_paired_features(original_img, processed_img)
        
        if features.size == 0:
            return jsonify({'error': 'Could not extract features from images'}), 400
        
        # Predict
        features_scaled = SCALER_PAIRED.transform(features.reshape(1, -1))
        pred_proba_all = MODEL_PAIRED.predict_proba(features_scaled)[0]
        
        # Create results
        results = []
        for i, class_index in enumerate(MODEL_PAIRED.classes_):
            if class_index in TECHNIQUE_NAMES:
                results.append({
                    'technique': TECHNIQUE_NAMES[class_index],
                    'probability': float(pred_proba_all[i])
                })
        
        # Sort by probability
        results.sort(key=lambda x: x['probability'], reverse=True)
        
        # Convert images to base64 for display
        original_b64 = image_to_base64(original_img)
        processed_b64 = image_to_base64(processed_img)
        
        # Clean up temp files
        os.remove(original_path)
        os.remove(processed_path)
        
        return jsonify({
            'success': True,
            'prediction': results[0]['technique'],
            'confidence': results[0]['probability'],
            'all_predictions': results,
            'original_image': original_b64,
            'processed_image': processed_b64
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/train', methods=['POST'])
def train():
    """Train models endpoint"""
    try:
        success = train_and_save_models()
        if success:
            # Reload models
            global MODEL_PAIRED, SCALER_PAIRED, TECHNIQUE_NAMES
            _, _, MODEL_PAIRED, SCALER_PAIRED, TECHNIQUE_NAMES = load_models()
            return jsonify({'success': True, 'message': 'Models trained successfully!'})
        else:
            return jsonify({'success': False, 'message': 'Training failed'}), 500
    except Exception as e:
        return jsonify({'success': False, 'message': f'Training error: {str(e)}'}), 500

if __name__ == '__main__':
    # Check if models exist, if not train them
    if not os.path.exists('models/model_paired_image.pkl'):
        print("Models not found. Training models...")
        train_and_save_models()
    
    # Load models
    try:
        _, _, MODEL_PAIRED, SCALER_PAIRED, TECHNIQUE_NAMES = load_models()
        print("Models loaded successfully!")
    except:
        print("Could not load models. Please train them using the web interface.")
    
    print("\n" + "="*60)
    print("Starting Flask Web Application")
    print("="*60)
    print("\nOpen your browser and go to: http://127.0.0.1:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)