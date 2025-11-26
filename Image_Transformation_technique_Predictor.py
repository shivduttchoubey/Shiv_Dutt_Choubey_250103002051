# Trains model, shows evaluation metrics, provides CLI to test the model.

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys

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
            8: 'contrast_decrease',       # New
            9: 'histogram_equalization',
            10: 'gamma_correction_bright', # Renamed
            11: 'gamma_correction_dark',   # New
            12: 'bilateral_filter',        # Renumbered
            13: 'rotation',                # Renumbered
            14: 'gaussian_noise',          # Renumbered
            15: 'salt_pepper_noise',       # Renumbered
            16: 'box_blur',                # New
            17: 'motion_blur'              # New
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

            elif technique_id == 8:  # contrast decrease (New)
                return cv2.convertScaleAbs(img, alpha=0.5, beta=0)

            elif technique_id == 9:  # histogram equalization
                if len(img.shape) == 3:
                    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
                    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                else:
                    return cv2.equalizeHist(img)

            elif technique_id == 10: # gamma correction (Bright)
                inv_gamma = 1.0 / 1.5 # gamma = 1.5
                table = np.array([((i / 255.0) ** inv_gamma) * 255
                                for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(img, table)

            elif technique_id == 11: # gamma correction (Dark) (New)
                inv_gamma = 1.0 / 0.5 # gamma = 0.5
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
                noise = np.random.normal(0, 25, img.shape).astype(np.int16) # Use int16 for temp calc
                noisy_img = img.astype(np.int16) + noise
                return np.clip(noisy_img, 0, 255).astype(np.uint8)

            elif technique_id == 15:  # salt and pepper noise
                noisy = img.copy()
                prob = 0.02
                rnd = np.random.rand(img.shape[0], img.shape[1])
                noisy[rnd < prob/2] = 0
                noisy[rnd > 1 - prob/2] = 255
                return noisy

            elif technique_id == 16: # box_blur (New)
                return cv2.blur(img, (7, 7))

            elif technique_id == 17: # motion_blur (New)
                size = 15
                kernel_motion_blur = np.zeros((size, size))
                kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
                kernel_motion_blur = kernel_motion_blur / size
                return cv2.filter2D(img, -1, kernel_motion_blur)

            return img
        except cv2.error as e:
            print(f"OpenCV error applying technique {technique_id}: {e}", file=sys.stderr)
            return None # Return None on failure
        except Exception as e:
            print(f"Error applying technique {technique_id}: {e}", file=sys.stderr)
            return None # Return None on failure

# ============================================================================
# PART 2: FEATURE EXTRACTION
# ============================================================================

class ImageFeatureExtractor:
    """Extract features from single images and image pairs"""

    # Pre-calculate expected feature length to handle failures
    # 7 (stats) + 32 (hist) + 1 (edge) + 1 (lap) + 3 (fft) + 4 (texture) = 48
    EXPECTED_SINGLE_FEATURES = 48

    def extract_single_image_features(self, img):
        """Extract features from a single image (without original)"""
        features = []

        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if gray.size == 0 or gray.shape[0] < 2 or gray.shape[1] < 2:
            print("Warning: Single image is empty or too small for feature extraction.", file=sys.stderr)
            return np.array([0] * self.EXPECTED_SINGLE_FEATURES) # Return zero vector

        try:
            # Statistical features
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.median(gray),
                np.min(gray),
                np.max(gray),
                np.percentile(gray, 25),
                np.percentile(gray, 75)
            ])

            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6) # Add epsilon to avoid divide by zero
            features.extend(hist.tolist())

            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.size + 1e-6)
            features.append(edge_density)

            # Laplacian variance (blur detection)
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features.append(lap_var)

            # Frequency domain features
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            features.extend([
                np.mean(magnitude),
                np.std(magnitude),
                np.median(magnitude)
            ])

            # Texture features (GLCM-like)
            dx = np.diff(gray, axis=1)
            dy = np.diff(gray, axis=0)
            features.extend([
                np.mean(np.abs(dx)),
                np.std(np.abs(dx)),
                np.mean(np.abs(dy)),
                np.std(np.abs(dy))
            ])

            if len(features) != self.EXPECTED_SINGLE_FEATURES:
                print(f"Warning: Feature mismatch. Expected {self.EXPECTED_SINGLE_FEATURES}, got {len(features)}", file=sys.stderr)
                # This case should not happen if logic is correct, but as a fallback:
                return (np.array(features + [0] * self.EXPECTED_SINGLE_FEATURES))[:self.EXPECTED_SINGLE_FEATURES]

            return np.array(features)

        except Exception as e:
            print(f"Error in extract_single_image_features: {e}", file=sys.stderr)
            return np.array([0] * self.EXPECTED_SINGLE_FEATURES) # Return zero vector on error

    def extract_paired_features(self, original, processed):
        """Extract features from image pair (original + processed)"""
        features = []

        try:
            # Convert to grayscale
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original

            if len(processed.shape) == 3:
                proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            else:
                proc_gray = processed

            if orig_gray.size == 0 or proc_gray.size == 0:
                print("Warning: Paired images have empty gray images.", file=sys.stderr)
                return np.array([]) # Return empty, will be skipped

            if orig_gray.shape != proc_gray.shape:
                 # Resize processed to match original if needed
                 proc_gray = cv2.resize(proc_gray, (orig_gray.shape[1], orig_gray.shape[0]))

            # Difference metrics
            diff = cv2.absdiff(orig_gray, proc_gray)
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.median(diff),
                np.max(diff)
            ])

            # Statistical differences
            features.extend([
                np.mean(proc_gray) - np.mean(orig_gray),
                np.std(proc_gray) - np.std(orig_gray),
                np.median(proc_gray) - np.median(orig_gray)
            ])

            # Histogram comparison
            hist_orig = cv2.calcHist([orig_gray], [0], None, [32], [0, 256])
            hist_proc = cv2.calcHist([proc_gray], [0], None, [32], [0, 256])
            cv2.normalize(hist_orig, hist_orig, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist_proc, hist_proc, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            hist_corr = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CORREL)
            hist_chisq = cv2.compareHist(hist_orig, hist_proc, cv2.HISTCMP_CHISQR)
            features.extend([hist_corr, hist_chisq])

            # Edge comparison
            edges_orig = cv2.Canny(orig_gray, 50, 150)
            edges_proc = cv2.Canny(proc_gray, 50, 150)
            edge_diff = np.sum(cv2.absdiff(edges_orig, edges_proc)) / (edges_orig.size + 1e-6)
            features.append(edge_diff)

            # Structural similarity components
            mu1 = np.mean(orig_gray)
            mu2 = np.mean(proc_gray)
            sigma1 = np.std(orig_gray)
            sigma2 = np.std(proc_gray)
            features.extend([mu1, mu2, sigma1, sigma2])

            # Add single image features for processed image
            single_features = self.extract_single_image_features(processed)
            features.extend(single_features.tolist())

            return np.array(features)

        except Exception as e:
            print(f"Error in extract_paired_features: {e}", file=sys.stderr)
            return np.array([]) # Return empty, will be skipped


# ============================================================================
# PART 3: DATA LOADING AND PREPARATION
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

    expected_paired_features = -1 # Will be set by first valid sample

    for tech_id in technique_names.keys():
        print(f"Generating samples for: {technique_names[tech_id]}")
        samples_generated = 0

        for i in range(n_samples_per_class):
            if samples_generated >= n_samples_per_class:
                break

            # Generate random base image
            base = np.random.randint(0, 256, (*img_size, 3), dtype=np.uint8)

            # Apply technique
            processed = generator.apply_technique(base, tech_id)

            # Handle edge detection special case (single channel)
            if processed is not None and len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

            # Handle empty/invalid processed images
            if processed is None or processed.size == 0:
                print(f"Warning: Skipping sample {i} for tech {tech_id} due to empty processed image.", file=sys.stderr)
                continue

            # Extract features
            single_feat = extractor.extract_single_image_features(processed)
            paired_feat = extractor.extract_paired_features(base, processed)

            # Ensure features are not empty and valid
            if single_feat.size != extractor.EXPECTED_SINGLE_FEATURES:
                print(f"Warning: Skipping sample {i} for tech {tech_id} due to single feature mismatch.", file=sys.stderr)
                continue

            if paired_feat.size == 0:
                print(f"Warning: Skipping sample {i} for tech {tech_id} due to empty paired features.", file=sys.stderr)
                continue

            # Set and check expected paired feature length
            if expected_paired_features == -1:
                expected_paired_features = paired_feat.size
            elif paired_feat.size != expected_paired_features:
                print(f"Warning: Skipping sample {i} for tech {tech_id} due to paired feature mismatch. Got {paired_feat.size}, expected {expected_paired_features}", file=sys.stderr)
                continue

            X_single.append(single_feat)
            X_paired.append(paired_feat)
            y.append(tech_id)
            samples_generated += 1

    return np.array(X_single), np.array(X_paired), np.array(y), technique_names


# ============================================================================
# PART 4: MODEL TRAINING
# ============================================================================

def train_models(X_single, X_paired, y, test_size=0.2):
    """Train both single-image and paired-image models"""
    print("\n" + "="*60)
    print("TRAINING SINGLE-IMAGE MODEL (without original)")
    print("="*60)

    # Split data
    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
        X_single, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    scaler_single = StandardScaler()
    X_s_train_scaled = scaler_single.fit_transform(X_s_train)
    X_s_test_scaled = scaler_single.transform(X_s_test)

    # Train model
    model_single = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_iter_no_change=10 # Add early stopping
    )
    model_single.fit(X_s_train_scaled, y_s_train)

    # Evaluate
    y_s_pred = model_single.predict(X_s_test_scaled)
    acc_single = np.mean(y_s_pred == y_s_test)
    print(f"\nSingle-Image Model Accuracy: {acc_single:.4f}")

    print("\n" + "="*60)
    print("TRAINING PAIRED-IMAGE MODEL (with original)")
    print("="*60)

    # Split data
    X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(
        X_paired, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    scaler_paired = StandardScaler()
    X_p_train_scaled = scaler_paired.fit_transform(X_p_train)
    X_p_test_scaled = scaler_paired.transform(X_p_test)

    # Train model
    model_paired = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_iter_no_change=10 # Add early stopping
    )
    model_paired.fit(X_p_train_scaled, y_p_train)

    # Evaluate
    y_p_pred = model_paired.predict(X_p_test_scaled)
    acc_paired = np.mean(y_p_pred == y_p_test)
    print(f"\nPaired-Image Model Accuracy: {acc_paired:.4f}")

    return (model_single, scaler_single, X_s_test_scaled, y_s_test, y_s_pred), \
           (model_paired, scaler_paired, X_p_test_scaled, y_p_test, y_p_pred)


# ============================================================================
# PART 5: EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(y_test, y_pred, technique_names, model_name):
    """Detailed evaluation with confusion matrix"""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print('='*60)

    # Classification report
    print("\nClassification Report:")
    class_labels = [technique_names[i] for i in sorted(technique_names.keys())]
    print(classification_report(y_test, y_pred,
                                target_names=class_labels, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    # In a .py script, plt.show() will block execution until the window is closed
    print(f"\nDisplaying Confusion Matrix for {model_name}. Close the plot window to continue...")
    plt.show()


# ============================================================================
# PART 6: MODEL PERSISTENCE
# ============================================================================

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
    print(f"  - model_single_image.pkl (without original)")
    print(f"  - model_paired_image.pkl (with original)")
    print(f"  - Scalers and technique names")

def load_models(save_dir='models'):
    """Load trained models"""
    model_single = joblib.load(f'{save_dir}/model_single_image.pkl')
    scaler_single = joblib.load(f'{save_dir}/scaler_single_image.pkl')
    model_paired = joblib.load(f'{save_dir}/model_paired_image.pkl')
    scaler_paired = joblib.load(f'{save_dir}/scaler_paired_image.pkl')
    technique_names = joblib.load(f'{save_dir}/technique_names.pkl')

    return model_single, scaler_single, model_paired, scaler_paired, technique_names


# ============================================================================
# PART 7: INFERENCE FUNCTIONS
# ============================================================================

def predict_single_image(img_path, model, scaler, technique_names):
    """Predict technique from single image (without original)"""
    extractor = ImageFeatureExtractor()

    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")

    features = extractor.extract_single_image_features(img)
    if features.size != extractor.EXPECTED_SINGLE_FEATURES:
        raise ValueError("Could not extract valid features from single image.")

    features_scaled = scaler.transform(features.reshape(1, -1))

    # Get probabilities for all classes
    pred_proba_all = model.predict_proba(features_scaled)[0]

    print(f"\n--- Prediction Results (Single Image) ---")

    print("Full Probability Distribution:")
    # Get sorted list of (name, prob) tuples
    probs_list = []
    for i, class_index in enumerate(model.classes_): # Loop through class indices
        if class_index in technique_names:
            name = technique_names[class_index]
            probability = pred_proba_all[i]
            probs_list.append((name, probability))
        else:
            print(f"Warning: Class index {class_index} not in technique_names map.", file=sys.stderr)

    # Sort by probability descending
    probs_list.sort(key=lambda x: x[1], reverse=True)

    # Print sorted list
    print(f"Predicted Technique: {probs_list[0][0]} (Confidence: {probs_list[0][1]:.2%})")
    for name, probability in probs_list:
        print(f"  {name}: {probability:.2%}")

    return probs_list[0][0], pred_proba_all

def predict_paired_images(original_path, processed_path, model, scaler, technique_names):
    """Predict technique from image pair"""
    extractor = ImageFeatureExtractor()

    original = cv2.imread(original_path)
    processed = cv2.imread(processed_path)

    if original is None:
        raise ValueError(f"Could not load original image: {original_path}")
    if processed is None:
        raise ValueError(f"Could not load processed image: {processed_path}")

    features = extractor.extract_paired_features(original, processed)
    if features.size == 0:
        raise ValueError("Could not extract valid features from paired images.")

    features_scaled = scaler.transform(features.reshape(1, -1))

    # Get probabilities for all classes
    pred_proba_all = model.predict_proba(features_scaled)[0]

    print(f"\n--- Prediction Results (Paired Images) ---")

    print("Full Probability Distribution:")
    # Get sorted list of (name, prob) tuples
    probs_list = []
    for i, class_index in enumerate(model.classes_): # Loop through class indices
        if class_index in technique_names:
            name = technique_names[class_index]
            probability = pred_proba_all[i]
            probs_list.append((name, probability))
        else:
            print(f"Warning: Class index {class_index} not in technique_names map.", file=sys.stderr)

    # Sort by probability descending
    probs_list.sort(key=lambda x: x[1], reverse=True)

    # Print sorted list
    print(f"Predicted Technique: {probs_list[0][0]} (Confidence: {probs_list[0][1]:.2%})")
    for name, probability in probs_list:
        print(f"  {name}: {probability:.2%}")

    return probs_list[0][0], pred_proba_all


# ============================================================================
# PART 8: MAIN PIPELINE
# ============================================================================

def run_complete_pipeline(n_samples_per_class=100):
    """Run the complete ML pipeline"""

    print("="*60)
    print("IMAGE PROCESSING TECHNIQUE DETECTION PIPELINE")
    print("="*60)

    # Step 1: Generate dataset
    X_single, X_paired, y, technique_names = create_synthetic_dataset(n_samples_per_class)

    if len(y) == 0:
      print("ERROR: No data was generated. Check Part 1 & 3.", file=sys.stderr)
      return None, None, None, None, None # Return Nones

    print(f"\n✓ Dataset created:")
    print(f"  - Total samples: {len(y)}")
    print(f"  - Classes: {len(technique_names)}")
    print(f"  - Single-image features: {X_single.shape[1]}")
    print(f"  - Paired-image features: {X_paired.shape[1]}")

    # Step 2: Train models
    single_results, paired_results = train_models(X_single, X_paired, y)

    model_single, scaler_single, X_s_test, y_s_test, y_s_pred = single_results
    model_paired, scaler_paired, X_p_test, y_p_test, y_p_pred = paired_results

    # Step 3: Evaluate
    evaluate_model(y_s_test, y_s_pred, technique_names, "Single-Image Model")
    evaluate_model(y_p_test, y_p_pred, technique_names, "Paired-Image Model")

    # Step 4: Save models
    save_models(model_single, scaler_single, model_paired, scaler_paired, technique_names)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nYou can now use the models for prediction.")

    # Return models for subsequent use
    return model_single, scaler_single, model_paired, scaler_paired, technique_names

# ============================================================================
# PART 9: PREDICT ON USER IMAGES
# ============================================================================

def run_prediction_on_user_images(model_paired, scaler_paired, technique_names):
    """
    Loads models and prompts user for original and processed images
    to run paired prediction.
    """
    print("="*60)
    print("Running Prediction on User Images")
    print("="*60)

    loaded_models = False

    # Check if models were passed from training
    if model_paired is not None and scaler_paired is not None and technique_names is not None:
        print("Models from training are in memory.")
        loaded_models = True
    else:
        # If not, try loading from disk
        print("Models not in memory, loading from disk...")
        try:
            _, _, model_paired, scaler_paired, technique_names = load_models()
            print("✓ Models loaded successfully.")
            loaded_models = True
        except FileNotFoundError:
            print("\nERROR: Models not found in 'models/' directory.", file=sys.stderr)
            print("Please run the script to train and save the models first.")
        except Exception as e:
             print(f"\nAn error occurred loading models: {e}", file=sys.stderr)

    if not loaded_models:
        print("\nSkipping prediction as models are not available.")
        return

    try:
        # Get file paths from user
        print("\nPlease provide the file path for the ORIGINAL image:")
        original_path = input("Original image path: ").strip().strip("'\"") # Clean up path

        print("\nPlease provide the file path for the PROCESSED image:")
        processed_path = input("Processed image path: ").strip().strip("'\"") # Clean up path

        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original image not found at: {original_path}")
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed image not found at: {processed_path}")

        # Display the images for verification
        print("\nDisplaying uploaded images for verification... Close the plot window to continue.")
        original_input_img = cv2.imread(original_path)
        processed_input_img = cv2.imread(processed_path)

        if original_input_img is None or processed_input_img is None:
            raise ValueError("Could not read one or both images. Please ensure they are valid image files.")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original_input_img, cv2.COLOR_BGR2RGB))
        plt.title('Uploaded Original')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(processed_input_img, cv2.COLOR_BGR2RGB))
        plt.title('Uploaded Processed')
        plt.axis('off')
        plt.show() # This will block, user must close plot to continue

        # Run prediction
        print("\nRunning paired-image model prediction...")
        predict_paired_images(original_path, processed_path, model_paired, scaler_paired, technique_names)

    except Exception as e:
        print(f"\nAn error occurred during the prediction process: {e}", file=sys.stderr)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to run the training pipeline and then the
    interactive prediction.
    """
    # --- PART 8: Run Training ---
    try:
        model_single, scaler_single, model_paired, scaler_paired, technique_names = \
            run_complete_pipeline(n_samples_per_class=100) # Or set a different number
    except Exception as e:
        print(f"\nAn error occurred during the training pipeline: {e}", file=sys.stderr)
        print("Aborting.")
        return

    # --- PART 9: Run Prediction ---
    # Check if training was successful
    if model_paired is not None:
        # Ask user if they want to predict now
        while True:
            try:
                choice = input("\nTraining complete. Do you want to predict on new images now? (y/n): ").lower()
                if choice == 'y':
                    run_prediction_on_user_images(model_paired, scaler_paired, technique_names)
                    break
                elif choice == 'n':
                    print("Exiting. You can re-run the script to predict later (it will load models from disk).")
                    break
                else:
                    print("Invalid choice. Please enter 'y' or 'n'.")
            except EOFError:
                print("\nExiting.") # Handle Ctrl+D
                break
    else:
        print("\nTraining pipeline did not complete successfully. Skipping prediction.")


if __name__ == "__main__": 
    main()