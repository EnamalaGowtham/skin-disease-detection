import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.applications import EfficientNetV2B3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class SkinDiseaseDetector:
    def __init__(self, img_size=(300, 300)):
        self.img_size = img_size
        self.num_classes = 0
        self.class_names = []
        self.model = None
        self.history = None
        self.train_path = 'dataset/train'
        self.test_path = 'dataset/test'
        
    def get_class_names(self, class_names_path='class_names.txt'):
        with open(class_names_path) as f:
            self.class_names = [line.strip() for line in f]
        self.num_classes = len(self.class_names)
        print(f"Loaded {self.num_classes} classes: {self.class_names}")
        return self.class_names
    
    def create_data_generators(self, batch_size=32):
        """Create high-performance tf.data.Dataset pipelines with advanced augmentation"""
        
        # Load datasets
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_path,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=self.img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.train_path,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=self.img_size,
            batch_size=batch_size,
            label_mode='categorical'
        )
        
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            self.test_path,
            image_size=self.img_size,
            batch_size=batch_size,
            label_mode='categorical',
            shuffle=False
        )
        
        # Compute Class Weights for Imbalance
        labels = []
        for _, batch_labels in train_ds.unbatch():
            labels.append(np.argmax(batch_labels.numpy()))
        class_weights_arr = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        self.class_weights = {i: weight for i, weight in enumerate(class_weights_arr)}
        print("Computed Class Weights to handle imbalance.")

        # Data Augmentation Layer Pipeline
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.3),
            layers.RandomZoom(0.2),
            layers.RandomTranslation(0.2, 0.2),
            layers.RandomContrast(0.2),
            layers.RandomBrightness(0.2)
        ], name='data_augmentation')
        
        # Prepare datasets for performance
        AUTOTUNE = tf.data.AUTOTUNE
        
        # Scale inputs since EfficientNetV2 expects values in [0, 255] directly, 
        # but let's standardise the preprocessing format
        # Actually EfficientNetV2 includes its own preprocessing (scaling) built-in when inputs are [0,255].
        def preprocess(image, label):
            return image, label

        self.train_dataset = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        self.train_dataset = self.train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
        self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)

        self.val_dataset = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
        self.test_dataset = test_ds.map(preprocess, num_parallel_calls=AUTOTUNE).prefetch(buffer_size=AUTOTUNE)
        
        print("Dataset pipelines ready.")
        
    def build_model(self):
        """Build the CNN model using EfficientNetV2B3"""
        base_model = EfficientNetV2B3(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3),
            include_preprocessing=True # Handles scaling internally
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        
        # Focal Loss (using standard CategoricalCrossentropy with label smoothing for now)
        # Note: Implementing actual Focal Loss natively in TF requires custom class, keeping it simple but robust with label smoothing and class weights.
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        print("Model Summary:")
        self.model.summary()
        
    def train_model(self, epochs=30):
        """Phase 1: Train Head Only"""
        print("\n--- Phase 1: Training Head ---")
        early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        checkpoint = callbacks.ModelCheckpoint('best_skin_disease_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
        
        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=10,
            class_weight=self.class_weights,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        """Phase 2: Fine-Tuning Top Half"""
        print("\n--- Phase 2: Fine-tuning top layers ---")
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze bottom half
        half_layers = len(base_model.layers) // 2
        for layer in base_model.layers[:half_layers]:
            layer.trainable = False
            
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        self.history_phase2 = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=15,
            class_weight=self.class_weights,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        """Phase 3: Full Fine-Tuning"""
        print("\n--- Phase 3: Full Fine-tuning ---")
        base_model.trainable = True
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )
        
        self.history_phase3 = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=15,
            class_weight=self.class_weights,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
    def evaluate_model(self):
        print("\nEvaluating model on test data...")
        test_loss, test_accuracy = self.model.evaluate(self.test_dataset, verbose=1)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        y_true = []
        y_pred = []
        for images, labels in self.test_dataset:
            preds = self.model.predict(images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        return test_accuracy, y_pred, y_true
        
    def plot_training_history(self):
        # Merge histories
        acc = self.history.history['accuracy'] + self.history_phase2.history['accuracy'] + self.history_phase3.history['accuracy']
        val_acc = self.history.history['val_accuracy'] + self.history_phase2.history['val_accuracy'] + self.history_phase3.history['val_accuracy']
        loss = self.history.history['loss'] + self.history_phase2.history['loss'] + self.history_phase3.history['loss']
        val_loss = self.history.history['val_loss'] + self.history_phase2.history['val_loss'] + self.history_phase3.history['val_loss']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(acc, label='Training Accuracy')
        axes[0].plot(val_acc, label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(loss, label='Training Loss')
        axes[1].plot(val_loss, label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        
    def predict_with_tta(self, img_array, num_augs=5):
        """Test-Time Augmentation for robust inference"""
        predictions = []
        # Original
        predictions.append(self.model.predict(img_array, verbose=0))
        
        # Augmentations (Horizontal flip, Vertical flip, Rotations)
        for _ in range(num_augs):
            aug_img = tf.image.random_flip_left_right(img_array[0])
            aug_img = tf.image.random_brightness(aug_img, 0.1)
            aug_img = tf.expand_dims(aug_img, axis=0)
            predictions.append(self.model.predict(aug_img, verbose=0))
            
        avg_prediction = np.mean(predictions, axis=0)
        return avg_prediction
    
    def generate_gradcam(self, img_array, predicted_class=None):
        """Generate Grad-CAM heatmap showing which regions the model focuses on.
        
        Args:
            img_array: preprocessed image array with shape (1, H, W, 3)
            predicted_class: class index to generate heatmap for (None = use top prediction)
        Returns:
            heatmap: numpy array (H, W) with values in [0, 1]
            overlay: numpy array (H, W, 3) RGB image with heatmap overlay
        """
        if self.model is None:
            return None, None
        
        # Find the last convolutional layer in the base model
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.Model):
                # This is the base model — search inside it
                for sub_layer in reversed(layer.layers):
                    if len(sub_layer.output_shape) == 4:  # Conv layer has 4D output
                        last_conv_layer = sub_layer
                        break
                if last_conv_layer:
                    break
            elif len(layer.output_shape) == 4:
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            return None, None
        
        # Build a model that outputs both the conv layer output and the final predictions
        grad_model = tf.keras.Model(
            inputs=self.model.input,
            outputs=[last_conv_layer.output, self.model.output]
        )
        
        # Compute gradients
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor)
            if predicted_class is None:
                predicted_class = tf.argmax(predictions[0])
            loss = predictions[:, predicted_class]
        
        # Gradient of the predicted class w.r.t. the conv layer output
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of the gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # ReLU and normalize
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to match original image size
        heatmap_resized = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
        
        # Create colored overlay
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        original = np.uint8(img_array[0] if img_array[0].max() > 1 else img_array[0] * 255)
        overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        
        return heatmap_resized, overlay
        
    def save_model(self, model_path='skin_disease_model.h5'):
        self.model.save(model_path, save_format='h5')
        
    def save_tflite(self, model_path='skin_disease_model.tflite'):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(model_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite Model saved to {model_path}")
        
    def load_model(self, model_path='skin_disease_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

def main():
    detector = SkinDiseaseDetector()
    detector.get_class_names()
    detector.create_data_generators(batch_size=32)
    detector.build_model()
    detector.train_model()
    test_acc, y_pred, y_true = detector.evaluate_model()
    detector.plot_training_history()
    detector.plot_confusion_matrix(y_true, y_pred)
    detector.save_model()
    detector.save_tflite()
    print("Training completed!")

if __name__ == "__main__":
    main()