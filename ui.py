import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from pathlib import Path
import matplotlib.pyplot as plt
import time
import random

#BUILT WITH STREAMLIT // streamlit run ui.py

st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide"
)

st.title("Traffic Sign Classifier")
st.markdown("Upload traffic sign images or evaluate models on test datasets")


class SevenClassDataset:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.classes = ['20_speed_limit', '60_speed_limit', 'bike_lane',
                        'children_crossing', 'do_not_enter',
                        'pedestrian_crossing', 'stop']
        self.num_classes = len(self.classes)

        self.gtsrb_class_mapping = {
            '20_speed_limit': 0, '60_speed_limit': 3, 'bike_lane': 29,
            'children_crossing': 28, 'do_not_enter': 17,
            'pedestrian_crossing': 27, 'stop': 14
        }
        self.gtsrb_to_7class = {0: 0, 3: 1, 29: 2, 28: 3, 17: 4, 27: 5, 14: 6}

    def load_dataset(self, target_size=(128, 128), split='test', use_gtsrb_mapping=False):
        split_names = {'train': 'Train', 'test': 'Test', 'val': 'Val'}
        split_path = self.data_path / split_names.get(split, 'Test')

        images = []
        labels = []

        if split_path.exists():
            for class_name in self.classes:
                class_path = split_path / class_name
                if class_path.exists():
                    for img_file in class_path.glob('*.*'):
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img = cv2.resize(img, target_size)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            images.append(img)

                            if use_gtsrb_mapping:
                                gtsrb_label = self.gtsrb_class_mapping[class_name]
                                labels.append(gtsrb_label)
                            else:
                                class_idx = self.classes.index(class_name)
                                labels.append(class_idx)

        if images:
            images = np.array(images)
            labels = np.array(labels)

        return images, labels


def map_gtsrb_to_7class(labels_gtsrb, mapping_dict):
    return np.array([mapping_dict[label] for label in labels_gtsrb])


#Model architectures
def create_resnet_model(num_classes=7, input_size=(128, 128)):
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(*input_size, 3)
    )
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    inputs = tf.keras.Input((*input_size, 3))
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def create_vgg_model(input_shape=(32, 32, 3), num_classes=7):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


#Initialize models
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_models():
    models = {}
    model_configs = {}

    st.info("Loading models... This may take a few seconds.")

    #BaseResNet
    if Path("models/phase2_fineTunedResNet.h5").exists():
        try:
            model = create_resnet_model()
            model.load_weights("models/phase2_fineTunedResNet.h5")
            models['BaseResNet'] = model
            model_configs['BaseResNet'] = {'input_size': (128, 128), 'preprocess': 'resnet',
                                           'needs_gtsrb_mapping': False}
            st.success("BaseResNet loaded")
        except Exception as e:
            st.error(f"BaseResNet failed: {e}")

    #BaseResnetZeroDCE
    if Path("models/baseresnet_zerodce_complete.h5").exists():
        try:
            model = create_resnet_model()
            model.load_weights("models/baseresnet_zerodce_complete.h5")
            models['BaseResNet_ZeroDCE'] = model
            model_configs['BaseResNet_ZeroDCE'] = {'input_size': (128, 128), 'preprocess': 'resnet',
                                                   'needs_gtsrb_mapping': False}
            st.success("BaseResNet ZeroDCE loaded")
        except Exception as e:
            st.error(f"BaseResNet ZeroDCE failed: {e}")

    #BaseVGG
    if Path("models/basevgg_complete.h5").exists():
        try:
            model = create_vgg_model()
            model.load_weights("models/basevgg_complete.h5")
            models['BaseVGG'] = model
            model_configs['BaseVGG'] = {'input_size': (32, 32), 'preprocess': 'vgg', 'needs_gtsrb_mapping': False}
            st.success("BaseVGG loaded")
        except Exception as e:
            st.error(f"BaseVGG failed: {e}")

    #ResNet MultiPhase
    multiphase_path = "models/resnet_multiphase_complete.h5"
    phase1_path = "models/gtsrb_model_phase1final_continued.h5"

    if Path(multiphase_path).exists() and Path(phase1_path).exists():
        try:
            def create_resnet_multiphase_model():
                base_model = tf.keras.applications.ResNet50(
                    weights=None,
                    include_top=False,
                    input_shape=(64, 64, 3)
                )

                inputs = tf.keras.Input(shape=(64, 64, 3))
                x = base_model(inputs, training=False)
                x = layers.GlobalAveragePooling2D()(x)
                x = layers.Dense(512, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.5)(x)
                x = layers.Dense(256, activation='relu')(x)
                outputs = layers.Dense(7, activation='softmax')(x)

                model = tf.keras.Model(inputs, outputs)

                model.load_weights(multiphase_path)

                for layer in model.layers[:-30]:
                    layer.trainable = False

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                return model

            model = create_resnet_multiphase_model()
            models['ResNet_MultiPhase'] = model
            model_configs['ResNet_MultiPhase'] = {'input_size': (64, 64), 'preprocess': 'resnet',
                                                  'needs_gtsrb_mapping': True}
            st.success("ResNet MultiPhase loaded")
        except Exception as e:
            st.error(f"ResNet MultiPhase failed: {e}")
    else:
        st.warning("ResNet MultiPhase files not found")

    if models:
        st.success(f"Successfully loaded {len(models)} models: {', '.join(models.keys())}")
    else:
        st.error("No models were loaded successfully!")

    return models, model_configs



#Load models
models, model_configs = load_models()

#Sidebar
st.sidebar.title("Configuration")

if not models:
    st.sidebar.error("No models loaded! Check your model files.")
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", list(models.keys()))
selected_dataset = st.sidebar.selectbox("Select Dataset", ['datanozerodce', 'datazerodce'])

#Model info in sidebar
st.sidebar.header("Model Information")
if selected_model in model_configs:
    config = model_configs[selected_model]
    st.sidebar.write(f"**Input Size:** {config['input_size']}")
    st.sidebar.write(f"**Preprocessing:** {config['preprocess']}")
    st.sidebar.write(f"**Uses GTSRB:** {config['needs_gtsrb_mapping']}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Image Prediction",
    "Model Evaluation",
    "Random Sample Predictions",
    "Failure Case Analysis"
])

with tab1:
    st.header("Single Image Prediction")
    uploaded_file = st.file_uploader("Choose a traffic sign image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

        with col2:
            if st.button("Predict Image"):
                with st.spinner("Predicting..."):
                    model = models[selected_model]
                    config = model_configs[selected_model]

                    # Preprocess
                    image_resized = cv2.resize(image_rgb, config['input_size'])
                    if config['preprocess'] == 'resnet':
                        image_processed = image_resized.astype('float32')
                        image_processed = tf.keras.applications.resnet50.preprocess_input(image_processed)
                    else:
                        image_processed = image_resized.astype('float32') / 255.0

                    image_processed = np.expand_dims(image_processed, axis=0)

                    # Predict
                    prediction = model.predict(image_processed, verbose=0)[0]
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[predicted_class]

                    classes = ['20_speed_limit', '60_speed_limit', 'bike_lane',
                               'children_crossing', 'do_not_enter',
                               'pedestrian_crossing', 'stop']

                    st.success(f"**Prediction:** {classes[predicted_class]}")
                    st.info(f"**Confidence:** {confidence:.2%}")

                    # Show confidence scores
                    fig, ax = plt.subplots(figsize=(10, 6))
                    y_pos = np.arange(len(classes))
                    colors = ['lightcoral'] * len(classes)
                    colors[predicted_class] = 'lightgreen'
                    bars = ax.barh(y_pos, prediction, color=colors)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(classes)
                    ax.set_xlabel('Confidence Score')
                    ax.set_title('Prediction Confidence Scores')
                    ax.set_xlim(0, 1)

                    # Add value labels on bars
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                                f'{width:.3f}', ha='left', va='center')

                    st.pyplot(fig)

with tab2:
    st.header("Model Evaluation")

    if st.button("Run Full Evaluation"):
        with st.spinner("Evaluating model on test dataset..."):
            model = models[selected_model]
            config = model_configs[selected_model]

            # Load dataset
            dataset = SevenClassDataset(selected_dataset)
            images, labels = dataset.load_dataset(
                target_size=config['input_size'],
                split='test',
                use_gtsrb_mapping=config['needs_gtsrb_mapping']
            )

            if images is not None and len(images) > 0:
                # Preprocess
                if config['preprocess'] == 'resnet':
                    images_processed = images.astype('float32')
                    images_processed = tf.keras.applications.resnet50.preprocess_input(images_processed)
                else:
                    images_processed = images.astype('float32') / 255.0

                # Convert labels
                if config['needs_gtsrb_mapping']:
                    labels_7class = map_gtsrb_to_7class(labels, dataset.gtsrb_to_7class)
                else:
                    labels_7class = labels

                labels_categorical = tf.keras.utils.to_categorical(labels_7class, 7)

                # Evaluate
                test_loss, test_accuracy = model.evaluate(images_processed, labels_categorical, verbose=0)

                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                with col2:
                    st.metric("Test Loss", f"{test_loss:.4f}")
                with col3:
                    st.metric("Test Images", len(images))

                st.success(
                    f"Evaluation complete! Model achieved {test_accuracy:.2%} accuracy on {len(images)} test images.")
            else:
                st.error(f"No test images found in {selected_dataset}/Test/")

with tab3:
    st.header("Random Sample Predictions")
    num_samples = st.slider("Number of random samples to show", min_value=1, max_value=12, value=6)

    if st.button("Show Random Predictions"):
        with st.spinner("Loading random samples and making predictions..."):
            model = models[selected_model]
            config = model_configs[selected_model]

            dataset = SevenClassDataset(selected_dataset)
            images, labels = dataset.load_dataset(
                target_size=config['input_size'],
                split='test',
                use_gtsrb_mapping=config['needs_gtsrb_mapping']
            )

            if images is not None and len(images) > 0:
                if config['needs_gtsrb_mapping']:
                    labels_7class = map_gtsrb_to_7class(labels, dataset.gtsrb_to_7class)
                else:
                    labels_7class = labels

                if len(images) < num_samples:
                    num_samples = len(images)

                random_indices = random.sample(range(len(images)), num_samples)
                sample_images = images[random_indices]
                sample_labels = labels_7class[random_indices]

                if config['preprocess'] == 'resnet':
                    images_processed = sample_images.astype('float32')
                    images_processed = tf.keras.applications.resnet50.preprocess_input(images_processed)
                else:
                    images_processed = sample_images.astype('float32') / 255.0

                predictions = model.predict(images_processed, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                confidences = np.max(predictions, axis=1)

                classes = ['20_speed_limit', '60_speed_limit', 'bike_lane',
                           'children_crossing', 'do_not_enter',
                           'pedestrian_crossing', 'stop']

                correct_predictions = np.sum(predicted_classes == sample_labels)
                sample_accuracy = correct_predictions / num_samples

                st.subheader(f"Random Sample Results ({num_samples} images)")
                st.write(
                    f"**Accuracy on these samples:** {sample_accuracy:.1%} ({correct_predictions}/{num_samples} correct)")

                cols = 3
                rows = (num_samples + cols - 1) // cols
                fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

                if num_samples == 1:
                    axes_flat = [axes]
                else:
                    axes_flat = axes.flatten()

                for i in range(num_samples):
                    ax = axes_flat[i]
                    ax.imshow(sample_images[i])
                    true_label = classes[sample_labels[i]]
                    pred_label = classes[predicted_classes[i]]
                    is_correct = (predicted_classes[i] == sample_labels[i])
                    color = 'green' if is_correct else 'red'
                    ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidences[i]:.2f}", color=color,
                                 fontsize=10)
                    ax.axis('off')

                for i in range(num_samples, len(axes_flat)):
                    axes_flat[i].axis('off')

                plt.tight_layout()
                st.pyplot(fig)

                st.subheader("Detailed Results")
                results_data = [{'Sample': i + 1, 'True Label': classes[sample_labels[i]],
                                 'Predicted Label': classes[predicted_classes[i]],
                                 'Confidence': f"{confidences[i]:.3f}",
                                 'Correct': 'Yes' if predicted_classes[i] == sample_labels[i] else 'No'} for i in
                                range(num_samples)]
                st.table(results_data)
            else:
                st.error(f"No test images found in {selected_dataset}/Test/")

with tab4:
    st.header("🔍 Failure Case Analysis")
    st.markdown(
        "This section isolates **misclassifications** and groups them by class to show a representative sample of errors.")

    if st.button("Identify Failure Cases"):
        with st.spinner("Analyzing all test images for errors..."):
            model = models[selected_model]
            config = model_configs[selected_model]

            dataset = SevenClassDataset(selected_dataset)
            images, labels = dataset.load_dataset(
                target_size=config['input_size'],
                split='test',
                use_gtsrb_mapping=config['needs_gtsrb_mapping']
            )

            if images is not None and len(images) > 0:
                # Label Mapping & Preprocessing
                labels_7class = map_gtsrb_to_7class(labels, dataset.gtsrb_to_7class) if config[
                    'needs_gtsrb_mapping'] else labels

                if config['preprocess'] == 'resnet':
                    images_processed = tf.keras.applications.resnet50.preprocess_input(images.astype('float32'))
                else:
                    images_processed = images.astype('float32') / 255.0

                # Predictions
                predictions = model.predict(images_processed, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
                confidences = np.max(predictions, axis=1)

                classes = ['20_speed_limit', '60_speed_limit', 'bike_lane', 'children_crossing', 'do_not_enter',
                           'pedestrian_crossing', 'stop']

                # Logic to find failures
                failure_indices = np.where(predicted_classes != labels_7class)[0]

                if len(failure_indices) > 0:
                    st.warning(f"Found {len(failure_indices)} misclassifications out of {len(images)} images.")

                    # --- NEW REPRESENTATIVE SAMPLING LOGIC ---
                    # We want to show up to 6 examples per class to ensure we see "everything"
                    samples_per_class = 6
                    selected_indices = []

                    for class_idx in range(len(classes)):
                        # Find failures belonging to this specific true class
                        class_failures = [idx for idx in failure_indices if labels_7class[idx] == class_idx]
                        # Take the first few (or random.sample if you prefer variety)
                        selected_indices.extend(class_failures[:samples_per_class])

                    # Limit final display to prevent UI crashing (e.g., 48 total)
                    display_indices = selected_indices[:48]

                    num_to_show = len(display_indices)
                    cols = 4
                    rows = (num_to_show + cols - 1) // cols

                    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
                    axes_flat = axes.flatten() if num_to_show > 1 else [axes]

                    for i, idx in enumerate(display_indices):
                        ax = axes_flat[i]
                        ax.imshow(images[idx])

                        true_label = classes[labels_7class[idx]]
                        pred_label = classes[predicted_classes[idx]]

                        ax.set_title(f"TRUE: {true_label}\nPRED: {pred_label}\nConf: {confidences[idx]:.2f}",
                                     color='red', fontsize=10, fontweight='bold')
                        ax.axis('off')

                    for j in range(i + 1, len(axes_flat)):
                        axes_flat[j].axis('off')

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Full searchable table for all failures
                    st.subheader("📊 Complete Error Data")
                    failure_list = [
                        {"True Label": classes[labels_7class[idx]],
                         "Predicted Label": classes[predicted_classes[idx]],
                         "Confidence": round(float(confidences[idx]), 3)}
                        for idx in failure_indices
                    ]
                    st.dataframe(failure_list, use_container_width=True)
                else:
                    st.success("Perfect Accuracy! No failure cases found.")
            else:
                st.error("Test dataset not found.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit - Traffic Sign Classification")
