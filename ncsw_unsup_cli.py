import tensorflow as tf
from tensorflow.keras import layers, models, Input, backend as K
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2

# -------------------------------------------------------------------------
# 1. N-pair Contrastive Loss (NT-Xent)
# -------------------------------------------------------------------------
class NPairsLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=0.1, name="n_pairs_loss"):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        # y_true is ignored as this is unsupervised
        # y_pred shape: (batch_size * 2, projection_dim)
        # The first half of the batch are view 1, the second half are view 2
        
        batch_size = tf.shape(y_pred)[0] // 2
        z_i = y_pred[:batch_size]
        z_j = y_pred[batch_size:]
        
        # Normalize embeddings
        z_i = tf.math.l2_normalize(z_i, axis=1)
        z_j = tf.math.l2_normalize(z_j, axis=1)
        
        # Concatenate for similarity matrix calculation
        z = tf.concat([z_i, z_j], axis=0) # (2N, D)
        
        # Compute cosine similarity matrix
        sim_matrix = tf.matmul(z, z, transpose_b=True) # (2N, 2N)
        sim_matrix = sim_matrix / self.temperature
        
        # Mask out self-similarity
        mask = tf.eye(2 * batch_size, dtype=tf.bool)
        labels = tf.range(2 * batch_size)
        labels = (labels + batch_size) % (2 * batch_size)
        
        # Remove self-similarity from similarity matrix
        sim_matrix = tf.boolean_mask(sim_matrix, ~mask)
        sim_matrix = tf.reshape(sim_matrix, (2 * batch_size, 2 * batch_size - 1))
        
        # Labels for cross-entropy
        # The positive pair for i is (i + batch_size) % (2N)
        # Since we removed the self-similarity (diagonal), the positive index shifts
        # If the positive index was > i, it decreases by 1
        labels = tf.range(batch_size)
        labels = tf.concat([labels + (batch_size - 1), labels], axis=0)

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, sim_matrix, from_logits=True)
        return tf.reduce_mean(loss)

# -------------------------------------------------------------------------
# 2. Data Generator with Augmentation
# -------------------------------------------------------------------------
class ContrastivejhDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, batch_size=4, input_shape=(1024, 1024, 3)):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.indices = np.arange(len(self.image_paths))

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[k] for k in batch_indices]
        
        batch_x1 = []
        batch_x2 = []
        
        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            
            # Augmentation 1
            img1 = self.augment(img)
            batch_x1.append(img1)
            
            # Augmentation 2
            img2 = self.augment(img)
            batch_x2.append(img2)
            
        return np.array(batch_x1), np.array(batch_x2)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def augment(self, image):
        # Specific augmentations can be added here (random crop, color jitter, flip)
        # For simplicity, we implement random flip and brightness
        
        # Random Horizontal Flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            
        # Random Vertical Flip
        if np.random.rand() > 0.5:
             image = cv2.flip(image, 0)
             
        # Normalize
        image = image.astype(np.float32) / 255.0
        return image

# -------------------------------------------------------------------------
# 3. Model Architecture (Encoder + Projection Head)
# -------------------------------------------------------------------------
def wscn_conv_block(x, filters, kernel_size=(3, 3), padding='same', use_separable=False):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.2)(x)
    
    if use_separable:
        x = layers.SeparableConv2D(filters, kernel_size, padding=padding)(x)
    else:
        x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
        
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def build_wscn_encoder(input_shape=(1024, 1024, 3)):
    inputs = Input(shape=input_shape)

    # --- Feature Extraction (Downsampling) ---
    e1 = wscn_conv_block(inputs, 16, use_separable=True)
    p1 = layers.MaxPooling2D((2, 2))(e1)

    e2 = wscn_conv_block(p1, 32, use_separable=True)
    p2 = layers.MaxPooling2D((2, 2))(e2)

    e3 = wscn_conv_block(p2, 64, use_separable=True)
    p3 = layers.MaxPooling2D((2, 2))(e3)

    e4 = wscn_conv_block(p3, 128, use_separable=True)
    p4 = layers.MaxPooling2D((2, 2))(e4)

    # Bridge (Encoder Output)
    b1 = wscn_conv_block(p4, 256, use_separable=False)
    
    # Global Pooling
    x = layers.GlobalAveragePooling2D()(b1)
    
    # Projection Head
    projection = layers.Dense(128, activation='relu')(x)
    projection = layers.Dense(64)(projection) # Embedding dimension

    model = models.Model(inputs=inputs, outputs=projection, name="WSCN_Encoder_Pretrain")
    return model

# -------------------------------------------------------------------------
# 4. Custom Training Loop for Contrastive Learning
# -------------------------------------------------------------------------
class ContrastiveModel(models.Model):
    def __init__(self, encoder, temperature=0.1):
        super(ContrastiveModel, self).__init__()
        self.encoder = encoder
        self.loss_fn = NPairsLoss(temperature)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, data):
        # We assume data is a tuple of (view1, view2)
        # But this method is mainly for inference/prediction if needed
        # Training logic is in train_step
        return self.encoder(data)

    def train_step(self, data):
        # Unpack the data
        # data is what generator yields: (batch_x1, batch_x2)
        # In tf.data or keras Sequence, data might come wrapped
        if isinstance(data, tuple) and len(data) == 2:
            view1, view2 = data
        else:
            # Handle potential wrapping
            view1, view2 = data[0]

        with tf.GradientTape() as tape:
            z1 = self.encoder(view1, training=True)
            z2 = self.encoder(view2, training=True)
            
            # Concatenate for loss calculation
            # We want shape (2N, D) where 2N is batch_size * 2
            predictions = tf.concat([z1, z2], axis=0)
            
            # Calculate loss
            # y_true is dummy, but Keras loss expects it not to be None.
            # We can pass predictions as y_true because our custom loss ignores y_true.
            loss = self.loss_fn(predictions, predictions)

        # Compute gradients
        trainable_vars = self.encoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
        
    @property
    def metrics(self):
        return [self.loss_tracker]

# -------------------------------------------------------------------------
# 5. CLI & Main Execution
# -------------------------------------------------------------------------
import argparse

def train(args):
    # A. Data Loading
    print("1. Loading dataset...")
    input_shape = (1024, 1024, 3)
    dataset_dir = args.image_dir
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist.")
        return
        
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, '*')))
    if len(image_paths) == 0:
        print("Error: No images found.")
        return
        
    print(f"Found {len(image_paths)} images.")
    
    # Create Data Generator
    data_gen = ContrastivejhDataGenerator(image_paths, batch_size=args.batch_size, input_shape=input_shape)
    
    # B. Model Creation
    print("\n2. Creating model...")
    encoder = build_wscn_encoder(input_shape)
    contrastive_model = ContrastiveModel(encoder, temperature=0.1)
    
    # Compile
    contrastive_model.compile(optimizer='adam')
    
    # C. Training
    print(f"\n3. Starting pre-training for {args.epochs} epochs...")
    history = contrastive_model.fit(data_gen, epochs=args.epochs)
    
    # Save Encoder
    print(f"\n4. Saving pre-trained encoder to {args.model_path}...")
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    encoder.save(args.model_path)
    print(f"Encoder saved to {args.model_path}")
    
    # Plot Loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('Contrastive Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('loss_plot.png')
    print("Loss plot saved to loss_plot.png")

def inference(args):
    print("1. Loading dataset for inference...")
    input_shape = (1024, 1024, 3)
    dataset_dir = args.image_dir
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist.")
        return
        
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, '*')))
    if len(image_paths) == 0:
        print("Error: No images found.")
        return
        
    print(f"Found {len(image_paths)} images.")
    
    # Load Model
    print(f"\n2. Loading model from {args.model_path}...")
    if not os.path.exists(args.model_path):
         print(f"Error: Model file {args.model_path} does not exist. Please train first.")
         return
         
    model = models.load_model(args.model_path)
    
    # Extract Embeddings
    print("\n3. Extracting embeddings...")
    embeddings = []
    
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_shape[1], input_shape[0]))
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # (1, H, W, 3)
        
        emb = model.predict(img, verbose=0)
        embeddings.append(emb[0])
        
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save Embeddings
    np.save(args.output, embeddings)
    print(f"Embeddings saved to {args.output}")

    # Save as Image
    # Normalize to 0-255
    norm_embeddings = (embeddings - np.min(embeddings)) / (np.max(embeddings) - np.min(embeddings) + 1e-8)
    norm_embeddings = (norm_embeddings * 255).astype(np.uint8)
    
    # Save image
    output_image_path = os.path.splitext(args.output)[0] + '.png'
    # Embeddings are (N, D). To save as image, we might want to resize or just save as is.
    # If D is small (e.g., 64), it might be too thin. Let's replicate rows for better visibility if N is small.
    # For now, just save raw (N, D).
    cv2.imwrite(output_image_path, norm_embeddings)
    print(f"Embeddings image saved to {output_image_path}")

def main():
    parser = argparse.ArgumentParser(description="WSCN Unsupervised Pre-training CLI")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], help='Mode: train or inference')
    parser.add_argument('--image_dir', type=str, default='dataset/images', help='Path to image directory')
    parser.add_argument('--model_path', type=str, default='models/pretrained_encoder.h5', help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--output', type=str, default='embeddings.npy', help='Output file for inference embeddings')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)

if __name__ == "__main__":
    main()
