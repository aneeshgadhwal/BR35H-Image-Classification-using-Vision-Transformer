# importing the libraries

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
from keras import ops
import numpy as np
import matplotlib.pyplot as plt

# data loading

# height, width, channels
input_shape = (224,224)

# visualize the image and its region of interest and crop it out

import cv2, os
dir_no = '/kaggle/input/brain-tumor-detection/no'
dir_yes = '/kaggle/input/brain-tumor-detection/yes'
y = []
X_train = []
for i in os.listdir(dir_no):
    y.append(0)
    path_to_image = dir_no + '/' + i
    #print(path_to_image)
    #break
    image = cv2.imread(path_to_image)
    print(image.shape)
    plt.subplot(2,2,1)
    plt.title("1")
    plt.imshow(image)
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise (optional but recommended)
    gray_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(gray_blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it corresponds to the MRI scan)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box coordinates
    x1, y1, w, h = cv2.boundingRect(largest_contour)
    print(x1,y1,w,h)

    # Crop the original color image using the bounding box coordinates
    cropped_image = image[y1:y1+h, x1:x1+w]
    print("cropped image shape", cropped_image.shape)
    plt.subplot(2,2,2)
    plt.title("2")
    plt.imshow(cropped_image)
    # return the numpy array as h,w,c
    image = cv2.resize(image, input_shape)
    print(image.shape)
    plt.subplot(2,2,3)
    plt.title("3")
    plt.imshow(image)
    cropped_image = cv2.resize(cropped_image, input_shape)
    print(cropped_image.shape)
    plt.subplot(2,2,4)
    plt.title("4")
    plt.imshow(cropped_image)
    #print(image)
    break
    X_train.append(image)

for i in os.listdir(dir_yes):
    y.append(1)
    path_to_image = dir_yes + '/' + i
    #print(path_to_image)
    #break
    image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    # return the numpy array as h,w,c
    
    break
    X_train.append(image)

# prepaing the training data
import cv2, os
dir_no = '/kaggle/input/brain-tumor-detection/no'
dir_yes = '/kaggle/input/brain-tumor-detection/yes'
y = []
X_train = []
for i in os.listdir(dir_no):
    y.append(0)
    path_to_image = dir_no + '/' + i
    #print(path_to_image)
    #break
    image = cv2.imread(path_to_image)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise (optional but recommended)
    gray_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(gray_blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it corresponds to the MRI scan)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box coordinates
    x1, y1, w, h = cv2.boundingRect(largest_contour)
    #print(x1,y1,w,h)

    # Crop the original color image using the bounding box coordinates
    cropped_image = image[y1:y1+h, x1:x1+w]
    
    cropped_image = cv2.resize(cropped_image, input_shape)
    
    X_train.append(cropped_image)
    

for i in os.listdir(dir_yes):
    y.append(1)
    path_to_image = dir_yes + '/' + i
    image = cv2.imread(path_to_image)
    # return the numpy array as h,w,c
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise (optional but recommended)
    gray_blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(gray_blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assuming it corresponds to the MRI scan)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box coordinates
    x1, y1, w, h = cv2.boundingRect(largest_contour)
    #print(x1,y1,w,h)

    # Crop the original color image using the bounding box coordinates
    cropped_image = image[y1:y1+h, x1:x1+w]
    
    cropped_image = cv2.resize(cropped_image, input_shape)
    X_train.append(cropped_image)

print(len(X_train))
print(len(y))
X_train = np.array(X_train)
y = np.array(y)



print("shape of features are {}".format(X_train.shape))
print("shape of target are {}".format(y.shape))

num_classes = 2
input_shape = X_train[0].shape
#print(input_shape)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_train, y, random_state=42, test_size=0.1)
print("shape of training sample are {}, {}".format(x_train.shape, y_train.shape))
print("shape of testing sample are {}, {}".format(x_test.shape, y_test.shape))


# Variable and hyperparameters

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 100  
image_size = 224  
patch_size = 16 
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 2
mlp_head_units = [
    2048,
    1024,
]

# Data augmentation layer

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)


# mlp layer

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=keras.activations.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# image to patches class

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config
    
# Embedding patches layers

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def build(self):
        #print("it is a automatic paramter that is passed to the ", input_shape)
        self.class_token = self.add_weight(
            shape = (1, 1, projection_dim),
            initializer="glorot_uniform",
            trainable=True,
            dtype="float32",
            name="class_token_build",
        )
    
    def call(self, patch):
        positions = ops.expand_dims(
            ops.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        print("projected_patches_shape", projected_patches)
        
        

        batch_size = ops.shape(projected_patches)[0]

        cls = ops.broadcast_to(self.class_token, [batch_size, 1, ops.shape(projected_patches)[-1]])
        cls = ops.cast(cls, dtype=projected_patches.dtype)
        print("class token shape ", cls.shape)

        cls = ops.concatenate([cls, projected_patches], axis=1)
        print("class token + linear porjection patches shape ", cls.shape)

        encoded = cls + self.position_embedding(positions)
        print("(class token + linear projection patches) + position embeddings shape", encoded.shape)
        
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config
    

# vit classifer creation function

def create_vit_classifier():
    inputs = keras.Input(shape=input_shape, batch_size=batch_size)
    print("inputs shape", inputs.shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    print("augmented shape", augmented.shape)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    print("patches shape", patches.shape)
    
    encoded_patches = PatchEncoder(num_patches+1, projection_dim)(patches)
    print("encoded_patches shape", encoded_patches.shape)
    
    
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(1, activation='sigmoid')(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# model compile and fitting

def run_experiment(model):
    optimizer = keras.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint.weights.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )
    
    model.load_weights(checkpoint_filepath)
    _, accuracy = model.evaluate(x_test, y_test, batch_size=256)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history


vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)


def plot_history(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()


plot_history("loss")
plot_history("accuracy")
