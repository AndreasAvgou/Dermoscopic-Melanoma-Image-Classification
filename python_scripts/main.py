import numpy as np
import tensorflow as tf
from data_preprocessing import get_data_generators
from model_definition import build_model
from utils import CollectBatchStats, plot_history

# Configuration
DATA_ROOT = '/content/drive/MyDrive/release_v0/images'
BATCH_SIZE = 32
IMAGE_SHAPE = (64, 64)
EPOCHS = 70

def main():
    # 1. Prepare Data
    print("Loading data...")
    train_gen, val_gen = get_data_generators(DATA_ROOT, IMAGE_SHAPE, BATCH_SIZE)

    # 2. Build and Compile Model
    print("Building model...")
    model = build_model(input_shape=(64, 64, 3), num_classes=34)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer='Adam',
        metrics=['accuracy']
    )
    model.summary()

    # 3. Train
    print("Starting training...")
    batch_stats_callback = CollectBatchStats()
    history = model.fit(
        train_gen,
        steps_per_epoch=np.ceil(train_gen.n / BATCH_SIZE),
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[batch_stats_callback],
        verbose=1
    )

    # 4. Evaluate
    print("Evaluating...")
    train_loss, train_acc = model.evaluate(train_gen, steps=45)
    test_loss, test_acc = model.evaluate(val_gen, steps=32)
    print(f'Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}')

    # 5. Visualization
    plot_history(history)

if __name__ == "__main__":
    # Ensure Colab drive is mounted if running in Colab environment
    # from google.colab import drive
    # drive.mount('/content/drive')
    main()