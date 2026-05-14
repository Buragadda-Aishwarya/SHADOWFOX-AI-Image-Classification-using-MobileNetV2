import matplotlib.pyplot as plt
import os
from classifier import ShadowFaxClassifier

def run_inference(image_filename):
    print(f"\n[🚀] Processing: {image_filename}")
    
    engine = ShadowFaxClassifier()
    
    if not os.path.exists(image_filename):
        print(f"[❌] Error: File {image_filename} not found.")
        return

    # Process and Predict
    original_img, processed_array = engine.process_image(image_filename)
    results = engine.predict(processed_array)

    print("\n[🎯] Predicted Tags:")
    for i, (imagenet_id, label, prob) in enumerate(results):
        print(f"{i+1}. {label.replace('_', ' ').title()} ({prob*100:.2f}%)")

    # Display Result
    plt.imshow(original_img)
    plt.title(f"Top Prediction: {results[0][1]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # To run: place an image named 'test.jpg' in the root or change this path
    test_image = "test.jpg" 
    run_inference(test_image)
