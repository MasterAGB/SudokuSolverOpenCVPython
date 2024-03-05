def pre_preprocess_image(img, size=(32, 32)):
    img_resized = cv2.resize(img, size)
    _, img_processed = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY)
    return img_processed


import cv2
import numpy as np

import json
import os

db_path = "number_db.json"


def load_db():
    if os.path.exists(db_path):
        with open(db_path, "r") as file:
            db = json.load(file)
    else:
        db = {}
    return db


def save_db(db):
    with open(db_path, "w") as file:
        json.dump(db, file)


THRESH = 0.8;


def recognize_number(image, db, similarity_threshold=-1):
    if (similarity_threshold < 0):
        similarity_threshold = THRESH

    image_flat = image.flatten()
    best_match_key = None
    best_similarity = -1
    best_variant_index = -1  # Index of the best matching variant

    for key, variants in db.items():
        for index, variant in enumerate(variants):
            variant_array = np.array(variant["image"])
            if image_flat.shape == variant_array.shape:
                similarity = similarity_score(image_flat, variant_array)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_key = key
                    best_variant_index = index

    if best_similarity > similarity_threshold:
        return best_match_key, best_similarity, best_variant_index
    return None, best_similarity, best_variant_index


def adjust_variant(image_flat, db, number_key, variant_index):
    variant_array = np.array(db[number_key][variant_index]["image"])
    adjusted_variant = ((variant_array + np.array(image_flat)) / 2).tolist()
    db[number_key][variant_index]["image"] = adjusted_variant
    print(f"Adjusted variant for {number_key} to better match new input.")


def update_db(image, number, db, update_threshold=-1):
    if (update_threshold < 0):
        update_threshold = THRESH

    image_flat = image.flatten().tolist()
    closest_number, best_similarity, variant_index = recognize_number(image, db)

    if best_similarity < update_threshold:
        # Check if the number key exists in the database; if not, initialize it with an empty list
        if number not in db:
            db[number] = []  # Initialize with an empty list
            print(f"Creating new entry for {number} in database.")
        db[number].append({"image": image_flat})
        print(f"Adding new variant for {number}.")
    elif best_similarity < 1 and variant_index != -1:
        # Ensure you handle adjustment correctly; ensure the adjust_variant function is implemented to handle this
        adjust_variant(image_flat, db, number, variant_index)
    else:
        print(f"Existing variant of {number} is similar enough; not adding a new variant.")
    save_db(db)


def similarity_score(image_flat, variant_array):
    """
    Computes a similarity score between two images based on grayscale differences.
    """
    # Normalize the difference to be between 0 and 1, where 1 is identical
    diff = np.abs(image_flat - variant_array)
    similarity = 1 - np.mean(diff) / 255
    return similarity


def custom_ocr_engine(processed_image):
    optimized_processed_image = pre_preprocess_image(processed_image)

    db = load_db()
    recognized_number, confidence, potential_number = recognize_number(optimized_processed_image, db)

    if recognized_number is None:
        cv2.imshow("Unknown Number", optimized_processed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Prompting the user for input if the number is not recognized

        number = input(f"Please enter the digit shown in the image {potential_number} - {confidence:.2f}: ")
        if number:  # Check if the user entered a value
            update_db(optimized_processed_image, number, db)
            print(f"Added as new variant for {number}.")
            return number  # Return the input number as a string
        else:
            return ""  # Return an empty string if no input was provided
    else:
        print(f"Recognized digit: {recognized_number} with confidence: {confidence:.2f}")
        # here we are updating the values a bit to make it better fit next time
        update_db(optimized_processed_image, recognized_number, db)
        return str(recognized_number)  # Ensure we return a string representation of the number


def decodeCell(cell):
    return custom_ocr_engine(cell)
