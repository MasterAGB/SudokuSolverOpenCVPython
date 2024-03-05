from decodingTools import decodeCell  # This would be your Sudoku solving function
from sudoku_solver import solve_sudoku  # This would be your Sudoku solving function
from sudoku_solver import optimize_recognized_digits  # This would be your Sudoku solving function



##1:
def preprocess_image(original_image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to smooth the image
    # You can adjust the kernel size (here: (5, 5)) and sigma values as needed
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to binarize the image
    # The maxValue, adaptiveMethod, thresholdType, blockSize, and C value can be adjusted
    # 11 and 2 ?
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 39, 2)

    # cv2.imshow('Processed Image', adaptive_thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return adaptive_thresh


##2:

def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e., top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return both the warped image and the transformation matrix
    return warped, M


def find_sudoku_grid(preprocessed_image):
    # Find contours in the preprocessed image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the Sudoku grid is the largest contour by area
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    if best_cnt is not None:
        # Approximate the contour
        peri = cv2.arcLength(best_cnt, True)
        approx = cv2.approxPolyDP(best_cnt, 0.02 * peri, True)

        if len(approx) == 4:  # The Sudoku grid should roughly be a square

            warped, M = four_point_transform(preprocessed_image, approx.reshape(4, 2))

            return warped, approx.reshape(4, 2), M  # Correctly return 3 values here

    return None, None, None  # Ensure to return None for all three if no suitable grid is found


##3:

import numpy as np

import cv2
import numpy as np


def trim_image(image):
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Use adaptive thresholding to accommodate varying lighting conditions
    thresh = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Proceed if any contours were found
    if contours:
        # Optionally filter contours by area (e.g., remove very small contours)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]  # Adjust the threshold as needed

        if contours:  # Check again in case the filter removed all contours
            # Find the bounding box for the combined contours
            all_contours = np.vstack(contours).reshape(-1, 2)
            x, y, w, h = cv2.boundingRect(all_contours)

            # Crop the original (unblurred) image
            cropped = image[y:y + h, x:x + w]

            # Display steps for debugging
            # cv2.imshow("Original", image)
            # cv2.waitKey(0)
            # cv2.imshow("Blurred", blurred_image)
            # cv2.waitKey(0)
            # cv2.imshow("Threshold", thresh)
            # cv2.waitKey(0)
            # cv2.imshow("Cropped", cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            return cropped

    # Return the original image if no suitable contours were found
    return image


def trim_borders_from_center(cell, expansion_threshold=250):
    """
    Starts with a central box half the size of the cell and expands until white space is encountered.

    :param cell: Grayscale cell image with white background and black content.
    :param expansion_threshold: Value above which a pixel is considered white.
    :return: Cropped cell image centered around the content.
    """
    # Determine initial box size (half of the image dimensions)
    height, width = cell.shape
    start_x, end_x = width // 4, 3 * width // 4
    start_y, end_y = height // 4, 3 * height // 4

    # Convert to binary for easier processing
    _, binary_cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Function to check if expansion is possible in a direction
    def can_expand(direction, start_x, end_x, start_y, end_y):
        if direction == 'left' and start_x > 0:
            return np.any(binary_cell[start_y:end_y, start_x - 1] < expansion_threshold)
        elif direction == 'right' and end_x < width - 1:
            return np.any(binary_cell[start_y:end_y, end_x + 1] < expansion_threshold)
        elif direction == 'up' and start_y > 0:
            return np.any(binary_cell[start_y - 1, start_x:end_x] < expansion_threshold)
        elif direction == 'down' and end_y < height - 1:
            return np.any(binary_cell[end_y + 1, start_x:end_x] < expansion_threshold)
        return False

    # Expand the box until white space is encountered
    while can_expand('left', start_x, end_x, start_y, end_y):
        start_x -= 1
    while can_expand('right', start_x, end_x, start_y, end_y):
        end_x += 1
    while can_expand('up', start_x, end_x, start_y, end_y):
        start_y -= 1
    while can_expand('down', start_x, end_x, start_y, end_y):
        end_y += 1

    # Crop the original grayscale image according to the final expanded box
    cropped_cell = cell[start_y:end_y + 1, start_x:end_x + 1]

    return cropped_cell


# Configure Pytesseract path to the executable
# pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
def extract_and_recognize_cells(grid_image):
    processed_cells = []  # To store processed cell images for visualization

    cells = np.zeros((9, 9), dtype=int)  # Initialize the Sudoku grid as a 9x9 matrix of zeros
    cell_height, cell_width = grid_image.shape[0] // 9, grid_image.shape[1] // 9

    for row in range(9):
        processed_row = []  # Store processed cells for the current row

        for col in range(9):
            # Extract the individual cell from the grid image
            start_y, start_x = row * cell_height, col * cell_width

            cell = grid_image[start_y:start_y + cell_height, start_x:start_x + cell_width]

            # Optionally visualize the cell before processing
            # cv2.imshow(f'Cell [{row},{col}]', cell)
            # cv2.waitKey(0)  # Wait for a key press to continue
            # cv2.destroyAllWindows()  # Close the window

            # Preprocess the cell for OCR: you might need to adjust preprocessing based on your input images
            cell = cv2.bitwise_not(cell)  # Invert colors if needed
            cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            # Trim borders containing the grid lines
            cell = trim_borders_from_center(cell)

            # print(cell.shape)
            # Optionally visualize the cell before processing
            # cv2.imshow(f'Cell [{row},{col}]', cell)
            # cv2.waitKey(0)  # Wait for a key press to continue
            # cv2.destroyAllWindows()  # Close the window

            # Extend the image by 3 white pixels on each side
            padding = 3
            cell = cv2.copyMakeBorder(cell, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                      value=[255, 255, 255])
            cell = trim_image(cell);

            processed_row.append(cell)

            # Check if the cell is not completely white
            if np.any(cell < 255):  # This checks if there's any black pixel

                # Use Pytesseract to recognize the digit in the cell
                digit_text = decodeCell(cell)

                # Clean the recognized text and convert to an integer if a digit is found
                digit_text = digit_text.strip()
                if digit_text.isdigit():
                    print("Found:" + digit_text);
                    cells[row, col] = int(digit_text)
                else:
                    print("not Found - unrecognized:" + digit_text)
                    cells[row, col] = 0  # Empty cell
            else:
                print("not Found - empty:" + digit_text)
                cells[row, col] = 0  # Empty cell5

        # Append the row of processed cells to the main list
        processed_cells.append(processed_row)

        # Convert processed_cells to a NumPy array for easier handling
    # processed_cells = np.array(processed_cells)

    # After processing all cells and obtaining results, visualize them
    visualize_processed_cells(processed_cells)

    return cells


def visualize_processed_cells(processed_cells):
    # Define the size for each cell in the visualization
    vis_cell_size = (40, 40)  # Width, Height
    gap = 1  # Gap size in pixels
    rows, cols = 9, 9
    border_color = (0, 0, 255)  # Red border in BGR format

    # Calculate the dimensions of the full image including gaps
    full_image_height = rows * (vis_cell_size[1] + gap) + gap
    full_image_width = cols * (vis_cell_size[0] + gap) + gap

    # Create a new image with a red background
    full_image = np.full((full_image_height, full_image_width, 3), border_color, dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            # Get the top-left corner for this cell on the full image
            top_left_y = row * (vis_cell_size[1] + gap) + gap
            top_left_x = col * (vis_cell_size[0] + gap) + gap

            # Extract the processed cell (need to handle both binary and BGR conversion)
            cell = processed_cells[row][col]
            if len(cell.shape) == 2:  # If grayscale, convert to BGR
                cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
            # Resize the cell for uniformity
            cell_resized = cv2.resize(cell, vis_cell_size, interpolation=cv2.INTER_AREA)

            # Place the resized cell in the full image
            full_image[top_left_y:top_left_y + vis_cell_size[1],
            top_left_x:top_left_x + vis_cell_size[0]] = cell_resized

    # Display the full image
    cv2.imshow('Processed Cells Visualization', full_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


##4:
def find_cell_center(cell_corners):
    # Calculate the center of the cell as the average of the corner points
    center_x = sum([corner[0] for corner in cell_corners]) / 4
    center_y = sum([corner[1] for corner in cell_corners]) / 4
    return int(center_x), int(center_y)

def annotate_solution(original_image, solution, cell_positions, color, M_inv):
    """
    Draws the solution digits on the original image, taking into account the grid's transformation.

    :param original_image: The original Sudoku image.
    :param solution: A 9x9 numpy array with the solution digits.
    :param cell_positions: A list of lists, each containing four tuples (x, y) of the cell corners.
    :param color: Color of the text to draw.
    :param M_inv: The inverse transformation matrix used to map points from the warped grid back to the original image.
    """
    for row in range(9):
        for col in range(9):
            digit = solution[row, col]
            if digit > 0:  # If the cell is not empty
                # Calculate cell center and size based on its corners
                cell_corners = cell_positions[row * 9 + col]
                cell_center = np.mean(cell_corners, axis=0)
                cell_width = np.linalg.norm(np.array(cell_corners[1]) - np.array(cell_corners[0]))
                cell_height = np.linalg.norm(np.array(cell_corners[3]) - np.array(cell_corners[0]))
                cell_size = min(cell_width, cell_height)  # Use the smaller dimension to ensure the text fits

                # Dynamically adjust font scale and thickness based on cell size
                font_scale = cell_size / 40  # Adjust this value based on your preference
                thickness = max(1, int(cell_size / 20))  # Adjust this value based on your preference

                # Transform the center point back to the original image's coordinate system
                original_center = cv2.perspectiveTransform(np.array([[cell_center]], dtype='float32'), M_inv)[0][0]

                # Calculate the size of the text to be drawn
                text_size = cv2.getTextSize(str(digit), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

                # Calculate the bottom-left corner of the text from the center in the original image
                text_origin = (int(original_center[0] - text_size[0] / 2), int(original_center[1] + text_size[1] / 2))

                # Draw the digit on the original image
                cv2.putText(original_image, str(digit), text_origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return original_image



def order_points(pts):
    # Initialize a list of coordinates that will be ordered
    rect = np.zeros((4, 2), dtype="float32")

    # The ordering of points in the rect should be [top-left, top-right, bottom-right, bottom-left]
    # Calculate the sum and difference of the points
    s = np.sum(pts, axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def calculate_transformed_cell_positions(grid_corners, M_inv):
    """
    Calculate each cell's position in the original image coordinates
    by using the inverse transformation matrix (M_inv).
    """
    cell_positions = []
    step = 1 / 9
    for row in range(9):
        for col in range(9):
            # Calculate normalized position within the grid for current cell
            top_left_norm = np.dot(M_inv, [col * step, row * step, 1])
            top_right_norm = np.dot(M_inv, [(col + 1) * step, row * step, 1])
            bottom_left_norm = np.dot(M_inv, [col * step, (row + 1) * step, 1])
            bottom_right_norm = np.dot(M_inv, [(col + 1) * step, (row + 1) * step, 1])

            # Convert from homogeneous coordinates
            top_left = (top_left_norm[:2] / top_left_norm[2]).astype(int)
            top_right = (top_right_norm[:2] / top_right_norm[2]).astype(int)
            bottom_left = (bottom_left_norm[:2] / bottom_left_norm[2]).astype(int)
            bottom_right = (bottom_right_norm[:2] / bottom_right_norm[2]).astype(int)

            # Store the calculated corners for this cell
            cell_positions.append([top_left, top_right, bottom_right, bottom_left])

    return cell_positions


def calculate_cell_positions(original_image, grid_corners):
    # First, correct the perspective distortion
    corrected_image, M = four_point_transform(original_image, np.array(grid_corners))
    M_inv = np.linalg.inv(M)

    # Now that the image is a square, calculate positions
    cell_positions = []
    for row in range(9):
        for col in range(9):
            cell_size_col = corrected_image.shape[1] // 9
            cell_size_row = corrected_image.shape[0] // 9

            cell_x, cell_y = col * cell_size_col, row * cell_size_row
            # For each cell, now store all four corners
            cell_positions.append([
                (cell_x, cell_y),  # Top-left
                (cell_x + cell_size_col, cell_y),  # Top-right
                (cell_x + cell_size_col, cell_y + cell_size_row),  # Bottom-right
                (cell_x, cell_y + cell_size_row)  # Bottom-left
            ])

    return cell_positions


def main(image_path):
    original_image = cv2.imread(image_path)
    processed_image = preprocess_image(original_image)
    # cv2.imshow('Processed Image', processed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Make sure to replace 'preprocessed_image' with your actual preprocessed image variable
    warped_grid, grid_corners, M = find_sudoku_grid(processed_image)
    if warped_grid is not None:
        print("Sudoku grid found.")
        cv2.imshow('Warped Sudoku Grid', warped_grid)

        M_inv = np.linalg.inv(M)
        print("Inverse Matrix, M_inv:", M_inv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print("Sudoku grid not found.")

    cell_positions = calculate_cell_positions(original_image, grid_corners)
    print("Grid and cell positions.")
    print(grid_corners)
    print(cell_positions)

    recognized_digits = extract_and_recognize_cells(warped_grid)

    print("Recognized digits")
    print(recognized_digits)

    recognized_digits = optimize_recognized_digits(recognized_digits)
    print("Optimized digits")
    print(recognized_digits)

    annotated_image = annotate_solution(original_image, recognized_digits, cell_positions, (255, 0, 255), M_inv)
    cv2.imshow('recognized Sudoku digits', annotated_image)

    # Example usage
    # Make sure to replace these variables with the actual data
    solution = solve_sudoku(recognized_digits)

    annotated_image = annotate_solution(original_image, solution, cell_positions, (0, 0, 255), M_inv)
    cv2.imshow('Solved Sudoku digits', annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "sudoku2fake.jpg"
    main(image_path)
