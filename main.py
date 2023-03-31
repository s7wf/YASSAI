import cv2
import pytesseract
import pandas as pd
import numpy as np

def process_frame(frame):
    # Define the expected size and position of the card in the image
    card_x = 100  # Adjust this to match the position of the card in your images
    card_y = 100  # Adjust this to match the position of the card in your images
    card_w = 500  # Adjust this to match the size of the card in your images
    card_h = 300  # Adjust this to match the size of the card in your images

    # Apply a mask to the image to isolate the area where the card is located
    mask = np.zeros_like(frame)
    cv2.rectangle(mask, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 255, 255), -1)
    masked_frame = cv2.bitwise_and(frame, mask)

    # Use OCR to detect and extract the text in row 1
    row1_roi = masked_frame[card_y + 10:card_y + 50, card_x:card_x + card_w]
    row1_text = extract_text(row1_roi)

    # Detect the 8 columns in row 2
    row2_roi = masked_frame[card_y + 70:card_y + 100, card_x:card_x + card_w]
    row2_gray = cv2.cvtColor(row2_roi, cv2.COLOR_BGR2GRAY)
    _, row2_thresh = cv2.threshold(row2_gray, 100, 255, cv2.THRESH_BINARY_INV)
    row2_contours, _ = cv2.findContours(row2_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out any contours that do not match the expected size and shape of the pink lines
    pink_lines = []
    for contour in row2_contours:  # Change row3_contours to row2_contours
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        if 0.5 < aspect_ratio < 2 and w > card_w * 0.05 and h > card_h * 0.04:
            pink_lines.append(contour)

    # Detect the white background in each column in row 3
    row3_thresh = row2_thresh  # Use row2_thresh instead of row3_thresh since row3_thresh is not defined
    row3_bg = cv2.bitwise_not(row3_thresh)
    row3_bg_contours, _ = cv2.findContours(row3_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Match the pink lines with their corresponding white background using the bounding boxes
    rows = []
    for i, pink_line in enumerate(pink_lines):
        pink_x, pink_y, pink_w, pink_h = cv2.boundingRect(pink_line)
        for white_bg in row3_bg_contours:
            white_x, white_y, white_w, white_h = cv2.boundingRect(white_bg)
            if abs(pink_x - white_x) < card_w * 0.05 and abs(pink_w - white_w) < card_w * 0.03:
                rows.append((pink_line, white_bg))
                break

    # Extract the text from each column and row
    results = {}
    for i, (pink_line, white_bg) in enumerate(rows):
        pink_x, pink_y, pink_w, pink_h = cv2.boundingRect(pink_line)
        cells = []
        cell_width = pink_w // cols
        cell_height = pink_h
        for j in range(cols):
            cell = image[pink_y:pink_y + cell_height, pink_x + j * cell_width:pink_x + (j + 1) * cell_width]
            cells.append(cell)
        results[i] = cells

    return cells
    return row1_text, results, color_confidence

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def find_largest_contour_and_corners(contours):
    largest_contour = max(contours, key=cv2.contourArea)
    corners = cv2.approxPolyDP(largest_contour, 0.02 * cv2.arcLength(largest_contour, True), True)

    return largest_contour, corners

def four_point_transform(image, points):
    points = order_points(points)
    (tl, tr, br, bl) = points

    # Calculate the width of the new image
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    # Calculate the height of the new image
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    # Create a new array of points for the transformed image
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Calculate the perspective transform matrix and warp the image
    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh_frame = cv2.threshold(gray_frame, 127, 255, cv2.THRESH_BINARY_INV)
    edged = cv2.Canny(thresh_frame, 50, 150)
    _, edged = cv2.threshold(edged, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = None
    corners = None

    if contours:
        largest_contour, corners = find_largest_contour_and_corners(contours)

    return largest_contour, corners

def draw_results(image, row1_text, results, color_confidence):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    # Draw the row1_text on the image
    cv2.putText(image, row1_text, (100, 50), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    # Draw the colored boxes and their text on the image
    for i, (color, confidence) in enumerate(color_confidence.items()):
        x, y, w, h = cv2.boundingRect(results[i])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        cv2.putText(image, f"{color}: {confidence:.2f}", (x, y - 10), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return image

def increase_contrast(frame, alpha=1.5, beta=30):
    return cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)

    # Define the region of interest (ROI) that covers the drug test card
    # You can adjust these values based on the size and position of the card in the camera's view
    x1, y1, x2, y2 = 200, 200, 500, 600
    roi = frame[y1:y2, x1:x2]

    # Increase the contrast of the ROI
    roi = increase_contrast(roi)

    # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to create a binary image
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Invert the binary image
    inverted_binary = cv2.bitwise_not(binary)

    # Apply edge detection to the inverted binary image
    edged = cv2.Canny(inverted_binary, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If a contour is found in the ROI, adjust its coordinates to the original frame
    if contours:
        contour = contours[0]
        x, y, w, h = cv2.boundingRect(contour)
        x += x1
        y += y1
        contour = np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)

    return contour

def capture_image():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        largest_contour, corners = preprocess_frame(frame)

        if largest_contour is not None and corners is not None and len(corners) == 4:
            # Draw the contour and corners on the frame
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow('Position the Card and Press Space to Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            if corners is not None and len(corners) == 4:
                corners = np.array([corner[0] for corner in corners], dtype="float32")
                break

    cap.release()
    cv2.destroyAllWindows()

    return frame, corners

def get_dominant_color(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = np.float32(image.reshape(-1, 3))

    n_colors = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.2)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]

    return dominant

def get_color_name(dominant_color):
    colors = {
        'red': [255, 0, 0],
        'green': [0, 255, 0],
        'blue': [0, 0, 255],
        # Add more colors if needed
    }

    color_name = 'unknown'
    min_distance = float('inf')

    for name, color in colors.items():
        distance = np.linalg.norm(np.array(dominant_color) - np.array(color))

        if distance < min_distance:
            min_distance = distance
            color_name = name

    return color_name

def correct_perspective(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        corners = cv2.approxPolyDP(largest_contour, epsilon, True)

        if len(corners) == 4:
            corners = np.array([corner[0] for corner in corners])
            return four_point_transform(image, corners)

    # If the perspective correction was not possible, return the original image
    return image

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Change the kernel size from (0, 0) to (5, 5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    print("Input image:", image)  # Add this line to print the input image
    image_np = np.array(image, dtype=np.uint8)
    processed_image = preprocess_image(image_np)

    # Set Tesseract configurations
    config = '--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    # Extract text from the image
    text = pytesseract.image_to_string(processed_image, config=config)

    return text.strip()

def analyze_colors(image, contours):
    color_confidence = {}

    for contour in contours:
        # Calculate the bounding rectangle for the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest (ROI) for the contour
        roi = image[y:y + h, x:x + w]

        # Calculate the dominant color in the ROI
        dominant_color = get_dominant_color(roi)

        # Get the color name
        color_name = get_color_name(dominant_color)

        # Calculate the confidence for the color
        confidence = cv2.contourArea(contour) / (w * h)

        # Store the confidence in the color_confidence dictionary
        color_confidence[color_name] = confidence

    return color_confidence

def detect_colored_areas(image, lower_color, upper_color):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color range for the filter
    lower_range = np.array(lower_color)
    upper_range = np.array(upper_color)

    # Create a mask using the specified color range
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # Apply the mask to the image
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    # Convert the filtered image to grayscale
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

    # Find contours in the grayscale image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def split_image_into_cells(image, rows, cols):
    cells = []
    h, w = image.shape
    cell_width = w // cols
    cell_height = h // rows

    for i in range(rows):
        row = []
        for j in range(cols):
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            print(f"Cell ({i}, {j}): {cell}")  # Add this print statement
            row.append(cell)
        cells.append(row)

    return cells


def analyze_colored_boxes(image):
    lower_red = [0, 50, 50]
    upper_red = [10, 255, 255]

    lower_blue = [110, 50, 50]
    upper_blue = [130, 255, 255]

    red_contours = detect_colored_areas(image, lower_red, upper_red)
    blue_contours = detect_colored_areas(image, lower_blue, upper_blue)

    red_confidence = analyze_colors(image, red_contours)
    blue_confidence = analyze_colors(image, blue_contours)

    color_confidence = {**red_confidence, **blue_confidence}

    # Combine the contours for red and blue
    all_contours = red_contours + blue_contours

    return color_confidence, all_contours

def analyze_test_windows(control_windows, test_windows):
    results = []
    for control_line, test_line in zip(control_windows, test_windows):

        # Measure the intensity of the control and test lines
        # You can use different methods to measure the intensity, e.g., color, brightness, etc.
        control_intensity = measure_line_intensity(control_line)
        test_intensity = measure_line_intensity(test_line)

        # Compare the intensities to determine if the test is positive or negative
        if test_intensity >= control_intensity:
            result = 'Negative'
        else:
            result = 'Positive'

        results.append(result)

    return results

def measure_line_intensity(line_img):
    # Placeholder function to measure the intensity of a line
    # You can implement your own method to measure the intensity
    intensity = np.mean(line_img)
    return intensity

def main():
    image = capture_image()

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Extract the text from the image
    text = extract_text(preprocessed_image)
    print("Extracted text:", text)

    # Analyze the colors in the image
    lower_color = [0, 100, 100]
    upper_color = [10, 255, 255]
    contours = detect_colored_areas(image, lower_color, upper_color)
    color_confidence = analyze_colors(image, contours)
    print("Color confidence:", color_confidence)

    # Split the image into cells
    rows = 3
    cols = 8
    cells = split_image_into_cells(preprocessed_image, rows, cols)

    # Process the cells
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            print(f"Processing cell ({i}, {j})")
            cell_result = process_frame(cell)
            print("Cell result:", cell_result)

if __name__ == "__main__":
    main()
    # Capture and process the image
    captured_image, corners = capture_image()
    corrected_image = four_point_transform(captured_image, corners)
    row1_text, results, color_confidence = process_frame(corrected_image)

    # Draw the results on the image and save it
    output_image = draw_results(corrected_image, row1_text, results, color_confidence)
    cv2.imwrite("output.png", output_image)

    # Display the output image
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
