import cv2
import pytesseract
import pandas as pd
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

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

def capture_image():
    def increase_contrast(frame, alpha=1.2, beta=0):
        return cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)

    def preprocess_frame(frame):
        def increase_contrast(frame, alpha=1.5, beta=30):
            return cv2.addWeighted(frame, alpha, np.zeros(frame.shape, frame.dtype), 0, beta)

        frame = increase_contrast(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use adaptive thresholding to create a binary image
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Invert the binary image
        inverted_binary = cv2.bitwise_not(binary)

        # Apply edge detection
        edged = cv2.Canny(inverted_binary, 50, 150)
        return edged

    def find_largest_contour_and_corners(edged_frame):
        contours, _ = cv2.findContours(edged_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            corners = cv2.approxPolyDP(largest_contour, epsilon, True)
            return largest_contour, corners
        return None, None

    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        edged_frame = preprocess_frame(frame)
        largest_contour, corners = find_largest_contour_and_corners(edged_frame)

        if largest_contour is not None and corners is not None and len(corners) == 4:
            # Draw the contour and corners on the frame
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow('Position the Card and Press Space to Capture', frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            if corners is not None and len(corners) == 4:
                corners = np.array([corner[0] for corner in corners])
                frame = four_point_transform(frame, corners)
                cv2.imshow('Transformed Image', frame)  # Display the transformed image
                cv2.waitKey(2000)  # Wait for 2 seconds (2000 ms)
                cv2.destroyAllWindows()  # Close the window
            break

    cap.release()
    cv2.destroyAllWindows()
    return frame

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Change the kernel size from (0, 0) to (5, 5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    config = '--psm 6'
    text = pytesseract.image_to_string(image, config=config)
    return text

def analyze_colors(image):
    # This function should be implemented to analyze the colored blocks and provide confidence for each color.
    pass

def split_image_into_cells(image, rows, cols):
    cells = []
    h, w = image.shape
    cell_width = w // cols
    cell_height = h // rows

    for i in range(rows):
        row = []
        for j in range(cols):
            cell = image[i * cell_height:(i + 1) * cell_height, j * cell_width:(j + 1) * cell_width]
            row.append(cell)
        cells.append(row)

    return cells

def main():
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Replace with the correct path for your system

    image = capture_image()
    processed_image = process_image(image)

    # Split the image into cells
    rows, cols = 7, 2
    cells = split_image_into_cells(processed_image, rows, cols)

    # Extract text from each cell and store it in a nested list
    table_data = []
    for row in cells:
        table_row = []
        for cell in row:
            text = extract_text(cell)
            table_row.append(text)
        table_data.append(table_row)

    # Create a pandas DataFrame from the nested list
    table = pd.DataFrame(table_data)

    print(table)

if __name__ == '__main__':
    main()