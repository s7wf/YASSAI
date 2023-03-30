# YASSAI

This Python program uses OpenCV and Tesseract OCR to process drug test cards. It captures the card image, applies perspective transformation, extracts text from cells, and organizes data in a pandas DataFrame. A placeholder function for color analysis is provided for future implementation.

This code is a Python program that uses OpenCV and Tesseract OCR to process and analyze images of drug test cards. The program captures an image of the card from a live camera feed and automatically detects the card's position using edge detection and contour analysis. Once the card is detected and the user presses the spacebar, the program applies a perspective transformation to the image to obtain a top-down view of the card.

After preprocessing the captured image, the program splits the image into cells according to the number of rows and columns specified. It then uses Tesseract OCR to extract text from each cell and stores the extracted text in a nested list, which is then converted into a pandas DataFrame for better organization and display.

The current version of the program is focused on text extraction, but a placeholder function analyze_colors is provided for future implementation of color analysis for the colored blocks present on the drug test card.

This program can be used as a foundation for further development and customization to suit the specific requirements of drug test card analysis and interpretation.
