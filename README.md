# Data preparation
The aim of data preparation is to generate the training and testing datasets in the form of input and target data pairs. The input data here is the raw image after a set of simple preprocessing, including black level substraction and demosacing. While the target here is the neutral color of the image, which is extracted using a color checker. The data preparation process involves the following steps:
1. Capture real world scenes with different lighting conditions and backgrounds, `take the image without color checker first, follwed by the one with color checker`.
2. Dump the raw image to a folder (by the software or by hand), `e.g. raw_images`. Then run 
`
python create_dataset.py
`.
It will automatically split the scenes and color checkers into different folders with some preprocessing.
3. Run the modified `labelme` tool with command `python -m labelme` to extract the color checker from the image. Note that only four corners are required to be clicked. The labeling order should be in clockwise direction, `ending with the most white patch`.
4. Once all color checker are extracted, run `extract_color_checker.py` script, and it will automatically extract the neutral color of each image.

# Training