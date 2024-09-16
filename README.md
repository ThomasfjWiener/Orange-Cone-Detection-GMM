# Orange-Cone-Detection-GMM
Development of Gaussian Mixture Model from scratch from numpy, trained using Baum-Welch to efficiently detect orange cones in images

## Instructions for running test script
The best performing model (specifically, its lookup table for prediction) is stored in `lookup_table_12_components.npy` and the results are output to `test_12_components_GMM_output.txt`. <br>

To run the test script for inference, cd to `Project1_submission` and then run `python test_script.py`. <br>
- HOWEVER, BEFORE RUNNING the script, please set the `folder` variable on line 14 in test_script.py equal to the path to the directory of test images. <br>
- I have included cv2 images that are shown as the script is run to demonstrate the processing steps. Just press any key (space bar for example) to continue the script <br>
- the final bounding box images are placed in the `Output_images` directory after running `test_script.py` <br>
- I have saved THREE models/learning algorithms for testing. *The First Two were submitted in the original Code submission on Thursday, and the Third is my final submission for the updated code for the writeup*: <br>
&nbsp; The first one uses the built-in Gaussian Mixture model code from sklearn.mixture. I have it to verify the correctness and efficiency of the rest of my project pipeline (e.g. hand-labeling, shape-statistics, test speed, etc.). When running test_script.py, the outputs of this model will be written to `test_builtinGMM_output.txt`. *NOTE: Running this model will likely no longer work, and therefore has been commented out* <br>
&nbsp; The second model is my own naive implementation of the Gaussian Mixture Model that uses 1 gaussian component for the orange cone class. When running test_script.py, the outputs of this model will be written to `test_naive_implementation_output.txt`. *NOTE: Running this model will likely no longer work, and therefore has been commented out* <br>
&nbsp; The THIRD AND FINAL model is my best performing model. When running test_script.py, the outputs of this model will be written to `test_12_components_GMM_output.txt`.

<br>

