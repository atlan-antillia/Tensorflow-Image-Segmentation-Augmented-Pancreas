<h2>Tensorflow-Image-Segmentation-Cropped-Pancreas (Updated:2025/04/17)</h2>

Toshiyuki A. Arai<br>
Software Laboratory antillia.com<br><br>

<li>2025/04/17: Updated to use the latest Tensorflow-Image-Segmentation-API</li>
<br>

This is the second experiment of Image Segmentation for Pancreas 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and <a href="https://drive.google.com/file/d/141HL4nRT8LZoqtxk2un6cM-LDuZa7jtO/view?usp=sharing">
Cropped-Pancreas-ImageMask-Dataset</a>, which was derived by us from the original dataset
<a href="https://www.kaggle.com/datasets/salihayesilyurt/pancreas-ct">
<b>Pancreas-CT</b></a>
<br>
<br>
Please see also our first experiment <a href="https://github.com/atlan-antillia/Tensorflow-Image-Segmentation-Pancreas">
Tensorflow-Image-Segmentation-Pancreas
</a>
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0001-1110.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0001-1110.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0001-1110.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0002-cropped-0-1103.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0002-cropped-0-1103.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0002-cropped-0-1103.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0002-cropped-1-1096.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0002-cropped-1-1096.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0002-cropped-1-1096.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this PancreasSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>

The original image dataset used here has been taken from the following kaggle.com web site.<br>
<a href="https://www.kaggle.com/datasets/salihayesilyurt/pancreas-ct">
Pancreas-CT</a><br>
Created by Sean Berryman, last modified by Tracy Nolan on Sep 16, 2020<br>
<br>
<b>About Dataset</b><br>
Summary:<br>

The National Institutes of Health Clinical Center performed 82 abdominal contrast enhanced 3D CT <br>
scans (~70 seconds after intravenous contrast injection in portal-venous) from 53 male and 27 <br>
female subjects. Seventeen of the subjects are healthy kidney donors scanned prior to nephrectomy.<br> 
The remaining 65 PANCREAS_s were selected by a radiologist from PANCREAS_s who neither had major <br>
abdominal pathologies nor pancreatic cancer lesions. Subjects' ages range from 18 to 76 years with <br>
a mean age of 46.8 ± 16.7. The CT scans have resolutions of 512x512 pixels with varying pixel sizes <br>
and slice thickness between 1.5 − 2.5 mm, acquired on Philips and Siemens MDCT scanners <br>
(120 kVp tube voltage).<br>
<br>
A medical student manually performed slice-by-slice segmentations of the pancreas as ground-truth <br>
and these were verified/modified by an experienced radiologist.<br>

Reference: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/LICENSE">LICENSE</a>

<br>
<br>
<h3>
<a id="2">
2 Cropped-Pancreas ImageMask Dataset
</a>
</h3>
 If you would like to train this Pancreas Segmentation model by yourself,
 please download our 512x512 pixels dataset from the google drive  
<a href="https://drive.google.com/file/d/141HL4nRT8LZoqtxk2un6cM-LDuZa7jtO/view?usp=sharing">
Cropped-Pancreas-ImageMask-Dataset</a>
</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Cropped-Pancreas
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
On detail of this Cropped-Pancreas dataset, please refer to our repository:
<a href="https://github.com/atlan-antillia/Pancreas-ImageMask-Dataset">Pancreas-ImageMask-Dataset</a><br>

As mentioned in <a href="https://github.com/atlan-antillia/Pancreas-ImageMask-Dataset">Pancreas-ImageMask-Dataset</a>, 
by applying two types of center cropping operations to the original images and masks files, the number of those files in this dataset has increased 
three-fold from the previous Non-Cropped Pancreas dataset.
However, simply increasing the number of image and mask files does not necessarily lead to direct improvement in the segmentation accuracy.
<br>
<br>
<b>Cropped-Pancreas Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/Cropped-Pancreas_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough to use for a training set of our segmentation model.

<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We trained Pancreas TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (98,99,100)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was terminated at epoch 100.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/train_console_output_at_epoch_100.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/eval/train_losses.png" width="520" height="auto"><br>
<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Pancreas.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/evaluate_console_output_at_epoch_100.png" width="720" height="auto">
<br><br>Image-Segmentation-Pancreas

<a href="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/evaluation.csv">evaluation.csv</a><br>
The loss (bce_dice_loss) to this Pancreas/test was low, but dice_coef not so high as shown below.
<br>
<pre>
loss,0.0875
dice_coef,0.8383
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Pancreas.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/asset/mini_test_output.png" width="1024" height="auto"><br>

<br>
<hr>
<b>Enlarged images and masks (512x512pixels)</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0001-1132.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0001-1132.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0001-1132.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0001-cropped-0-1117.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0001-cropped-0-1117.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0001-cropped-0-1117.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0001-cropped-0-1162.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0001-cropped-0-1162.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0001-cropped-0-1162.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0002-cropped-0-1103.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0002-cropped-0-1103.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0002-cropped-0-1103.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0002-cropped-1-1081.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0002-cropped-1-1081.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0002-cropped-1-1081.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/images/PANCREAS_0002-cropped-1-1096.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test/masks/PANCREAS_0002-cropped-1-1096.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Cropped-Pancreas/mini_test_output/PANCREAS_0002-cropped-1-1096.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Accurate pancreas segmentation using multi-level pyramidal pooling residual U-Net with adversarial mechanism</b><br>
Li, M., Lian, F., Wang, C. et al. <br>
BMC Med Imaging 21, 168 (2021). https://doi.org/10.1186/s12880-021-00694-1<br>
<pre>
https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-021-00694-1
</pre>

<b>2. Automated pancreas segmentation and volumetry using deep neural network on computed tomography</b><br>
Sang-Heon Lim, Young Jae Kim, Yeon-Ho Park, Doojin Kim, Kwang Gi Kim & Doo-Ho Lee<br>
Sci Rep 12, 4075 (2022). https://doi.org/10.1038/s41598-022-07848-3<br>
<pre>
https://www.nature.com/articles/s41598-022-07848-3#Sec11
</pre>

