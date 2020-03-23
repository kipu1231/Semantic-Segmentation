# Semantic-Segmentation
Implementation of two different models to achieve semantic segmentation on Image Data using Pytorch. The first model serves as a baseline for the development of the advanced model. The baseline model achieved a meanIoU of 0.6586 and the advanced model achieved a meanIoU of 0.6814.

The project was part of the 'Deep Learning for Computer Vision' lecture at National Taiwan University.

### Results
![alt text](https://github.com/kipu1231/Semantic-Segmentation/blob/master/Results/Comparison.png)

# Usage

### Dataset
In order to download the used dataset, a shell script is provided and can be used by the following command.

    bash ./get_dataset.sh
    
The shell script will automatically download the dataset and store the data in a folder called `semseg_data`. 

### Packages
The project is done with python3.6. For used packages, please refer to the requirments.txt for more details. All packages can be installed with the following command.

    pip3 install -r requirements.txt
    
### Training
The models can be trained using the following command. To distinguish the training of the baseline and the advanced model, the respective model needs to be initialised (baseline: model.Net(args) / advanced: model_best.Net(args)).

    bash train.sh semseg_data

### Segmentation
To perform segmantic segmantation with the trained models, the provided evaluation script can be run by using the following command for first the baseline and second the advanced model.

    bash test_model_baseline.sh semseg_data/val/img PredictionDir
    bash test_model_best.sh semseg_data/val/img PredictionDir

### Evaluation
To evaluate the models, you can run the provided evaluation script by using the following command.

    python3 mean_iou_evaluate.py <--pred PredictionDir> <--labels GroundTruthDir>

 - `<PredictionDir>` should be the directory to your predicted semantic segmentation map 
 - `<GroundTruthDir>` should be the directory of ground truth 

### Visualization
To visualization the ground truth or predicted semantic segmentation map in an image, the provided visualization script can be run by using the following command.

    python3 viz_mask.py <--img_path xxxx.png> <--seg_path xxxx.png>
