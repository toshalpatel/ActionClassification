# CS5242-Breakfast
Breakfast actions classification. 

The goal of this test case is to classify each segment to any of the 47 sub-actions (excluding SIL). During the training stage, you can obtain the segments based on the labels. For the test stage, we give you the video segments. Your task is to experiment with different ways of segment representations and to perform action classification on top of these representations. You should submit your classification results for each segment in the test set in csv format.

## DATA

In this “data” folder you can find the I3D features that are computed for each frame. Note that this folder includes both the training and testing videos. 

data/xxxx.gz file: Each .gz file ( corresponds to a video training videos: P16-P54, test videos: P03-P15). Please extract all the .gz files to the data folder. Each feature matrix in the .gz file has a dimension of Ni∗D
N_i * D, where N_i is the total number of frames in the video i,D= 400 and is the dimension of the I3D features.

## Labels

In the “groundTruth” folder which can be found under the files, you can find the frame-wise sub-action labels for each video.
groundTruth/xxxx.txt: Each .txt file corresponds to the frame labels that have a dimension of N_i * 1, where Ni is the total number of frames in the video i. Please extract all the txt files to the groundTruth folder. “silence (SIL)” label corresponds to the background and only appears in the beginning frames and the end frames. In this project, you are asked to ignore these segments.

## SPLITS

In the “splits” folder which can be found under the files, you can find the dataset splits. The dataset is composed of four splits and you are asked to test your models on split1 and train them on the other splits. You can use the read_datasetBreakfast.py file to load the data by using these split bundle file. 

splits/mapping_bf.txt: a mapping file for the action numbers and the actual labels. \
splits/train.split1.bundle: list of the training videos. \
splits/test.split1.bundle: list of the test videos.\
Please extract all the files to the these folder.

### read_datasetBreakfast.py description

data_feat: a list variable that contains the features for all the training videos. Each item in this list is a tensor with a dimension of Ni * D. \
data_labels: a list variable that contains all of the training labels, each item in this variable is a list variable of length = the number of training segment for this video clip. \
You need to install the pytorch, numpy package to run this file and may need a few minutes to load the data.\

You can use the template_code.py to load data and write your own code.

## SUBMISSION

For the submission, you are required to process the test videos and predict a label for each action segment.

segment.txt: we provide the frame locations of each action segment. Each row corresponds to a text video in the order given in 'splits/test.split1.bundle'. For example, the first row includes '30 150 428 575 705'. As we are ignoring the SIL action, the first segment starts at frame 30 and ends at frame 150, similarly the second segment starts at 151 and ends at 428. There are four segments in this example and you are asked to predict a class label for each segment as follows 4, 1, 2, 3. You are asked to create a csv file and fill it with your prediction results in the order you make the predictions. You will submit this file. See the 'Evaluation' tab for the submission format.\
training_segment.txt is similar to the test_segment.txt, we provide it in order to reduce the difficulty of the project.

## Reference

Carreira J, Zisserman A. Quo vadis, action recognition? a new model and the kinetics dataset[C]//proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 6299-6308.

### competition link:
https://www.kaggle.com/c/cs5242project/overview
