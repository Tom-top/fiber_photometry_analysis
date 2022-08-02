# Fiber Photometry Analysis

Python toolkit for the analysis of fiber photometry data and related video recordings.

## Installation

### Downloading the code

* Method 1 : Installing this toolkit by cloning it with Git :

	For this, run the following commands :

	```
	cd folder/to/clone/into
	git clone https://github.com/Tom-top/Fiber_Photometry_Analysis
	```

	If you don't have Git on your computer you can install it following this [link](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

* Method 2 : Installing by manually downloading the toolkit :

	You can manually download the source code from this [link](https://github.com/Tom-top/Fiber_Photometry_Analysis).

### Installing the requiered Python packages

Once the code is on your computer, the easiest way to get this toolkit running is to install [Anaconda](https://www.anaconda.com/distribution/).

To install all the pre-requiered libraries for the code to run there are two main options :

* Cloning the provided virtual environment :

	For this, find the _Environement_File.yml_ file and run this in a terminal :

  	```
  	conda env create --n envname -f path/to/the/Environment_File.yml
	```

* Installing all the pre-requiered libraries manually :

  	I suggest running this toolkit in an integrated development environement. I personnaly use [Spyder](https://www.spyder-ide.org/). You can install Spyder and all the other pre-requiered libraries using Anaconda very easily by executing the following commands :

	```
	conda install spyder
	conda install pandas
	conda install xlrd
	conda install matplotlib
	conda install seaborn
	conda install numpy
	conda install -c conda-forge moviepy
	conda install -c anaconda scipy
	conda install -c anaconda scikit-learn 
	```

## Testing the pipeline

### Downloading the test dataset

Some test data is available at this [link](https://www.dropbox.com/sh/9h9albv5oqoueg8/AAAXPXlLm6Uy8UvHY5enZxaEa?dl=0). 
Once downloaded, place the folder in the path/to/the/fiber_photometry_analysis folder.

### Running the tests

If you are using a virtual environement, make sure to activate it, by running 

```
conda activate name_of_virtual_environement
```

Launch Spyder and open the _Pipeline.py_ file.

The first step is to change the photometry_lib variable and try to run the first section of the code with all the imports (using Cntrl+Enter).

```
photometry_lib = path/to/the/fiber_photometry_analysis/folder.
```

If there is no problem, you are all set, and ready to analyze some photometry data!

The pipeline is separated into three major sections :

#### The first one allows you to pre-process the raw photometry data to get the dF/F or (Z-Score) dF/F of the photometry recording.
#### The second one allows you to pre-process the behavior data, merging small behavioral bouts or filtering small artefactual bouts for instance.
#### The third one allows you to align the pre-processed photometry data to the behavioral data and generate a variety of useful plots and video clips.

* Description of the second section of the code : Setting the working directory and the related files

	In this section the user has to change the _experiment_ and the _mouse_ variables (if not running the test, otherwise leave intact). All the files useful for the pipeline will bare the name "*_experiment_mouse.*".
	The expected files for the pipeline are :

	- "behavior_manual_experiment_mouse.ext" (csv, xlsx extensions allowed)
	- "photometry_experiment_mouse.ext" (csv, xlsx extensions allowed)
	- "video_experiment_mouse.ext" (avi, mp4 extensions allowed)

	By running this section, all the main path variables will be set.

* Description of the third section of the code : Loading all the parameters

	Running this section will load all the main analysis parameters.

* Description of the third section of the code : Conversion of the photometry data into a numpy format

	Running this section will convert the "photometry_experiment_mouse.ext" to "photometry_experiment_mouse.npy" format.

* Description of the fourth section of the code : Preprocessing of the photometry data

	Running this section will run the full pre-processing pipeline on the photometry data and generate all the related plots.
	This part was heavily inspired by the code from [this paper](https://www.jove.com/t/60278/multi-fiber-photometry-to-record-neural-activity-freely-moving), with the code from Ekaterina Martianova available [here](https://github.com/katemartian/Photometry_data_processing).

	Briefly, after loading the raw data, the data is smoothed and an estimation of the baseline is computed for each signal (calcium independant (isosbestic) and calcium dependant). The baseline is then substracted from the signal to correct for signal decay and bleaching (baseline correction). The corrected signals are then normalized using standardization ((signal - mean) / standard dev) and a regression is performed between the two signals to align them. Finally, the two signals are substracted to obtain the zdF/F (or the dF/F if the standardization step was not performed).

	The result should look like this :

	<a href="url"><img src="https://github.com/Tom-top/Fiber_Photometry_Analysis/blob/master/Images/dFF.png" align="center"></a>

	To get a more detailed step by step explanation please open the Jupyter notebook : _Pipeline_Jupyter.ipynb_

* Description of the fifth section of the code : Reading the Video data

	Running this section will load the video clip and display the first frame (check).

* Description of the sixth section of the code : Cropping the video field of view (for future display)

	Running this section will crop the video clip and display the first frame (check). This step is for pure aesthetic reasons (for future video plots).

* Description of the seventh section of the code : Importing behavioral data

	Running this section will load the behavioral data corresponding to _args["behavior_to_segment"]_ from the "behavior_manual_experiment_mouse.ext" file, and plot the pre-processed photometry data with overlayed behavioral data.

	The result should look like this :

	<a href="url"><img src="https://github.com/Tom-top/Fiber_Photometry_Analysis/blob/master/Images/dF_&_raw_behavior.png" align="center"></a>

* Description of the eigth section of the code : Merge behavioral bouts that are close together

	The algorithm will merge any behavioral bouts that are closer than the _args["peak_merging_distance"]_ distance, and plot the pre-processed photometry data with overlayed filtered behavioral data.

	The result should look like this :

	<a href="url"><img src="https://github.com/Tom-top/Fiber_Photometry_Analysis/blob/master/Images/dF_&_merged_behavior.png" align="center"></a>

* Description of the ninth section of the code : Detect major behavioral bouts based on size

	The algorithm will exclude any behavioral bouts that is smaller than the _args["minimal_bout_length"]_ length, and plot the pre-processed photometry data with overlayed filtered behavioral data.

	The result should look like this :

	<a href="url"><img src="https://github.com/Tom-top/Fiber_Photometry_Analysis/blob/master/Images/dF_&_filtered_behavior.png" align="center"></a>

* Description of the tenth section of the code : Extract photometry data around major bouts of behavior

	This section will extract the pre-processed photometry data from the filetered behavioral bouts to generate a peri-event plot.

	The result should look like this :

	<a href="url"><img src="https://github.com/Tom-top/Fiber_Photometry_Analysis/blob/master/Images/Peri_Event_Plot_Individual.png" align="center"></a>

* Description of the eleventh section of the code : Generate bar plot for the peri-event data

	This section will use the peri-event data and compute the Area Under the Curve (AUC) before and after the initiation of the selected behavior to analyze.
	The results are displayed in the form of a bar plot.

	The result should look like this :

	<a href="url"><img src="https://github.com/Tom-top/Fiber_Photometry_Analysis/blob/master/Images/AUC.png" align="center" width="490" height="490"></a>

* Description of the twelfth section of the code : Creates a video with aligned photometry and behavior

	This section creates a video clip with aligned behavioral and photometry data along side the recording of the animal.
	Here, the optic fiber is implanted on top of the barrel cortex. The mouse was previously injected with an AAV expressing GCaMP7s, and the segmented behavior is whisking.

	The result should look like this :

	![](https://github.com/Tom-top/Fiber_Photometry_Analysis/blob/master/Images/Live_Video_Plot.gif)

## Future

In the close future, will be added to this module :

- A code for semi-automated segmentation of behavior (by pressing different keys of the computer), and saving the results in a file directly readable by this code.
- A code for close-loop recording of behavior and photometry.
- Interfacing the results given by the previously written [Mouse Tracker](https://github.com/Tom-top/TopMouseTracker) with this module.

And many more...

## Authors & Contact

* **Thomas TOPILKO**
thomas.topilko@gmail.com
* **Charly ROUSSEAU**
* **Grace HOUSER**
grace.anne.houser@gmail.com

## License

N.A
