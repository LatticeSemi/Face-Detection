1. Training Environment Setup Using Docker
------------------------------------------
Please refer "installdocker.txt" to install docker based on operating system.
To build docker image: (This will take around 15-20 minutes)
  For Linux:
  ----------
     - Open terminal
     - cd facedet_training
     - sudo docker build -t face_det .
  For Windows:
  ------------
     - Open Docker Quickstart terminal
     - cd facedet_training
     - docker build -t face_det .


2. Dataset format
-----------------
The dataset should be in the following format:
  1. Training Images:
	-- <dataset_dir>/train/face/*.(jpg/png) - Positive Images
	-- <dataset_dir>/train/none/*.(jpg/png) - Negative Images
  2. Validation Images:
	-- <dataset_dir>/val/face/*.(jpg/png) - Positive Images
	-- <dataset_dir>/val/none/*.(jpg/png) - Negative Images
  3. Test Images:
	-- <dataset_dir>/test/face/*.(jpg/png) - Positive Images
	-- <dataset_dir>/test/none/*.(jpg/png) - Negative Images
  4. Label:
	-- <dataset_dir>/label.txt


3. Shared directory between docker container and host
-----------------------------------------------------
The shared folder should contain:
  1. dataset directory in above mentioned format
  2. train_logs directory to store training logs and final .pb file
  
4. Training
-----------
1. Run docker image and select the dataset path
Follow the main step 1 to build the docker image
	For Linux:
	----------
	  - $cd facedet_training
	  - $python input.py
	  	arguments:
		   - shared folder(mentioned in above step)
		   - Docker image name(face_det) that has been built
	  - Docker shell will open from the script.
	For Windows:
	------------
	  - Open the Docker Quickstart terminal.
		- Make sure that Xlaunch to use display in docker. If it is not installed, please refer "installdocker.txt"
          - $cd facedet_training
	  - $export DISPLAY=<Machine IP address>:0.0 #Same as mentioned in link: https://dev.to/darksmile92/run-gui-app-in-linux-docker-container-on-windows-host-4kde 
          - $docker run --rm -it -p 8888:8888 -p 6006:6006 -v <path to shared folder>:<path to shared folder> -e DISPLAY=$DISPLAY --net=host face_det

2. There are two ways one can trigger training:
    1. Using Jupyter notebook GUI
	- To use GUI based training run command prompted in the docker shell to lauch jupyter notebook
	- For linux, jupyter notebook will run at "http://localhost:8888" and tensorboard will run at "http://localhost:6006" in the host browser.
        - For windows, jupyter notebook will run at "http://192.168.99.100:8888" and tensorboard will run at "http://192.168.99.100:6006" in the host browser. 192.168.99.100 is docker-machine default ip. Use command "docker-machine ip default" to get it.
	- Token id should be given which is available in the jupyter notebook link.
	- Open train.ipynb file
	- After notebook is opened, click "Kernel-> Restart & Run All" option from menubar to load the GUI
	- The final .pb is copied to shared_folder/train_logs to access it from host machine. After that it can be used in SensAI tool.

    2. Using Docker shell
	- Copy all the folders from <dataset_dir> to ./Data
        - $python build_image_data.py       :create train-00000-of-00001 and validation-00000-of-00001 inside Data\100\tfrecord\train\)
	- $python facedet_train.py          :create checkpoint inside TrainLog folder and save best checkpoint in TrainLog\best folder
	- $python trainckpt2inferencepb.py  :create model.ckpt-#####_frozenforInference.pb in TrainLog
	- Copy .pb to shared folder to access it from host

4. Hint and troubleshotting:
----------------------------
 - If have access error during training, give permissions to the shared folder to resolve the issue.
