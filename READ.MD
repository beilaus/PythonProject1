# HOW TO RUN

Use python 3.8. 

```commandline
pip3 install requirements.txt

python3 main.py
```

-results will be saved to file
-download output_video.mp4 to view most recent result uploaded (conf>0.6)


# Work and approach:
-Used windows, researched for a viable dataset of cones, installed all the required applications for the suitable environment. Proceeded to install the ai model.
-Trained the ai through a sophisticated guide on Colab, imported the weights result into the project and added a command block to process video files.
-Altered the data to exclude confidence scales below 0.6 (can be changed through main.py)

