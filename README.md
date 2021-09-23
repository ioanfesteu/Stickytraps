# Stickytraps
## Streamlit webapp for counting and monitoring greenhouse whiteflies
This web application counts the number of **whiteflies** (Trialeurodes vaporariorum) sticked on yellow traps allong with their predators **Macrolophus pygmaeus**

Dataset used to train the model: https://www.kaggle.com/friso1987/yellow-sticky-traps

Inspired from https://github.com/roboflow-ai/streamlit-bccd 

### How to use:
  * Create and activate a venv: python3 -m venv venv and source venv/bin/activate
  * Install requirements: pip3 install -r requirements.txt
  * Run streamlit from app folder: streamlit run kernels_app.py
  
Sample images for inferencing can be found in "samples" folder.

![Detected soybeans in image](/detected/detected_small.jpg)
