import streamlit as st
import requests
import base64
from base64 import decodebytes
import io
from io import BytesIO
from patchify import patchify, unpatchify
from PIL import Image, ImageDraw, ImageFont
from PIL.ExifTags import TAGS
from PIL.ExifTags import GPSTAGS
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pydeck as pdk
import constants

# objects_no = 0
# confidences_all = []

API_KEY = constants.PUBLIC_API_KEY

def get_exif(image):
    #img = Image.open(image)
    # try:
    #     image.verify()
    # except:
    #     pass
    return image._getexif()

def get_geotagging(exif):
    if not exif:
        raise ValueError("No EXIF metadata found")
    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")
            for (key, val) in GPSTAGS.items():
                if key in exif[idx]:
                    geotagging[val] = exif[idx][key]
    return geotagging

# cut original image to smaller patches to be sent for inference
def patch_image(image):
    img = np.array(image) 
    # st.write(img.shape)
    patches = patchify(img, (432, 432, 3), step=432)
    #st.write("Patches: {}".format(patches.shape))
    return patches

# reconstruct the original image from individual patches
def unpatch_image(patches):
    # image_shape = (5184,3456,3)
    image_shape = (3456,5184,3)
    reconstructed_image = unpatchify(patches, image_shape)
    return reconstructed_image
    
def flatten(confidences_all):
    return [item for sublist in confidences_all for item in sublist]

@st.cache(show_spinner=False, suppress_st_warning=True)
def fetch_files(patches):
    start = time.time()
    objects_no = 0
    confidences_all = []
    results = []
    progress_bar = [*range(0,96,1)] # prepare the progress bar range. Must match the number of patches!!!
    percent_complete = iter(progress_bar) # set up as iterator
    my_bar = st.progress(0)
    with st.spinner('Wait for inference'):
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                single_patch_img = patches[i, j, 0, :, :, :]
                image = Image.fromarray(single_patch_img.astype('uint8'), 'RGB')

                # Convert to JPEG Buffer.
                buffered = io.BytesIO()
                image.save(buffered, quality=90, format='JPEG')

                # Base 64 encode.
                img_str = base64.b64encode(buffered.getvalue())
                img_str = img_str.decode('ascii')

                ## Construct the URL to retrieve JSON.
                ############# PUBLIC STICKY TRAPS DATASET ##################
                upload_url = ''.join([
                    "https://detect.roboflow.com/stickytraps/",
                    "2?api_key=" + API_KEY,
                    f'&overlap={overlap_threshold * 100}',
                    f'&confidence={confidence_threshold * 100}'
                ])

                ## POST to the API.
                r = requests.post(upload_url,
                                  data=img_str,
                                  headers={
                    'Content-Type': 'application/x-www-form-urlencoded'
                })

                ## Save the JSON.
                output_dict = r.json()

                ## Generate list of confidences.
                confidences = [box['confidence'] for box in output_dict['predictions']]
                # classes = [box['class'] for box in output_dict['predictions']]

                ######### DRAW BOXES AROUND DETECTED OBJECTS ##############
                preds = r.json()
                detections = preds['predictions']
                results.append(preds)
                objects_no += len(detections)

                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()

                for box in detections:
                    color = "#05ff05"
                    x1 = box['x'] - box['width'] / 2
                    x2 = box['x'] + box['width'] / 2
                    y1 = box['y'] - box['height'] / 2
                    y2 = box['y'] + box['height'] / 2

                    draw.rectangle([
                        x1, y1, x2, y2
                    ], outline=color, width=4)

                    if True:
                        text = box['class']
                        text_size = font.getsize(text)

                        # set button size + 10px margins
                        button_size = (text_size[0]+20, text_size[1]+20)
                        button_img = Image.new('RGBA', button_size, color)
                        # put text on button with 10px margins
                        button_draw = ImageDraw.Draw(button_img)
                        button_draw.text((10, 10), text, font=font, fill=(255,255,255,255))

                        # put button on source image in position (0, 0)
                        image.paste(button_img, (int(x1), int(y1-30)))

                img = np.array(image)
                patches[i, j, 0, :, :, :] = img
                confidences_all.append(confidences)

                percent = next(percent_complete)
                if percent > 100:
                    percent = 100
                my_bar.progress(percent)
    # with st.expander("Raw inference results"):
    #     st.write(classes)
    end = time.time()
    st.success(f'Done in : {round(end-start)} seconds')
    return patches, objects_no, confidences_all




####################### REMOVE THE HAMBURGER ####################
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
#st.markdown(hide_st_style, unsafe_allow_html=True)

############################# Sidebar ###########################
# Add in location to select image.
st.sidebar.write('#### Select an image to upload.')
uploaded_file = st.sidebar.file_uploader('',
                                         type=['png', 'jpg', 'jpeg'],
                                         accept_multiple_files=False)

st.sidebar.write("")
st.sidebar.write('[Find the public dataset on Roboflow.](https://app.roboflow.com/public-yupt3/stickytraps/overview)')
st.sidebar.write("")

## Add in sliders.
confidence_threshold = st.sidebar.slider('Confidence threshold: What is the minimum acceptable confidence level for displaying a bounding box?', 0.0, 1.0, 0.5, 0.01)
st.sidebar.write("")
overlap_threshold = st.sidebar.slider('Overlap threshold: What is the maximum amount of overlap permitted between visible bounding boxes?', 0.0, 1.0, 0.5, 0.01)

st.sidebar.write("")
image = Image.open('./images/agrilogic.png')
st.sidebar.image(image, use_column_width=True)
#st.sidebar.write("")

#st.sidebar.markdown("Social links")
link = '[GitHub](https://github.com/ioanfesteu/Stickytraps)'
# st.sidebar.markdown("<h1 style='text-align: center; color: red;'> link </h1>", unsafe_allow_html=True)
st.sidebar.markdown(link, unsafe_allow_html=True)


############################# Main app ###########################
## Title.
st.write('# Whiteflies Counter ')

## Pull in default image or user-selected image.
if uploaded_file is None:
    # Default image.
    image = Image.open('./samples/1236.jpg')
   
else:
    # User-selected image.
    image = Image.open(uploaded_file)

slot1 = st.empty()
slot1.image(image)
# slot1.image(image, use_column_width=True)

exif = get_exif(image)
geotags = get_geotagging(exif)

# Patch the image and print the shape of the corresponding array
patches = patch_image(image)

# Send the patches for inference
patches, objects_no, confidences_all = fetch_files(patches)

# Reconstruct the image from marked patches
reconstructed_image = unpatch_image(patches)
slot1.image(reconstructed_image, use_column_width=True)

image = Image.fromarray(reconstructed_image.astype('uint8'), 'RGB')
image.save("./detected/detected.jpg")

# Download detected image with boxes
with open("./detected/detected.jpg", "rb") as file:
    btn = st.download_button(
        label="Download",
        data=file,
        file_name="detected.jpg",
        mime="image/jpeg"
    )

#st.write(f"Inference time: {round(end-start)} seconds")
st.write(f"### Number of bugs detected: {objects_no}")
# st.write(f"### Bugs types: {objects_no}")
#unpatch_image(patches)

## Histogram in main app.
with st.expander("Histogram of Confidence Levels"):
    # st.write('### Histogram of Confidence Levels')
    fig, ax = plt.subplots()
    ax.hist(flatten(confidences_all), bins=10, range=(0.0,1.0))
    st.pyplot(fig)

# # Display the image geaotagg in main app.
lat = []
lon = []
# st.write(geotags)
try:
    for values in geotags["GPSLatitude"]:
        lat.append(values)
    for values in geotags["GPSLongitude"]:
        lon.append(values)

    latitude = round(lat[0] + lat[1]/60.0 + lat[2]/3600.0, 6)
    longitude = round(lon[0] + lon[1]/60.0 + lon[2]/3600.0, 6)


    with st.expander("Geotagged image"):
        # df = pd.DataFrame(
        # np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
        # columns=['lat', 'lon'])
        df = pd.DataFrame(
                [[latitude, longitude]],
                columns=['lat', 'lon'])

        st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=pdk.ViewState(
            latitude=latitude,
            longitude=longitude,
            zoom=16,
            pitch=0,
            ),
        layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=5,
            ),
        ],
        ))
except:
    with st.expander("Geotagged image"):
        st.write("No GPS data available")