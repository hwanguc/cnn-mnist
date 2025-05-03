# ---------------------------------------------
# Streamlit front-end for MNIST digit recognition
# ---------------------------------------------

# Author: Han Wang
# Date: 2025-05-01
# Description: This script is a Streamlit application that allows users to draw a digit on a canvas, predict the digit using the pre-trained CNN model, and log the results to a CSV file.


from streamlit_drawable_canvas import st_canvas
import os

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2
import torch.nn as nn
import torchvision

# Set paths:

img_save_path = os.path.join(os.getcwd(), 'img_save')
os.makedirs(img_save_path, exist_ok=True)


# Model definition:
def cnn_model():
    return nn.Sequential(
        nn.ZeroPad2d(2),
        nn.Conv2d(1, 16, 5, 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.LazyLinear(10),
        nn.Softmax(dim=1)           # you used Softmax when training
    )

st.write('# MNIST Digit Recogniser')
st.write('## Using a `PyTorch` CNN')

Network = cnn_model()                                            # build layers
state_dict = torch.load('./checkpoint/250501_mnist_cnn_pt1_cpu.pt',
                        map_location="cpu")                      # load weights
Network.load_state_dict(state_dict)
Network.eval()      

st.write('### Draw a digit in 0-9 in the box below')

# Canvas:
## Specify canvas parameters in application
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 9)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

## Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color='#FFFFFF',
    background_color='#000000',
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=200,
    width=200,
    drawing_mode='freedraw',
    key="canvas",
)

# Record the hand drawn digit:
if canvas_result.image_data is not None:
    
    
    ## Get the numpy array (4-channel RGBA 100,100,4)
    input_numpy_array = np.array(canvas_result.image_data)
    
    
    ## Get the RGBA PIL image
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    input_image.save(os.path.join(img_save_path, "user_input.png"))
    
    ## Convert it to grayscale
    input_image_gs = input_image.convert('L')
    input_image_gs_np = np.array(input_image_gs) 

    ## Create a temporary image for opencv to read it
    input_image_gs.save(os.path.join(img_save_path, "temp_for_cv2.jpg"))
    image = cv2.imread(os.path.join(img_save_path, "temp_for_cv2.jpg"), 0)
    
    ## Start creating a bounding box
    height, width = image.shape
    x,y,w,h = cv2.boundingRect(image)

    ## Create new blank image and shift ROI to new coordinates
    ROI = image[y:y+h, x:x+w]
    mask = np.zeros([ROI.shape[0]+10,ROI.shape[1]+10])
    width, height = mask.shape
    x = width//2 - ROI.shape[0]//2 
    y = height//2 - ROI.shape[1]//2 
    mask[y:y+h, x:x+w] = ROI
    
    ## Check if centering/masking was successful
    output_image = Image.fromarray(mask) # mask has values in [0-255] as expected
    ### Now we need to resize, but it causes problems with default arguments as it changes the range of pixel values to be negative or positive
    ### compressed_output_image = output_image.resize((22,22))
    ### Therefore, we use the following:
    compressed_output_image = output_image.resize((22,22), Image.BILINEAR) # PIL.Image.NEAREST or PIL.Image.BILINEAR also performs good

    convert_tensor = torchvision.transforms.ToTensor()
    tensor_image = convert_tensor(compressed_output_image)
    ### Another problem we face is that in the above ToTensor() command, we should have gotten a normalized tensor with pixel values in [0,1]
    ### Here, we need to normalize manually.
    tensor_image = tensor_image/255.
    
    ## Padding
    tensor_image = torch.nn.functional.pad(tensor_image, (3,3,3,3), "constant", 0)
    ## Normalization shoudl be done after padding i guess
    convert_tensor = torchvision.transforms.Normalize((0.1307), (0.3081)) # Mean and std of MNIST
    tensor_image = convert_tensor(tensor_image)
    ### Shape of tensor image is (1,28,28)
    

    ## Save the processed image:
    plt.imsave(os.path.join(img_save_path, "processed_tensor.png"),tensor_image.detach().cpu().numpy().reshape(28,28), cmap='gray')


    ## Compute the predictions
    device='cpu'
    with torch.no_grad():
        # input image for network should be (1,1,28,28)
        output0 = Network(torch.unsqueeze(tensor_image, dim=0).to(device=device))
        # Need to apply Softmax here to get probabilities
        m = torch.nn.Softmax(dim=1)
        output0 = m(output0)
        # st.write(output0)
        certainty, output = torch.max(output0[0], 0)
        certainty = certainty.clone().cpu().item()
        output = output.clone().cpu().item()    
        certainty1, output1 = torch.topk(output0[0],3)
        certainty1 = certainty1.clone().cpu()#.item()
        output1 = output1.clone().cpu()#.item()
#     print(certainty)
    st.write('### Prediction') 
    st.write('### '+str(output))

    st.write('### Certainty')    
    st.write('#### '+str(round(certainty1[0].item()*100,2)) +'%')
    st.write('### Top 3 candidates')
    top3_100 = ', '.join(f'{round(x, 3)}' for x in output1.tolist())
    st.write('#### '+str(top3_100))
    st.write('### Certainties')
    certainty1_100 = ', '.join(f'{round(x * 100, 1)}%' for x in certainty1.tolist())
    st.write('#### '+str(certainty1_100))

    ## Record the click to submit the ground truth to both the postgresql database and a local csv backup.

    ### a helper func to connect to sql server:
    import datetime, csv, pathlib, pandas as pd, altair as alt, psycopg2
    from dotenv import load_dotenv
    load_dotenv()                                     # makes .env vars visible

    @st.cache_resource
    def get_conn():
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "mnist"),
            user=os.getenv("DB_USER", "postgresmnist"),
            password=os.getenv("DB_PASS", "evbxg361"),
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS inference_log (
                    id SERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ,
                    pred INT,
                    truth INT,
                    confidence REAL
                );
            """)
        return conn

    conn = get_conn()

    ### write the user-input ground truth to the server and to csv.
    st.divider()

    st.markdown(
        """
        <style>
            .record-label {
                font-size: 28px !important;
                font-weight: 600;          
                margin-bottom: 0.25rem;
                display: block;
            }
        </style>

        <label class="record-label">Record correct digit</label>
        """,
        unsafe_allow_html=True,
    )

    true_digit = st.number_input(label="",
                                min_value=0, max_value=9, step=1, key="truth",
                                label_visibility="collapsed")

    if st.button("Submit example"):
        ts  = datetime.datetime.utcnow()
        row = (ts, int(output), int(true_digit), round(certainty*100, 2))

        # → PostgreSQL
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO inference_log (ts, pred, truth, confidence) "
                "VALUES (%s, %s, %s, %s)", row
            )

        # → CSV backup
        csv_path = pathlib.Path(img_save_path) / "inference_log.csv"
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp", "pred", "truth", "confidence"])
            w.writerow(row)

        st.success("Example recorded to PostgreSQL server and CSV file!")

    ### a history chart for the user-input ground truth read from the sql server:
    df = pd.read_sql(
        "SELECT ts AS timestamp, pred, truth, confidence "
        "FROM inference_log ORDER BY id DESC LIMIT 200",  # load a window
        conn
    )

    if df.empty:
        st.info("No submissions logged yet. Submit a few examples to see history.")
    else:
        st.markdown("#### Recent submissions")
        st.dataframe(df.head(20), use_container_width=True)

        # running accuracy
        df["correct"] = df["pred"] == df["truth"]
        run_acc = df["correct"].mean() * 100

        st.markdown(
            """
            <style>
                .record-label {
                    font-size: 28px !important;
                    font-weight: 600;          
                    margin-bottom: 0.25rem;
                    display: block;
                }
            </style>

            <label class="record-label">Running accuracy</label>
            """,
            unsafe_allow_html=True,
        )

        st.metric("", f"{run_acc:.1f}%",
                  label_visibility="collapsed")



        # confidence bar chart
        st.markdown("#### Confidence history")
        chart = (
            alt.Chart(df.reset_index())
            .mark_bar()
            .encode(
                x=alt.X("index:O", title="Example #"),
                y=alt.Y("confidence:Q", title="Confidence (%)",
                        scale=alt.Scale(domain=[0, 100])),
                color=alt.condition(
                    alt.datum.correct,
                    alt.value("#4caf50"),  # green if correct
                    alt.value("#f44336")   # red  if wrong
                ),
                tooltip=["timestamp", "pred", "truth", "confidence"]
            )
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)