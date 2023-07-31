from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from helper.helper import get_predictions, add_boxes_to_predictions, byte_to_image, image_to_byte, store_images_confidences
from ultralytics import YOLO
from fastapi.responses import StreamingResponse
import pandas as pd
import os

###################################################################
####### FastAPI ###################################################
###################################################################

app = FastAPI(
    title="Plastic Detection API",
    description="API for plastic detection",
    version="0.1/July'23",
    
)

@app.on_event("startup")
def startup_event():
    print("Starting up")
    if not os.path.exists('Inference'):
        os.makedirs('Inference')
    print("Startup complete . . .") 
       
    


###################################################################
######## Main Work ################################################
###################################################################

@app.post("/predict_to_json")
async def predict_to_json(file: UploadFile = File(...)):
    """Predict plastic in image and return json"""
   
    
    
    # predictions = get_predictions(img, model)
    # res = pd.DataFrame(predictions)
    # print(len(res['Confidence'].values.tolist()))
    # print(res)
    
    
    pass


@app.post("/predict_to_image")
def predict_to_image(file: UploadFile = File(...)):
    model = YOLO("YOLO_Custom_v8m.pt")
    img = byte_to_image(file.file.read())
    predictions = get_predictions(img, model)
    # print(predictions)
    processed = add_boxes_to_predictions(predictions, img)
    return StreamingResponse(content=image_to_byte(processed) , media_type="image/jpeg")


@app.post("/predict_save_image_and_json")
def predict_and_save(file: UploadFile = File(...)):
    model = YOLO("YOLO_Custom_v8m.pt")
    img = byte_to_image(file.file.read())
    predictions = get_predictions(img, model)
    processed = add_boxes_to_predictions(predictions, img)
    processed_image_byte = image_to_byte(processed)
    store_images_confidences(file_name = file.filename, image=processed, details = predictions)
    return StreamingResponse(content=processed_image_byte , media_type="image/jpeg")
    