from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from helper.helper import get_predictions, add_boxes_to_predictions, byte_to_image, image_to_byte
from ultralytics import YOLO
from fastapi.responses import StreamingResponse

###################################################################
####### FastAPI ###################################################
###################################################################

app = FastAPI(
    title="Plastic Detection API",
    description="API for plastic detection",
    version="0.1/July'23",
    
)


###################################################################
######## Main Work ################################################
###################################################################

@app.post("/predict_to_json")
async def predict_to_json(file: bytes = File(...)):
    """Predict plastic in image and return json"""
    
    pass


@app.post("/predict_to_image")
def predict_to_image(file: bytes = File(...)):
    model = YOLO("yolov8n.pt")
    img = byte_to_image(file)
    predictions = get_predictions(img, model)
    print(predictions)
    processed = add_boxes_to_predictions(predictions, img)
    return StreamingResponse(content=image_to_byte(processed) , media_type="image/jpeg")