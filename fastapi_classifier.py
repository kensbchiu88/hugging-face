import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
from transformers import pipeline
import cv2
import threading
import json
import paho.mqtt.client as mqtt
from datetime import datetime
from pydantic import BaseModel

class GetPipelinesResponseModel(BaseModel):
    id: int
    state: str
    avg_fps: float
    start_time: str
    source_uri: str
    destination_mqtt_host: str
    destination_mqtt_topic: str
    device_name: str
    message: str

    

#app = FastAPI()
app = FastAPI(swagger_ui_parameters={"syntaxHighlight.theme": "obsidian"})
 
# Load the Hugging Face model and processor
# model_name = "Salesforce/blip-image-captioning-base"
# processor = BlipProcessor.from_pretrained(model_name)
# model = BlipForConditionalGeneration.from_pretrained(model_name)
# load pipe
image_classifier = pipe = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")
 
# Endpoint to handle image upload and predictions
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Check if the uploaded file is an image
    #if not file.content_type.startswith("image/"):
    #    raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
 
    # Read the image file
    contents = await file.read()
 
    # Preprocess the image (resize, convert to tensor, etc.)
    image = preprocess_image(contents)
 
    # Use Hugging Face model to get predictions
    # input_ids = processor(image, return_tensors="pt").input_ids
    # outputs = model.generate(input_ids)
    # caption = processor.decode(outputs[0], skip_special_tokens=True)
    outputs = image_classifier(image, candidate_labels=["green", "red", "yellow", "blue", "none"])
 
    # return JSONResponse(content={"prediction": caption})
    return outputs

@app.post("/pipeline")
async def process_url(data: dict):
    global pipeline_sequential_id
    uri = data.get("uri")
    if uri:
        if uri in running_threads: 
            return {"message": f"{uri} already in progress"}            
        else:
            event = threading.Event()
            stop_events[uri] = event
            task_thread = threading.Thread(target=classfy_frame_from_rtsp_thread, args=(event,data))
            running_threads[uri] = task_thread
            pipeline_id_to_uri[pipeline_sequential_id] = uri
            uri_to_pipeline_id[uri] = pipeline_sequential_id
            put_thread_info(data)
            pipeline_sequential_id += 1
            task_thread.start()
            return {"message": f"Processing URI: {uri}"}
    else:
        return {"error": "URI field not found in the input JSON data"}

@app.delete("/pipeline/{instance_id}")
async def stop_predict(instance_id: int):
    uri = pipeline_id_to_uri.get(instance_id)
    if uri:
        event = stop_events.get(uri)
        if event:
            event.set()
            running_thread = running_threads.get(uri)
            if running_thread:
                running_thread.join(10)
                if not running_thread.is_alive():
                    del running_threads[uri]
                    del stop_events[uri]
            return {"message": f"Stop Processing URI: {uri}"}
        else:
            return {"message": f"{uri} not in progress"}
    else:
        return {"error": "URI field not found in the input JSON data"}
    
@app.get("/pipelines/", response_model=list[GetPipelinesResponseModel])    
async def show_pipelines():
    values_as_list = list(thread_info.values())
    return values_as_list

def preprocess_image(image_bytes):
    # Example: Resize the image to the required input size
    img = Image.open(io.BytesIO(image_bytes))
    #vimg = img.resize((256, 256))
    # Additional preprocessing if needed
    return img

def classfy_frame_from_rtsp_thread(stop_event, pipeline_request_dict):
    pipeline_id = uri_to_pipeline_id.get(pipeline_request_dict['uri'])
    print(f"pipeline_id: {pipeline_id}")
    # Validate the pipeline request
    is_valid_request, message = validate_pipeline_request(pipeline_request_dict)
    if(not is_valid_request):
        print(message)
        set_pipeline_error(pipeline_id, message)
        return 

    # Connect MQTT Broker
    if is_enable_mqtt:
        mqtt_client = mqtt.Client()
        try:
            mqtt_client.username_pw_set("admin", "admin")
            mqtt_client.connect(pipeline_request_dict['mqtt_host'], 1883)
        except:
            print("Error connecting to MQTT broker.")
            set_pipeline_error(pipeline_id, "Error connecting to MQTT broker.")
            return

    # Open the RTSP stream using FFmpeg
    video = cv2.VideoCapture(pipeline_request_dict['uri'], cv2.CAP_FFMPEG)

    if not video.isOpened():
        print("Error opening video stream.")
        set_pipeline_error(pipeline_id, "Error opening video stream.")
        return
    
    avg_fps = video.get(cv2.CAP_PROP_FPS)
    thread_info[pipeline_id]['avg_fps'] = avg_fps
    fps = round(avg_fps)

    frame_count = 0
    interval = 2  # 間格秒數

    try:
        while not stop_event.is_set():
            # Read the first frame
            ret, frame = video.read()

            if not ret:
                print("Error reading frame from video stream.")
                set_pipeline_error(pipeline_id, "Error reading frame from video stream.")
                break
            
            frame_count += 1

            if frame_count % (interval * fps) == 0:
                # Save the frame as an image file
                #image_name = f"output_images/frame_{frame_count}.jpg"
                #cv2.imwrite(image_name, frame)
                #print(f"Frame saved as {image_name}")

                # 將 OpenCV 的幀轉換為 Pillow 的 Image
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # 使用 Hugging Face model 進行預測
                outputs = image_classifier(image_pil, candidate_labels=["green", "red", "yellow", "blue", "none"])
                print(f"Response: {outputs}")

                result_label, result_score = get_label_and_score_with_max_score(outputs)
                print(f"label: {result_label}, score: {result_score}")

                # Publish the result
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Remove the last 3 characters
                payload = {
                    "device_name": pipeline_request_dict['device_name'],
                    "label": result_label,
                    "score": result_score,
                    "Timestamp": timestamp
                }

                if is_enable_mqtt:
                    mqtt_client.publish(pipeline_request_dict['mqtt_topic'], json.dumps(payload))
                print(f"publish message: {json.dumps(payload)}")

        thread_info[pipeline_id]['state'] = "COMPLETED" 
    except KeyboardInterrupt:
        print("Capturing stopped by user.")
        set_pipeline_error(pipeline_id, "Capturing stopped by user.")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        set_pipeline_error(pipeline_id, sys.exc_info()[0])
    finally:
        # 释放资源
        video.release()
        #mqtt_client.disconnect()

def validate_pipeline_request(pipeline_request_dict): 
    uri = pipeline_request_dict['uri']
    mqtt_host = pipeline_request_dict['mqtt_host']
    mqtt_topic = pipeline_request_dict['mqtt_topic']
    device_name = pipeline_request_dict['device_name']
    if uri is None and mqtt_host is None and mqtt_topic is None and device_name is None:
        return False, "One or more required fields(uri, mqtt_host, mqtt_topic, device_name) are missing."
    else:
        return True, ""

def get_label_and_score_with_max_score(data):
    # 使用 max 函數找到 score 最高的字典
    max_score_dict = max(data, key=lambda x: x['score'])

    # 獲取對應的 label 和 score
    label_with_max_score = max_score_dict['label']
    score_with_max_score = max_score_dict['score']

    return label_with_max_score, score_with_max_score

def put_thread_info(api_request: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Remove the last 3 characters
    info = {
        "id": pipeline_sequential_id,
        "state": "RUNNING",
        "avg_fps": 0,
        "start_time": timestamp,    
        "source_uri": api_request['uri'],
        "destination_mqtt_host": api_request['mqtt_host'],
        "destination_mqtt_topic": api_request['mqtt_topic'],
        "device_name": api_request['device_name'],
        "message": ""
    }
    thread_info[pipeline_sequential_id] = info              

def set_pipeline_error(pipeline_id, message):
    thread_info[pipeline_id]['state'] = "ERROR"
    thread_info[pipeline_id]['message'] = message

# Run the FastAPI app
if __name__ == "__main__":

    stop_events = {}
    running_threads = {}
    thread_info = {}
    pipeline_id_to_uri = {}
    uri_to_pipeline_id = {}
    pipeline_sequential_id = 1
    is_enable_mqtt = False

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)