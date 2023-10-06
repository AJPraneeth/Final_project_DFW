
from flask import Flask, request, render_template, jsonify 
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import os
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the Keras model and define class labels
model = load_model("Model/Classification.h5")
class_labels = ["both", "infection", "ischemia", "none"]

# Load the wound segmentation model
segmentation_model = load_model('Model/woundsegmentation.h5')
segmentation_model_sticker = load_model('Model/stickersegmentation.h5')


# Function to process the uploaded image and predict the label
def process_image(image_path):
    try:
        # Load and preprocess the image
        # image = cv2.imread(image_path)
        # image = cv2.resize(image, (224, 224))
        # image = image.astype("float") / 255.0
        # image = img_to_array(image)
        # image = np.expand_dims(image, axis=0)

        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Unable to load the image."
        image = cv2.resize(image, (224, 224))
        image = image.reshape(-1,224,224,3)
        
        

        # Make a prediction
        predictions = model.predict(image)
        print(predictions)
        predicted_class = class_labels[np.argmax(predictions)]

        return predicted_class
    except Exception as e:
        error_message = str(e)
        return error_message

# Function to preprocess and segment the uploaded image
def segment_wound(image_path):
    try:
        # Load and preprocess the image
        image = cv2.imread(image_path)
          # Check if the image was loaded successfully
        if image is None:
            return None, "Error: Unable to load the image."
        image = cv2.resize(image, (256, 256))  # Resize to the model's input size
        image = image / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)

        # Perform wound segmentation using the model
        segmented_image = segmentation_model.predict(image)
        print(segmented_image)

     

        # Convert the predicted mask to an 8-bit format for further processing
        predicted_mask = (segmented_image[0, :, :, 0] * 255).astype(np.uint8)

        # Apply Median Filtering to the predicted mask
        median_filtered_mask = cv2.medianBlur(predicted_mask, 5)  # Adjust kernel size (5) as needed

        # Define the kernel for morphological operations
        kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed

        # Erosion operation to remove noise
        eroded_mask = cv2.erode(median_filtered_mask, kernel, iterations=1)

        # Dilation operation to restore the mask
        dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=1)

        # Convert the dilated mask to the appropriate data type for Otsu's thresholding
        dilated_mask_uint8 = dilated_mask.astype(np.uint8)

        
        # Apply Otsu's thresholding method to the dilated mask
        _, binary_mask = cv2.threshold(dilated_mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the contour with the largest area (whitest area)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create an empty mask with the same size as the input image
        whitest_area_mask = np.zeros_like(binary_mask)

        # Draw the largest contour (whitest area) on the mask
        cv2.drawContours(whitest_area_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)   

         # Count white pixels in the wound mask
        white_pixel_count_wound = np.sum(whitest_area_mask == 255)

        print("White Pixel Count (wound):", white_pixel_count_wound)

        # print (dilated_mask_uint8)
        # return dilated_mask_uint8
        return whitest_area_mask,white_pixel_count_wound
    except Exception as e:
        error_message = str(e)
        return None

def segment_sticker(image_path):
    try:
       
         # Load and preprocess the image
        image = cv2.imread(image_path)

          # Check if the image was loaded successfully
        if image is None:
            return None, "Error: Unable to load the image."
        
        image = cv2.resize(image, (256, 256))  # Resize to the model's input size
        image = image / 255.0  # Normalize

        # Predict the mask using the model
        image_input = np.expand_dims(np.array([image]), axis=-1)
        predicted_mask = segmentation_model_sticker.predict(image_input)[0, :, :, 0]
        # Calculate and print the range of values in the prediction array
        min_value = np.min(predicted_mask)
        max_value = np.max(predicted_mask)
        print(f"Minimum Prediction Value: {min_value}")
        print(f"Maximum Prediction Value: {max_value}")

        # Threshold the predicted mask
        threshold = 0.5
        binary_mask = np.where(predicted_mask > threshold, 255, 0).astype(np.uint8)

          # Count white pixels in the sticker mask
        white_pixel_count = np.sum(binary_mask == 255)

        print("White Pixel Count (Sticker):", white_pixel_count)
       
        return binary_mask,white_pixel_count 
    except Exception as e:
        error_message = str(e)
        print(error_message)
        return None

def wound_area(stickerWhiteCount,woundWhiteCount):
    try :
        if(stickerWhiteCount==0):
            return 0.0
        realStickerWidth=1.9
        realStickerheight=1.9
        realWoundArea=realStickerWidth*realStickerheight
        areaOfStickerOnePixel=realWoundArea/stickerWhiteCount
        print('areaOfStickerOnePixel',areaOfStickerOnePixel,'\n')
        
        areaOfWound=woundWhiteCount*areaOfStickerOnePixel
        print('area of wound',areaOfWound)
        return areaOfWound
    
    except Exception as e:
        error_message = str(e)
        print(error_message)
        return None

@app.route('/')
def index():
    return render_template ("index.html")
    
@app.route('/model', methods=['GET', 'POST'])
def dfwModels():
     if request.method == 'POST':
        try:
            if 'img' not in request.files:
                return jsonify({'error': "No image selected."})

            image_file = request.files['img']

            # Ensure the file has a valid extension
            if image_file and '.' in image_file.filename:
                extension = image_file.filename.rsplit('.', 1)[1].lower()
                if extension not in ['jpg', 'jpeg', 'png']:
                    return jsonify({'error': "Invalid file format. Please upload a JPG, JPEG, or PNG image."}) 

            # Determine the base directory
            base_directory = os.path.dirname(__file__)

            # Create a folder to store uploaded images (if it doesn't exist)
            uploads_folder = os.path.join(base_directory, "uploads")
            os.makedirs(uploads_folder, exist_ok=True)
            img_path = os.path.join(uploads_folder, image_file.filename)

            image_file.save(img_path)

            #Process Classsification
            classification = process_image(img_path)
            #process wound segmentation
            # dilated_mask_uint8 = segment_wound(img_path)
            woundSegmentMask,woundWhiteCount = segment_wound(img_path)
            #Process sticker segementation
            stickerSegmentmask,stickertWhiteCount = segment_sticker(img_path)

            if classification is None:
                return  jsonify({'error':"image not load correctly (Classifiction)"})
            
            if (stickerSegmentmask & woundSegmentMask) is not None:
            #sticker
                # Save the segmented image to a file or convert it to base64 for displaying in HTML
                segmented_imagesticker_path = os.path.join(uploads_folder, "segmented_sticker_" + image_file.filename)
                cv2.imwrite(segmented_imagesticker_path, stickerSegmentmask)

                # Encode the images as base64 strings
                _, input_imagestiker_encoded = cv2.imencode('.png', cv2.imread(img_path))
                input_image_base64 = base64.b64encode(input_imagestiker_encoded).decode('utf-8')

                _, segmented_image_sticker_encoded = cv2.imencode('.png', stickerSegmentmask)
                segmented_image_sticker_base64 = base64.b64encode(segmented_image_sticker_encoded).decode('utf-8')

            #wound segmentation

                # Save the segmented image to a file or convert it to base64 for displaying in HTML
                segmented_wound_image_path = os.path.join(uploads_folder, "segmented_wound_" + image_file.filename)
                cv2.imwrite(segmented_wound_image_path, woundSegmentMask)

                # Encode the images as base64 strings
                _, input_imagewound_encoded = cv2.imencode('.png', cv2.imread(img_path))
                input_image_base64 = base64.b64encode(input_imagewound_encoded).decode('utf-8')

                _, segmented_imagewound_encoded = cv2.imencode('.png', woundSegmentMask)
                segmented_image_wound_base64 = base64.b64encode(segmented_imagewound_encoded).decode('utf-8')
                           
            #wound area
                woundAreaMeasurement=wound_area(stickertWhiteCount,woundWhiteCount)
                   
                 # Return JSON response with base64-encoded segmented image
                return jsonify({'classification': classification,
                                'sticker':segmented_image_sticker_base64,
                                'wound':segmented_image_wound_base64,
                                'area':woundAreaMeasurement})
            else:
                return jsonify({'error': "Error during segmentation."})
            

        except Exception as e:
            error_message = str(e)
            return jsonify({'error': error_message})





if __name__ == "__main__":  
    app.run(debug=True, host='0.0.0.0')
