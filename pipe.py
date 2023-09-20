import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
import base64
import io
from flask import Flask, request, jsonify
from flask_mysqldb import MySQL

app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'test'

mysql = MySQL(app)

# Load the YOLO model
model = YOLO('best.pt')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()

    if 'base64_image' not in data:
        return jsonify({'error': 'Please provide a base64 image.'}), 400

    base64_image = data['base64_image']

    # Decode the base64 string to bytes
    image_data = base64.b64decode(base64_image)

    # Convert bytes to a PIL image
    img = Image.open(io.BytesIO(image_data))

    # Perform object detection on the image
    result = model.predict(img, conf=0.45, save=False, show=False)

    # Create a PIL image and draw on it
    for r in result:
        im_array = r.plot()  # Plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        draw = ImageDraw.Draw(im)

        # Get the coordinates of the detected objects
        arrxy = r.boxes.xyxy
        coordinates = np.array(arrxy)

        x_coords = (coordinates[:, 0] + coordinates[:, 2]) / 2
        y_coords = (coordinates[:, 1] + coordinates[:, 3]) / 2
        midpoints = np.column_stack((x_coords, y_coords))

        # Draw numbers inside the detected pipes
        for i, (x, y) in enumerate(midpoints, start=1):
            size_str = str(i)  # Sequential number

            # Position the text inside the detected pipe
            text_x = int(x)
            text_y = int(y)

            # Add the number without a circle
            draw.text((text_x - 10, text_y - 10), size_str, fill=(255, 0, 0))

        # Calculate the total count of detected pipes
        total_count = len(midpoints)

        # Add the total count to the image
        total_count_str = f"Total Pipes: {total_count}"
        print(total_count_str)

        draw.text((500, 400), total_count_str, fill=(255, 0, 0))

        # Save the image with the total count in the filename
        save_filename = f'results_Total_pipes{total_count}.jpg'
        im.save(save_filename)

    # Convert the final image to base64
    buffered = io.BytesIO()
    im.save(buffered, format="JPEG")  # Change format as needed
    base64_image_result = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as JSON
    response_data = {
        'total_count': total_count,
        'base64_image_result': base64_image_result
    }
    return jsonify(response_data)

@app.route('/search', methods=['POST'])
def search_by_invoice_number():
    try:
        data = request.get_json()
        invoice_number = data.get("invoice_number")

        # Query the MySQL database to search for data based on the invoice number
        cur = mysql.connection.cursor()
        cur.execute("SELECT from_address,to_address,date_time,invoice_number,totalcount,received_base64_image FROM pipecount WHERE invoice_number = %s", (invoice_number,))
        result = cur.fetchone()
        cur.close()

        if result:
            # Convert the result to a dictionary for JSON response
            result_dict = {
                'from_address': result[0],
                'to_address': result[1],
                'date_time': result[2],
                'invoice_number': result[3],
                'totalcount': result[4],
                'received_base64_image': result[5],
            }
            return jsonify(result_dict), 200
        else:
            return jsonify({"error": "Invoice number not found"}), 404
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to process the request"}), 500


@app.route('/detect1', methods=['POST'])
def detect1():
    try:
        data = request.get_json()
        from_address = data.get("from_address")
        to_address = data.get("to_address")
        date_time = data.get("date_time")
        invoice_number = data.get("invoice_number")
        totalcount = data.get("totalcount")
        received_base64_image = data.get("base64")

        # Insert data into MySQL
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO pipecount (from_address, to_address, date_time, invoice_number, totalcount, received_base64_image)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (from_address, to_address, date_time, invoice_number, totalcount, received_base64_image))
        mysql.connection.commit()

        # Close the cursor
        cur.close()

        # You can send a response back to the Flutter app if needed
        response_data = {"message": "Data received and saved to MySQL successfully"}
        return jsonify(response_data), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Failed to process the request"}), 500
    
    

if __name__ == '__main__':
    app.run(debug=True)
