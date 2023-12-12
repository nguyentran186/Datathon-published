from flask import Flask, request, Response, send_file, make_response
from io import BytesIO
import zipfile
from PIL import Image
from flask_cors import CORS

from util.dis_hm import distribution_heatmap
from util.pose_hm import pose_heatmap

app = Flask(__name__)

CORS(app)

@app.route('/api/dis_hm', methods=['POST'])
def get_dis_hm():
    print(request.files)
    vid = request.files['file']
    vid_path = 'temp.mp4'
    vid.save(vid_path)
    img = distribution_heatmap(vid_path)
    response = Response(img.tobytes(), content_type='image/jpeg')
    return response 


@app.route('/api/process', methods=['POST'])
def get_pose_hm():
    print(request.files)
    vid = request.files['file']
    vid_path = 'temp.mp4'
    vid.save(vid_path)
    temp_data = pose_heatmap(vid_path)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for i in range (0, 4):
            img_bytes = temp_data[i]
            zip_file.writestr(f'image_{i}.png', img_bytes.tobytes())
     # Move the cursor to the beginning of the buffer
    zip_buffer.seek(0)

    # Create a Flask response with the ZIP archive
    response = make_response(send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='images.zip'))

    return response
    # response = Response(temp_data[2].tobytes(), content_type='image/jpeg')
    # return response

# @app.route('/api/high', methods=['POST'])
# def get_high():
#     data = g.getattr('im', 'No data found')
#     return Response(data.tobytes(), content_type='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, port = 5000)