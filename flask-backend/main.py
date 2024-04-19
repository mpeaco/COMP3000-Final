from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    flash,
    redirect,
    url_for,
    send_file,
)
from PIL import Image
import io
from flask_cors import CORS
from deepforest import main
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from base64 import b64encode
import base64
from io import BytesIO  # Converts data from Database into bytes

app = Flask(__name__)
CORS(app, origins="*")
basedir = "sqlite:///" + os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "data.sqlite"
)

app.config["SQLALCHEMY_DATABASE_URI"] = basedir
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "dev"
db = SQLAlchemy(app)




# Picture table. By default the table name is filecontent
class FileContent(db.Model):
   
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)  # Actual data, needed for Download
    """rendered_data = db.Column(
        db.Text, nullable=False
    )  # Data to render the pic in browser"""
    trees = db.Column(db.Integer, nullable=False)
    pic_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return "Pic Name: {self.name} Data: {self.data} text: {self.text} created on: {self.pic_date} location: {self.location}"


deepforest_model = main.deepforest()
deepforest_model.use_release()


@app.route("/image_upload", methods=["POST"])
def image_upload():
    print("Image upload request received\n")
    try:
        # Check if the POST request has a file
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]

        if file and file.filename.endswith((".tif", ".tiff", ".jpg", ".png", ".jpeg")):
            image = Image.open(io.BytesIO(file.read()))
            image = np.array(image, dtype=np.float32)

            plot = deepforest_model.predict_image(image, return_plot=True)
            plot_df = deepforest_model.predict_image(image, return_plot=False)
            number_of_trees = plot_df.shape[0]
            buffer = BytesIO()
            # plt.savefig(buffer)

            # Store the file and result into the database
            print(file.filename)
            file_to_store = FileContent(
                name=file.filename, data=image, trees=number_of_trees
            )
            # Check if the data for that image is already processed
            existing_image = FileContent.query.filter_by(
                name=file_to_store.name
            ).first()

            if existing_image:
                return jsonify({"message": "Image already exists in the database"})

            db.session.add(file_to_store)  # Store the file in database
            db.session.commit()

            return jsonify(
                {
                    "result": "Image processed successfully: "
                    + "\n"
                    + "Number of trees: {}".format(number_of_trees)
                }
            )

        else:
            print("Invalid file type\n")
            return jsonify(
                {"error": "Invalid file type. Please select a .tif or .tiff file."}
            )

    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"})


@app.route("/history", methods=["GET"])
def query_db():
    all_im = FileContent.query.all()
    print(type(all_im))
    if all_im == None:
        return jsonify({"Result": "No stored images when querying the database"})

    images_data = []
    for image in all_im:
        images_data.append(
            {
                "name": image.name,
                "trees": image.trees,
                "pic_date": image.pic_date,
            }
        )

    return jsonify({"images": images_data})


# Main
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print("***** Db created *****\n")
    app.run(port=8000)
