'''from flask import Flask, render_template, request, redirect, url_for
import os
from processor import DEMProcessor
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/input'
OUTPUT_FOLDER = 'static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(OUTPUT_FOLDER, f"output_{filename}")

            file.save(input_path)

            # Process the image
            processor = DEMProcessor()
            processor.run(input_path, 'static/output')

            

            return render_template('index.html', input_image=input_path, output_image=output_path)

    return render_template('index.html', input_image=None, output_image=None)

if __name__ == '__main__':
    app.run(debug=True)'''

from flask import Flask, render_template, request, redirect
import os
from processor import DEMProcessor
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/input"
OUTPUT_FOLDER = "static/output"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)

        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(input_path)

        # ---- RUN SFS PIPELINE (UNCHANGED LOGIC) ----
        processor = DEMProcessor()
        gray_path, color_path = processor.run(input_path, OUTPUT_FOLDER)

        # ---- OPTIONAL RF REFINEMENT ----
        ml_output_path = os.path.join(OUTPUT_FOLDER, "ml_refined_dem.png")

        '''
        try:
            from ml.predict_rf import refine_dem
            refine_dem(input_path, gray_path, ml_output_path)
            ml_available = True
        except Exception as e:
            print("RF skipped:", e)

            ml_available = False
            '''
        try:
            from ml.predict_rf import refine_dem
            ml_gray, ml_color = refine_dem(input_path, gray_path, OUTPUT_FOLDER)
            ml_available = True
        except Exception as e:
            print("RF skipped:", e)
            ml_gray, ml_color = None, None
            ml_available = False


        return render_template(
    "index.html",
    input_image=f"input/{filename}",
    #gray_dem="output/gray_dem.png",
    #color_dem="output/color_dem.png",
    ml_gray="output/ml_gray_dem.png" if ml_available else None,
    ml_color="output/ml_color_dem.png" if ml_available else None

)


    return render_template("index.html")
    

if __name__ == "__main__":
    app.run(debug=True)
