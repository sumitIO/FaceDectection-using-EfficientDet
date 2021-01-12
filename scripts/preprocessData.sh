# Clone the tensorflow-object-detection-api
git clone https://github.com/tensorflow/models.git

# Install the Object Detection API
cd models/research/

# make sure to install the protoc compliler if using Linux/Mac o windows download from official site
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

# Test the installation.
python models/research/object_detection/builders/model_builder_tf2_test.py

# bash script to run python file getData
# get Data retrive the datasets as zip and extract them.
python3 getData.py

# to remove the downloaded zip files as we have no requiremt of them any more. Uncomment below if you want to keep them as zip on disk.
rm ~/data/train.zip
rm ~/data/val.zip

python3 annotateData.py
python3 convertData_to_csv.py

echo "now use generate_tfrecod.sh file to convert data in TFRecord format"