# change dir to data
cd Utility
# change dir one level up
python annotateData.py
python3 convertData_to_csv.py

echo "now use generate_tfrecod.sh file to convert data in TFRecord format"