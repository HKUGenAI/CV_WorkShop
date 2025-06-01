conda create --name detectionworkshop python=3.12
conda activate detectionworkshop
pip install -r requirements_yolo
pip install -r requirements_GD
pip install -r requirements_GD   <-- you have to do this a second time, first time some package is not install and it show error somehow trust me!!
pip install --upgrade supervision
