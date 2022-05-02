mkdir -p models

# Download the generation model
gdown https://drive.google.com/uc?id=1vVhRgLtsQDAOmxYhY5PMPnxxHUyCOdQU
mkdir -p models/generation
mv model.tar.gz models/generation

# Download the answering model
mkdir models/answering
cd models/answering
gdown https://drive.google.com/uc?id=1q2Z3FPP9AYNz0RJKHMlaweNhmLQoyPA8
unzip model.zip
rm model.zip
cd ../..

# Download the LERC models
gdown https://drive.google.com/uc?id=193K7v6pjOtuXdlMenQW-RzF6ft-xY2qd
gdown https://drive.google.com/uc?id=1fWBahDT-O1mpsbND300cuZuF73mfObzH
mkdir -p models/lerc
mv model.tar.gz pretraining.tar.gz models/lerc


wget https://storage.googleapis.com/qafacteval/quip-512-mocha.tar.gz 
tar -xzvf quip-512-mocha.tar.gz
mv quip-512-mocha models/
rm quip-512-mocha.tar.gz
