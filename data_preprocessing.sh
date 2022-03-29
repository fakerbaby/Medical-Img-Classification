cd data
unzip -O CP936 '*.zip' # -O CP936 unzip解决解压中文乱码问题
rm -rf *.zip
cd ../
python preprocess.py

cd data
rm -rf 0[0-2]*

