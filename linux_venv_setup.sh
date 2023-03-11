# init virtual env
python -m venv venv

source venv/bin/activate

# install pillow deps
sudo apt update
sudo apt-get install libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
    libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
    libharfbuzz-dev libfribidi-dev libxcb1-dev -y

pip install meson ninja

# clone pillow
git clone https://github.com/python-pillow/Pillow

cd Pillow/depends
# make sudo happy
sed -i 's/sudo/sudo -E env PATH=$PATH/g' install_raqm.sh
# install script
chmod +x ./install_raqm.sh && ./install_raqm.sh

# install everything
pip install -r requirements_generate_font_dataset.txt

#clean up
cd ../..
rm -rf Pillow

