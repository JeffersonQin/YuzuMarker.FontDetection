sudo apt update && sudo apt install python3-venv -y

# init virtual env
python3 -m venv venv

source venv/bin/activate

# install pillow deps
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

#clean up
cd ../..
rm -rf Pillow

# install everything
pip install -r requirements_generate_font_dataset.txt

# download wordlist (added since my cluster has bad public network connection)
wget https://www.mit.edu/~ecprice/wordlist.10000
mv wordlist.10000 wordlist.txt

