#!bin/bash
FILEID="1WYyy67rXuT3jimOnD2PeFKwIT7efGp6n"
FILENAME="COVID-19-20.zip"
URL='https://docs.google.com/uc?export=download&id='$FILEID

gdown $URL -O $FILENAME
unzip -qq COVID-19-20.zip && rm COVID-19-20.zip
mkdir images && cp -a COVID-19-20_v2/Train/. images/ && rm -rf COVID-19-20_v2/