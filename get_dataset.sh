# Download dataset from Dropbox
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Lp3KS9Gh1LZx6_WVQsSd5H0iHmFAsmFn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Lp3KS9Gh1LZx6_WVQsSd5H0iHmFAsmFn" -O hw2_data.zip && rm -rf /tmp/cookies.txt

# Unzip the downloaded zip file
mkdir semseg_data
unzip ./hw2_data.zip -d semseg_data

# Remove the downloaded zip file
rm ./hw2_data.zip