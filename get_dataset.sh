dir="./dataset"

if [ ! -d "$dir" ]; then

    echo "Downloading dataset, please wait until it finishes."

    start=$(date +%s)

    mkdir -p "$dir"

    download_successful=0
    attempt=1
    max_attempts=10
    
    while [ $download_successful -eq 0 ]; do

        if [ $attempt -gt $max_attempts ]; then
            echo "Unable to get dataset, servers are not responding, please try executing the 'get_dataset.sh' file at a couple of minutes or another time."
            rm -rf "$dir"
            exit 1
        fi
        
        echo "Trying to get dataset attempt $attempt/$max_attempts..."

        curl -L -o "$dir"/open-sprayer-images.zip https://www.kaggle.com/api/v1/datasets/download/gavinarmstrong/open-sprayer-images
        if [ $? -eq 0 ]; then
            download_successful=1
        else
            echo "Connection error, retrying in 10 seconds..."
            sleep 10
        fi
        
        attempt=$((attempt + 1))
    done

    total_files=$(unzip -Z1 "$dir"/open-sprayer-images.zip | grep -E "^(Docknet/train/|Docknet/valid/)" | wc -l)
    
    echo "Unpacking files... ($total_files files total)"
    counter=0

    for file in $(unzip -Z1 "$dir"/open-sprayer-images.zip | grep -E "^(Docknet/train/|Docknet/valid/)"); do
        unzip -q "$dir"/open-sprayer-images.zip "$file" -d "$dir"
        counter=$((counter + 1))
        stdbuf -o0 echo -ne "Unpacking files... ($counter/$total_files) files unpacked\r"
    done
    echo ""

    rm "$dir"/open-sprayer-images.zip

    rm -rf "$dir"/Docknet/Docknet
    mv "$dir"/Docknet/* "$dir"
    rm -rf "$dir"/Docknet

    end=$(date +%s)
    duration=$((end - start))

    echo "Dataset downloaded and unpacked suscefully in $duration seconds."
fi