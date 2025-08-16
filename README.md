# Gray-Wolf-Feature-Optimization

<p align="justify">    
<b>Convolutional Neural Networks</b> applies kernels to extract characteristics from images such as edges, textures and patterns, this information is perform by computing derivates from the forward propagation method which executes a weighted sum through the whole model, once the prediction is done the error takes notes of the expected and obtained result on a loss function also known as fitness taking place the gradient descend learning method which minimizes the error in each iteration. 
</p>

<p align="justify">
Gradient descend is a well known machine learning algorithm however has limitations on the complexity of the network called vanished gradient and does not find the optimial solutions for non convex functions or problems which have multiple or equal solutions being sensitive of the inicial weights.
</p>

<p align="justify">
Metaheuristic algorithms by approaching different biology or non-biology methods explore in a search space limited by upper and lower bound variables a possibly new solution, close, equal or better than gradient descend, these algorithms could be applied to adapt the network architecture, adjust learning rate, regularizes, weights, coefficients or other hyperparameters that have a huge number of combinations or values.
</p>

<p align="justify">
Taking notes of how Convolutional Neural Networks extract features before the result is selected by the dense layers the amount and quality of data could be compromise, by <b> adding a metaheuristic optimizations gray wolf on the final extracted features is perform a dimensionality reduction choosing the most relevant data</b>, the performance of the selected features are test on a support vector machine ensemble boosting model, this means the data flows from the input through the last convolution layer which is applied a metaheuristic algorithm and only trains with the remaining data a support vector machine, through epochs the selected features change according the fitness function considering the native loss function and the amount of features selections above the total.
</p>

https://github.com/user-attachments/assets/e8376dcf-178e-43b8-9faf-20b00298978f

## Summary

- Hardware Requirements
- How to Install
  - $${\color{orange}1.- \space Docker \space Desktop}$$
  - $${\color{orange}2.- \space WSL2}$$
  - $${\color{orange}3.- \space Nvidia \space Container \space Toolkit}$$
  - $${\color{orange}4.- \space Verify \space Docker \space Installation}$$
  - $${\color{orange}5.- \space Run \space Container \space (Choose \space Environment}$$
      - $${\color{lightskyblue}Ubuntu \space Terminal}$$
      - $${\color{lightskyblue}Visual \space Studio \space Code \space (DevContainer)}$$

## Extra Resources

- [CNN Architectures](https://github.com/Pathinker/Gray-Wolf-Feature-Optimization/tree/main/cnn)
- [SVM Concepts](https://github.com/Pathinker/Gray-Wolf-Feature-Optimization/tree/main/boosting_ensemble)
- [Code Documentation](https://github.com/Pathinker/Gray-Wolf-Feature-Optimization/tree/main/metaheuristic)

> [!TIP]
> Learn how to make your own virtual environments for machine or deep learning with Docker support [here](https://github.com/Pathinker/Docker-Tensorflow-WSL2).

# Hardware Requirements

<p align="justify">
$${\color{orange}- \space Intel \space or \space AMD \space Processor:}$$ The Docker image is compatible only with the AMD64 (x86_64) architecture. It could be cross-platform with ARM, but doing so would double the image size for non-targeted devices.
</p>

<p align="justify">
$${\color{orange}- \space 8 \space VRAM \space Nvidia \space Graphic \space Card:}$$ Required to load and run models using hardware acceleration by GPU on TensorFlow, lower VRAM models could run lesser weighted models only with lower images per batch or get errors due data segmentation and not enough memory.
</p>

> [!CAUTION]
> Nvidia GPU is also need it to run the parallel optimizations code applied with pycuda library that runs on Nvidia Cuda platform.

# How to Install

<p align="justify">
In order to execute and replicate the results the project was virtualized through Docker being necessary only to install the tools to manage the virtual environment, also a dev container was made for visual studio code requiring less commands and a more friendly IDE to test and play on.
</p>

## $${\color{orange}1.- \space Docker \space Desktop}$$

<p align="justify">
Download and procced a to install <a href = "https://www.docker.com"> Docker desktop </a> from the official website, once done enable the WSL2 connectivity currently on "<b>General</b>" and click on "<b>Use the WSL2 based engine</b>" or similar.
</p>

## $${\color{orange}2.- \space WSL2}$$

<p align="justify">
WSL2 is the windows subsystem for Linux, this will allow to integrate and use the kernel of many Linux distributions without need of virtual machine and being native on windows, to open it you must do the following path:
</p>

- Enable "**Virtual Machine Platform**" and "**Windows Subsystem for Linux**" on windows features, could be found in "**Turn Windows features on or off**"
- Open windows PowerShell as an admin, then write "**wsl --install**"
- Login on Ubuntu distribution as a new user, then write "**sudo apt update && sudo apt upgrade -y**" to download all updates.

More information on <a href = "https://learn.microsoft.com/en-us/windows/wsl/install"> Microsoft </a> official website.

## $${\color{orange}3.- \space Nvidia \space Container \space Toolkit}$$

<p align="justify">

<p align="justify">
Nvidia Container Toolkit allows the containers made by Docker the use of all graphic cards naturally this gives the support for tensorflow to utilize a GPU on training models, however is also need it to use nvidia cuda platform to write parallel code and run it on the graphic card, this feature is applied with pycuda to makes faster the exploitation process of the metaheuristic algorithm.
</p>

Copy the following commands to download and install the files in the end restart the system.

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update
```

```
sudo apt-get install -y nvidia-container-toolkit
```

More information on <a href = "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"> Nvidia </a> official website.

## $${\color{orange}4.- \space Verify \space Docker \space Installation}$$

- Go to the Ubuntu distribution download on WSL2 and type "**```docker --version```**" if everything is all right you should see the current version of Docker you got on your system.
- Run the next Docker image "**```docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi```**" if you see your GPU, Drivers and Cuda version everything was installed correct.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c1109a57-f7b6-4b87-98f8-50d7c4c668fe" alt="image">
</p>

## $${\color{orange}5.- \space Run \space Container \space (Choose \space Environment)}$$

<p align="justify">
Now you have two options to run the code, one being run it on the native WSL2 Ubuntu terminal which is already installed or add one extension on visual studio to get a more comfortable develop environment.
</p>

### $${\color{lightskyblue}Ubuntu \space Terminal}$$

<p align="justify">
Run the next command and wait around 30 minutes for the Docker image to be fully downloaded, <b>this will allow you to access the training results</b>.
</p>

```
docker run -it --rm --gpus all --name GWO pathinker/gray-wolf-feature-optimization:1.0
```

The following meaning of the arguments sent are the next:

- **it:** Allows and interactive terminal, it will allow to show feedback and logs from the terminal.
- **rm:** Removes the container once it is close.
- **gpus all:** Gives access to the container all the GPUs on your device using Docker Container Toolkit
- **name:** Names the container.

<p align="justify">
For now, you will need to write python3 and the complete route of files to execute them and other commands for editing, viewing the files, however all codes will be fully operational.
</p>

> [!CAUTION]
> Note: All changes made once the program is running will be lost due the missing of a volume that shares the data among the virtual environment and the host device.

### $${\color{lightskyblue}Visual \space Studio \space Code}$$

<p align="justify">
To run it on Visual Studio Code you will need to search for extensions and type "<b><a href = "https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers">Dev Containers</a></b>" from Microsoft, wait until it is fully operational and then do the next shortcut "<b>Ctrl + Shift + P</b>", write and click on "<b>>Dev Containers: Rebuild and Reopen in Container</b>". Afterwards you will need to wait around the same time of the Ubuntu terminal setup.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5c8dd9f-792c-4b8e-9a95-9fd4aa7d2883" alt="image">
</p>

<p align="justify">
Using the dev container on Visual Studio Code will provide a few extensions to enable python debugging, fast code runner buttons, access to your git SSH keys to clone and modify the repository, zsh terminal, data persistence between changes and not opening the WSL2 or Ubuntu terminal.
</p>

> [!IMPORTANT]
> Docker engine must be running always if you want to reopen the project in both cases.