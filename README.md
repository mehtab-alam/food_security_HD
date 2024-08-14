# Food security indicators predictions using heterogenous data


## Dataset

How to download the dataset

- For **MAC/Linux** users, Open the terminal and navigate to your code directory, e.g. (cd food_security_HD). Run the below script to download the data:

```sh
./data.sh
```
- For **Windows** users, Download the data using [Link](https://www.googleapis.com/drive/v3/files/1VJFM0wuljsc2Dhdxus8h0IdcE9-0iJJu?alt=media&key=AIzaSyBo55XtefB47P_CPLKosGvnpEi3pQs5lCk). Unzip the data and paste it in your code (**food_security_HD**) folder.


## Configuration

The [configuration.py](https://github.com/mehtab-alam/food_security_HD/blob/master/configuration.py) file contains the basic configuration files to setup the directories, variables according to their own dataset.


## Installing Necessary Libraries

```sh
pip install pandas
pip install -U scikit-learn
pip install GDAL
pip3 install torch torchvision
```
