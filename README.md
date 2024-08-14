# Food security indicators predictions using heterogenous data


# 1. Dataset

How to download the dataset

- For **MAC/Linux** users, Open the terminal and navigate to your code directory, e.g. (cd food_security_HD). Run the below script to download the data:

```sh
./data.sh
```
- For **Windows** users, Download the data using [Link](https://www.googleapis.com/drive/v3/files/1VJFM0wuljsc2Dhdxus8h0IdcE9-0iJJu?alt=media&key=AIzaSyBo55XtefB47P_CPLKosGvnpEi3pQs5lCk). Unzip the data and paste it in your code (**food_security_HD**) folder.


# 2. Configuration

The [configuration.py](https://github.com/mehtab-alam/food_security_HD/blob/master/configuration.py) file contains the basic configuration files to setup the directories, variables according to their own dataset.


# 3. Installing Necessary Libraries

```sh
pip install pandas
pip install -U scikit-learn
pip install GDAL
pip3 install torch torchvision
```

# 4. How to run the application

The application offers multiple running configurations, including:
1. **country:** Options include burkina_faso, rwanda, and tanzania.
2. **algorithm:** Choice between classification or regression.
3. **tt_split:** Data split selection of temporal, spatio-temporal, or percentage methods for dividing input data.

In the article, we presented the classification results of *temporal*, and *spatio-temporal* split. To perform this, please run the following commands:

## 4.1 Temporal Split:

1. For Burkina Faso 
```sh
 python main.py -country=burkina_faso -algorithm=classification -tt_split=temporal 
```
The output will be found at *'../output/burkina_faso/results/[model]/temporal'*. The *[model]* directory are the following: lstm, cnn.  

2. For Rwanda
```sh
 python main.py -country=rwanda -algorithm=classification -tt_split=temporal 
```
The output will be found at *'../output/rwanda/results/[model]/temporal'*. For instance, the spatial distribution and the confusion matrices for FCS (FCS -> SCA) and HDDS (HDDS -> SDA) are as follow:

**Confusion Matrix:**
<p align="center">
  <img src="results/class_sca_confusion_matrix.png" alt="Image 1" width="30%" />
  <img src="results/class_sda_confusion_matrix.png" alt="Image 2" width="30%" />
</p>

**Spatial Distribution Comparison of FCS (Actual vs. Predicted Values):**
<table align="center">
  <tr>
    <td align="center">
      <b>FCS</b>
    </td>
    <td align="center">
      <b>HDDS</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="results/class_sca.png" alt="Image 1" width="100%" />
    </td>
    <td align="center">
      <img src="results/class_sda.png" alt="Image 2" width="100%" />
    </td>
  </tr>
</table>

3. For Tanzania 
```sh
 python main.py -country=burkina_faso -algorithm=classification -tt_split=temporal 
```
The output will be found at *'../output/tanzania/results/[model]/temporal'*. 


**Note:** For each country, it may take **15-20** minutes to complete the experiment.

## 4.2 Spatio-temporal Split:

1. For Burkina Faso 
```sh
 python main.py -country=burkina_faso -algorithm=classification -tt_split=spatio-temporal 
```
The output will be found at *'../output/burkina_faso/results/[model]/spatio-temporal'*. The *[model]* directory are the following: lstm, cnn.  

2. For Rwanda
```sh
 python main.py -country=rwanda -algorithm=classification -tt_split=spatio-temporal 
```
The output will be found at *'../output/rwanda/results/[model]/spatio-temporal'*.  

3. For Tanzania 
```sh
 python main.py -country=burkina_faso -algorithm=classification -tt_split=spatio-temporal 
```
The output will be found at *'../output/tanzania/results/[model]/spatio-temporal'*. 


