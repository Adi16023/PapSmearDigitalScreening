
## Requirements

To install the required dependencies run the following using the Command Prompt:

`pip install -r requirements.txt`

# Implementing the code for Cervical Cytology data

1. Herlev Pap Smear dataset by Jantzen et al.: http://mde-lab.aegean.gr/index.php/downloads  
2. Mendeley Liquid Based Cytology dataset by Hussain
et al.: https://data.mendeley.com/datasets/zddtpgzv63/4    
3. SIPaKMeD Pap Smear dataset by Plissiti et al.: https://www.cs.uoi.gr/~marina/sipakmed.html  

Structure the directory as follows:

```

.
+-- data
|   +-- .
|   +-- train
|   +-- val
+-- extract_features.py
+-- fitnessFUNs.py
+-- GWO.py
+-- main.py
+-- resnet50.csv
+-- selector.py
+-- solution.py
+-- transfer_functions_benchmark.py

```

To extract ResNet-50 features run the following script:

`python extract_features.py`

Similarly the script can be modified for extracting features from other models.

Run the following code for the feature set optimization:

`python main.py --num_csv 2`

Set `num_csv` to the number of features csv files you have. You will be asked to enter the names of the csv files upon executing the above code. Execute `python main.py -h` to get the details of all the available arguments.
