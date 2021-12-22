# Fraud Detection Project
IEEE-CIS Fraud Detection challenge was first hosted by Kaggle in 2019. The idea was for competitors to develop a model 
to detect fraud from customer transactions. While IEEE-CIS already have a fraud prevention system in place, researchers
were looking for ways to improve the current figure being saved by the system, and improve the customer experience.

## Usage
Clone this repository to your computer. 
To view explorations navigate to the project directory cd IEEE-CIS Fraud Detection from 
your terminal then cd into the `notebooks` directory. This directory contains data analysis
and the pipeline we converted into a package. To run the notebooks, you'll have
to install the [data](https://www.kaggle.com/c/ieee-fraud-detection/data) into a directory
called data. The directory must live at the same level as the `notebooks` and `packages`
directory. 

To use the sample the deployed model locally through the API, navigate to the project 
directory from your terminal then cd into `packages/fraud_detection_api`. From here, 
run the following command: 
`py -m tox -e run`
This will create a localhost link, simply click it or copy and paste it into your 
browser. Then select the docs option and go to the `predict` heading. There is already
an example instance there, but you may play around with the values.

## Extending This Work
Some ideas to extend this work:
- Replace the model 
- Add monitoring 

