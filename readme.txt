supervise.py           train and test the dataset with full label
representation.py          use SimCLR learn representation without label.  encoder f+ projector g
linearvalidate.py frozen encoder f and finetune classifier layers with full label to check the performance of representation learning
weak10.py frozen encoder f and finetune classifier layers with 10% label (600 samples) to check the performance of representation learning