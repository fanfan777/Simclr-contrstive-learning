# Simlar_templete
supervise.py           train and test the dataset with full label <br>
representation.py          use SimCLR learn representation without label.  encoder f+ projector g <br>
linearvalidate.py frozen encoder f and finetune classifier layers with full label to check the performance of representation learning <br>
weak10.py frozen encoder f and finetune classifier layers with 10% label (600 samples) to check the performance of representation learning <br>
