# ReceiptVQA_exp
## Setup

1. Clone the repository:
    ```
    git clone https://github.com/phong-lt/ReceiptVQA_exp
    ```
2. Install the required packages:
    ```
    pip install -r /ReceiptVQA_exp/requirements.txt
    ```

## Usage

To run the main script:
```bash
python ReceiptVQA_exp/run.py \
	# config file path
	--config-file ReceiptVQA_exp/config/latr.yaml \
 
	# mode: train - to train models, eval - to evaluate models, predict - to predict trained models
	--mode train \

	# evaltype: last - evaluate lattest saved model, best - evaluate best-score saved model 
	--evaltype last \
	
	# predicttype: last - predict lattest saved model, best - predict best-score saved model 
	--predicttype best \
```

## Contact

- Thanh-Phong Le: [21520395@gm.uit.edu.vn](mailto:21520395@gm.uit.edu.vn)