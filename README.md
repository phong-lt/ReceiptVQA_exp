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
 
	# mode: train - pretrain/train models, eval - evaluate models, predict - predict trained models
	--mode train \

	# evaltype: last - evaluate lattest saved model, best - evaluate best-err saved model 
	--evaltype last \
	
	# predicttype: last - predict lattest saved model, best - predict best-err saved model 
	--predicttype best \
```