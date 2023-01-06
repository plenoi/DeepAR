# DeepAR
DeepAR can be widely used for performing high-throughput identification of AR antagonists in an economic manner.

## Dependency
The packages that this program depends on are <br> 
`scikit-learn==0.24.1 or higher`. <br>
`jpype1` <br>
`torch==1.9.1` <br>
`xgboost==0.90` <br> <br>

You can run following command in terminal.<br>
`pip install scikit-learn==0.24.1` <br>
`pip install jpype1` <br>
`pip install torch==1.9.1` <br>
`pip install xgboost==0.90` <br>

## How to use DeepAR
1. Copy your SMILES file into `./input` and change the name to smiles.csv<br>
2. Run command<br>
`python DeepAR.py`
3. The result including SMILE, label and probability will be saved in `./output/predicted_result.csv`
