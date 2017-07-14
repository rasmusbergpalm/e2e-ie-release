# End-to-End Information Extraction without Token-Level Supervision

Code for the paper End-to-End Information Extraction without Token-Level Supervision

The results presented in the paper can be found in emnlp-results

### Installing dependencies

 * `pip install -r requirements.txt`

### Training and evaluating the multi-output pointer net models

 * `python train.py <atis|movie|restaurant>` will train a model. The best model (evaluated on the val set) will be saved as best.npz in the work dir
 * `python test.py  <atis|movie|restaurant>` will load best.npz and evaluate the model on the test set

### Training and evaluating the Bi-LSTM baseline

 * Set the dataset in `baselines/bilstm/runner.py`
 * `python baselines/bilstm/runner.py` trains a model, evaluate on val set, and for every improvement evaluate on test set.

### Boostrap sampling

 * `python evaluate.py actual1.json actual2.json expected.json` will run bootstrap sampling to estimate p-values for actual1.json vs. actual2.json

