# Game-of-Arrows


## Environment Setup
Note that different library version may affect test results. 

```
export HF_ENDPOINT="https://hf-mirror.com"

conda env create -f environment.yml

conda activate game-of-arrows1

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://mirrors.aliyun.com/pytorch-wheels/cu113/
```

## Code Structure
* **results:**  The results of attack and defense. (Please download the *xxx* weights from if you are interested in our results)
    * **train_results:** The results of training from public pretrained models
    * **arrowmatch_results**: The results of ARROWMATCH across different datasets.
* **data:** The dataset obtained by the adversary through querying the victim model accounts for 1% of all the training data.
* **utils:** The functions in *ARROWMATCH* & *ARROWCLOAK*
* **code:**  
    * evaluate_model.py
    * arrowmatch.py
    * blackbox_test.py
    * arrowcloak.py
* **train:**
    * train.py
* **tee:** The code for sgx experiments



## Experiments

**Train public model**
```
# Train
./train.sh --gpus 2,3 --dataset mnli --output_dir "results/train_results
```

**Eval model:** 
Select the different parameters and run the following scripts to evaluate the results.
```
./evaluate_model.sh --dataset "sst2" --obfus "translinkguard" --gpus 0,1              #for ARROWMATCH results
 
 ./evaluate_model.sh --dataset "sst2" --obfus "none"  --gpus 0,1                      #for training results

```

**Try ARROWMATCH:** 
Select different datasets and different obfuscation methods to verify the effectiveness of *ARROWMATCH*.

Make sure the training results from public have been saved. 

```
./arrowmatch.sh --gpus 0,1 --dataset sst2  --obfus translinkguard
```

**Try ARROWCLOAK:** Select different datasets to verify the defense of *ARROWCLOAK*.

Make sure the training results from public have been saved. 

```
./arrowcloak.sh --gpus 0,1 --dataset sst2
```

**Test black-box baseline:** The results of finetuning public model with recovery dataset, which represents the situation that adversary can not get any private information.
```
./blackbox_test.sh --gpus 0,1 --dataset sst2 --obfus translinkguard
```

**Try SGX experiment**

**NOTE**: The code can only run on a machine with SGX hardware, so please **Make sure your hardware supports SGX.**

```
cd tee

make 

./tee_code/run.sh
```
