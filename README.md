To Run follow these steps

```shell
# Make venv
python -m venv ./.venv

# Activate venv
./.venv/scripts/activate

pip install -r requirements.txt

python ./train.py --task biosnap --epochs 30 --batch-size 12
```
