# Automated Program Repair Using Quantized Language Models and Parameter-Efficient Fine-Tuning
- [https://github.com/KNU-PLML-Lab/P14-PEFT-LLM-APR](https://github.com/KNU-PLML-Lab/P14-PEFT-LLM-APR)
- [https://github.com/knu-plml/apr-using-qlora](https://github.com/knu-plml/apr-using-qlora)

## Dependencies
- conda
(inherit from Defects4J 2.0.1)
- Java 1.8
- Git >= 1.9
- SVN >= 1.8
- Perl >= 5.0.12

## How to start
- My env: Win11 > wsl2 > Ubuntu 24.04 LTS
```bash
git clone [THIS_REPOSITORY]
# init submodules (Defects4J 2.0.1)
git submodule update --init --recursive

# Download Java 1.8
# Manual download: https://www.oracle.com/java/technologies/downloads/?er=221886#java8
# or
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt update
sudo apt install openjdk-8-jdk

# Init Defects4J
cd defects4j
sudo apt install cpanminus unzip build-essential
# Local cpanm binary (https://metacpan.org/pod/App::cpanminus#Downloading-the-standalone-executable)
#curl -L https://cpanmin.us/ -o cpanm
#chmod +x cpanm
# For Local
#cpanm --local-lib=~/perl5 local::lib && eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
cpanm --installdeps .
./init.sh

# Register Defects4J PATH
export PATH=$PATH:$(pwd)/framework/bin
# ... or add to .bashrc
#echo 'export PATH="$PATH:'$(pwd)'/framework/bin"' >> ~/.bashrc
#source ~/.bashrc

# Install Perl modules
sudo cpanm DBI
sudo cpanm DBD::SQLite


# Check Defects4J
defects4j info -p Lang
cd ..

# Init Conda
conda env update --prefix ./.conda --file environment.yml
conda activate ./.conda

# Run the project
python main.py --task train
```

## Additional tools & Notes
- https://bitbucket.org/rjust/fault-localization-data/src/master/

```bash
# Sample Defects4J (Lang project, 1st bug)
defects4j checkout -p Lang -v 1b -w Lang_1_buggy
defects4j checkout -p Lang -v 1f -w Lang_1_fixed

###### jasper init
cd clm/jasper
mkdir target
javac -cp ".:lib/*" -d target src/main/java/clm/jasper/*.java src/main/java/clm/codet5/*.java src/main/java/clm/codegen/*.java src/main/java/clm/plbart/*.java src/main/java/clm/incoder/*.java src/main/java/clm/finetuning/*.java
```

extra...
```bash
# for matplotlib
sudo apt install libxcb-cursor0
```

## Reference
- https://llama.meta.com/docs/how-to-guides/fine-tuning/
- https://github.com/microsoft/CodeXGLUE


## Memo
```
conda env update --name p14 --file environment.yml
pip install git+https://github.com/huggingface/transformers
pip install git+https://github.com/bitsandbytes-foundation/bitsandbytes
pip install transformers -U
pip install 'accelerate>=0.26.0'
pip install 'bitsandbytes>=0.41.3'
pip install flash-attn --no-build-isolation --no-cache-dir
pip install fuzzywuzzy
pip install bitsandbytes --no-build-isolation --no-cache-dir
pip install peft --no-build-isolation --no-cache-dir
pip install sentencepiece
pip install protobuf

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub
sudo add-apt-repository 'deb https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/ /'
sudo apt-get update
sudo apt-get -y install cuda
sudo apt install nvidia-cuda-toolkit
nvcc --version

ln -s /usr/lib/wsl/lib/libcuda.so /home/stgr/miniconda3/envs/p14_3/lib/libcuda.so
```

### 오리지널 codegen
```bash
cd clm-apr/codegen_finetune

# --max_steps 1875 \ --num_train_epochs 1 \
# --eval_steps 187 \ 
CUDA_VISIBLE_DEVICES=0 python finetune.py

```

### apex 설치
```bash
git clone https://github.com/NVIDIA/apex
cd apex
python setup.py install
```
