# PL&ML YKL Project 14
- [https://github.com/KNU-PLML-Lab/P14-PEFT-LLM-APR](https://github.com/KNU-PLML-Lab/P14-PEFT-LLM-APR)
- [https://github.com/knu-plml/apr-using-qlora](https://github.com/knu-plml/apr-using-qlora)

## Dependencies
- conda
(inherit from Defects4J 2.0.1)
- Java 1.8
- Git >= 1.9
- SVN >= 1.8
- Perl >= 5.0.12

## How to setup
- Tested env: Ubuntu 24.04 LTS
```bash
git clone [THIS_REPOSITORY]

# init submodules (Defects4J 2.0.1)
git submodule update --init --recursive

# Java 1.8
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt update
sudo apt install openjdk-8-jdk

# Defects4J
cd defects4j
sudo apt install cpanminus unzip build-essential
cpanm --installdeps .
./init.sh

# Defects4J PATH
export PATH=$PATH:$(pwd)/framework/bin

# Install Perl modules
sudo cpanm DBI
sudo cpanm DBD::SQLite

# Check Defects4J
defects4j info -p Lang
cd ..

# Init Conda
conda env update --prefix ./.conda --file env14_5.yml
conda activate ./.conda

# Run the project
python main.py --task train
```
- For more command information, please refer to `./docs/` folder.