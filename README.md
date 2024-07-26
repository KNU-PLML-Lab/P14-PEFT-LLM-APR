# PL&ML YKL Project 14
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
```