pushd .
cd ..
mkdir -p models
cd models

git clone https://huggingface.co/codellama/CodeLlama-7b-hf CodeLlama-7b-hf
git clone https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf CodeLlama-7b-Instruct-hf

popd