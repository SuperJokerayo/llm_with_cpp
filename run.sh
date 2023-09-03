#!/bin/bash

# clone project
git clone https://github.com/SuperJokerayo/llm_with_cpp.git

# clone sentencepiece tokenizer
git clone https://github.com/google/sentencepiece.git ./third_party/sentencepiece/

# download checkpoint
wget -P ./checkpoints/  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

# compile
mkdir build
cd build
cmake
make -j $nproc