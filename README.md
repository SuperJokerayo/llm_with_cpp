# llm_with_cpp
LLM inference with cpp.

Only for fun.

Referenced from [llama2.c](https://github.com/leloykun/llama2.cpp/tree/master)

## Usage
1. Clone this project.

```bash
git clone https://github.com/SuperJokerayo/llm_with_cpp.git
```

2. Install [sentencepiece](https://github.com/google/sentencepiece) tokenizer.

```bash
sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev

git clone https://github.com/google/sentencepiece.git ./third_party/sentencepiece/
```

3. Download LLM checkpoint from [llama2.c](https://github.com/leloykun/llama2.cpp/tree/master) repo. For example:

```bash
# download OG model which has 15M parameters
wget -P ./checkpoints/  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```

4. Compile this project with cmake tool and run.

```bash
mkdir build
cd build
cmake
make -j $nproc
```
then, the executable file is put in `./bin`

and you can run with:

```bash
./bin/run
```

5. The config parameters are written in `config.ini`, and contum config is supported.

## License

Have a look at the [license file](./LICENSE) for details.