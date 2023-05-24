# LLaMA-AdapterQ
Quantization of the LLaMa-Adapter model to work on consumer GPUs

**Why?**

People prosposed a novel approach to fine tune the LLaMA model here. By adding an adapter to the Attention module we can achieve better results by fine tuning it with very little propmts. The new model seems to outperform a lot of previous proposed methods. 

This is good since it is a step towards the democratization of LLMs. Now developers, researchers, and hobbists can finally fine tune their own LLMs without breaking the bank. This is especially meaningful if you take into consideration data privacy, properatary data, data businesses or people won't or can't share with OpenAI or any other companies. There are many use cases that a public LLM like ChatGPT simply can't address. 

However chanlleges still remain, one of which is GPU requirements. When I tried to load the LLaMA Adapter model on my consumer GPUs, it ran out of memory. A used A100 GPU is selling for ~$7,500 on ebay last time I checked. 4 of them would cost as much as Toyato Corolla. Democratization of LLM means very little if most developers, researchers or hobbists can't afford it. 

The problem this Repo aims to solve is to take it further. It quantizes the LLaMA-Adapter model so it runs on my 1080Ti with 11GB memory. This means finally you will be able to do interference or fine tuning LLMs on consumer GPUs. 

Finally I use the Lit-llama's quantization code in tihs repo. So credit goes to them. 

**Set up**

`git clone https://github.com/davyuan/LLaMA-AdapterQ
cd LLaMA-AdapterQ
pip install -r requirements`

You are all set! 🎉

**Quantize the LLaMA 7B model**

First you will need to download the LLaMA weights first. YOu can refer to the doc at /howto/download_weights.md for more information.
Then you can run the following to quantize it to 8-bit in my example.

`python quantize/gptq.py --checkpoint_path lit-llama.pth --tokenizer_path tokenizer.model --output_path llama-7b-gptq.8bit.pth --dtype bfloat16  --quantize gptq.int8
`

**Run the inference**

With the quantized model, we will able to load it onto a consumer GPU. Now the last thing we need is the weights from the LLaMA-Adapter model, which can be downloaded from:
https://github.com/ZrrSkywalker/LLaMA-Adapter/releases/download/v.1.0.0/llama_adapter_len10_layer30_release.pth

`python generate.py --quantize gptq.int8 --prompt "who is Athena?" --checkpoint_path llama-7b-gptq.8bit.pth --tokenizer_path checkpoints/lit-llama/tokenizer.model --adapter_path ../LLaMA-Adapter/adapter/llama_adapter_len10_layer30_release.pth`

`Loading model ...

Time to load model: 5.46 seconds.

Global seed set to 1234

who is Athena?
The goddess of wisdom, war, courage, strategy, and heroic endeavour, and courage. Athenaeducing, creativity, cleverness, justice, and battle strategic, she is often associated with anadorn

Time for inference 1: 73.85 sec total, 0.68 tokens/sec
Memory used: 9.06 GB`

**Reference**

LLaMA-Adapter model: https://github.com/ZrrSkywalker/LLaMA-Adapter

Lit-Llama model: https://github.com/Lightning-AI/lit-llama
