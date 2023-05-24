# LLaMA-AdapterQ
Quantization of the LLaMa-Adapter model to work on consumer GPUs

**Why?**

People prosposed a novel approach to fine tune the LLaMA model here. By adding an adapter to the Attention module we can achieve better results by fine tuning it with very little propmts. The new model seems to outperform a lot of previous proposed methods. 

This is good since it is a step towards the democratization of LLMs. Now developers, researchers, and hobbists can finally fine tune their own LLMs without breaking the bank. This is especially meaningful if you take into consideration data privacy, business requirements and etc. There are many use cases that a public LLM like ChatGPT simply can't address. 

However chanlleges still remain, and when I tried to load the LLaMA Adapter model on my consumer GPUs, it ran out of memory. A used A100 GPU is selling for ~$7,500 on ebay last time I checked. Democratization of LLM means very little if most developers, researchers or hobbists can't afford it. 

The problem this Repo aims to solve is to take it further. It quantizes the LLaMA-Adapter model so it runs on my 1080Ti with 11GB memory. This means finally you will be able to do interference or fine tuning on consumer GPUs. 

Finally I use the Lit-llama's quantization code in tihs repo. So credit goes to them. 

**Set up**

**Reference**
LLaMA-Adapter model: https://github.com/ZrrSkywalker/LLaMA-Adapter
Lit-Llama model: https://github.com/Lightning-AI/lit-llama
