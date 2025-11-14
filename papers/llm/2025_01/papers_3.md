# llm - 2025_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14291v1">Advances in Temporal Point Processes: Bayesian, Deep, and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Temporal point processes (TPPs) are stochastic process models used to characterize event sequences occurring in continuous time. Traditional statistical TPPs have a long-standing history, with numerous models proposed and successfully applied across diverse domains. In recent years, advances in deep learning have spurred the development of neural TPPs, enabling greater flexibility and expressiveness in capturing complex temporal dynamics. The emergence of large language models (LLMs) has further sparked excitement, offering new possibilities for modeling and analyzing event sequences by leveraging their rich contextual understanding. This survey presents a comprehensive review of recent research on TPPs from three perspectives: Bayesian, deep learning, and LLM approaches. We begin with a review of the fundamental concepts of TPPs, followed by an in-depth discussion of model design and parameter estimation techniques in these three frameworks. We also revisit classic application areas of TPPs to highlight their practical relevance. Finally, we outline challenges and promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14275v1">Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Advances in Large Language Models (LLMs) have sparked interest in their ability to solve Olympiad-level math problems. However, the training and evaluation of these models are constrained by the limited size and quality of available datasets, as creating large-scale data for such advanced problems requires extensive effort from human experts. In addition, current benchmarks are prone to contamination, leading to unreliable evaluations. In this paper, we present an automated pipeline that leverages the rich resources of the Art of Problem Solving (AoPS) forum, which predominantly features Olympiad-level problems and community-driven solutions. Using open-source LLMs, we develop a method to extract question-answer pairs from the forum, resulting in AoPS-Instruct, a dataset of more than 600,000 high-quality QA pairs. Our experiments demonstrate that fine-tuning LLMs on AoPS-Instruct improves their reasoning abilities across various benchmarks. Moreover, we build an automatic pipeline that introduces LiveAoPSBench, an evolving evaluation set with timestamps, derived from the latest forum data, providing a contamination-resistant benchmark for assessing LLM performance. Notably, we observe a significant decline in LLM performance over time, suggesting their success on older examples may stem from pre-training exposure rather than true reasoning ability. Our work presents a scalable approach to creating and maintaining large-scale, high-quality datasets for advanced math reasoning, offering valuable insights into the capabilities and limitations of LLMs in this domain. Our benchmark and code is available at https://github.com/DSL-Lab/aops
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12612v2">Auto-Cypher: Improving LLMs on Cypher generation via LLM-supervised generation-verification framework</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 Accepted at NAACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Graph databases like Neo4j are gaining popularity for handling complex, interconnected data, over traditional relational databases in modeling and querying relationships. While translating natural language into SQL queries is well-researched, generating Cypher queries for Neo4j remains relatively underexplored. In this work, we present an automated, LLM-Supervised, pipeline to generate high-quality synthetic data for Text2Cypher. Our Cypher data generation pipeline introduces LLM-As-Database-Filler, a novel strategy for ensuring Cypher query correctness, thus resulting in high quality generations. Using our pipeline, we generate high quality Text2Cypher data - SynthCypher containing 29.8k instances across various domains and queries with varying complexities. Training open-source LLMs like LLaMa-3.1-8B, Mistral-7B, and QWEN-7B on SynthCypher results in performance gains of up to 40% on the Text2Cypher test split and 30% on the SPIDER benchmark, adapted for graph databases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12692v2">SparseInfer: Training-free Prediction of Activation Sparsity for Fast LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Leveraging sparsity is crucial for optimizing large language model inference. however, modern LLMs employing SiLU as their activation function exhibit minimal activation sparsity. Recent research has proposed replacing SiLU with ReLU to induce significant activation sparsity and showed no downstream task accuracy degradation through fine tuning. However, taking full advantage of it required training a predictor to estimate this sparsity. In this paper, we introduce SparseInfer, a simple, light weight, and training free predictor for activation sparsity of ReLU field LLMs, in which activation sparsity is predicted by comparing only the sign bits of inputs and weights. To compensate for possible prediction inaccuracy, an adaptive tuning of the predictor's conservativeness is enabled, which can also serve as a control knob for optimizing LLM inference. The proposed method achieves approximately faster inference speed over the state of the art, with negligible accuracy loss of within 1%p.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14205v1">Serving Long-Context LLMs at the Mobile Edge: Test-Time Reinforcement Learning-based Model Caching and Inference Offloading</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can perform zero-shot learning on unseen tasks and few-shot learning on complex reasoning tasks. However, resource-limited mobile edge networks struggle to support long-context LLM serving for LLM agents during multi-round interactions with users. Unlike stateless computation offloading and static service offloading in edge computing, optimizing LLM serving at edge servers is challenging because LLMs continuously learn from context which raises accuracy, latency, and resource consumption dynamics. In this paper, we propose a joint model caching and inference offloading framework that utilizes test-time deep reinforcement learning (T2DRL) to optimize deployment and execution strategies for long-context LLM serving. In this framework, we analyze the performance convergence and design an optimization problem considering the utilization of context windows in LLMs. Furthermore, the T2DRL algorithm can learn in both the training phase and the testing phase to proactively manage cached models and service requests and adapt to context changes and usage patterns during execution. To further enhance resource allocation efficiency, we propose a double Dutch auction (DDA) mechanism, which dynamically matches supply and demand while maximizing social welfare. Finally, experimental results demonstrate that the T2DRL algorithm can reduce system costs by at least 30% compared to baselines while guaranteeing the performance of LLM agents in real-world perception and reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16376v1">HWPQ: Hessian-free Weight Pruning-Quantization For LLM Compression And Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-01-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across numerous domains. However, the high time complexity of existing pruning and quantization methods significantly hinders their effective deployment on resource-constrained consumer or edge devices. In this study, we propose a novel Hessian-free Weight Pruning-Quantization (HWPQ) method. HWPQ eliminates the need for computationally intensive Hessian matrix calculations by introducing a contribution-based weight metric, which evaluates the importance of weights without relying on second-order derivatives. Additionally, we employ the Exponentially Weighted Moving Average (EWMA) technique to bypass weight sorting, enabling the selection of weights that contribute most to LLM accuracy and further reducing time complexity. Our approach is extended to support 2:4 structured sparsity pruning, facilitating efficient execution on modern hardware accelerators. Experimental results demonstrate that HWPQ significantly enhances the compression performance of LLaMA2. Compared to state-of-the-art quantization and pruning frameworks, HWPQ achieves average speedups of 5.97x (up to 20.75x) in quantization time and 12.29x (up to 56.02x) in pruning time, while largely preserving model accuracy. Furthermore, we observe a 1.50x inference speedup compared to the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06252v3">Transformer-Squared: Self-adaptive LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 To appear at the 13th International Conference on Learning Representations (ICLR 2025)
    </div>
    <details class="paper-abstract">
      Self-adaptive large language models (LLMs) aim to solve the challenges posed by traditional fine-tuning methods, which are often computationally intensive and static in their ability to handle diverse tasks. We introduce Transformer-Squared, a novel self-adaptation framework that adapts LLMs for unseen tasks in real-time by selectively adjusting only the singular components of their weight matrices. During inference, Transformer-Squared employs a two-pass mechanism: first, a dispatch system identifies the task properties, and then task-specific 'expert' vectors, trained using reinforcement learning, are dynamically mixed to obtain targeted behavior for the incoming prompt. Our method consistently outperforms ubiquitous approaches such as LoRA, with fewer parameters and greater efficiency. Furthermore, Transformer-Squared demonstrates versatility across different LLM architectures and modalities, including vision-language tasks. Transformer-Squared represents a significant leap forward, offering a scalable, efficient solution for enhancing the adaptability and task-specific performance of LLMs, paving the way for truly dynamic, self-organizing AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10989v3">Liger Kernel: Efficient Triton Kernels for LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 17 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) efficiently at scale presents a formidable challenge, driven by their ever-increasing computational demands and the need for enhanced performance. In this work, we introduce Liger-Kernel, an open-sourced set of Triton kernels developed specifically for LLM training. With kernel optimization techniques like kernel operation fusing and input chunking, our kernels achieve on average a 20% increase in training throughput and a 60% reduction in GPU memory usage for popular LLMs compared to HuggingFace implementations. In addition, Liger-Kernel is designed with modularity, accessibility, and adaptability in mind, catering to both casual and expert users. Comprehensive benchmarks and integration tests are built in to ensure compatibility, performance, correctness, and convergence across diverse computing environments and model architectures. The source code is available under a permissive license at: github.com/linkedin/Liger-Kernel.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.18617v1">DarkMind: Latent Chain-of-Thought Backdoor in Customized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-24
      | 💬 21 pages, 9 figures, 13 tables
    </div>
    <details class="paper-abstract">
      With the growing demand for personalized AI solutions, customized LLMs have become a preferred choice for businesses and individuals, driving the deployment of millions of AI agents across various platforms, e.g., GPT Store hosts over 3 million customized GPTs. Their popularity is partly driven by advanced reasoning capabilities, such as Chain-of-Thought, which enhance their ability to tackle complex tasks. However, their rapid proliferation introduces new vulnerabilities, particularly in reasoning processes that remain largely unexplored. We introduce DarkMind, a novel backdoor attack that exploits the reasoning capabilities of customized LLMs. Designed to remain latent, DarkMind activates within the reasoning chain to covertly alter the final outcome. Unlike existing attacks, it operates without injecting triggers into user queries, making it a more potent threat. We evaluate DarkMind across eight datasets covering arithmetic, commonsense, and symbolic reasoning domains, using five state-of-the-art LLMs with five distinct trigger implementations. Our results demonstrate DarkMind effectiveness across all scenarios, underscoring its impact. Finally, we explore potential defense mechanisms to mitigate its risks, emphasizing the need for stronger security measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14876v2">Precise and Robust Sidewalk Detection: Leveraging Ensemble Learning to Surpass LLM Limitations in Urban Environments</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      This study aims to compare the effectiveness of a robust ensemble model with the state-of-the-art ONE-PEACE Large Language Model (LLM) for accurate detection of sidewalks. Accurate sidewalk detection is crucial in improving road safety and urban planning. The study evaluated the model's performance on Cityscapes, Ade20k, and the Boston Dataset. The results showed that the ensemble model performed better than the individual models, achieving mean Intersection Over Union (mIOU) scores of 93.1\%, 90.3\%, and 90.6\% on these datasets under ideal conditions. Additionally, the ensemble model maintained a consistent level of performance even in challenging conditions such as Salt-and-Pepper and Speckle noise, with only a gradual decrease in efficiency observed. On the other hand, the ONE-PEACE LLM performed slightly better than the ensemble model in ideal scenarios but experienced a significant decline in performance under noisy conditions. These findings demonstrate the robustness and reliability of the ensemble model, making it a valuable asset for improving urban infrastructure related to road safety and curb space management. This study contributes positively to the broader context of urban health and mobility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13299v1">Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted in NAACL 2025
    </div>
    <details class="paper-abstract">
      Materials discovery and design are essential for advancing technology across various industries by enabling the development of application-specific materials. Recent research has leveraged Large Language Models (LLMs) to accelerate this process. We explore the potential of LLMs to generate viable hypotheses that, once validated, can expedite materials discovery. Collaborating with materials science experts, we curated a novel dataset from recent journal publications, featuring real-world goals, constraints, and methods for designing real-world applications. Using this dataset, we test LLM-based agents that generate hypotheses for achieving given goals under specific constraints. To assess the relevance and quality of these hypotheses, we propose a novel scalable evaluation metric that emulates the process a materials scientist would use to evaluate a hypothesis critically. Our curated dataset, proposed method, and evaluation framework aim to advance future research in accelerating materials discovery and design with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02076v7">LongGenBench: Benchmarking Long-Form Generation in Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 ICLR 2025; Github: https://github.com/mozhu621/LongGenBench/
    </div>
    <details class="paper-abstract">
      Current benchmarks like Needle-in-a-Haystack (NIAH), Ruler, and Needlebench focus on models' ability to understand long-context input sequences but fail to capture a critical dimension: the generation of high-quality long-form text. Applications such as design proposals, technical documentation, and creative writing rely on coherent, instruction-following outputs over extended sequences - a challenge that existing benchmarks do not adequately address. To fill this gap, we introduce LongGenBench, a novel benchmark designed to rigorously evaluate large language models' (LLMs) ability to generate long text while adhering to complex instructions. Through tasks requiring specific events or constraints within generated text, LongGenBench evaluates model performance across four distinct scenarios, three instruction types, and two generation-lengths (16K and 32K tokens). Our evaluation of ten state-of-the-art LLMs reveals that, despite strong results on Ruler, all models struggled with long text generation on LongGenBench, particularly as text length increased. This suggests that current LLMs are not yet equipped to meet the demands of real-world, long-form text generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06394v4">GameArena: Evaluating LLM Reasoning through Live Computer Games</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Evaluating the reasoning abilities of large language models (LLMs) is challenging. Existing benchmarks often depend on static datasets, which are vulnerable to data contamination and may get saturated over time, or on binary live human feedback that conflates reasoning with other abilities. As the most prominent dynamic benchmark, Chatbot Arena evaluates open-ended questions in real-world settings, but lacks the granularity in assessing specific reasoning capabilities. We introduce GameArena, a dynamic benchmark designed to evaluate LLM reasoning capabilities through interactive gameplay with humans. GameArena consists of three games designed to test specific reasoning capabilities (e.g., deductive and inductive reasoning), while keeping participants entertained and engaged. We analyze the gaming data retrospectively to uncover the underlying reasoning processes of LLMs and measure their fine-grained reasoning capabilities. We collect over 2000 game sessions and provide detailed assessments of various reasoning capabilities for five state-of-the-art LLMs. Our user study with 100 participants suggests that GameArena improves user engagement compared to Chatbot Arena. For the first time, GameArena enables the collection of step-by-step LLM reasoning data in the wild.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.07341v2">A Guide To Effectively Leveraging LLMs for Low-Resource Text Summarization: Data Augmentation and Semi-supervised Approaches</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted to NAACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Existing approaches for low-resource text summarization primarily employ large language models (LLMs) like GPT-3 or GPT-4 at inference time to generate summaries directly; however, such approaches often suffer from inconsistent LLM outputs and are difficult to adapt to domain-specific data in low-resource scenarios. In this work, we propose two novel methods to effectively utilize LLMs for low-resource text summarization: 1) MixSumm, an LLM-based data augmentation regime that synthesizes high-quality documents (short and long) for few-shot text summarization, and 2) PPSL, a prompt-based pseudolabeling strategy for sample-efficient semi-supervised text summarization. Specifically, MixSumm leverages the open-source LLaMA-3-70b-Instruct model to generate new documents by mixing topical information derived from a small seed set, and PPSL leverages the LLaMA-3-70b-Instruct model to generate high-quality pseudo-labels in a semi-supervised learning setup. We evaluate our methods on the TweetSumm, WikiHow, and ArXiv/PubMed datasets and use L-Eval, a LLaMA-3-based evaluation metric, and ROUGE scores to measure the quality of generated summaries. Our experiments on extractive and abstractive summarization show that MixSumm and PPSL achieve competitive ROUGE scores as a fully supervised method with 5% of the labeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11747v2">Optimizing Pretraining Data Mixtures with LLM-Estimated Utility</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models improve with increasing amounts of high-quality training data. However, leveraging larger datasets requires balancing quality, quantity, and diversity across sources. After evaluating nine baseline methods under both compute- and data-constrained scenarios, we find token-count heuristics outperform manual and learned mixes, indicating that simple approaches accounting for dataset size and diversity are surprisingly effective. Building on this insight, we propose two complementary approaches: UtiliMax, which extends token-based heuristics by incorporating utility estimates from reduced-scale ablations, achieving up to a 10.6x speedup over manual baselines; and Model Estimated Data Utility (MEDU), which leverages LLMs to estimate data utility from small samples, matching ablation-based performance while reducing computational requirements by $\sim$200x. Together, these approaches establish a new framework for automated, compute-efficient data mixing that is robust across training regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14073v1">LLMs are Vulnerable to Malicious Prompts Disguised as Scientific Language</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) have been deployed in various real-world settings, concerns about the harm they may propagate have grown. Various jailbreaking techniques have been developed to expose the vulnerabilities of these models and improve their safety. This work reveals that many state-of-the-art proprietary and open-source LLMs are vulnerable to malicious requests hidden behind scientific language. Specifically, our experiments with GPT4o, GPT4o-mini, GPT-4, LLama3-405B-Instruct, Llama3-70B-Instruct, Cohere, Gemini models on the StereoSet data demonstrate that, the models' biases and toxicity substantially increase when prompted with requests that deliberately misinterpret social science and psychological studies as evidence supporting the benefits of stereotypical biases. Alarmingly, these models can also be manipulated to generate fabricated scientific arguments claiming that biases are beneficial, which can be used by ill-intended actors to systematically jailbreak even the strongest models like GPT. Our analysis studies various factors that contribute to the models' vulnerabilities to malicious requests in academic language. Mentioning author names and venues enhances the persuasiveness of some models, and the bias scores can increase as dialogues progress. Our findings call for a more careful investigation on the use of scientific data in the training of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13387v3">CLEAR: Towards Contextual LLM-Empowered Privacy Policy Analysis and Risk Generation for Large Language Model Applications</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      The rise of end-user applications powered by large language models (LLMs), including both conversational interfaces and add-ons to existing graphical user interfaces (GUIs), introduces new privacy challenges. However, many users remain unaware of the risks. This paper explores methods to increase user awareness of privacy risks associated with LLMs in end-user applications. We conducted five co-design workshops to uncover user privacy concerns and their demand for contextual privacy information within LLMs. Based on these insights, we developed CLEAR (Contextual LLM-Empowered Privacy Policy Analysis and Risk Generation), a just-in-time contextual assistant designed to help users identify sensitive information, summarize relevant privacy policies, and highlight potential risks when sharing information with LLMs. We evaluated the usability and usefulness of CLEAR across two example domains: ChatGPT and the Gemini plugin in Gmail. Our findings demonstrated that CLEAR is easy to use and improves users' understanding of data practices and privacy risks. We also discussed LLM's duality in posing and mitigating privacy risks, offering design and policy implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14046v1">LLM-guided Instance-level Image Manipulation with Diffusion U-Net Cross-Attention Maps</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Presented at BMVC 2024
    </div>
    <details class="paper-abstract">
      The advancement of text-to-image synthesis has introduced powerful generative models capable of creating realistic images from textual prompts. However, precise control over image attributes remains challenging, especially at the instance level. While existing methods offer some control through fine-tuning or auxiliary information, they often face limitations in flexibility and accuracy. To address these challenges, we propose a pipeline leveraging Large Language Models (LLMs), open-vocabulary detectors, cross-attention maps and intermediate activations of diffusion U-Net for instance-level image manipulation. Our method detects objects mentioned in the prompt and present in the generated image, enabling precise manipulation without extensive training or input masks. By incorporating cross-attention maps, our approach ensures coherence in manipulated images while controlling object positions. Our method enables precise manipulations at the instance level without fine-tuning or auxiliary information such as masks or bounding boxes. Code is available at https://github.com/Palandr123/DiffusionU-NetLLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13912v1">Analysis of Indic Language Capabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 17 pages, 2 figures, 5 tables
    </div>
    <details class="paper-abstract">
      This report evaluates the performance of text-in text-out Large Language Models (LLMs) to understand and generate Indic languages. This evaluation is used to identify and prioritize Indic languages suited for inclusion in safety benchmarks. We conduct this study by reviewing existing evaluation studies and datasets; and a set of twenty-eight LLMs that support Indic languages. We analyze the LLMs on the basis of the training data, license for model and data, type of access and model developers. We also compare Indic language performance across evaluation datasets and find that significant performance disparities in performance across Indic languages. Hindi is the most widely represented language in models. While model performance roughly correlates with number of speakers for the top five languages, the assessment after that varies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03797v2">NESTFUL: A Benchmark for Evaluating LLMs on Nested Sequences of API Calls</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      The resurgence of autonomous agents built using large language models (LLMs) to solve complex real-world tasks has brought increased focus on LLMs' fundamental ability of tool or function calling. At the core of these agents, an LLM must plan, execute, and respond using external tools, APIs, and custom functions. Research on tool calling has gathered momentum, but evaluation benchmarks and datasets representing the complexity of the tasks have lagged behind. In this work, we focus on one such complexity, nested sequencing, with the goal of extending existing benchmarks and evaluation. Specifically, we present NESTFUL, a benchmark to evaluate LLMs on nested sequences of API calls, i.e., sequences where the output of one API call is passed as input to a subsequent call. NESTFUL contains 1800+ nested sequences where all the function calls are executable. Experimental results on multiple models and settings show that the best-performing model on the dataset has a full sequence match accuracy of 25% and win-rate of 34% necessitating a large scope for improvement in the nested sequencing aspect of function calling. Our analysis of these results provides possible future research directions for the community, in addition to a benchmark to track progress. We have released the NESTFUL dataset under the Apache 2.0 license at https://github.com/IBM/NESTFUL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13884v1">Exploring Finetuned Audio-LLM on Heart Murmur Features</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 5 pages, 1 figure, and 3 tables. Submitted to IEEE/ACM Conference on Connected Health: Applications, Systems , and Engineering Technologies
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) for audio have excelled in recognizing and analyzing human speech, music, and environmental sounds. However, their potential for understanding other types of sounds, particularly biomedical sounds, remains largely underexplored despite significant scientific interest. In this study, we focus on diagnosing cardiovascular diseases using phonocardiograms, i.e., heart sounds. Most existing deep neural network (DNN) paradigms are restricted to heart murmur classification (healthy vs unhealthy) and do not predict other acoustic features of the murmur such as timing, grading, harshness, pitch, and quality, which are important in helping physicians diagnose the underlying heart conditions. We propose to finetune an audio LLM, Qwen2-Audio, on the PhysioNet CirCor DigiScope phonocardiogram (PCG) dataset and evaluate its performance in classifying 11 expert-labeled murmur features. Additionally, we aim to achieve more noise-robust and generalizable system by exploring a preprocessing segmentation algorithm using an audio representation model, SSAMBA. Our results indicate that the LLM-based model outperforms state-of-the-art methods in 8 of the 11 features and performs comparably in the remaining 3. Moreover, the LLM successfully classifies long-tail murmur features with limited training data, a task that all previous methods have failed to classify. These findings underscore the potential of audio LLMs as assistants to human cardiologists in enhancing heart disease diagnosis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19599v3">Take Caution in Using LLMs as Human Surrogates: Scylla Ex Machina</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Recent studies suggest large language models (LLMs) can exhibit human-like reasoning, aligning with human behavior in economic experiments, surveys, and political discourse. This has led many to propose that LLMs can be used as surrogates or simulations for humans in social science research. However, LLMs differ fundamentally from humans, relying on probabilistic patterns, absent the embodied experiences or survival objectives that shape human cognition. We assess the reasoning depth of LLMs using the 11-20 money request game. Nearly all advanced approaches fail to replicate human behavior distributions across many models. Causes of failure are diverse and unpredictable, relating to input language, roles, and safeguarding. These results advise caution when using LLMs to study human behavior or as surrogates or simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13802v1">Enhancing LLMs for Governance with Human Oversight: Evaluating and Aligning LLMs on Expert Classification of Climate Misinformation for Detecting False or Misleading Claims about Climate Change</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted to the AI Governance Workshop at AAAI 2025
    </div>
    <details class="paper-abstract">
      Climate misinformation is a problem that has the potential to be substantially aggravated by the development of Large Language Models (LLMs). In this study we evaluate the potential for LLMs to be part of the solution for mitigating online dis/misinformation rather than the problem. Employing a public expert annotated dataset and a curated sample of social media content we evaluate the performance of proprietary vs. open source LLMs on climate misinformation classification task, comparing them to existing climate-focused computer-assisted tools and expert assessments. Results show (1) state-of-the-art (SOTA) open-source models substantially under-perform in classifying climate misinformation compared to proprietary models, (2) existing climate-focused computer-assisted tools leveraging expert-annotated datasets continues to outperform many of proprietary models, including GPT-4o, and (3) demonstrate the efficacy and generalizability of fine-tuning GPT-3.5-turbo on expert annotated dataset in classifying claims about climate change at the equivalency of climate change experts with over 20 years of experience in climate communication. These findings highlight 1) the importance of incorporating human-oversight, such as incorporating expert-annotated datasets in training LLMs, for governance tasks that require subject-matter expertise like classifying climate misinformation, and 2) the potential for LLMs in facilitating civil society organizations to engage in various governance tasks such as classifying false or misleading claims in domains beyond climate change such as politics and health science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13778v1">Explainable XR: Understanding User Behaviors of XR Environments using LLM-assisted Analytics Framework</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 11 pages, 8 figures. This is the author's version of the article that has been accepted for publication in IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      We present Explainable XR, an end-to-end framework for analyzing user behavior in diverse eXtended Reality (XR) environments by leveraging Large Language Models (LLMs) for data interpretation assistance. Existing XR user analytics frameworks face challenges in handling cross-virtuality - AR, VR, MR - transitions, multi-user collaborative application scenarios, and the complexity of multimodal data. Explainable XR addresses these challenges by providing a virtuality-agnostic solution for the collection, analysis, and visualization of immersive sessions. We propose three main components in our framework: (1) A novel user data recording schema, called User Action Descriptor (UAD), that can capture the users' multimodal actions, along with their intents and the contexts; (2) a platform-agnostic XR session recorder, and (3) a visual analytics interface that offers LLM-assisted insights tailored to the analysts' perspectives, facilitating the exploration and analysis of the recorded XR session data. We demonstrate the versatility of Explainable XR by demonstrating five use-case scenarios, in both individual and collaborative XR applications across virtualities. Our technical evaluation and user studies show that Explainable XR provides a highly usable analytics solution for understanding user actions and delivering multifaceted, actionable insights into user behaviors in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14804v2">Can LLMs Solve longer Math Word Problems Better?</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Math Word Problems (MWPs) play a vital role in assessing the capabilities of Large Language Models (LLMs), yet current research primarily focuses on questions with concise contexts. The impact of longer contexts on mathematical reasoning remains under-explored. This study pioneers the investigation of Context Length Generalizability (CoLeG), which refers to the ability of LLMs to solve MWPs with extended narratives. We introduce Extended Grade-School Math (E-GSM), a collection of MWPs featuring lengthy narratives, and propose two novel metrics to evaluate the efficacy and resilience of LLMs in tackling these problems. Our analysis of existing zero-shot prompting techniques with proprietary LLMs along with open-source LLMs reveals a general deficiency in CoLeG. To alleviate these issues, we propose tailored approaches for different categories of LLMs. For proprietary LLMs, we introduce a new instructional prompt designed to mitigate the impact of long contexts. For open-source LLMs, we develop a novel auxiliary task for fine-tuning to enhance CoLeG. Our comprehensive results demonstrate the effectiveness of our proposed methods, showing improved performance on E-GSM. Additionally, we conduct an in-depth analysis to differentiate the effects of semantic understanding and reasoning efficacy, showing that our methods improves the latter. We also establish the generalizability of our methods across several other MWP benchmarks. Our findings highlight the limitations of current LLMs and offer practical solutions correspondingly, paving the way for further exploration of model generalizability and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13746v1">EICopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      The paper introduces EICopilot, an novel agent-based solution enhancing search and exploration of enterprise registration data within extensive online knowledge graphs like those detailing legal entities, registered capital, and major shareholders. Traditional methods necessitate text-based queries and manual subgraph explorations, often resulting in time-consuming processes. EICopilot, deployed as a chatbot via Baidu Enterprise Search, improves this landscape by utilizing Large Language Models (LLMs) to interpret natural language queries. This solution automatically generates and executes Gremlin scripts, providing efficient summaries of complex enterprise relationships. Distinct feature a data pre-processing pipeline that compiles and annotates representative queries into a vector database of examples for In-context learning (ICL), a comprehensive reasoning pipeline combining Chain-of-Thought with ICL to enhance Gremlin script generation for knowledge graph search and exploration, and a novel query masking strategy that improves intent recognition for heightened script accuracy. Empirical evaluations demonstrate the superior performance of EICopilot, including speed and accuracy, over baseline methods, with the \emph{Full Mask} variant achieving a syntax error rate reduction to as low as 10.00% and an execution correctness of up to 82.14%. These components collectively contribute to superior querying capabilities and summarization of intricate datasets, positioning EICopilot as a groundbreaking tool in the exploration and exploitation of large-scale knowledge graphs for enterprise information search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13731v1">Pseudocode-Injection Magic: Enabling LLMs to Tackle Graph Computational Tasks</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      Graph computational tasks are inherently challenging and often demand the development of advanced algorithms for effective solutions. With the emergence of large language models (LLMs), researchers have begun investigating their potential to address these tasks. However, existing approaches are constrained by LLMs' limited capability to comprehend complex graph structures and their high inference costs, rendering them impractical for handling large-scale graphs. Inspired by human approaches to graph problems, we introduce a novel framework, PIE (Pseudocode-Injection-Enhanced LLM Reasoning for Graph Computational Tasks), which consists of three key steps: problem understanding, prompt design, and code generation. In this framework, LLMs are tasked with understanding the problem and extracting relevant information to generate correct code. The responsibility for analyzing the graph structure and executing the code is delegated to the interpreter. We inject task-related pseudocodes into the prompts to further assist the LLMs in generating efficient code. We also employ cost-effective trial-and-error techniques to ensure that the LLM-generated code executes correctly. Unlike other methods that require invoking LLMs for each individual test case, PIE only calls the LLM during the code generation phase, allowing the generated code to be reused and significantly reducing inference costs. Extensive experiments demonstrate that PIE outperforms existing baselines in terms of both accuracy and computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13687v1">Question Answering on Patient Medical Records with Private Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Healthcare systems continuously generate vast amounts of electronic health records (EHRs), commonly stored in the Fast Healthcare Interoperability Resources (FHIR) standard. Despite the wealth of information in these records, their complexity and volume make it difficult for users to retrieve and interpret crucial health insights. Recent advances in Large Language Models (LLMs) offer a solution, enabling semantic question answering (QA) over medical data, allowing users to interact with their health records more effectively. However, ensuring privacy and compliance requires edge and private deployments of LLMs. This paper proposes a novel approach to semantic QA over EHRs by first identifying the most relevant FHIR resources for a user query (Task1) and subsequently answering the query based on these resources (Task2). We explore the performance of privately hosted, fine-tuned LLMs, evaluating them against benchmark models such as GPT-4 and GPT-4o. Our results demonstrate that fine-tuned LLMs, while 250x smaller in size, outperform GPT-4 family models by 0.55% in F1 score on Task1 and 42% on Meteor Task in Task2. Additionally, we examine advanced aspects of LLM usage, including sequential fine-tuning, model self-evaluation (narcissistic evaluation), and the impact of training data size on performance. The models and datasets are available here: https://huggingface.co/genloop
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13677v1">HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that fundamentally reimagines LLM safety by decoupling it from refusal prefixes through the use of humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests while maintaining engaging interactions. Our approach effectively addresses the common "over-defense" issues in existing safety mechanisms, demonstrating superior robustness against various attack vectors while preserving natural and high-quality interactions on legitimate tasks. Our findings suggest that innovations at the data level are even more fundamental than the alignment algorithm itself in achieving effective LLM safety, opening new directions for developing more resilient and user-friendly AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13669v1">How to Complete Domain Tuning while Keeping General Ability in LLM: Adaptive Layer-wise and Element-wise Regularization</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong general-purpose language capabilities. However, fine-tuning these models on domain-specific tasks often leads to catastrophic forgetting, where the model overwrites or loses essential knowledge acquired during pretraining. This phenomenon significantly limits the broader applicability of LLMs. To address this challenge, we propose a novel approach to compute the element-wise importance of model parameters crucial for preserving general knowledge during fine-tuning. Our method utilizes a dual-objective optimization strategy: (1) regularization loss to retain the parameter crucial for general knowledge; (2) cross-entropy loss to adapt to domain-specific tasks. Additionally, we introduce layer-wise coefficients to account for the varying contributions of different layers, dynamically balancing the dual-objective optimization. Extensive experiments on scientific, medical, and physical tasks using GPT-J and LLaMA-3 demonstrate that our approach mitigates catastrophic forgetting while enhancing model adaptability. Compared to previous methods, our solution is approximately 20 times faster and requires only 10%-15% of the storage, highlighting the practical efficiency. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13545v1">LLMs Can Plan Only If We Tell Them</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant capabilities in natural language processing and reasoning, yet their effectiveness in autonomous planning has been under debate. While existing studies have utilized LLMs with external feedback mechanisms or in controlled environments for planning, these approaches often involve substantial computational and development resources due to the requirement for careful design and iterative backprompting. Moreover, even the most advanced LLMs like GPT-4 struggle to match human performance on standard planning benchmarks, such as the Blocksworld, without additional support. This paper investigates whether LLMs can independently generate long-horizon plans that rival human baselines. Our novel enhancements to Algorithm-of-Thoughts (AoT), which we dub AoT+, help achieve state-of-the-art results in planning benchmarks out-competing prior methods and human baselines all autonomously.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11433v2">One Does Not Simply Meme Alone: Evaluating Co-Creativity Between LLMs and Humans in the Generation of Humor</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 to appear in: 30th International Conference on Intelligent User Interfaces IUI 25 March 2427 2025 Cagliari Italy
    </div>
    <details class="paper-abstract">
      Collaboration has been shown to enhance creativity, leading to more innovative and effective outcomes. While previous research has explored the abilities of Large Language Models (LLMs) to serve as co-creative partners in tasks like writing poetry or creating narratives, the collaborative potential of LLMs in humor-rich and culturally nuanced domains remains an open question. To address this gap, we conducted a user study to explore the potential of LLMs in co-creating memes - a humor-driven and culturally specific form of creative expression. We conducted a user study with three groups of 50 participants each: a human-only group creating memes without AI assistance, a human-AI collaboration group interacting with a state-of-the-art LLM model, and an AI-only group where the LLM autonomously generated memes. We assessed the quality of the generated memes through crowdsourcing, with each meme rated on creativity, humor, and shareability. Our results showed that LLM assistance increased the number of ideas generated and reduced the effort participants felt. However, it did not improve the quality of the memes when humans collaborated with LLM. Interestingly, memes created entirely by AI performed better than both human-only and human-AI collaborative memes in all areas on average. However, when looking at the top-performing memes, human-created ones were better in humor, while human-AI collaborations stood out in creativity and shareability. These findings highlight the complexities of human-AI collaboration in creative tasks. While AI can boost productivity and create content that appeals to a broad audience, human creativity remains crucial for content that connects on a deeper level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04454v2">Inner-Probe: Discovering Copyright-related Data Generation in LLM Architecture</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) utilize extensive knowledge databases and show powerful text generation ability. However, their reliance on high-quality copyrighted datasets raises concerns about copyright infringements in generated texts. Current research often employs prompt engineering or semantic classifiers to identify copyrighted content, but these approaches have two significant limitations: (1) Challenging to identify which specific sub-dataset (e.g., works from particular authors) influences an LLM's output. (2) Treating the entire training database as copyrighted, hence overlooking the inclusion of non-copyrighted training data. We propose InnerProbe, a lightweight framework designed to evaluate the influence of copyrighted sub-datasets on LLM-generated texts. Unlike traditional methods relying solely on text, we discover that the results of multi-head attention (MHA) during LLM output generation provide more effective information. Thus, InnerProbe performs sub-dataset contribution analysis using a lightweight LSTM-based network trained on MHA results in a supervised manner. Harnessing such a prior, InnerProbe enables non-copyrighted text detection through a concatenated global projector trained with unsupervised contrastive learning. InnerProbe demonstrates 3x improved efficiency compared to semantic model training in sub-dataset contribution analysis on Books3, achieves 15.04%-58.7% higher accuracy over baselines on the Pile, and delivers a 0.104 increase in AUC for non-copyrighted data filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13480v1">Adaptive Testing for LLM-Based Applications: A Diversity-based Approach</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The recent surge of building software systems powered by Large Language Models (LLMs) has led to the development of various testing frameworks, primarily focused on treating prompt templates as the unit of testing. Despite the significant costs associated with test input execution and output assessment, the curation of optimized test suites is yet overlooked in these tools, which calls for tailored test selection or prioritization strategies. In this paper, we show that diversity-based testing techniques, such as Adaptive Random Testing (ART) with appropriate string distance metrics, can be effectively applied to the testing of prompt templates. Our proposed adaptive testing approach adjusts the conventional ART process to this context by selecting new test inputs based on scores derived from existing test suite and their labelling results. Our results, obtained using various implementations that explore several string-based distances, confirm that our approach enables the discovery of failures with reduced testing budgets and promotes the generation of more varied outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19825v2">Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Preprint version, under review
    </div>
    <details class="paper-abstract">
      Today's large language models (LLMs) can solve challenging question-answering tasks, and prompt engineering techniques, such as chain-of-thought (CoT), have gained attention for enhancing the explanation and correctness of outputs. However, many models and techniques tend to produce excessively verbose and lengthy answers, leading to issues with both conciseness and generation time. To address this, this paper analyzes the impact of output lengths on LLM inference pipelines by introducing and proposing novel metrics to evaluate the \textit{correct conciseness} of a model and related prompting techniques. Then, we examine the impact of controlling output length through a refined prompt engineering strategy, Constrained-CoT (CCoT), which encourages the model to produce more concise outputs. To better understand the effects of such a prompt, we also introduce two additional scores for analyzing the conciseness, measured in terms of redundancy and information flow in generated answers. Experiments on pretrained LLMs and multiple datasets demonstrate the benefits of the proposed metrics and the effectiveness of CCoT across different models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14162v4">DIRAS: Efficient LLM Annotation of Document Relevance in Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 NAACL 2025 Long
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) is widely employed to ground responses to queries on domain-specific documents. But do RAG implementations leave out important information when answering queries that need an integrated analysis of information (e.g., Tell me good news in the stock market today.)? To address these concerns, RAG developers need to annotate information retrieval (IR) data for their domain of interest, which is challenging because (1) domain-specific queries usually need nuanced definitions of relevance beyond shallow semantic relevance; and (2) human or GPT-4 annotation is costly and cannot cover all (query, document) pairs (i.e., annotation selection bias), thus harming the effectiveness in evaluating IR recall. To address these challenges, we propose DIRAS (Domain-specific Information Retrieval Annotation with Scalability), a manual-annotation-free schema that fine-tunes open-sourced LLMs to consider nuanced relevance definition and annotate (partial) relevance labels with calibrated relevance scores. Extensive evaluation shows that DIRAS enables smaller (8B) LLMs to achieve GPT-4-level performance on annotating and ranking unseen (query, document) pairs, and is helpful for real-world RAG development. All code, LLM generations, and human annotations can be found in \url{https://github.com/EdisonNi-hku/DIRAS}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11885v3">Med-R$^2$: Crafting Trustworthy LLM Physicians through Retrieval and Reasoning of Evidence-Based Medicine</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. However, despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 14.87\% improvement over vanilla RAG methods and even a 3.59\% enhancement compared to fine-tuning strategies, without incurring additional training costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13984v1">Comprehensive Modeling and Question Answering of Cancer Clinical Practice Guidelines using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      The updated recommendations on diagnostic procedures and treatment pathways for a medical condition are documented as graphical flows in Clinical Practice Guidelines (CPGs). For effective use of the CPGs in helping medical professionals in the treatment decision process, it is necessary to fully capture the guideline knowledge, particularly the contexts and their relationships in the graph. While several existing works have utilized these guidelines to create rule bases for Clinical Decision Support Systems, limited work has been done toward directly capturing the full medical knowledge contained in CPGs. This work proposes an approach to create a contextually enriched, faithful digital representation of National Comprehensive Cancer Network (NCCN) Cancer CPGs in the form of graphs using automated extraction and node & relationship classification. We also implement semantic enrichment of the model by using Large Language Models (LLMs) for node classification, achieving an accuracy of 80.86% and 88.47% with zero-shot learning and few-shot learning, respectively. Additionally, we introduce a methodology for answering natural language questions with constraints to guideline text by leveraging LLMs to extract the relevant subgraph from the guideline knowledge base. By generating natural language answers based on subgraph paths and semantic information, we mitigate the risk of incorrect answers and hallucination associated with LLMs, ensuring factual accuracy in medical domain Question Answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16454v2">Catastrophic Failure of LLM Unlearning via Quantization</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 25 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable proficiency in generating text, benefiting from extensive training on vast textual corpora. However, LLMs may also acquire unwanted behaviors from the diverse and sensitive nature of their training data, which can include copyrighted and private content. Machine unlearning has been introduced as a viable solution to remove the influence of such problematic content without the need for costly and time-consuming retraining. This process aims to erase specific knowledge from LLMs while preserving as much model utility as possible. Despite the effectiveness of current unlearning methods, little attention has been given to whether existing unlearning methods for LLMs truly achieve forgetting or merely hide the knowledge, which current unlearning benchmarks fail to detect. This paper reveals that applying quantization to models that have undergone unlearning can restore the "forgotten" information. To thoroughly evaluate this phenomenon, we conduct comprehensive experiments using various quantization techniques across multiple precision levels. We find that for unlearning methods with utility constraints, the unlearned model retains an average of 21\% of the intended forgotten knowledge in full precision, which significantly increases to 83\% after 4-bit quantization. ... Our code is available at: \href{https://github.com/zzwjames/FailureLLMUnlearning}{https://github.com/zzwjames/FailureLLMUnlearning}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11478v2">Each Graph is a New Language: Graph Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
    </div>
    <details class="paper-abstract">
      Recent efforts leverage Large Language Models (LLMs) for modeling text-attributed graph structures in node classification tasks. These approaches describe graph structures for LLMs to understand or aggregate LLM-generated textual attribute embeddings through graph structure. However, these approaches face two main limitations in modeling graph structures with LLMs. (i) Graph descriptions become verbose in describing high-order graph structure. (ii) Textual attributes alone do not contain adequate graph structure information. It is challenging to model graph structure concisely and adequately with LLMs. LLMs lack built-in mechanisms to model graph structures directly. They also struggle with complex long-range dependencies between high-order nodes and target nodes. Inspired by the observation that LLMs pre-trained on one language can achieve exceptional performance on another with minimal additional training, we propose \textbf{G}raph-\textbf{D}efined \textbf{L}anguage for \textbf{L}arge \textbf{L}anguage \textbf{M}odel (GDL4LLM). This novel framework enables LLMs to transfer their powerful language understanding capabilities to graph-structured data. GDL4LLM translates graphs into a graph language corpus instead of graph descriptions and pre-trains LLMs on this corpus to adequately understand graph structures. During fine-tuning, this corpus describes the structural information of target nodes concisely with only a few tokens. By treating graphs as a new language, GDL4LLM enables LLMs to model graph structures adequately and concisely for node classification tasks. Extensive experiments on three real-world datasets demonstrate that GDL4LLM outperforms description-based and textual attribute embeddings-based baselines by efficiently modeling different orders of graph structure with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.03518v4">Improving LLM Abilities in Idiomatic Translation</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Preprint for LoResLM Workshop at COLING 2025
    </div>
    <details class="paper-abstract">
      For large language models (LLMs) like NLLB and GPT, translating idioms remains a challenge. Our goal is to enhance translation fidelity by improving LLM processing of idiomatic language while preserving the original linguistic style. This has a significant social impact, as it preserves cultural nuances and ensures translated texts retain their intent and emotional resonance, fostering better cross-cultural communication. Previous work has utilized knowledge bases like IdiomKB by providing the LLM with the meaning of an idiom to use in translation. Although this method yielded better results than a direct translation, it is still limited in its ability to preserve idiomatic writing style across languages. In this research, we expand upon the knowledge base to find corresponding idioms in the target language. Our research performs translations using two methods: The first method employs the SentenceTransformers model to semantically generate cosine similarity scores between the meanings of the original and target language idioms, selecting the best idiom (Cosine Similarity method). The second method uses an LLM to find a corresponding idiom in the target language for use in the translation (LLM-generated idiom method). As a baseline, we performed a direct translation without providing additional information. Human evaluations on the English -> Chinese, and Chinese -> English show the Cosine Similarity Lookup method out-performed others in all GPT4o translations. To further build upon IdiomKB, we developed a low-resource Urdu dataset containing Urdu idioms and their translations. Despite dataset limitations, the Cosine Similarity Lookup method shows promise, potentially overcoming language barriers and enabling the exploration of diverse literary works in Chinese and Urdu.(LoResLM @ COLING Preprint)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13331v1">Qrazor: Reliable and effortless 4-bit llm quantization by significant data razoring</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      Large-scale language models (LLMs) have demonstrated outstanding performance in language processing tasks, yet their deployment is often hindered by high memory demands and computational complexity. Although low-bit quantization techniques, such as 4-bit quantization, present a potential solution, they frequently lead to significant accuracy degradation or require substantial effort for such aggressive quantization approaches. To overcome these challenges, we introduce QRazor, a reliable and effortless quantization scheme designed to enable 4-bit quantization for weights, activations, and KV cache in transformer-based LLMs. The scheme involves two main stages: quantization and compression. During the quantization stage, weights, activations, and KV cache values are quantized with wider 8 or 16-bit integers as a basis to achieve nearly identical accuracy to the original full-precision LLM models, using the absolute max scaling. Subsequently, all data are compressed to 4-bit using our proposed significant data razoring (SDR) technique, which retains only the four most salient bits while discarding the others. Furthermore, we present an integer-based arithmetic unit dedicated to QRazor, enabling direct low-precision arithmetic operations without decompressing the SDR data. Despite the reduced quantization effort, QRazor achieves LLM accuracies better or comparable to state-of-the-art 4-bit methods. By also validating the hardware efficiency, our decompression-free arithmetic unit achieves 61.2% and 57.8% reduction in area and power consumption, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16372v1">Low-Rank Adapters Meet Neural Architecture Search for LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 AAAI-25 Workshop on Connecting Low-rank Representations in AI
    </div>
    <details class="paper-abstract">
      The rapid expansion of Large Language Models (LLMs) has posed significant challenges regarding the computational resources required for fine-tuning and deployment. Recent advancements in low-rank adapters have demonstrated their efficacy in parameter-efficient fine-tuning (PEFT) of these models. This retrospective paper comprehensively discusses innovative approaches that synergize low-rank representations with Neural Architecture Search (NAS) techniques, particularly weight-sharing super-networks. Robust solutions for compressing and fine-tuning large pre-trained models are developed by integrating these methodologies. Our analysis highlights the potential of these combined strategies to democratize the use of LLMs, making them more accessible for deployment in resource-constrained environments. The resulting models exhibit reduced memory footprints and faster inference times, paving the way for more practical and scalable applications of LLMs. Models and code are available at https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10245v3">From Text to Emoji: How PEFT-Driven Personality Manipulation Unleashes the Emoji Potential in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 Findings paper of NAACL 2025 and NeurIPS 2024 Workshop on Behavioral Machine Learning
    </div>
    <details class="paper-abstract">
      The manipulation of the personality traits of large language models (LLMs) has emerged as a key area of research. Methods like prompt-based In-Context Knowledge Editing (IKE) and gradient-based Model Editor Networks (MEND) have been explored but show irregularity and variability; IKE depends on the prompt, leading to variability and sensitivity, while MEND yields inconsistent and gibberish outputs. To address this, we employed Opinion QA Based Parameter-Efficient Fine-Tuning (PEFT), specifically Quantized Low-Rank Adaptation (QLoRA), to manipulate the Big Five personality traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. After PEFT, models such as Mistral-7B-Instruct and LLaMA-2-7B-chat began generating emojis, even though no emojis were present in the PEFT data. For instance, LLaMA-2-7B-chat generated emojis in 99.5% of extraversion-related test instances, while Mistral-7B-Instruct did so in 92.5% of openness-related test instances. ICL Explainability analysis indicated that the LLMs used emojis intentionally to express these traits. Mechanistic Interpretability analysis showed that this latent behaviour of LLMs could be traced to specific neurons that became activated or amplified after PEFT. This paper provides a number of novel contributions. First, introducing an Opinion QA dataset for PEFT-driven personality manipulation; second, developing metric models to benchmark LLM personality traits; third, demonstrating PEFT's superiority over IKE in personality manipulation; and finally, analysing and validating emoji usage through explainability methods such as Mechanistic Interpretability and In-context learning Explainability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2501.12557v1">Understanding the LLM-ification of CHI: Unpacking the Impact of LLMs at CHI through a Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-01-23
      | 💬 This is a preprint version of the paper conditionally accepted to CHI'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been positioned to revolutionize HCI, by reshaping not only the interfaces, design patterns, and sociotechnical systems that we study, but also the research practices we use. To-date, however, there has been little understanding of LLMs' uptake in HCI. We address this gap via a systematic literature review of 153 CHI papers from 2020-24 that engage with LLMs. We taxonomize: (1) domains where LLMs are applied; (2) roles of LLMs in HCI projects; (3) contribution types; and (4) acknowledged limitations and risks. We find LLM work in 10 diverse domains, primarily via empirical and artifact contributions. Authors use LLMs in five distinct roles, including as research tools or simulated users. Still, authors often raise validity and reproducibility concerns, and overwhelmingly study closed models. We outline opportunities to improve HCI research with and on LLMs, and provide guiding questions for researchers to consider the validity and appropriateness of LLM-related work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13080v1">Refining Input Guardrails: Enhancing LLM-as-a-Judge Efficiency Through Chain-of-Thought Fine-Tuning and Alignment</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 16 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated powerful capabilities that render them valuable in different applications, including conversational AI products. It is paramount to ensure the security and reliability of these products by mitigating their vulnerabilities towards malicious user interactions, which can lead to the exposure of great risks and reputational repercussions. In this work, we present a comprehensive study on the efficacy of fine-tuning and aligning Chain-of-Thought (CoT) responses of different LLMs that serve as input moderation guardrails. We systematically explore various tuning methods by leveraging a small set of training data to adapt these models as proxy defense mechanisms to detect malicious inputs and provide a reasoning for their verdicts, thereby preventing the exploitation of conversational agents. We rigorously evaluate the efficacy and robustness of different tuning strategies to generalize across diverse adversarial and malicious query types. Our experimental results outline the potential of alignment processes tailored to a varied range of harmful input queries, even with constrained data resources. These techniques significantly enhance the safety of conversational AI systems and provide a feasible framework for deploying more secure and trustworthy AI-driven interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12980v1">Implicit Causality-biases in humans and LLMs as a tool for benchmarking LLM discourse capabilities</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 38 pages, 8 figures
    </div>
    <details class="paper-abstract">
      In this paper, we compare data generated with mono- and multilingual LLMs spanning a range of model sizes with data provided by human participants in an experimental setting investigating well-established discourse biases. Beyond the comparison as such, we aim to develop a benchmark to assess the capabilities of LLMs with discourse biases as a robust proxy for more general discourse understanding capabilities. More specifically, we investigated Implicit Causality verbs, for which psycholinguistic research has found participants to display biases with regard to three phenomena:\ the establishment of (i) coreference relations (Experiment 1), (ii) coherence relations (Experiment 2), and (iii) the use of particular referring expressions (Experiments 3 and 4). With regard to coreference biases we found only the largest monolingual LLM (German Bloom 6.4B) to display more human-like biases. For coherence relation, no LLM displayed the explanation bias usually found for humans. For referring expressions, all LLMs displayed a preference for referring to subject arguments with simpler forms than to objects. However, no bias effect on referring expression was found, as opposed to recent studies investigating human biases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12972v1">Accessible Smart Contracts Verification: Synthesizing Formal Models with Tamed LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      When blockchain systems are said to be trustless, what this really means is that all the trust is put into software. Thus, there are strong incentives to ensure blockchain software is correct -- vulnerabilities here cost millions and break businesses. One of the most powerful ways of establishing software correctness is by using formal methods. Approaches based on formal methods, however, induce a significant overhead in terms of time and expertise required to successfully employ them. Our work addresses this critical disadvantage by automating the creation of a formal model -- a mathematical abstraction of the software system -- which is often a core task when employing formal methods. We perform model synthesis in three phases: we first transpile the code into model stubs; then we "fill in the blanks" using a large language model (LLM); finally, we iteratively repair the generated model, on both syntactical and semantical level. In this way, we significantly reduce the amount of time necessary to create formal models and increase accessibility of valuable software verification methods that rely on them. The practical context of our work was reducing the time-to-value of using formal models for correctness audits of smart contracts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11935v2">Web vs. LLMs: An Empirical Study of Learning Behaviors of CS2 Students</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      LLMs such as ChatGPT have been widely adopted by students in higher education as tools for learning programming and related concepts. However, it remains unclear how effective students are and what strategies students use while learning with LLMs. Since the majority of students' experiences in online self-learning have come through using search engines such as Google, evaluating AI tools in this context can help us address these gaps. In this mixed methods research, we conducted an exploratory within-subjects study to understand how CS2 students learn programming concepts using both LLMs as well as traditional online methods such as educational websites and videos to examine how students approach learning within and across both scenarios. We discovered that students found it easier to learn a more difficult concept using traditional methods than using ChatGPT. We also found that students ask fewer follow-ups and use more keyword-based queries for search engines while their prompts to LLMs tend to explicitly ask for information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12862v1">Mutation-Guided LLM-based Test Generation at Meta</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 Submitted to FSE 2025 Industry Track
    </div>
    <details class="paper-abstract">
      This paper describes Meta's ACH system for mutation-guided LLM-based test generation. ACH generates relatively few mutants (aka simulated faults), compared to traditional mutation testing. Instead, it focuses on generating currently undetected faults that are specific to an issue of concern. From these currently uncaught faults, ACH generates tests that can catch them, thereby `killing' the mutants and consequently hardening the platform against regressions. We use privacy concerns to illustrate our approach, but ACH can harden code against {\em any} type of regression. In total, ACH was applied to 10,795 Android Kotlin classes in 7 software platforms deployed by Meta, from which it generated 9,095 mutants and 571 privacy-hardening test cases. ACH also deploys an LLM-based equivalent mutant detection agent that achieves a precision of 0.79 and a recall of 0.47 (rising to 0.95 and 0.96 with simple pre-processing). ACH was used by Messenger and WhatsApp test-a-thons where engineers accepted 73% of its tests, judging 36% to privacy relevant. We conclude that ACH hardens code against specific concerns and that, even when its tests do not directly tackle the specific concern, engineers find them useful for their other benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12826v1">Open or Closed LLM for Lesser-Resourced Languages? Lessons from Greek</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 NLP, Modern Greek, benchmark, machine learning, language resources
    </div>
    <details class="paper-abstract">
      Natural Language Processing (NLP) for lesser-resourced languages faces persistent challenges, including limited datasets, inherited biases from high-resource languages, and the need for domain-specific solutions. This study addresses these gaps for Modern Greek through three key contributions. First, we evaluate the performance of open-source (Llama-70b) and closed-source (GPT-4o mini) large language models (LLMs) on seven core NLP tasks with dataset availability, revealing task-specific strengths, weaknesses, and parity in their performance. Second, we expand the scope of Greek NLP by reframing Authorship Attribution as a tool to assess potential data usage by LLMs in pre-training, with high 0-shot accuracy suggesting ethical implications for data provenance. Third, we showcase a legal NLP case study, where a Summarize, Translate, and Embed (STE) methodology outperforms the traditional TF-IDF approach for clustering \emph{long} legal texts. Together, these contributions provide a roadmap to advance NLP in lesser-resourced languages, bridging gaps in model evaluation, task innovation, and real-world impact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12774v1">LLMs as Repositories of Factual Knowledge: Limitations and Solutions</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      LLMs' sources of knowledge are data snapshots containing factual information about entities collected at different timestamps and from different media types (e.g. wikis, social media, etc.). Such unstructured knowledge is subject to change due to updates through time from past to present. Equally important are the inconsistencies and inaccuracies occurring in different information sources. Consequently, the model's knowledge about an entity may be perturbed while training over the sequence of snapshots or at inference time, resulting in inconsistent and inaccurate model performance. In this work, we study the appropriateness of Large Language Models (LLMs) as repositories of factual knowledge. We consider twenty-four state-of-the-art LLMs that are either closed-, partially (weights), or fully (weight and training data) open-source. We evaluate their reliability in responding to time-sensitive factual questions in terms of accuracy and consistency when prompts are perturbed. We further evaluate the effectiveness of state-of-the-art methods to improve LLMs' accuracy and consistency. We then propose "ENtity-Aware Fine-tuning" (ENAF), a soft neurosymbolic approach aimed at providing a structured representation of entities during fine-tuning to improve the model's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12702v1">Paradigm-Based Automatic HDL Code Generation Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 accepted by ISQED2025. arXiv admin note: text overlap with arXiv:2407.18326
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated the ability to generate hardware description language (HDL) code for digital circuits, they still face the hallucination problem, which can result in the generation of incorrect HDL code or misinterpretation of specifications. In this work, we introduce a human-expert-inspired method to mitigate the hallucination of LLMs and enhance their performance in HDL code generation. We begin by constructing specialized paradigm blocks that consist of several steps designed to divide and conquer generation tasks, mirroring the design methodology of human experts. These steps include information extraction, human-like design flows, and the integration of external tools. LLMs are then instructed to classify the type of circuit in order to match it with the appropriate paradigm block and execute the block to generate the HDL codes. Additionally, we propose a two-phase workflow for multi-round generation, aimed at effectively improving the testbench pass rate of the generated HDL codes within a limited number of generation and verification rounds. Experimental results demonstrate that our method significantly enhances the functional correctness of the generated Verilog code
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12697v1">Combining Knowledge Graph and LLMs for Enhanced Zero-shot Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Zero-shot visual question answering (ZS-VQA), an emerged critical research area, intends to answer visual questions without providing training samples. Existing research in ZS-VQA has proposed to leverage knowledge graphs or large language models (LLMs), respectively, as external information sources to help VQA model comprehend images and questions. However, LLMs often struggle in accurately interpreting specific question meanings. Meanwhile, although knowledge graph has rich entity relationships, it is challenging to effectively connect entities to individual image content for visual question answers. In this paper, we propose a novel design to combine knowledge graph and LLMs for zero-shot visual question answer. Our approach uses LLMs' powerful understanding capabilities to accurately interpret image content through a strategic question search mechanism. Meanwhile, the knowledge graph is used to expand and connect users' queries to the image content for better visual question answering. An optimization algorithm is further used to determine the optimal weights for the loss functions derived from different information sources, towards a globally optimal set of candidate answers. Experimental results on two benchmark datasets demonstrate that our model achieves state-of-the-art (SOTA) performance. Both source code and benchmark data will be released for public access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07098v2">A Multi-Agent Approach for REST API Testing with Semantic Graphs and LLM-Driven Inputs</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 To be published in the 47th IEEE/ACM International Conference on Software Engineering (ICSE 2025)
    </div>
    <details class="paper-abstract">
      As modern web services increasingly rely on REST APIs, their thorough testing has become crucial. Furthermore, the advent of REST API documentation languages, such as the OpenAPI Specification, has led to the emergence of many black-box REST API testing tools. However, these tools often focus on individual test elements in isolation (e.g., APIs, parameters, values), resulting in lower coverage and less effectiveness in fault detection. To address these limitations, we present AutoRestTest, the first black-box tool to adopt a dependency-embedded multi-agent approach for REST API testing that integrates multi-agent reinforcement learning (MARL) with a semantic property dependency graph (SPDG) and Large Language Models (LLMs). Our approach treats REST API testing as a separable problem, where four agents -- API, dependency, parameter, and value agents -- collaborate to optimize API exploration. LLMs handle domain-specific value generation, the SPDG model simplifies the search space for dependencies using a similarity score between API operations, and MARL dynamically optimizes the agents' behavior. Our evaluation of AutoRestTest on 12 real-world REST services shows that it outperforms the four leading black-box REST API testing tools, including those assisted by RESTGPT (which generates realistic test inputs using LLMs), in terms of code coverage, operation coverage, and fault detection. Notably, AutoRestTest is the only tool able to trigger an internal server error in the Spotify service. Our ablation study illustrates that each component of AutoRestTest -- the SPDG, the LLM, and the agent-learning mechanism -- contributes to its overall effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12599v1">Kimi k1.5: Scaling Reinforcement Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      Language model pretraining with next token prediction has proved effective for scaling compute but is limited to the amount of available training data. Scaling reinforcement learning (RL) unlocks a new axis for the continued improvement of artificial intelligence, with the promise that large language models (LLMs) can scale their training data by learning to explore with rewards. However, prior published work has not produced competitive results. In light of this, we report on the training practice of Kimi k1.5, our latest multi-modal LLM trained with RL, including its RL training techniques, multi-modal data recipes, and infrastructure optimization. Long context scaling and improved policy optimization methods are key ingredients of our approach, which establishes a simplistic, effective RL framework without relying on more complex techniques such as Monte Carlo tree search, value functions, and process reward models. Notably, our system achieves state-of-the-art reasoning performance across multiple benchmarks and modalities -- e.g., 77.5 on AIME, 96.2 on MATH 500, 94-th percentile on Codeforces, 74.9 on MathVista -- matching OpenAI's o1. Moreover, we present effective long2short methods that use long-CoT techniques to improve short-CoT models, yielding state-of-the-art short-CoT reasoning results -- e.g., 60.8 on AIME, 94.6 on MATH500, 47.3 on LiveCodeBench -- outperforming existing short-CoT models such as GPT-4o and Claude Sonnet 3.5 by a large margin (up to +550%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02406v2">Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly difficult as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by LLM $A$ or $B$ (where $B$ can be a human)? We model LLM-generated text as a sequential stochastic process with complete dependence on history and design zero-shot statistical tests to distinguish between (i) the text generated by two different sets of LLMs $A$ (in-house) and $B$ (non-sanctioned) and also (ii) LLM-generated and human-generated texts. We prove that the type I and type II errors for our tests decrease exponentially in the text length. In designing our tests, we derive concentration inequalities on the difference between log-perplexity and the average entropy of the string under $A$. Specifically, for a given string, we demonstrate that if the string is generated by $A$, the log-perplexity of the string under $A$ converges to the average entropy of the string under $A$, except with an exponentially small probability in string length. We also show that if $B$ generates the text, except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. Lastly, we present preliminary experimental results to support our theoretical results. By enabling guaranteed (with high probability) finding of the origin of harmful LLM-generated text with arbitrary size, we can help combat misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15023v2">PaperWave: Listening to Research Papers as Conversational Podcasts Scripted by LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Listening to audio content, such as podcasts and audiobooks, is one way for people to engage with knowledge. Listening affords people more mobility than reading by seeing, thereby broadening their learning opportunities. This study explores the potential applications of large language models (LLMs) to adapt text documents to audio content and addresses the lack of listening-friendly materials for niche content, such as research papers. LLMs can generate scripts of audio content in various styles tailored to specific needs, such as full-content duration or speech types (monologue or dialogue). To explore this potential, we developed PaperWave as a prototype that transforms academic paper PDFs into conversational podcasts. Our two-month investigation, involving 11 participants (including the authors), employed an autobiographical design, a field study, and a design workshop. The findings highlight the importance of considering listener interaction with their environment when designing document-to-audio systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12573v1">Leveraging LLMs to Create a Haptic Devices' Recommendation System</a></div>
    <div class="paper-meta">
      📅 2025-01-22
    </div>
    <details class="paper-abstract">
      Haptic technology has seen significant growth, yet a lack of awareness of existing haptic device design knowledge hinders development. This paper addresses these limitations by leveraging advancements in Large Language Models (LLMs) to develop a haptic agent, focusing specifically on Grounded Force Feedback (GFF) devices recommendation. Our approach involves automating the creation of a structured haptic device database using information from research papers and product specifications. This database enables the recommendation of relevant GFF devices based on user queries. To ensure precise and contextually relevant recommendations, the system employs a dynamic retrieval method that combines both conditional and semantic searches. Benchmarking against the established UEQ and existing haptic device searching tools, the proposed haptic recommendation agent ranks in the top 10\% across all UEQ categories with mean differences favoring the agent in nearly all subscales, and maintains no significant performance bias across different user groups, showcasing superior usability and user satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09879v2">Testing Refactoring Engine via Historical Bug Report driven LLM</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 Accepted at the 2nd ACM international conference on AI Foundation Models and Software Engineering (FORGE 2025)
    </div>
    <details class="paper-abstract">
      Refactoring is the process of restructuring existing code without changing its external behavior while improving its internal structure. Refactoring engines are integral components of modern Integrated Development Environments (IDEs) and can automate or semi-automate this process to enhance code readability, reduce complexity, and improve the maintainability of software products. Similar to traditional software systems such as compilers, refactoring engines may also contain bugs that can lead to unexpected behaviors. In this paper, we propose a novel approach called RETESTER, a LLM-based framework for automated refactoring engine testing. Specifically, by using input program structure templates extracted from historical bug reports and input program characteristics that are error-prone, we design chain-of-thought (CoT) prompts to perform refactoring-preserving transformations. The generated variants are then tested on the latest version of refactoring engines using differential testing. We evaluate RETESTER on two most popular modern refactoring engines (i.e., ECLIPSE, and INTELLIJ IDEA). It successfully revealed 18 new bugs in the latest version of those refactoring engines. By the time we submit our paper, seven of them were confirmed by their developers, and three were fixed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12557v1">Understanding the LLM-ification of CHI: Unpacking the Impact of LLMs at CHI through a Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 This is a preprint version of the paper conditionally accepted to CHI'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been positioned to revolutionize HCI, by reshaping not only the interfaces, design patterns, and sociotechnical systems that we study, but also the research practices we use. To-date, however, there has been little understanding of LLMs' uptake in HCI. We address this gap via a systematic literature review of 153 CHI papers from 2020-24 that engage with LLMs. We taxonomize: (1) domains where LLMs are applied; (2) roles of LLMs in HCI projects; (3) contribution types; and (4) acknowledged limitations and risks. We find LLM work in 10 diverse domains, primarily via empirical and artifact contributions. Authors use LLMs in five distinct roles, including as research tools or simulated users. Still, authors often raise validity and reproducibility concerns, and overwhelmingly study closed models. We outline opportunities to improve HCI research with and on LLMs, and provide guiding questions for researchers to consider the validity and appropriateness of LLM-related work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.12557v1">Understanding the LLM-ification of CHI: Unpacking the Impact of LLMs at CHI through a Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-01-22
      | 💬 This is a preprint version of the paper conditionally accepted to CHI'25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been positioned to revolutionize HCI, by reshaping not only the interfaces, design patterns, and sociotechnical systems that we study, but also the research practices we use. To-date, however, there has been little understanding of LLMs' uptake in HCI. We address this gap via a systematic literature review of 153 CHI papers from 2020-24 that engage with LLMs. We taxonomize: (1) domains where LLMs are applied; (2) roles of LLMs in HCI projects; (3) contribution types; and (4) acknowledged limitations and risks. We find LLM work in 10 diverse domains, primarily via empirical and artifact contributions. Authors use LLMs in five distinct roles, including as research tools or simulated users. Still, authors often raise validity and reproducibility concerns, and overwhelmingly study closed models. We outline opportunities to improve HCI research with and on LLMs, and provide guiding questions for researchers to consider the validity and appropriateness of LLM-related work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.15524v2">The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive potential to simulate human behavior. We identify a fundamental challenge in using them to simulate experiments: when LLM-simulated subjects are blind to the experimental design (as is standard practice with human subjects), variations in treatment systematically affect unspecified variables that should remain constant, violating the unconfoundedness assumption. Using demand estimation as a context and an actual experiment as a benchmark, we show this can lead to implausible results. While confounding may in principle be addressed by controlling for covariates, this can compromise ecological validity in the context of LLM simulations: controlled covariates become artificially salient in the simulated decision process, which introduces focalism. This trade-off between unconfoundedness and ecological validity is usually absent in traditional experimental design and represents a unique challenge in LLM simulations. We formalize this challenge theoretically, showing it stems from ambiguous prompting strategies, and hence cannot be fully addressed by improving training data or by fine-tuning. Alternative approaches that unblind the experimental design to the LLM show promise. Our findings suggest that effectively leveraging LLMs for experimental simulations requires fundamentally rethinking established experimental design practices rather than simply adapting protocols developed for human subjects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21548v2">MultiTok: Variable-Length Tokenization for Efficient LLMs Adapted from LZW Compression</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large language models have drastically changed the prospects of AI by introducing technologies for more complex natural language processing. However, current methodologies to train such LLMs require extensive resources including but not limited to large amounts of data, expensive machinery, and lengthy training. To solve this problem, this paper proposes a new tokenization method inspired by universal Lempel-Ziv-Welch data compression that compresses repetitive phrases into multi-word tokens. With MultiTok as a new tokenizing tool, we show that language models are able to be trained notably more efficiently while offering a similar accuracy on more succinct and compressed training data. In fact, our results demonstrate that MultiTok achieves a comparable performance to the BERT and GPT-2 standards as both a stand-alone tokenizer and an add-on to existing tokenizers while also providing close to 2.5x faster training with more than 30% less training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12456v1">Deploying Privacy Guardrails for LLMs: A Comparative Analysis of Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 This paper has been accepted at Deployable AI workshop at AAAI 2025
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) has revolutionized AI applications but poses significant challenges in safeguarding user privacy. Ensuring compliance with privacy regulations such as GDPR and CCPA while addressing nuanced privacy risks requires robust and scalable frameworks. This paper presents a detailed study of OneShield Privacy Guard, a framework designed to mitigate privacy risks in user inputs and LLM outputs across enterprise and open-source settings. We analyze two real-world deployments:(1) a multilingual privacy-preserving system integrated with Data and Model Factory, focusing on enterprise-scale data governance; and (2) PR Insights, an open-source repository emphasizing automated triaging and community-driven refinements. In Deployment 1, OneShield achieved a 0.95 F1 score in detecting sensitive entities like dates, names, and phone numbers across 26 languages, outperforming state-of-the-art tool such as StarPII and Presidio by up to 12\%. Deployment 2, with an average F1 score of 0.86, reduced manual effort by over 300 hours in three months, accurately flagging 8.25\% of 1,256 pull requests for privacy risks with enhanced context sensitivity. These results demonstrate OneShield's adaptability and efficacy in diverse environments, offering actionable insights for context-aware entity recognition, automated compliance, and ethical AI adoption. This work advances privacy-preserving frameworks, supporting user trust and compliance across operational contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v1">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 14 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve a strong performance with 67.41\% on BIRD benchmark (dev) without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12332v1">Automatic Labelling with Open-source LLMs using Dynamic Label Schema Integration</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 11 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Acquiring labelled training data remains a costly task in real world machine learning projects to meet quantity and quality requirements. Recently Large Language Models (LLMs), notably GPT-4, have shown great promises in labelling data with high accuracy. However, privacy and cost concerns prevent the ubiquitous use of GPT-4. In this work, we explore effectively leveraging open-source models for automatic labelling. We identify integrating label schema as a promising technology but found that naively using the label description for classification leads to poor performance on high cardinality tasks. To address this, we propose Retrieval Augmented Classification (RAC) for which LLM performs inferences for one label at a time using corresponding label schema; we start with the most related label and iterates until a label is chosen by the LLM. We show that our method, which dynamically integrates label description, leads to performance improvements in labelling tasks. We further show that by focusing only on the most promising labels, RAC can trade off between label quality and coverage - a property we leverage to automatically label our internal datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10622v3">A recent evaluation on the performance of LLMs on radiation oncology physics using questions of randomly shuffled options</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Purpose: We present an updated study evaluating the performance of large language models (LLMs) in answering radiation oncology physics questions, focusing on the recently released models. Methods: A set of 100 multiple-choice radiation oncology physics questions, previously created by a well-experienced physicist, was used for this study. The answer options of the questions were randomly shuffled to create "new" exam sets. Five LLMs -- OpenAI o1-preview, GPT-4o, LLaMA 3.1 (405B), Gemini 1.5 Pro, and Claude 3.5 Sonnet -- with the versions released before September 30, 2024, were queried using these new exam sets. To evaluate their deductive reasoning ability, the correct answer options in the questions were replaced with "None of the above." Then, the explain-first and step-by-step instruction prompts were used to test if this strategy improved their reasoning ability. The performance of the LLMs was compared with the answers from medical physicists. Results: All models demonstrated expert-level performance on these questions, with o1-preview even surpassing medical physicists with a majority vote. When replacing the correct answer options with 'None of the above', all models exhibited a considerable decline in performance, suggesting room for improvement. The explain-first and step-by-step instruction prompts helped enhance the reasoning ability of the LLaMA 3.1 (405B), Gemini 1.5 Pro, and Claude 3.5 Sonnet models. Conclusion: These recently released LLMs demonstrated expert-level performance in answering radiation oncology physics questions, exhibiting great potential to assist in radiation oncology physics education and training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12300v1">LLM-Assisted Knowledge Graph Completion for Curriculum and Domain Modelling in Personalized Higher Education Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted in the IEEE Global Engineering Education Conference (EDUCON2025), London, UK, 22-25 April, 2025
    </div>
    <details class="paper-abstract">
      While learning personalization offers great potential for learners, modern practices in higher education require a deeper consideration of domain models and learning contexts, to develop effective personalization algorithms. This paper introduces an innovative approach to higher education curriculum modelling that utilizes large language models (LLMs) for knowledge graph (KG) completion, with the goal of creating personalized learning-path recommendations. Our research focuses on modelling university subjects and linking their topics to corresponding domain models, enabling the integration of learning modules from different faculties and institutions in the student's learning path. Central to our approach is a collaborative process, where LLMs assist human experts in extracting high-quality, fine-grained topics from lecture materials. We develop a domain, curriculum, and user models for university modules and stakeholders. We implement this model to create the KG from two study modules: Embedded Systems and Development of Embedded Systems Using FPGA. The resulting KG structures the curriculum and links it to the domain models. We evaluate our approach through qualitative expert feedback and quantitative graph quality metrics. Domain experts validated the relevance and accuracy of the model, while the graph quality metrics measured the structural properties of our KG. Our results show that the LLM-assisted graph completion approach enhances the ability to connect related courses across disciplines to personalize the learning experience. Expert feedback also showed high acceptance of the proposed collaborative approach for concept extraction and classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12273v1">Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Tech Report. Github: https://github.com/InternLM/Condor
    </div>
    <details class="paper-abstract">
      The quality of Supervised Fine-Tuning (SFT) data plays a critical role in enhancing the conversational capabilities of Large Language Models (LLMs). However, as LLMs become more advanced, the availability of high-quality human-annotated SFT data has become a significant bottleneck, necessitating a greater reliance on synthetic training data. In this work, we introduce Condor, a novel two-stage synthetic data generation framework that incorporates World Knowledge Tree and Self-Reflection Refinement to produce high-quality SFT data at scale. Our experimental results demonstrate that a base model fine-tuned on only 20K Condor-generated samples achieves superior performance compared to counterparts. The additional refinement stage in Condor further enables iterative self-improvement for LLMs at various scales (up to 72B), validating the effectiveness of our approach. Furthermore, our investigation into the scaling for synthetic data in post-training reveals substantial unexplored potential for performance improvements, opening promising avenues for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12210v1">You Can't Eat Your Cake and Have It Too: The Performance Degradation of LLMs with Jailbreak Defense</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      With the rise of generative large language models (LLMs) like LLaMA and ChatGPT, these models have significantly transformed daily life and work by providing advanced insights. However, as jailbreak attacks continue to circumvent built-in safety mechanisms, exploiting carefully crafted scenarios or tokens, the safety risks of LLMs have come into focus. While numerous defense strategies--such as prompt detection, modification, and model fine-tuning--have been proposed to counter these attacks, a critical question arises: do these defenses compromise the utility and usability of LLMs for legitimate users? Existing research predominantly focuses on the effectiveness of defense strategies without thoroughly examining their impact on performance, leaving a gap in understanding the trade-offs between LLM safety and performance. Our research addresses this gap by conducting a comprehensive study on the utility degradation, safety elevation, and exaggerated-safety escalation of LLMs with jailbreak defense strategies. We propose USEBench, a novel benchmark designed to evaluate these aspects, along with USEIndex, a comprehensive metric for assessing overall model performance. Through experiments on seven state-of-the-art LLMs, we found that mainstream jailbreak defenses fail to ensure both safety and performance simultaneously. Although model-finetuning performs the best overall, their effectiveness varies across LLMs. Furthermore, vertical comparisons reveal that developers commonly prioritize performance over safety when iterating or fine-tuning their LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12162v1">AdaServe: SLO-Customized LLM Serving with Fine-Grained Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      This paper introduces AdaServe, the first LLM serving system to support SLO customization through fine-grained speculative decoding. AdaServe leverages the logits of a draft model to predict the speculative accuracy of tokens and employs a theoretically optimal algorithm to construct token trees for verification. To accommodate diverse SLO requirements without compromising throughput, AdaServe employs a speculation-and-selection scheme that first constructs candidate token trees for each request and then dynamically selects tokens to meet individual SLO constraints while optimizing throughput. Comprehensive evaluations demonstrate that AdaServe achieves up to 73% higher SLO attainment and 74% higher goodput compared to state-of-the-art systems. These results underscore AdaServe's potential to enhance the efficiency and adaptability of LLM deployments across varied application scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12152v1">Contextualizing Recommendation Explanations with LLMs: A User Study</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly prevalent in recommender systems, where LLMs can be used to generate personalized recommendations. Here, we examine how different LLM-generated explanations for movie recommendations affect users' perceptions of cognitive, affective, and utilitarian needs and consumption intentions. In a pre-registered, between-subject online experiment (N=759) and follow-up interviews (N=30), we compare (a) LLM-generated generic explanations, and (b) LLM-generated contextualized explanations. Our findings show that contextualized explanations (i.e., explanations that incorporate users' past behaviors) effectively meet users' cognitive needs while increasing users' intentions to watch recommended movies. However, adding explanations offers limited benefits in meeting users' utilitarian and affective needs, raising concerns about the proper design and implications of LLM-generated explanations. Qualitative insights from interviews reveal that referencing users' past preferences enhances trust and understanding but can feel excessive if overused. Furthermore, users with more active and positive engagement with the recommender system and movie-watching get substantial gains from contextualized explanations. Overall, our research clarifies how LLM-generated recommendations influence users' motivations and behaviors, providing valuable insights for the future development of user-centric recommender systems, a key element in social media platforms and online ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12134v1">Do LLMs Provide Links to Code Similar to what they Generate? A Study with Gemini and Bing CoPilot</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are currently used for various software development tasks, including generating code snippets to solve specific problems. Unlike reuse from the Web, LLMs are limited in providing provenance information about the generated code, which may have important trustworthiness and legal consequences. While LLM-based assistants may provide external links that are "related" to the generated code, we do not know how relevant such links are. This paper presents the findings of an empirical study assessing the extent to which 243 and 194 code snippets, across six programming languages, generated by Bing CoPilot and Google Gemini, likely originate from the links provided by these two LLM-based assistants. The study leverages automated code similarity assessments with thorough manual analysis. The study's findings indicate that the LLM-based assistants provide a mix of relevant and irrelevant links having a different nature. Specifically, although 66% of the links from Bing CoPilot and 28% from Google Gemini are relevant, LLMs-based assistants still suffer from serious "provenance debt".
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12128v1">Evaluating Efficiency and Engagement in Scripted and LLM-Enhanced Human-Robot Interactions</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted as a Late-Breaking Report to the 2025, 20th ACM/IEEE International Conference on Human-Robot Interaction (HRI)
    </div>
    <details class="paper-abstract">
      To achieve natural and intuitive interaction with people, HRI frameworks combine a wide array of methods for human perception, intention communication, human-aware navigation and collaborative action. In practice, when encountering unpredictable behavior of people or unexpected states of the environment, these frameworks may lack the ability to dynamically recognize such states, adapt and recover to resume the interaction. Large Language Models (LLMs), owing to their advanced reasoning capabilities and context retention, present a promising solution for enhancing robot adaptability. This potential, however, may not directly translate to improved interaction metrics. This paper considers a representative interaction with an industrial robot involving approach, instruction, and object manipulation, implemented in two conditions: (1) fully scripted and (2) including LLM-enhanced responses. We use gaze tracking and questionnaires to measure the participants' task efficiency, engagement, and robot perception. The results indicate higher subjective ratings for the LLM condition, but objective metrics show that the scripted condition performs comparably, particularly in efficiency and focus during simple tasks. We also note that the scripted condition may have an edge over LLM-enhanced responses in terms of response latency and energy consumption, especially for trivial and repetitive interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13126v1">Preference Curriculum: LLMs Should Always Be Pretrained on Their Preferred Data</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 18 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Current large language models (LLMs) generally utilize a consistent data distribution throughout the entire pretraining process. However, as the model's ability improves, it intuitively should be pretrained with differentiated data. To achieve it, we propose the Perplexity Difference based Preference Curriculum learning (PDPC) framework, which always perceives and uses the data preferred by LLMs to train and boost them. Firstly, we introduce the PD metric to measure the difference in how well strong and weak models fit the samples. Samples with high PD are more challenging for weak models to learn and are more suitable to be arranged in the later stage of pretraining. Secondly, we propose the PD preference function to approximate the model and predict the data preference of the LLM at any time, so as to complete the arrangement of the entire data offline and ensure continuous training without interruption. Experimental results on 1.3B and 3B models demonstrate that our PDPC significantly surpasses baselines. Notably, the 3B model achieved more substantial gains, with an increased average accuracy of over 4.1% across various benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02044v2">QROA: A Black-Box Query-Response Optimization Attack on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have surged in popularity in recent months, yet they possess concerning capabilities for generating harmful content when manipulated. This study introduces the Query-Response Optimization Attack (QROA), an optimization-based strategy designed to exploit LLMs through a black-box, query-only interaction. QROA adds an optimized trigger to a malicious instruction to compel the LLM to generate harmful content. Unlike previous approaches, QROA does not require access to the model's logit information or any other internal data and operates solely through the standard query-response interface of LLMs. Inspired by deep Q-learning and Greedy coordinate descent, the method iteratively updates tokens to maximize a designed reward function. We tested our method on various LLMs such as Vicuna, Falcon, and Mistral, achieving an Attack Success Rate (ASR) over 80\%. We also tested the model against Llama2-chat, the fine-tuned version of Llama2 designed to resist Jailbreak attacks, achieving good ASR with a suboptimal initial trigger seed. This study demonstrates the feasibility of generating jailbreak attacks against deployed LLMs in the public domain using black-box optimization methods, enabling more comprehensive safety testing of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01144v3">BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      The rapidly increasing size of large language models (LLMs) presents significant challenges in memory usage and computational costs. Quantizing both weights and activations can address these issues, with hardware-supported fine-grained scaling emerging as a promising solution to mitigate outliers. However, existing methods struggle to capture nuanced block data distributions. We propose BlockDialect, a block-wise fine-grained mixed format technique that assigns a per-block optimal number format from a formatbook for better data representation. Additionally, we introduce DialectFP4, a formatbook of FP4 variants (akin to dialects) that adapt to diverse data distributions. To leverage this efficiently, we propose a two-stage approach for online DialectFP4 activation quantization. Importantly, DialectFP4 ensures energy efficiency by selecting representable values as scaled integers compatible with low-precision integer arithmetic. BlockDialect achieves 10.78% (7.48%) accuracy gain on the LLaMA3-8B (LLaMA2-7B) model compared to MXFP4 format with lower bit usage per data, while being only 5.45% (2.69%) below full precision even when quantizing full-path matrix multiplication. Focusing on how to represent over how to scale, our work presents a promising path for energy-efficient LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11911v1">Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Temporal Knowledge Graph Forecasting (TKGF) aims to predict future events based on the observed events in history. Recently, Large Language Models (LLMs) have exhibited remarkable capabilities, generating significant research interest in their application for reasoning over temporal knowledge graphs (TKGs). Existing LLM-based methods have integrated retrieved historical facts or static graph representations into LLMs. Despite the notable performance of LLM-based methods, they are limited by the insufficient modeling of temporal patterns and ineffective cross-modal alignment between graph and language, hindering the ability of LLMs to fully grasp the temporal and structural information in TKGs. To tackle these issues, we propose a novel framework TGL-LLM to integrate temporal graph learning into LLM-based temporal knowledge graph model. Specifically, we introduce temporal graph learning to capture the temporal and relational patterns and obtain the historical graph embedding. Furthermore, we design a hybrid graph tokenization to sufficiently model the temporal patterns within LLMs. To achieve better alignment between graph and language, we employ a two-stage training paradigm to finetune LLMs on high-quality and diverse data, thereby resulting in better performance. Extensive experiments on three real-world datasets show that our approach outperforms a range of state-of-the-art (SOTA) methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11877v1">From Drafts to Answers: Unlocking LLM Potential via Aggregation Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 20 pages; work in progress
    </div>
    <details class="paper-abstract">
      Scaling data and model size has been proven effective for boosting the performance of large language models. In addition to training-time scaling, recent studies have revealed that increasing test-time computational resources can further improve performance. In this work, we introduce Aggregation Fine-Tuning (AFT), a supervised finetuning paradigm where the model learns to synthesize multiple draft responses, referred to as proposals, into a single, refined answer, termed aggregation. At inference time, a propose-and-aggregate strategy further boosts performance by iteratively generating proposals and aggregating them. Empirical evaluations on benchmark datasets show that AFT-trained models substantially outperform standard SFT. Notably, an AFT model, fine-tuned from Llama3.1-8B-Base with only 64k data, achieves a 41.3% LC win rate on AlpacaEval 2, surpassing significantly larger LLMs such as Llama3.1-405B-Instruct and GPT4. By combining sequential refinement and parallel sampling, the propose-and-aggregate framework scales inference-time computation in a flexible manner. Overall, These findings position AFT as a promising approach to unlocking additional capabilities of LLMs without resorting to increasing data volume or model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12624v5">Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Offering a promising solution to the scalability challenges associated with human evaluation, the LLM-as-a-judge paradigm is rapidly gaining traction as an approach to evaluating large language models (LLMs). However, there are still many open questions about the strengths and weaknesses of this paradigm, and what potential biases it may hold. In this paper, we present a comprehensive study of the performance of various LLMs acting as judges, focusing on a clean scenario in which inter-human agreement is high. Investigating thirteen judge models of different model sizes and families, judging answers of nine different 'examtaker models' - both base and instruction-tuned - we find that only the best (and largest) models achieve reasonable alignment with humans. However, they are still quite far behind inter-human agreement and their assigned scores may still differ with up to 5 points from human-assigned scores. In terms of their ranking of the nine exam-taker models, instead, also smaller models and even the lexical metric contains may provide a reasonable signal. Through error analysis and other studies, we identify vulnerabilities in judge models, such as their sensitivity to prompt complexity and length, and a tendency toward leniency. The fact that even the best judges differ from humans in this comparatively simple setup suggest that caution may be wise when using judges in more complex setups. Lastly, our research rediscovers the importance of using alignment metrics beyond simple percent alignment, showing that judges with high percent agreement can still assign vastly different scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11051v2">FLAME: Learning to Navigate with Multimodal LLM in Urban Environments</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted to AAAI 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated potential in Vision-and-Language Navigation (VLN) tasks, yet current applications face challenges. While LLMs excel in general conversation scenarios, they struggle with specialized navigation tasks, yielding suboptimal performance compared to specialized VLN models. We introduce FLAME (FLAMingo-Architected Embodied Agent), a novel Multimodal LLM-based agent and architecture designed for urban VLN tasks that efficiently handles multiple observations. Our approach implements a three-phase tuning technique for effective adaptation to navigation tasks, including single perception tuning for street view description, multiple perception tuning for route summarization, and end-to-end training on VLN datasets. The augmented datasets are synthesized automatically. Experimental results demonstrate FLAME's superiority over existing methods, surpassing state-of-the-art methods by a 7.3% increase in task completion on Touchdown dataset. This work showcases the potential of Multimodal LLMs (MLLMs) in complex navigation tasks, representing an advancement towards applications of MLLMs in the field of embodied intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11864v1">LLM-Agents Driven Automated Simulation Testing and Analysis of small Uncrewed Aerial Systems</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted as full paper at ICSE-2025
    </div>
    <details class="paper-abstract">
      Thorough simulation testing is crucial for validating the correct behavior of small Uncrewed Aerial Systems (sUAS) across multiple scenarios, including adverse weather conditions (such as wind, and fog), diverse settings (hilly terrain, or urban areas), and varying mission profiles (surveillance, tracking). While various sUAS simulation tools exist to support developers, the entire process of creating, executing, and analyzing simulation tests remains a largely manual and cumbersome task. Developers must identify test scenarios, set up the simulation environment, integrate the System under Test (SuT) with simulation tools, formulate mission plans, and collect and analyze results. These labor-intensive tasks limit the ability of developers to conduct exhaustive testing across a wide range of scenarios. To alleviate this problem, in this paper, we propose AutoSimTest, a Large Language Model (LLM)-driven framework, where multiple LLM agents collaborate to support the sUAS simulation testing process. This includes: (1) creating test scenarios that subject the SuT to unique environmental contexts; (2) preparing the simulation environment as per the test scenario; (3) generating diverse sUAS missions for the SuT to execute; and (4) analyzing simulation results and providing an interactive analytics interface. Further, the design of the framework is flexible for creating and testing scenarios for a variety of sUAS use cases, simulation tools, and SuT input requirements. We evaluated our approach by (a) conducting simulation testing of PX4 and ArduPilot flight-controller-based SuTs, (b) analyzing the performance of each agent, and (c) gathering feedback from sUAS developers. Our findings indicate that AutoSimTest significantly improves the efficiency and scope of the sUAS testing process, allowing for more comprehensive and varied scenario evaluations while reducing the manual effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00253v4">CodeHalu: Investigating Code Hallucinations in LLMs via Execution-based Verification</a></div>
    <div class="paper-meta">
      📅 2025-01-21
      | 💬 Accepted by AAAI 2025 main conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant progress in code generation, offering developers groundbreaking automated programming support. However, LLMs often generate code that is syntactically correct and even semantically plausible, but may not execute as expected or fulfill specified requirements. This phenomenon of hallucinations in the code domain has not been systematically explored. To advance the community's understanding and research on this issue, we introduce the concept of code hallucinations and propose a classification method for code hallucination based on execution verification. We categorize code hallucinations into four main types: mapping, naming, resource, and logic hallucinations, with each category further divided into different subcategories to understand and address the unique challenges faced by LLMs in code generation with finer granularity. Additionally, we present a dynamic detection algorithm called CodeHalu designed to detect and quantify code hallucinations. We also introduce the CodeHaluEval benchmark, which includes 8,883 samples from 699 tasks, to systematically and quantitatively evaluate code hallucinations. By evaluating 17 popular LLMs using this benchmark, we reveal significant differences in their accuracy and reliability in code generation, offering detailed insights for further improving the code generation capabilities of LLMs. The CodeHalu benchmark and code are publicly available at https://github.com/yuchen814/CodeHalu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08067v3">Reward-Augmented Data Enhances Direct Preference Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      Preference alignment in Large Language Models (LLMs) has significantly improved their ability to adhere to human instructions and intentions. However, existing direct alignment algorithms primarily focus on relative preferences and often overlook the qualitative aspects of responses. Striving to maximize the implicit reward gap between the chosen and the slightly inferior rejected responses can cause overfitting and unnecessary unlearning of the high-quality rejected responses. The unawareness of the reward scores also drives the LLM to indiscriminately favor the low-quality chosen responses and fail to generalize to responses with the highest rewards, which are sparse in data. To overcome these shortcomings, our study introduces reward-conditioned LLM policies that discern and learn from the entire spectrum of response quality within the dataset, helping extrapolate to more optimal regions. We propose an effective yet simple data relabeling method that conditions the preference pairs on quality scores to construct a reward-augmented dataset. This dataset is easily integrated with existing direct alignment algorithms and is applicable to any preference dataset. The experimental results across instruction-following benchmarks including AlpacaEval, MT-Bench, and Arena-Hard-Auto demonstrate that our approach consistently boosts the performance of DPO by a considerable margin across diverse models. Additionally, our method improves the average accuracy on various academic benchmarks. When applying our method to on-policy data, the resulting DPO model achieves SOTA results on AlpacaEval. Through ablation studies, we demonstrate that our method not only maximizes the utility of preference data but also mitigates the issue of unlearning, demonstrating its broad effectiveness beyond mere dataset expansion. Our code is available at https://github.com/shenao-zhang/reward-augmented-preference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11833v1">Is your LLM trapped in a Mental Set? Investigative study on how mental sets affect the reasoning capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-01-21
    </div>
    <details class="paper-abstract">
      In this paper, we present an investigative study on how Mental Sets influence the reasoning capabilities of LLMs. LLMs have excelled in diverse natural language processing (NLP) tasks, driven by advancements in parameter-efficient fine-tuning (PEFT) and emergent capabilities like in-context learning (ICL). For complex reasoning tasks, selecting the right model for PEFT or ICL is critical, often relying on scores on benchmarks such as MMLU, MATH, and GSM8K. However, current evaluation methods, based on metrics like F1 Score or reasoning chain assessments by larger models, overlook a key dimension: adaptability to unfamiliar situations and overcoming entrenched thinking patterns. In cognitive psychology, Mental Set refers to the tendency to persist with previously successful strategies, even when they become inefficient - a challenge for problem solving and reasoning. We compare the performance of LLM models like Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct and GPT-4o in the presence of mental sets. To the best of our knowledge, this is the first study to integrate cognitive psychology concepts into the evaluation of LLMs for complex reasoning tasks, providing deeper insights into their adaptability and problem-solving efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11779v1">Glinthawk: A Two-Tiered Architecture for High-Throughput LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM) have revolutionized natural language processing, but their inference demands substantial resources, while under-utilizing high-end accelerators like GPUs. A major bottleneck arises from the attention mechanism, which requires storing large key-value caches, limiting the maximum achievable throughput way below the available computing resources. Current approaches attempt to mitigate this issue through memory-efficient attention and paging mechanisms, but remained constrained by the assumption that all operations must be performed on high-end accelerators. In this work, we propose Glinthawk, a two-tiered architecture that decouples the attention mechanism from the rest of the Transformer model. This approach allows the memory requirements for attention to scale independently, enabling larger batch sizes and more efficient use of the high-end accelerators. We prototype Glinthawk with NVIDIA T4 GPUs as one tier and standard CPU VMs as the other. Compared to a traditional single-tier setup, it improves throughput by $5.9\times$ and reduces cost of generation by $2.8\times$. For longer sequence lengths, it achieves $16.3\times$ throughput improvement at $2.4\times$ less cost. Our evaluation shows that this architecture can tolerate moderate network latency with minimal performance degradation, making it highly effective for latency-tolerant, throughput-oriented applications such as batch processing. We shared our prototype publicly at \url{https://github.com/microsoft/glinthawk}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17167v1">QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM Quality Checks</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      We introduce QualityFlow, a dynamic agentic workflow for program synthesis. Given the English description of a programming problem and a set of unit tests, the model's goal is to synthesize the correct program that solves the problem and passes the tests. QualityFlow consists of multiple large language model (LLM) agents that resemble a software development team, including code generation, testing, and self-debugging. Existing program synthesis methods face three major limitations: assumption of visible unit test conformity, bottleneck of synthesized test quality, and deviation of self-debugging trajectory. To address them, we propose the LLM Quality Checker, which explicitly "imagines" whether the synthesized programs' execution would conform to the unit tests. The Quality Checks dynamically control the workflow, including actions to submit the final answer, clarify the problem statement, and revert previous workflow steps. As a result, our Quality Checker can precisely accept any correct program, mitigate faulty synthesized tests, and prevent potential workflow deviation. The success of the Quality Checker further enables Diversified Prompting, which encourages variations in LLM responses to maximize the possibility that a correct program appears and passes the quality check. In experiments, QualityFlow establishes the state-of-the-art results on four program synthesis benchmarks: MBPP, HumanEval, and the stricter evaluations of both MBPP and HumanEval from EvalPlus. Our systematic analysis shows that the dynamic workflow controlled by LLM quality checks can outperform static workflows and single-attempt zero-shot synthesis. The Quality Checker is the center of our investigation, and we dissect its individual performance and integrated impact on the workflow accuracy, as well as other ablations experiments to justify our workflow design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11721v1">Explain-Query-Test: Self-Evaluating LLMs Via Explanation and Comprehension Discrepancy</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in generating detailed and coherent explanations of complex concepts. However, the extent to which these models truly comprehend the concepts they articulate remains unclear. To assess the level of comprehension of a model relative to the content it generates, we implemented a self-evaluation pipeline where models: (i) given a topic generate an excerpt with information about the topic, (ii) given an excerpt generate question-answer pairs, and finally (iii) given a question generate an answer. We refer to this self-evaluation approach as Explain-Query-Test (EQT). Interestingly, the accuracy on generated questions resulting from running the EQT pipeline correlates strongly with the model performance as verified by typical benchmarks such as MMLU-Pro. In other words, EQT's performance is predictive of MMLU-Pro's, and EQT can be used to rank models without the need for any external source of evaluation data other than lists of topics of interest. Moreover, our results reveal a disparity between the models' ability to produce detailed explanations and their performance on questions related to those explanations. This gap highlights fundamental limitations in the internal knowledge representation and reasoning abilities of current LLMs. We release the code at https://github.com/asgsaeid/EQT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11709v1">Towards Detecting Prompt Knowledge Gaps for Improved LLM-guided Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become essential in software development, especially for issue resolution. However, despite their widespread use, significant challenges persist in the quality of LLM responses to issue resolution queries. LLM interactions often yield incorrect, incomplete, or ambiguous information, largely due to knowledge gaps in prompt design, which can lead to unproductive exchanges and reduced developer productivity. In this paper, we analyze 433 developer-ChatGPT conversations within GitHub issue threads to examine the impact of prompt knowledge gaps and conversation styles on issue resolution. We identify four main knowledge gaps in developer prompts: Missing Context, Missing Specifications, Multiple Context, and Unclear Instructions. Assuming that conversations within closed issues contributed to successful resolutions while those in open issues did not, we find that ineffective conversations contain knowledge gaps in 54.7% of prompts, compared to only 13.2% in effective ones. Additionally, we observe seven distinct conversational styles, with Directive Prompting, Chain of Thought, and Responsive Feedback being the most prevalent. We find that knowledge gaps are present in all styles of conversations, with Missing Context being the most repeated challenge developers face in issue-resolution conversations. Based on our analysis, we identify key textual and code related heuristics-Specificity, Contextual Richness, and Clarity-that are associated with successful issue closure and help assess prompt quality. These heuristics lay the foundation for an automated tool that can dynamically flag unclear prompts and suggest structured improvements. To test feasibility, we developed a lightweight browser extension prototype for detecting prompt gaps, that can be easily adapted to other tools within developer workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13120v1">Multilinguality in LLM-Designed Reward Functions for Restless Bandits: Effects on Task Performance and Fairness</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 Accepted at the AAAI-2025 Deployable AI Workshop
    </div>
    <details class="paper-abstract">
      Restless Multi-Armed Bandits (RMABs) have been successfully applied to resource allocation problems in a variety of settings, including public health. With the rapid development of powerful large language models (LLMs), they are increasingly used to design reward functions to better match human preferences. Recent work has shown that LLMs can be used to tailor automated allocation decisions to community needs using language prompts. However, this has been studied primarily for English prompts and with a focus on task performance only. This can be an issue since grassroots workers, especially in developing countries like India, prefer to work in local languages, some of which are low-resource. Further, given the nature of the problem, biases along population groups unintended by the user are also undesirable. In this work, we study the effects on both task performance and fairness when the DLM algorithm, a recent work on using LLMs to design reward functions for RMABs, is prompted with non-English language commands. Specifically, we run the model on a synthetic environment for various prompts translated into multiple languages. The prompts themselves vary in complexity. Our results show that the LLM-proposed reward functions are significantly better when prompted in English compared to other languages. We also find that the exact phrasing of the prompt impacts task performance. Further, as prompt complexity increases, performance worsens for all languages; however, it is more robust with English prompts than with lower-resource languages. On the fairness side, we find that low-resource languages and more complex prompts are both highly likely to create unfairness along unintended dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11623v1">Early evidence of how LLMs outperform traditional systems on OCR/HTR tasks for historical records</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      We explore the ability of two LLMs -- GPT-4o and Claude Sonnet 3.5 -- to transcribe historical handwritten documents in a tabular format and compare their performance to traditional OCR/HTR systems: EasyOCR, Keras, Pytesseract, and TrOCR. Considering the tabular form of the data, two types of experiments are executed: one where the images are split line by line and the other where the entire scan is used as input. Based on CER and BLEU, we demonstrate that LLMs outperform the conventional OCR/HTR methods. Moreover, we also compare the evaluated CER and BLEU scores to human evaluations to better judge the outputs of whole-scan experiments and understand influential factors for CER and BLEU. Combining judgments from all the evaluation metrics, we conclude that two-shot GPT-4o for line-by-line images and two-shot Claude Sonnet 3.5 for whole-scan images yield the transcriptions of the historical records most similar to the ground truth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15825v2">60 Data Points are Sufficient to Fine-Tune LLMs for Question-Answering</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode extensive world knowledge through pre-training on massive datasets, which can then be fine-tuned for the question-answering (QA) task. However, effective strategies for fine-tuning LLMs for the QA task remain largely unexplored. To address this gap, we categorize supervised fine-tuning (SFT) data based on the extent of knowledge memorized by the pretrained LLMs and conduct a series of empirical analyses. Our experiments, involving four LLMs from three different model families, focus on three key factors: the amount of data required for SFT, the impact of different SFT datasets on model performance, and how data requirements vary across LLMs. The results show that as few as 60 data points during the SFT stage can activate the knowledge encoded during pre-training, enabling LLMs to perform the QA task. Additionally, SFT with data of varying memory levels has a significant impact on LLM performance, with the optimal dataset differing based on the specific model being fine-tuned. Future research will delve deeper into the mechanisms underlying these phenomena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13955v1">Guided Persona-based AI Surveys: Can we replicate personal mobility preferences at scale using LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      This study explores the potential of Large Language Models (LLMs) to generate artificial surveys, with a focus on personal mobility preferences in Germany. By leveraging LLMs for synthetic data creation, we aim to address the limitations of traditional survey methods, such as high costs, inefficiency and scalability challenges. A novel approach incorporating "Personas" - combinations of demographic and behavioural attributes - is introduced and compared to five other synthetic survey methods, which vary in their use of real-world data and methodological complexity. The MiD 2017 dataset, a comprehensive mobility survey in Germany, serves as a benchmark to assess the alignment of synthetic data with real-world patterns. The results demonstrate that LLMs can effectively capture complex dependencies between demographic attributes and preferences while offering flexibility to explore hypothetical scenarios. This approach presents valuable opportunities for transportation planning and social science research, enabling scalable, cost-efficient and privacy-preserving data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11468v1">LLM supervised Pre-training for Multimodal Emotion Recognition in Conversations</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 ICASSP 2025; 5 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Emotion recognition in conversations (ERC) is challenging due to the multimodal nature of the emotion expression. In this paper, we propose to pretrain a text-based recognition model from unsupervised speech transcripts with LLM guidance. These transcriptions are obtained from a raw speech dataset with a pre-trained ASR system. A text LLM model is queried to provide pseudo-labels for these transcripts, and these pseudo-labeled transcripts are subsequently used for learning an utterance level text-based emotion recognition model. We use the utterance level text embeddings for emotion recognition in conversations along with speech embeddings obtained from a recently proposed pre-trained model. A hierarchical way of training the speech-text model is proposed, keeping in mind the conversational nature of the dataset. We perform experiments on three established datasets, namely, IEMOCAP, MELD, and CMU- MOSI, where we illustrate that the proposed model improves over other benchmarks and achieves state-of-the-art results on two out of these three datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11411v1">Beyond the Hype: Benchmarking LLM-Evolved Heuristics for Bin Packing</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 To appear in Applications of Evolutionary Computation 28th International Conference, EvoApplications 2025
    </div>
    <details class="paper-abstract">
      Coupling Large Language Models (LLMs) with Evolutionary Algorithms has recently shown significant promise as a technique to design new heuristics that outperform existing methods, particularly in the field of combinatorial optimisation. An escalating arms race is both rapidly producing new heuristics and improving the efficiency of the processes evolving them. However, driven by the desire to quickly demonstrate the superiority of new approaches, evaluation of the new heuristics produced for a specific domain is often cursory: testing on very few datasets in which instances all belong to a specific class from the domain, and on few instances per class. Taking bin-packing as an example, to the best of our knowledge we conduct the first rigorous benchmarking study of new LLM-generated heuristics, comparing them to well-known existing heuristics across a large suite of benchmark instances using three performance metrics. For each heuristic, we then evolve new instances won by the heuristic and perform an instance space analysis to understand where in the feature space each heuristic performs well. We show that most of the LLM heuristics do not generalise well when evaluated across a broad range of benchmarks in contrast to existing simple heuristics, and suggest that any gains from generating very specialist heuristics that only work in small areas of the instance space need to be weighed carefully against the considerable cost of generating these heuristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13952v1">The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility?</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Recent years have witnessed extensive efforts to enhance Large Language Models (LLMs) across various domains, alongside growing attention to their ethical implications. However, a critical challenge remains largely overlooked: LLMs must balance between rejecting harmful requests for safety and accommodating legitimate ones for utility. This paper presents a Direct Preference Optimization (DPO) based alignment framework that achieves better overall performance by addressing this ethical-utility trade-off, using chemical domain applications as a proof-of-concept. Our alignment pipeline starts with a GPT-assisted three-phase data generation scheme, in which we create LibraChemQA, a chemical question-answering dataset comprising 31.6k triplet instances. By incorporating an innovative balanced seed in the data generation process, our framework systematically considers both legitimate and illegitimate requests. The framework also introduces a rephrasing mechanism for efficient data augmentation that enhances the model's chemical comprehension. We further develop a novel hybrid evaluation scheme with LLM judges for precise assessment of both safety and utility. Experimental results demonstrate our model's substantial improvements in overall performance where both safety and utility are considered - our resulting model, LibraChem, outperforms leading LLMs including Claude-3, GPT-4o, and LLaMA-3 by margins of 13.44%, 7.16%, and 7.10% respectively on our released benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.11361v2">Opportunistically Parallel Lambda Calculus. Or, Lambda: The Ultimate LLM Scripting Language</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Scripting languages are widely used to compose external calls, such as foreign functions that perform expensive computations, remote APIs, and more recently, machine learning systems such as large language models (LLMs). The execution time of scripts is often dominated by waiting for these external calls, and large speedups can be achieved via parallelization and streaming. However, doing this manually is challenging, even for expert programmers. To address this, we propose a novel opportunistic evaluation strategy for scripting languages based on a core lambda calculus that automatically executes external calls in parallel, as early as possible. We prove that our approach is confluent, ensuring that it preserves the programmer's original intent, and that our approach eventually executes every external call. We implement this approach in a framework called EPIC, embedded in Python. We demonstrate its versatility and performance on several applications drawn from the LLM literature, including Tree-of-Throughts and tool use. Our experiments show that opportunistic evaluation improves total running time (up to $6.2\times$) and latency (up to $12.7\times$) compared to several state-of-the-art baselines, while performing very close (between $1.3\%$ and $18.5\%$ running time overhead) to hand-tuned manually optimized parallel Rust implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11241v1">Irony in Emojis: A Comparative Study of Human and LLM Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-01-20
    </div>
    <details class="paper-abstract">
      Emojis have become a universal language in online communication, often carrying nuanced and context-dependent meanings. Among these, irony poses a significant challenge for Large Language Models (LLMs) due to its inherent incongruity between appearance and intent. This study examines the ability of GPT-4o to interpret irony in emojis. By prompting GPT-4o to evaluate the likelihood of specific emojis being used to express irony on social media and comparing its interpretations with human perceptions, we aim to bridge the gap between machine and human understanding. Our findings reveal nuanced insights into GPT-4o's interpretive capabilities, highlighting areas of alignment with and divergence from human behavior. Additionally, this research underscores the importance of demographic factors, such as age and gender, in shaping emoji interpretation and evaluates how these factors influence GPT-4o's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11233v1">PlotEdit: Natural Language-Driven Accessible Chart Editing in PDFs via Multimodal LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-01-20
      | 💬 Accepted at ECIR 2025
    </div>
    <details class="paper-abstract">
      Chart visualizations, while essential for data interpretation and communication, are predominantly accessible only as images in PDFs, lacking source data tables and stylistic information. To enable effective editing of charts in PDFs or digital scans, we present PlotEdit, a novel multi-agent framework for natural language-driven end-to-end chart image editing via self-reflective LLM agents. PlotEdit orchestrates five LLM agents: (1) Chart2Table for data table extraction, (2) Chart2Vision for style attribute identification, (3) Chart2Code for retrieving rendering code, (4) Instruction Decomposition Agent for parsing user requests into executable steps, and (5) Multimodal Editing Agent for implementing nuanced chart component modifications - all coordinated through multimodal feedback to maintain visual fidelity. PlotEdit outperforms existing baselines on the ChartCraft dataset across style, layout, format, and data-centric edits, enhancing accessibility for visually challenged users and improving novice productivity.
    </details>
</div>
