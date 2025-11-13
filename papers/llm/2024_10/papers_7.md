# llm - 2024_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11242v2">Measuring and Enhancing Trustworthiness of LLMs in RAG through Grounded Attributions and Learning to Refuse</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      LLMs are an integral component of retrieval-augmented generation (RAG) systems. While many studies focus on evaluating the overall quality of end-to-end RAG systems, there is a gap in understanding the appropriateness of LLMs for the RAG task. To address this, we introduce Trust-Score, a holistic metric that evaluates the trustworthiness of LLMs within the RAG framework. Our results show that various prompting methods, such as in-context learning, fail to effectively adapt LLMs to the RAG task as measured by Trust-Score. Consequently, we propose Trust-Align, a method to align LLMs for improved Trust-Score performance. The LLaMA-3 family, aligned using our method, significantly outperforms open-source LLMs of similar sizes on ASQA (up 14.0), QAMPARI (up 28.9), and ELI5 (up 13.7). We also demonstrate the effectiveness of Trust-Align across different open-weight models, including the LLaMA series (1b to 8b), Qwen-2.5 series (0.5b to 7b), and Phi3.5 (3.8b). We release our code at \url{https://anonymous.4open.science/r/trust-align}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00159v2">LLMs hallucinate graphs too: a structural perspective</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      It is known that LLMs do hallucinate, that is, they return incorrect information as facts. In this paper, we introduce the possibility to study these hallucinations under a structured form: graphs. Hallucinations in this context are incorrect outputs when prompted for well known graphs from the literature (e.g. Karate club, Les Mis\'erables, graph atlas). These hallucinated graphs have the advantage of being much richer than the factual accuracy -- or not -- of a statement; this paper thus argues that such rich hallucinations can be used to characterize the outputs of LLMs. Our first contribution observes the diversity of topological hallucinations from major modern LLMs. Our second contribution is the proposal of a metric for the amplitude of such hallucinations: the Graph Atlas Distance, that is the average graph edit distance from several graphs in the graph atlas set. We compare this metric to the Hallucination Leaderboard, a hallucination rank that leverages 10,000 times more prompts to obtain its ranking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15992v2">Can LLM Graph Reasoning Generalize beyond Pattern Memorization?</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 17 pages, 6 figures. EMNLP 2024 Findings. Code and data is publicly available at https://github.com/MatthewYZhang/NLGift
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate great potential for problems with implicit graphical structures, while recent works seek to enhance the graph reasoning capabilities of LLMs through specialized instruction tuning. The resulting 'graph LLMs' are evaluated with in-distribution settings only, thus it remains underexplored whether LLMs are learning generalizable graph reasoning skills or merely memorizing patterns in the synthetic training data. To this end, we propose the NLGift benchmark, an evaluation suite of LLM graph reasoning generalization: whether LLMs could go beyond semantic, numeric, structural, reasoning patterns in the synthetic training data and improve utility on real-world graph-based tasks. Extensive experiments with two LLMs across four graph reasoning tasks demonstrate that while generalization on simple patterns (semantic, numeric) is somewhat satisfactory, LLMs struggle to generalize across reasoning and real-world patterns, casting doubt on the benefit of synthetic graph tuning for real-world tasks with underlying network structures. We explore three strategies to improve LLM graph reasoning generalization, and we find that while post-training alignment is most promising for real-world tasks, empowering LLM graph reasoning to go beyond pattern memorization remains an open research question.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.17003v3">Safety Layers in Aligned Large Language Models: The Key to LLM Security</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Aligned LLMs are secure, capable of recognizing and refusing to answer malicious questions. However, the role of internal parameters in maintaining such security is not well understood yet, further these models can be vulnerable to security degradation when fine-tuned with non-malicious backdoor or normal data. To address these challenges, our work uncovers the mechanism behind security in aligned LLMs at the parameter level, identifying a small set of contiguous layers in the middle of the model that are crucial for distinguishing malicious queries from normal ones, referred to as "safety layers". We first confirm the existence of these safety layers by analyzing variations in input vectors within the model's internal layers. Additionally, we leverage the over-rejection phenomenon and parameters scaling analysis to precisely locate the safety layers. Building on these findings, we propose a novel fine-tuning approach, Safely Partial-Parameter Fine-Tuning (SPPFT), that fixes the gradient of the safety layers during fine-tuning to address the security degradation. Our experiments demonstrate that the proposed approach can significantly preserve LLM security while maintaining performance and reducing computational resources compared to full fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08527v1">Scaling Laws for Predicting Downstream Performance in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Precise estimation of downstream performance in large language models (LLMs) prior to training is essential for guiding their development process. Scaling laws analysis utilizes the statistics of a series of significantly smaller sampling language models (LMs) to predict the performance of the target LLM. For downstream performance prediction, the critical challenge lies in the emergent abilities in LLMs that occur beyond task-specific computational thresholds. In this work, we focus on the pre-training loss as a more computation-efficient metric for performance estimation. Our two-stage approach consists of first estimating a function that maps computational resources (e.g., FLOPs) to the pre-training Loss using a series of sampling models, followed by mapping the pre-training loss to downstream task Performance after the critical "emergent phase". In preliminary experiments, this FLP solution accurately predicts the performance of LLMs with 7B and 13B parameters using a series of sampling LMs up to 3B, achieving error margins of 5% and 10%, respectively, and significantly outperforming the FLOPs-to-Performance approach. This motivates FLP-M, a fundamental approach for performance prediction that addresses the practical need to integrate datasets from multiple sources during pre-training, specifically blending general corpora with code data to accurately represent the common necessity. FLP-M extends the power law analytical function to predict domain-specific pre-training loss based on FLOPs across data sources, and employs a two-layer neural network to model the non-linear relationship between multiple domain-specific loss and downstream performance. By utilizing a 3B LLM trained on a specific ratio and a series of smaller sampling LMs, FLP-M can effectively forecast the performance of 3B and 7B LLMs across various data mixtures for most benchmarks within 10% error margins.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16180v2">Benchmarking Japanese Speech Recognition on ASR-LLM Setups with Multi-Pass Augmented Generative Error Correction</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      With the strong representational power of large language models (LLMs), generative error correction (GER) for automatic speech recognition (ASR) aims to provide semantic and phonetic refinements to address ASR errors. This work explores how LLM-based GER can enhance and expand the capabilities of Japanese language processing, presenting the first GER benchmark for Japanese ASR with 0.9-2.6k text utterances. We also introduce a new multi-pass augmented generative error correction (MPA GER) by integrating multiple system hypotheses on the input side with corrections from multiple LLMs on the output side and then merging them. To the best of our knowledge, this is the first investigation of the use of LLMs for Japanese GER, which involves second-pass language modeling on the output transcriptions generated by the ASR system (e.g., N-best hypotheses). Our experiments demonstrated performance improvement in the proposed methods of ASR quality and generalization both in SPREDS-U1-ja and CSJ data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08500v1">Aerial Vision-and-Language Navigation via Semantic-Topo-Metric Representation Guided LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Submitted to ICRA 2025
    </div>
    <details class="paper-abstract">
      Aerial Vision-and-Language Navigation (VLN) is a novel task enabling Unmanned Aerial Vehicles (UAVs) to navigate in outdoor environments through natural language instructions and visual cues. It remains challenging due to the complex spatial relationships in outdoor aerial scenes. In this paper, we propose an end-to-end zero-shot framework for aerial VLN tasks, where the large language model (LLM) is introduced as our agent for action prediction. Specifically, we develop a novel Semantic-Topo-Metric Representation (STMR) to enhance the spatial reasoning ability of LLMs. This is achieved by extracting and projecting instruction-related semantic masks of landmarks into a top-down map that contains the location information of surrounding landmarks. Further, this map is transformed into a matrix representation with distance metrics as the text prompt to the LLM, for action prediction according to the instruction. Experiments conducted in real and simulation environments have successfully proved the effectiveness and robustness of our method, achieving 15.9% and 12.5% improvements (absolute) in Oracle Success Rate (OSR) on AerialVLN-S dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11192v4">I Learn Better If You Speak My Language: Understanding the Superior Performance of Fine-Tuning Large Language Models with LLM-Generated Responses</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 The paper has been accepted to EMNLP 2024 (Main Conference)
    </div>
    <details class="paper-abstract">
      This paper explores an intriguing observation: fine-tuning a large language model (LLM) with responses generated by a LLM often yields better results than using responses generated by humans, particularly in reasoning tasks. We conduct an in-depth investigation to understand why this occurs. Contrary to the common belief that these instances is due to the more detailed nature of LLM-generated content, our study identifies another contributing factor: an LLM is inherently more "familiar" with LLM generated responses. This familiarity is evidenced by lower perplexity before fine-tuning. We design a series of experiments to understand the impact of the "familiarity" and our conclusion reveals that this "familiarity" significantly impacts learning performance. Training with LLM-generated responses not only enhances performance but also helps maintain the model's capabilities in other reasoning tasks after fine-tuning on a specific task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00151v3">Scheherazade: Evaluating Chain-of-Thought Math Reasoning in LLMs with Chain-of-Problems</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Benchmarks are critical for measuring progress of math reasoning abilities of Large Language Models (LLMs). However, existing widely-used benchmarks such as GSM8K have been rendered less useful as multiple cutting-edge LLMs achieve over 94% accuracy. While harder benchmarks have been proposed, their creation is often manual and expensive. We present Scheherazade, an automated approach for producing challenging mathematical reasoning benchmarks by logically chaining mathematical reasoning problems. We propose two different chaining methods, forward chaining and backward chaining, which require reasoning forward and backward through the chain respectively. We apply Scheherazade on GSM8K to create GSM8K-Scheherazade and evaluate 3 frontier LLMs and OpenAI's o1-preview on it. We show that while frontier models' performance declines precipitously at only a few questions chained, a preliminary evaluation suggests o1-preview performance persists up to 5 questions chained backwards. In addition, while all other models perform worse when problems are chained backwards, o1-preview performs better on backward-chained benchmarks. We will release the dataset and code publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06682v2">Enhancing Multimodal LLM for Detailed and Accurate Video Captioning using Multi-Round Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Videos contain a wealth of information, and generating detailed and accurate descriptions in natural language is a key aspect of video understanding. In this paper, we present video-SALMONN 2, an advanced audio-visual large language model (LLM) with low-rank adaptation (LoRA) designed for enhanced video (with paired audio) captioning through directed preference optimization (DPO). We propose new metrics to evaluate the completeness and accuracy of video descriptions, which are optimized using DPO. To further improve training, we introduce a novel multi-round DPO (mrDPO) approach, which involves periodically updating the DPO reference model, merging and re-initializing the LoRA module as a proxy for parameter updates after each training round (1,000 steps), and incorporating guidance from ground-truth video captions to stabilize the process. To address potential catastrophic forgetting of non-captioning abilities due to mrDPO, we propose rebirth tuning, which finetunes the pre-DPO LLM by using the captions generated by the mrDPO-trained model as supervised labels. Experiments show that mrDPO significantly enhances video-SALMONN 2's captioning accuracy, reducing global and local error rates by 40\% and 20\%, respectively, while decreasing the repetition rate by 35\%. The final video-SALMONN 2 model, with just 7 billion parameters, surpasses leading models such as GPT-4o and Gemini-1.5-Pro in video captioning tasks, while maintaining competitive performance to the state-of-the-art on widely used video question-answering benchmark among models of similar size. Upon acceptance, we will release the code, model checkpoints, and training and test data. Demos are available at \href{https://video-salmonn-2.github.io}{https://video-salmonn-2.github.io}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07471v2">SEAL: Safety-enhanced Aligned LLM Fine-tuning via Bilevel Data Selection</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Fine-tuning on task-specific data to boost downstream performance is a crucial step for leveraging Large Language Models (LLMs). However, previous studies have demonstrated that fine-tuning the models on several adversarial samples or even benign data can greatly comprise the model's pre-equipped alignment and safety capabilities. In this work, we propose SEAL, a novel framework to enhance safety in LLM fine-tuning. SEAL learns a data ranker based on the bilevel optimization to up rank the safe and high-quality fine-tuning data and down rank the unsafe or low-quality ones. Models trained with SEAL demonstrate superior quality over multiple baselines, with 8.5% and 9.7% win rate increase compared to random selection respectively on Llama-3-8b-Instruct and Merlinite-7b models. Our code is available on github https://github.com/hanshen95/SEAL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08437v1">$\forall$uto$\exists$$\lor\!\land$L: Autonomous Evaluation of LLMs for Truth Maintenance and Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      This paper presents $\forall$uto$\exists$$\lor\!\land$L, a novel benchmark for scaling Large Language Model (LLM) assessment in formal tasks with clear notions of correctness, such as truth maintenance in translation and logical reasoning. $\forall$uto$\exists$$\lor\!\land$L is the first benchmarking paradigm that offers several key advantages necessary for scaling objective evaluation of LLMs without human labeling: (a) ability to evaluate LLMs of increasing sophistication by auto-generating tasks at different levels of difficulty; (b) auto-generation of ground truth that eliminates dependence on expensive and time-consuming human annotation; (c) the use of automatically generated, randomized datasets that mitigate the ability of successive LLMs to overfit to static datasets used in many contemporary benchmarks. Empirical analysis shows that an LLM's performance on $\forall$uto$\exists$$\lor\!\land$L is highly indicative of its performance on a diverse array of other benchmarks focusing on translation and reasoning tasks, making it a valuable autonomous evaluation paradigm in settings where hand-curated datasets can be hard to obtain and/or update.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21281v1">The Social Impact of Generative LLM-Based AI</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 34 pages, 3 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Liking it or not, ready or not, we are likely to enter a new phase of human history in which Artificial Intelligence (AI) will dominate economic production and social life -- the AI Revolution. Before the actual arrival of the AI Revolution, it is time for us to speculate on how AI will impact the social world. In this article, we focus on the social impact of generative LLM-based AI (GELLMAI), discussing societal factors that contribute to its technological development and its potential roles in enhancing both between-country and within-country social inequality. There are good indications that the US and China will lead the field and will be the main competitors for domination of AI in the world. We conjecture the AI Revolution will likely give rise to a post-knowledge society in which knowledge per se will become less important than in today's world. Instead, individual relationships and social identity will become more important. So will soft skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15948v2">Teaching LLMs to Abstain across Languages via Multilingual Feedback</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Multilingual LLMs often have knowledge disparities across languages, with larger gaps in under-resourced languages. Teaching LLMs to abstain in the face of knowledge gaps is thus a promising strategy to mitigate hallucinations in multilingual settings. However, previous studies on LLM abstention primarily focus on English; we find that directly applying existing solutions beyond English results in up to 20.5% performance gaps between high and low-resource languages, potentially due to LLMs' drop in calibration and reasoning beyond a few resource-rich languages. To this end, we propose strategies to enhance LLM abstention by learning from multilingual feedback, where LLMs self-reflect on proposed answers in one language by generating multiple feedback items in related languages: we show that this helps identifying the knowledge gaps across diverse languages, cultures, and communities. Extensive experiments demonstrate that our multilingual feedback approach outperforms various strong baselines, achieving up to 9.2% improvement for low-resource languages across three black-box and open models on three datasets, featuring open-book, closed-book, and commonsense QA. Further analysis reveals that multilingual feedback is both an effective and a more equitable abstain strategy to serve diverse language speakers, and cultural factors have great impact on language selection and LLM abstention behavior, highlighting future directions for multilingual and multi-cultural reliable language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15951v2">Modular Pluralism: Pluralistic Alignment via Multi-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      While existing alignment paradigms have been integral in developing large language models (LLMs), LLMs often learn an averaged human preference and struggle to model diverse preferences across cultures, demographics, and communities. We propose Modular Pluralism, a modular framework based on multi-LLM collaboration for pluralistic alignment: it "plugs into" a base LLM a pool of smaller but specialized community LMs, where models collaborate in distinct modes to flexibility support three modes of pluralism: Overton, steerable, and distributional. Modular Pluralism is uniquely compatible with black-box LLMs and offers the modular control of adding new community LMs for previously underrepresented communities. We evaluate Modular Pluralism with six tasks and four datasets featuring questions/instructions with value-laden and perspective-informed responses. Extensive experiments demonstrate that Modular Pluralism advances the three pluralism objectives across six black-box and open-source LLMs. Further analysis reveals that LLMs are generally faithful to the inputs from smaller community LLMs, allowing seamless patching by adding a new community LM to better cover previously underrepresented communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07484v2">WALL-E: World Alignment by Rule Learning Improves World Model-based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 35 pages, including references and appendix. Code is available at https://github.com/elated-sawyer/WALL-E
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) directly serve as powerful world models for model-based agents? While the gaps between the prior knowledge of LLMs and the specified environment's dynamics do exist, our study reveals that the gaps can be bridged by aligning an LLM with its deployed environment and such "world alignment" can be efficiently achieved by rule learning on LLMs. Given the rich prior knowledge of LLMs, only a few additional rules suffice to align LLM predictions with the specified environment dynamics. To this end, we propose a neurosymbolic approach to learn these rules gradient-free through LLMs, by inducing, updating, and pruning rules based on comparisons of agent-explored trajectories and world model predictions. The resulting world model is composed of the LLM and the learned rules. Our embodied LLM agent "WALL-E" is built upon model-predictive control (MPC). By optimizing look-ahead actions based on the precise world model, MPC significantly improves exploration and learning efficiency. Compared to existing LLM agents, WALL-E's reasoning only requires a few principal rules rather than verbose buffered trajectories being included in the LLM input. On open-world challenges in Minecraft and ALFWorld, WALL-E achieves higher success rates than existing methods, with lower costs on replanning time and the number of tokens used for reasoning. In Minecraft, WALL-E exceeds baselines by 15-30% in success rate while costing 8-20 fewer replanning rounds and only 60-80% of tokens. In ALFWorld, its success rate surges to a new record high of 95% only after 6 iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04965v3">Beyond Perplexity: Multi-dimensional Safety Evaluation of LLM Compression</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 Findings of EMNLP 2024
    </div>
    <details class="paper-abstract">
      Increasingly, model compression techniques enable large language models (LLMs) to be deployed in real-world applications. As a result of this momentum towards local deployment, compressed LLMs will interact with a large population. Prior work on compression typically prioritize preserving perplexity, which is directly analogous to training loss. The impact of compression method on other critical aspects of model behavior\, -- \,particularly safety\, -- \,requires systematic assessment. To this end, we investigate the impact of model compression along four dimensions: (1) degeneration harm, i.e., bias and toxicity in generation; (2) representational harm, i.e., biases in discriminative tasks; (3) dialect bias; and(4) language modeling and downstream task performance. We examine a wide spectrum of LLM compression techniques, including unstructured pruning, semi-structured pruning, and quantization. Our analysis reveals that compression can lead to unexpected consequences. Although compression may unintentionally alleviate LLMs' degeneration harm, it can still exacerbate representational harm. Furthermore, increasing compression produces a divergent impact on different protected groups. Finally, different compression methods have drastically different safety impacts: for example, quantization mostly preserves bias while pruning degrades quickly. Our findings underscore the importance of integrating safety assessments into the development of compressed LLMs to ensure their reliability across real-world applications.\footnote{Our implementation and results are available here: \url{https://github.com/zhichaoxu-shufe/Beyond-Perplexity-Compression-Safety-Eval}}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09268v1">One Step at a Time: Combining LLMs and Static Analysis to Generate Next-Step Hints for Programming Tasks</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Students often struggle with solving programming problems when learning to code, especially when they have to do it online, with one of the most common disadvantages of working online being the lack of personalized help. This help can be provided as next-step hint generation, i.e., showing a student what specific small step they need to do next to get to the correct solution. There are many ways to generate such hints, with large language models (LLMs) being among the most actively studied right now. While LLMs constitute a promising technology for providing personalized help, combining them with other techniques, such as static analysis, can significantly improve the output quality. In this work, we utilize this idea and propose a novel system to provide both textual and code hints for programming tasks. The pipeline of the proposed approach uses a chain-of-thought prompting technique and consists of three distinct steps: (1) generating subgoals - a list of actions to proceed with the task from the current student's solution, (2) generating the code to achieve the next subgoal, and (3) generating the text to describe this needed action. During the second step, we apply static analysis to the generated code to control its size and quality. The tool is implemented as a modification to the open-source JetBrains Academy plugin, supporting students in their in-IDE courses. To evaluate our approach, we propose a list of criteria for all steps in our pipeline and conduct two rounds of expert validation. Finally, we evaluate the next-step hints in a classroom with 14 students from two universities. Our results show that both forms of the hints - textual and code - were helpful for the students, and the proposed system helped them to proceed with the coding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09252v1">ReasonPlanner: Enhancing Autonomous Planning in Dynamic Environments with Temporal Knowledge Graphs and LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Planning and performing interactive tasks, such as conducting experiments to determine the melting point of an unknown substance, is straightforward for humans but poses significant challenges for autonomous agents. We introduce ReasonPlanner, a novel generalist agent designed for reflective thinking, planning, and interactive reasoning. This agent leverages LLMs to plan hypothetical trajectories by building a World Model based on a Temporal Knowledge Graph. The agent interacts with the environment using a natural language actor-critic module, where the actor translates the imagined trajectory into a sequence of actionable steps, and the critic determines if replanning is necessary. ReasonPlanner significantly outperforms previous state-of-the-art prompting-based methods on the ScienceWorld benchmark by more than 1.8 times, while being more sample-efficient and interpretable. It relies solely on frozen weights thus requiring no gradient updates. ReasonPlanner can be deployed and utilized without specialized knowledge of Machine Learning, making it accessible to a wide range of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09247v1">Benchmark Inflation: Revealing LLM Performance Gaps Using Retro-Holdouts</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      The training data for many Large Language Models (LLMs) is contaminated with test data. This means that public benchmarks used to assess LLMs are compromised, suggesting a performance gap between benchmark scores and actual capabilities. Ideally, a private holdout set could be used to accurately verify scores. Unfortunately, such datasets do not exist for most benchmarks, and post-hoc construction of sufficiently similar datasets is non-trivial. To address these issues, we introduce a systematic methodology for (i) retrospectively constructing a holdout dataset for a target dataset, (ii) demonstrating the statistical indistinguishability of this retro-holdout dataset, and (iii) comparing LLMs on the two datasets to quantify the performance gap due to the dataset's public availability. Applying these methods to TruthfulQA, we construct and release Retro-Misconceptions, on which we evaluate twenty LLMs and find that some have inflated scores by as much as 16 percentage points. Our results demonstrate that public benchmark scores do not always accurately assess model properties, and underscore the importance of improved data practices in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09244v1">Using off-the-shelf LLMs to query enterprise data by progressively revealing ontologies</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Ontologies are known to improve the accuracy of Large Language Models (LLMs) when translating natural language queries into a formal query language like SQL or SPARQL. There are two ways to leverage ontologies when working with LLMs. One is to fine-tune the model, i.e., to enhance it with specific domain knowledge. Another is the zero-shot prompting approach, where the ontology is provided as part of the input question. Unfortunately, modern enterprises typically have ontologies that are too large to fit in a prompt due to LLM's token size limitations. We present a solution that incrementally reveals "just enough" of an ontology that is needed to answer a given question.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12859v1">Enhancing Long Context Performance in LLMs Through Inner Loop Query Mechanism</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Transformers have a quadratic scaling of computational complexity with input size, which limits the input context window size of large language models (LLMs) in both training and inference. Meanwhile, retrieval-augmented generation (RAG) besed models can better handle longer contexts by using a retrieval system to filter out unnecessary information. However, most RAG methods only perform retrieval based on the initial query, which may not work well with complex questions that require deeper reasoning. We introduce a novel approach, Inner Loop Memory Augmented Tree Retrieval (ILM-TR), involving inner-loop queries, based not only on the query question itself but also on intermediate findings. At inference time, our model retrieves information from the RAG system, integrating data from lengthy documents at various levels of abstraction. Based on the information retrieved, the LLM generates texts stored in an area named Short-Term Memory (STM) which is then used to formulate the next query. This retrieval process is repeated until the text in STM converged. Our experiments demonstrate that retrieval with STM offers improvements over traditional retrieval-augmented LLMs, particularly in long context tests such as Multi-Needle In A Haystack (M-NIAH) and BABILong.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.18216v2">LLM Task Interference: An Initial Study on the Impact of Task-Switch in Conversational History</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 20 pages, 13 figures, 20 tables, EMNLP Main Conference 2024
    </div>
    <details class="paper-abstract">
      With the recent emergence of powerful instruction-tuned large language models (LLMs), various helpful conversational Artificial Intelligence (AI) systems have been deployed across many applications. When prompted by users, these AI systems successfully perform a wide range of tasks as part of a conversation. To provide some sort of memory and context, such approaches typically condition their output on the entire conversational history. Although this sensitivity to the conversational history can often lead to improved performance on subsequent tasks, we find that performance can in fact also be negatively impacted, if there is a task-switch. To the best of our knowledge, our work makes the first attempt to formalize the study of such vulnerabilities and interference of tasks in conversational LLMs caused by task-switches in the conversational history. Our experiments across 5 datasets with 15 task switches using popular LLMs reveal that many of the task-switches can lead to significant performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02825v2">Ingest-And-Ground: Dispelling Hallucinations from Continually-Pretrained LLMs with RAG</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      This paper presents new methods that have the potential to improve privacy process efficiency with LLM and RAG. To reduce hallucination, we continually pre-train the base LLM model with a privacy-specific knowledge base and then augment it with a semantic RAG layer. Our evaluations demonstrate that this approach enhances the model performance (as much as doubled metrics compared to out-of-box LLM) in handling privacy-related queries, by grounding responses with factual information which reduces inaccuracies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12263v2">Defending Against Social Engineering Attacks in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) poses challenges in detecting and mitigating digital deception, as these models can emulate human conversational patterns and facilitate chat-based social engineering (CSE) attacks. This study investigates the dual capabilities of LLMs as both facilitators and defenders against CSE threats. We develop a novel dataset, SEConvo, simulating CSE scenarios in academic and recruitment contexts, and designed to examine how LLMs can be exploited in these situations. Our findings reveal that, while off-the-shelf LLMs generate high-quality CSE content, their detection capabilities are suboptimal, leading to increased operational costs for defense. In response, we propose ConvoSentinel, a modular defense pipeline that improves detection at both the message and the conversation levels, offering enhanced adaptability and cost-effectiveness. The retrieval-augmented module in ConvoSentinel identifies malicious intent by comparing messages to a database of similar conversations, enhancing CSE detection at all stages. Our study highlights the need for advanced strategies to leverage LLMs in cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09168v1">Hybrid Training Approaches for LLMs: Leveraging Real and Synthetic Data to Enhance Model Performance in Domain-Specific Applications</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 22 pages, 7 figures
    </div>
    <details class="paper-abstract">
      This research explores a hybrid approach to fine-tuning large language models (LLMs) by integrating real-world and synthetic data to boost model performance, particularly in generating accurate and contextually relevant responses. By leveraging a dataset combining transcribed real interactions with high-quality synthetic sessions, we aimed to overcome the limitations of scarce, noisy, and domain-specific real data. Synthetic personas and scenarios were employed to enhance training diversity. The study evaluated three models: a base foundational model, a model fine-tuned with real data, and a hybrid fine-tuned model. Experimental results showed that the hybrid model consistently outperformed the others in specific vertical applications, achieving the highest scores across all metrics. Further testing confirmed the hybrid model's superior adaptability and contextual understanding across diverse scenarios. These findings suggest that combining real and synthetic data can significantly improve the robustness and contextual sensitivity of LLMs, particularly in domain-specific and vertical use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09040v1">AttnGCG: Enhancing Jailbreaking Attacks on LLMs with Attention Manipulation</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      This paper studies the vulnerabilities of transformer-based Large Language Models (LLMs) to jailbreaking attacks, focusing specifically on the optimization-based Greedy Coordinate Gradient (GCG) strategy. We first observe a positive correlation between the effectiveness of attacks and the internal behaviors of the models. For instance, attacks tend to be less effective when models pay more attention to system prompts designed to ensure LLM safety alignment. Building on this discovery, we introduce an enhanced method that manipulates models' attention scores to facilitate LLM jailbreaking, which we term AttnGCG. Empirically, AttnGCG shows consistent improvements in attack efficacy across diverse LLMs, achieving an average increase of ~7% in the Llama-2 series and ~10% in the Gemma series. Our strategy also demonstrates robust attack transferability against both unseen harmful goals and black-box LLMs like GPT-3.5 and GPT-4. Moreover, we note our attention-score visualization is more interpretable, allowing us to gain better insights into how our targeted attention manipulation facilitates more effective jailbreaking. We release the code at https://github.com/UCSC-VLAA/AttnGCG-attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06800v3">LLM-Generated Black-box Explanations Can Be Adversarially Helpful</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 NeurIPS Regulatable ML Workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming vital tools that help us solve and understand complex problems by acting as digital assistants. LLMs can generate convincing explanations, even when only given the inputs and outputs of these problems, i.e., in a ``black-box'' approach. However, our research uncovers a hidden risk tied to this approach, which we call *adversarial helpfulness*. This happens when an LLM's explanations make a wrong answer look right, potentially leading people to trust incorrect solutions. In this paper, we show that this issue affects not just humans, but also LLM evaluators. Digging deeper, we identify and examine key persuasive strategies employed by LLMs. Our findings reveal that these models employ strategies such as reframing the questions, expressing an elevated level of confidence, and cherry-picking evidence to paint misleading answers in a credible light. To examine if LLMs are able to navigate complex-structured knowledge when generating adversarially helpful explanations, we create a special task based on navigating through graphs. Most LLMs are not able to find alternative paths along simple graphs, indicating that their misleading explanations aren't produced by only logical deductions using complex knowledge. These findings shed light on the limitations of the black-box explanation setting and allow us to provide advice on the safe usage of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12856v1">Optimized Biomedical Question-Answering Services with LLM and Multi-BERT Integration</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 10 pages, 12 figures, accepted and to be published in the proceedings of 2024 IEEE International Conference on Data Mining Workshops (ICDMW)
    </div>
    <details class="paper-abstract">
      We present a refined approach to biomedical question-answering (QA) services by integrating large language models (LLMs) with Multi-BERT configurations. By enhancing the ability to process and prioritize vast amounts of complex biomedical data, this system aims to support healthcare professionals in delivering better patient outcomes and informed decision-making. Through innovative use of BERT and BioBERT models, combined with a multi-layer perceptron (MLP) layer, we enable more specialized and efficient responses to the growing demands of the healthcare sector. Our approach not only addresses the challenge of overfitting by freezing one BERT model while training another but also improves the overall adaptability of QA services. The use of extensive datasets, such as BioASQ and BioMRC, demonstrates the system's ability to synthesize critical information. This work highlights how advanced language models can make a tangible difference in healthcare, providing reliable and responsive tools for professionals to manage complex information, ultimately serving the broader goal of improved care and data-driven insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.20086v3">Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 13 pages, 14 figures. Code and data at https://footprints.baulab.info/
    </div>
    <details class="paper-abstract">
      LLMs process text as sequences of tokens that roughly correspond to words, where less common words are represented by multiple tokens. However, individual tokens are often semantically unrelated to the meanings of the words/concepts they comprise. For example, Llama-2-7b's tokenizer splits the word "northeastern" into the tokens ['_n', 'ort', 'he', 'astern'], none of which correspond to semantically meaningful units like "north" or "east." Similarly, the overall meanings of named entities like "Neil Young" and multi-word expressions like "break a leg" cannot be directly inferred from their constituent tokens. Mechanistically, how do LLMs convert such arbitrary groups of tokens into useful higher-level representations? In this work, we find that last token representations of named entities and multi-token words exhibit a pronounced "erasure" effect, where information about previous and current tokens is rapidly forgotten in early layers. Using this observation, we propose a method to "read out" the implicit vocabulary of an autoregressive LLM by examining differences in token representations across layers, and present results of this method for Llama-2-7b and Llama-3-8B. To our knowledge, this is the first attempt to probe the implicit vocabulary of an LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08948v1">The Dynamics of Social Conventions in LLM populations: Spontaneous Emergence, Collective Biases and Tipping Points</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Social conventions are the foundation for social and economic life. As legions of AI agents increasingly interact with each other and with humans, their ability to form shared conventions will determine how effectively they will coordinate behaviors, integrate into society and influence it. Here, we investigate the dynamics of conventions within populations of Large Language Model (LLM) agents using simulated interactions. First, we show that globally accepted social conventions can spontaneously arise from local interactions between communicating LLMs. Second, we demonstrate how strong collective biases can emerge during this process, even when individual agents appear to be unbiased. Third, we examine how minority groups of committed LLMs can drive social change by establishing new social conventions. We show that once these minority groups reach a critical size, they can consistently overturn established behaviors. In all cases, contrasting the experimental results with predictions from a minimal multi-agent model allows us to isolate the specific role of LLM agents. Our results clarify how AI systems can autonomously develop norms without explicit programming and have implications for designing AI systems that align with human values and societal goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08911v1">Test-driven Software Experimentation with LASSO: an LLM Benchmarking Example</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Empirical software engineering faces a critical gap: the lack of standardized tools for rapid development and execution of Test-Driven Software Experiments (TDSEs) - that is, experiments that involve the execution of software subjects and the observation and analysis of their "de facto" run-time behavior. In this paper we present a general-purpose analysis platform called LASSO that provides a minimal set of domain-specific languages and data structures to conduct TDSEs. By empowering users with an executable scripting language to design and execute TDSEs, LASSO enables efficient evaluation of run-time semantics and execution characteristics in addition to statically determined properties. We present an example TDSE that demonstrates the practical benefits of LASSO's scripting capabilities for assessing the reliability of LLMs for code generation by means of a self-contained, reusable and extensible study script. The LASSO platform is freely available at: https://softwareobservatorium.github.io/, and a demo video is available on YouTube: https://youtu.be/tzY9oNTWXzw
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11557v4">A Quick, trustworthy spectral knowledge Q&A system leveraging retrieval-augmented generation on LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-11
      | 💬 16 pages,10 figures,3 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) has demonstrated significant success in a range of natural language processing (NLP) tasks within general domain. The emergence of LLM has introduced innovative methodologies across diverse fields, including the natural sciences. Researchers aim to implement automated, concurrent process driven by LLM to supplant conventional manual, repetitive and labor-intensive work. In the domain of spectral analysis and detection, it is imperative for researchers to autonomously acquire pertinent knowledge across various research objects, which encompasses the spectroscopic techniques and the chemometric methods that are employed in experiments and analysis. Paradoxically, despite the recognition of spectroscopic detection as an effective analytical method, the fundamental process of knowledge retrieval remains both time-intensive and repetitive. In response to this challenge, we first introduced the Spectral Detection and Analysis Based Paper(SDAAP) dataset, which is the first open-source textual knowledge dataset for spectral analysis and detection and contains annotated literature data as well as corresponding knowledge instruction data. Subsequently, we also designed an automated Q\&A framework based on the SDAAP dataset, which can retrieve relevant knowledge and generate high-quality responses by extracting entities in the input as retrieval parameters. It is worth noting that: within this framework, LLM is only used as a tool to provide generalizability, while RAG technique is used to accurately capture the source of the knowledge.This approach not only improves the quality of the generated responses, but also ensures the traceability of the knowledge. Experimental results show that our framework generates responses with more reliable expertise compared to the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08860v1">Audio Description Generation in the Era of LLMs and VLMs: A Review of Transferable Generative AI Technologies</a></div>
    <div class="paper-meta">
      📅 2024-10-11
    </div>
    <details class="paper-abstract">
      Audio descriptions (ADs) function as acoustic commentaries designed to assist blind persons and persons with visual impairments in accessing digital media content on television and in movies, among other settings. As an accessibility service typically provided by trained AD professionals, the generation of ADs demands significant human effort, making the process both time-consuming and costly. Recent advancements in natural language processing (NLP) and computer vision (CV), particularly in large language models (LLMs) and vision-language models (VLMs), have allowed for getting a step closer to automatic AD generation. This paper reviews the technologies pertinent to AD generation in the era of LLMs and VLMs: we discuss how state-of-the-art NLP and CV technologies can be applied to generate ADs and identify essential research directions for the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21280v1">TraderTalk: An LLM Behavioural ABM applied to Simulating Human Bilateral Trading Interactions</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      We introduce a novel hybrid approach that augments Agent-Based Models (ABMs) with behaviors generated by Large Language Models (LLMs) to simulate human trading interactions. We call our model TraderTalk. Leveraging LLMs trained on extensive human-authored text, we capture detailed and nuanced representations of bilateral conversations in financial trading. Applying this Generative Agent-Based Model (GABM) to government bond markets, we replicate trading decisions between two stylised virtual humans. Our method addresses both structural challenges, such as coordinating turn-taking between realistic LLM-based agents, and design challenges, including the interpretation of LLM outputs by the agent model. By exploring prompt design opportunistically rather than systematically, we enhance the realism of agent interactions without exhaustive overfitting or model reliance. Our approach successfully replicates trade-to-order volume ratios observed in related asset markets, demonstrating the potential of LLM-augmented ABMs in financial simulations
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13246v2">GSR-BENCH: A Benchmark for Grounded Spatial Reasoning Evaluation via Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 Accepted to NeurIPS 2024 Workshop on Compositional Learning
    </div>
    <details class="paper-abstract">
      The ability to understand and reason about spatial relationships between objects in images is an important component of visual reasoning. This skill rests on the ability to recognize and localize objects of interest and determine their spatial relation. Early vision and language models (VLMs) have been shown to struggle to recognize spatial relations. We extend the previously released What'sUp dataset and propose a novel comprehensive evaluation for spatial relationship understanding that highlights the strengths and weaknesses of 27 different models. In addition to the VLMs evaluated in What'sUp, our extensive evaluation encompasses 3 classes of Multimodal LLMs (MLLMs) that vary in their parameter sizes (ranging from 7B to 110B), training/instruction-tuning methods, and visual resolution to benchmark their performances and scrutinize the scaling laws in this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.00113v2">Wait, that's not an option: LLMs Robustness with Incorrect Multiple-Choice Options</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 Accepted for NeurIPS 2024 FM-EduAssess Workshop
    </div>
    <details class="paper-abstract">
      Decision-making under full alignment requires balancing between reasoning and faithfulness - a challenge for large language models (LLMs). This study explores whether LLMs prioritize following instructions over reasoning and truth when given "misleading" instructions, such as "Respond solely with A or B", even when neither option is correct. We introduce a new metric called "reflective judgment", which sheds new light on the relationship between the pre-training and post-training alignment schemes. In tasks ranging from basic arithmetic to domain-specific assessments, models like GPT-4o, o1-mini, or Claude 3 Opus adhered to instructions correctly but failed to reflect on the validity of the provided options. Contrary, models from the Llama 3.1 family (8B, 70B, 405B) or base Qwen2.5 (7B, 14B, 32B) families exhibit improved refusal rates with size, indicating a scaling effect. We also observed that alignment techniques, though intended to enhance reasoning, sometimes weakened the models' ability to reject incorrect instructions, leading them to follow flawed prompts uncritically. Finally, we have also conducted a parallel human study revealing similar patterns in human behavior and annotations. We highlight how popular RLHF datasets might disrupt either training or evaluation due to annotations exhibiting poor reflective judgement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08197v1">From Exploration to Mastery: Enabling LLMs to Master Tools via Self-Driven Interactions</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Tool learning enables Large Language Models (LLMs) to interact with external environments by invoking tools, serving as an effective strategy to mitigate the limitations inherent in their pre-training data. In this process, tool documentation plays a crucial role by providing usage instructions for LLMs, thereby facilitating effective tool utilization. This paper concentrates on the critical challenge of bridging the comprehension gap between LLMs and external tools due to the inadequacies and inaccuracies inherent in existing human-centric tool documentation. We propose a novel framework, DRAFT, aimed at Dynamically Refining tool documentation through the Analysis of Feedback and Trails emanating from LLMs' interactions with external tools. This methodology pivots on an innovative trial-and-error approach, consisting of three distinct learning phases: experience gathering, learning from experience, and documentation rewriting, to iteratively enhance the tool documentation. This process is further optimized by implementing a diversity-promoting exploration strategy to ensure explorative diversity and a tool-adaptive termination mechanism to prevent overfitting while enhancing efficiency. Extensive experiments on multiple datasets demonstrate that DRAFT's iterative, feedback-based refinement significantly ameliorates documentation quality, fostering a deeper comprehension and more effective utilization of tools by LLMs. Notably, our analysis reveals that the tool documentation refined via our approach demonstrates robust cross-model generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08189v1">SG-Nav: Online 3D Scene Graph Prompting for LLM-based Zero-shot Object Navigation</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 Accepted to NeurIPS 2024. Project page: https://bagh2178.github.io/SG-Nav/
    </div>
    <details class="paper-abstract">
      In this paper, we propose a new framework for zero-shot object navigation. Existing zero-shot object navigation methods prompt LLM with the text of spatially closed objects, which lacks enough scene context for in-depth reasoning. To better preserve the information of environment and fully exploit the reasoning ability of LLM, we propose to represent the observed scene with 3D scene graph. The scene graph encodes the relationships between objects, groups and rooms with a LLM-friendly structure, for which we design a hierarchical chain-of-thought prompt to help LLM reason the goal location according to scene context by traversing the nodes and edges. Moreover, benefit from the scene graph representation, we further design a re-perception mechanism to empower the object navigation framework with the ability to correct perception error. We conduct extensive experiments on MP3D, HM3D and RoboTHOR environments, where SG-Nav surpasses previous state-of-the-art zero-shot methods by more than 10% SR on all benchmarks, while the decision process is explainable. To the best of our knowledge, SG-Nav is the first zero-shot method that achieves even higher performance than supervised object navigation methods on the challenging MP3D benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08133v1">Assessing Episodic Memory in LLMs with Sequence Order Recall Tasks</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Current LLM benchmarks focus on evaluating models' memory of facts and semantic relations, primarily assessing semantic aspects of long-term memory. However, in humans, long-term memory also includes episodic memory, which links memories to their contexts, such as the time and place they occurred. The ability to contextualize memories is crucial for many cognitive tasks and everyday functions. This form of memory has not been evaluated in LLMs with existing benchmarks. To address the gap in evaluating memory in LLMs, we introduce Sequence Order Recall Tasks (SORT), which we adapt from tasks used to study episodic memory in cognitive psychology. SORT requires LLMs to recall the correct order of text segments, and provides a general framework that is both easily extendable and does not require any additional annotations. We present an initial evaluation dataset, Book-SORT, comprising 36k pairs of segments extracted from 9 books recently added to the public domain. Based on a human experiment with 155 participants, we show that humans can recall sequence order based on long-term memory of a book. We find that models can perform the task with high accuracy when relevant text is given in-context during the SORT evaluation. However, when presented with the book text only during training, LLMs' performance on SORT falls short. By allowing to evaluate more aspects of memory, we believe that SORT will aid in the emerging development of memory-augmented models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08115v1">Optima: Optimizing Effectiveness and Efficiency for LLM-Based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) based multi-agent systems (MAS) show remarkable potential in collaborative problem-solving, yet they still face critical challenges: low communication efficiency, poor scalability, and a lack of effective parameter-updating optimization methods. We present Optima, a novel framework that addresses these issues by significantly enhancing both communication efficiency and task effectiveness in LLM-based MAS through LLM training. Optima employs an iterative generate, rank, select, and train paradigm with a reward function balancing task performance, token efficiency, and communication readability. We explore various RL algorithms, including Supervised Fine-Tuning, Direct Preference Optimization, and their hybrid approaches, providing insights into their effectiveness-efficiency trade-offs. We integrate Monte Carlo Tree Search-inspired techniques for DPO data generation, treating conversation turns as tree nodes to explore diverse interaction paths. Evaluated on common multi-agent tasks, including information-asymmetric question answering and complex reasoning, Optima shows consistent and substantial improvements over single-agent baselines and vanilla MAS based on Llama 3 8B, achieving up to 2.8x performance gain with less than 10\% tokens on tasks requiring heavy information exchange. Moreover, Optima's efficiency gains open new possibilities for leveraging inference-compute more effectively, leading to improved inference-time scaling laws. By addressing fundamental challenges in LLM-based MAS, Optima shows the potential towards scalable, efficient, and effective MAS (https://chenweize1998.github.io/optima-project-page).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05292v5">How to Teach Programming in the AI Era? Using LLMs as a Teachable Agent for Debugging</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) now excel at generative skills and can create content at impeccable speeds. However, they are imperfect and still make various mistakes. In a Computer Science education context, as these models are widely recognized as "AI pair programmers," it becomes increasingly important to train students on evaluating and debugging the LLM-generated code. In this work, we introduce HypoCompass, a novel system to facilitate deliberate practice on debugging, where human novices play the role of Teaching Assistants and help LLM-powered teachable agents debug code. We enable effective task delegation between students and LLMs in this learning-by-teaching environment: students focus on hypothesizing the cause of code errors, while adjacent skills like code completion are offloaded to LLM-agents. Our evaluations demonstrate that HypoCompass generates high-quality training materials (e.g., bugs and fixes), outperforming human counterparts fourfold in efficiency, and significantly improves student performance on debugging by 12% in the pre-to-post test.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08255v1">Generalization from Starvation: Hints of Universality in LLM Knowledge Graph Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 14 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Motivated by interpretability and reliability, we investigate how neural networks represent knowledge during graph learning, We find hints of universality, where equivalent representations are learned across a range of model sizes (from $10^2$ to $10^9$ parameters) and contexts (MLP toy models, LLM in-context learning and LLM training). We show that these attractor representations optimize generalization to unseen examples by exploiting properties of knowledge graph relations (e.g. symmetry and meta-transitivity). We find experimental support for such universality by showing that LLMs and simpler neural networks can be stitched, i.e., by stitching the first part of one model to the last part of another, mediated only by an affine or almost affine transformation. We hypothesize that this dynamic toward simplicity and generalization is driven by "intelligence from starvation": where overfitting is minimized by pressure to minimize the use of resources that are either scarce or competed for against other tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08048v1">VerifierQ: Enhancing LLM Test Time Compute with Q-Learning-based Verifiers</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Recent advancements in test time compute, particularly through the use of verifier models, have significantly enhanced the reasoning capabilities of Large Language Models (LLMs). This generator-verifier approach closely resembles the actor-critic framework in reinforcement learning (RL). However, current verifier models in LLMs often rely on supervised fine-tuning without temporal difference learning such as Q-learning. This paper introduces VerifierQ, a novel approach that integrates Offline Q-learning into LLM verifier models. We address three key challenges in applying Q-learning to LLMs: (1) handling utterance-level Markov Decision Processes (MDPs), (2) managing large action spaces, and (3) mitigating overestimation bias. VerifierQ introduces a modified Bellman update for bounded Q-values, incorporates Implicit Q-learning (IQL) for efficient action space management, and integrates a novel Conservative Q-learning (CQL) formulation for balanced Q-value estimation. Our method enables parallel Q-value computation and improving training efficiency. While recent work has explored RL techniques like MCTS for generators, VerifierQ is among the first to investigate the verifier (critic) aspect in LLMs through Q-learning. This integration of RL principles into verifier models complements existing advancements in generator techniques, potentially enabling more robust and adaptive reasoning in LLMs. Experimental results on mathematical reasoning tasks demonstrate VerifierQ's superior performance compared to traditional supervised fine-tuning approaches, with improvements in efficiency, accuracy and robustness. By enhancing the synergy between generation and evaluation capabilities, VerifierQ contributes to the ongoing evolution of AI systems in addressing complex cognitive tasks across various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08014v1">LLM Cascade with Multi-Objective Optimal Consideration</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional capabilities in understanding and generating natural language. However, their high deployment costs often pose a barrier to practical applications, especially. Cascading local and server models offers a promising solution to this challenge. While existing studies on LLM cascades have primarily focused on the performance-cost trade-off, real-world scenarios often involve more complex requirements. This paper introduces a novel LLM Cascade strategy with Multi-Objective Optimization, enabling LLM cascades to consider additional objectives (e.g., privacy) and better align with the specific demands of real-world applications while maintaining their original cascading abilities. Extensive experiments on three benchmarks validate the effectiveness and superiority of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07962v1">Towards Assurance of LLM Adversarial Robustness using Ontology-Driven Argumentation</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 To be published in xAI 2024, late-breaking track
    </div>
    <details class="paper-abstract">
      Despite the impressive adaptability of large language models (LLMs), challenges remain in ensuring their security, transparency, and interpretability. Given their susceptibility to adversarial attacks, LLMs need to be defended with an evolving combination of adversarial training and guardrails. However, managing the implicit and heterogeneous knowledge for continuously assuring robustness is difficult. We introduce a novel approach for assurance of the adversarial robustness of LLMs based on formal argumentation. Using ontologies for formalization, we structure state-of-the-art attacks and defenses, facilitating the creation of a human-readable assurance case, and a machine-readable representation. We demonstrate its application with examples in English language and code translation tasks, and provide implications for theory and practice, by targeting engineers, data scientists, users, and auditors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07959v1">COMPL-AI Framework: A Technical Interpretation and LLM Benchmarking Suite for the EU Artificial Intelligence Act</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      The EU's Artificial Intelligence Act (AI Act) is a significant step towards responsible AI development, but lacks clear technical interpretation, making it difficult to assess models' compliance. This work presents COMPL-AI, a comprehensive framework consisting of (i) the first technical interpretation of the EU AI Act, translating its broad regulatory requirements into measurable technical requirements, with the focus on large language models (LLMs), and (ii) an open-source Act-centered benchmarking suite, based on thorough surveying and implementation of state-of-the-art LLM benchmarks. By evaluating 12 prominent LLMs in the context of COMPL-AI, we reveal shortcomings in existing models and benchmarks, particularly in areas like robustness, safety, diversity, and fairness. This work highlights the need for a shift in focus towards these aspects, encouraging balanced development of LLMs and more comprehensive regulation-aligned benchmarks. Simultaneously, COMPL-AI for the first time demonstrates the possibilities and difficulties of bringing the Act's obligations to a more concrete, technical level. As such, our work can serve as a useful first step towards having actionable recommendations for model providers, and contributes to ongoing efforts of the EU to enable application of the Act, such as the drafting of the GPAI Code of Practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.09265v2">Sharing Matters: Analysing Neurons Across Languages and Tasks in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized the field of natural language processing (NLP), and recent studies have aimed to understand their underlying mechanisms. However, most of this research is conducted within a monolingual setting, primarily focusing on English. Few studies attempt to explore the internal workings of LLMs in multilingual settings. In this study, we aim to fill the research gap by examining how neuron activation is shared across tasks and languages. We classify neurons into four distinct categories based on their responses to a specific input across different languages:all-shared, partial-shared, specific, and non-activated. This categorization is combined with a study of neuron attribution, i.e. the importance of a neuron w.r.t an output. Our analysis reveals the following insights: (i) the patterns of neuron sharing are significantly affected by the characteristics of tasks and examples; (ii) neuron sharing does not fully correspond with language similarity; (iii) shared neurons play a vital role in generating responses, especially those shared across all languages. These findings shed light on the internal workings of multilingual LLMs and pave the way to the future research. We will release the code to foster research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13968v3">Protecting Your LLMs with Information Bottleneck</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 Accepted by Neural Information Processing Systems (NeurIPS 2024)
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06992v2">SWE-Bench+: Enhanced Coding Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) in Software Engineering (SE) can offer assistance for coding. To facilitate a rigorous evaluation of LLMs in practical coding contexts, Carlos et al. introduced the SWE-bench dataset, which comprises 2,294 real-world GitHub issues and their corresponding pull requests, collected from 12 widely used Python repositories. Several impressive LLM-based toolkits recently are developed and evaluated on this dataset. However, a systematic evaluation of the quality of SWE-bench remains missing. In this paper, we addressed this gap by presenting an empirical analysis of the SWE-bench dataset. We conducted a manual screening of instances where SWEAgent + GPT-4 successfully resolved issues by comparing the model-generated patches with the actual pull requests. SWE-Agent+GPT-4 was at the top of SWE-bench leaderboard during the time of our study. Our analysis reveals some critical issues with the SWE-bench dataset: 1) 32.67% of the successful patches involve cheating as the solutions were directly provided in the issue report or the comments. We refer to as solution leakage problem. 2) 31.08% of the passed patches are suspicious patches due to weak test cases, i.e., the tests were not adequate to verify the correctness of a patch. When we filtered out these problematic issues, the resolution rate of SWE-Agent+GPT-4 dropped from 12.47% to 3.97%. We also observed that the same data quality issues also exist in the two variants of SWE-bench, i.e., SWE-bench Lite and SWE-Bench Verified. In addition, over 94% of the issues were created before LLM's knowledge cutoff dates, posing potential data leakage issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05864v2">From Tokens to Words: On the Inner Lexicon of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Natural language is composed of words, but modern LLMs process sub-words as input. A natural question raised by this discrepancy is whether LLMs encode words internally, and if so how. We present evidence that LLMs engage in an intrinsic detokenization process, where sub-word sequences are combined into coherent word representations. Our experiments show that this process takes place primarily within the early and middle layers of the model. They also show that it is robust to non-morphemic splits, typos and perhaps importantly-to out-of-vocabulary words: when feeding the inner representation of such words to the model as input vectors, it can "understand" them despite never seeing them during training. Our findings suggest that LLMs maintain a latent vocabulary beyond the tokenizer's scope. These insights provide a practical, finetuning-free application for expanding the vocabulary of pre-trained models. By enabling the addition of new vocabulary words, we reduce input length and inference iterations, which reduces both space and model latency, with little to no loss in model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02952v3">Visual Editing with LLM-based Tool Chaining: An Efficient Distillation Approach for Real-Time Applications</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      We present a practical distillation approach to fine-tune LLMs for invoking tools in real-time applications. We focus on visual editing tasks; specifically, we modify images and videos by interpreting user stylistic requests, specified in natural language ("golden hour"), using an LLM to select the appropriate tools and their parameters to achieve the desired visual effect. We found that proprietary LLMs such as GPT-3.5-Turbo show potential in this task, but their high cost and latency make them unsuitable for real-time applications. In our approach, we fine-tune a (smaller) student LLM with guidance from a (larger) teacher LLM and behavioral signals. We introduce offline metrics to evaluate student LLMs. Both online and offline experiments show that our student models manage to match the performance of our teacher model (GPT-3.5-Turbo), significantly reducing costs and latency. Lastly, we show that fine-tuning was improved by 25% in low-data regimes using augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15934v2">Automated test generation to evaluate tool-augmented LLMs as conversational AI agents</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 14 pages, 5 figures, Submitted to GenBench@EMNLP2024
    </div>
    <details class="paper-abstract">
      Tool-augmented LLMs are a promising approach to create AI agents that can have realistic conversations, follow procedures, and call appropriate functions. However, evaluating them is challenging due to the diversity of possible conversations, and existing datasets focus only on single interactions and function-calling. We present a test generation pipeline to evaluate LLMs as conversational AI agents. Our framework uses LLMs to generate diverse tests grounded on user-defined procedures. For that, we use intermediate graphs to limit the LLM test generator's tendency to hallucinate content that is not grounded on input procedures, and enforces high coverage of the possible conversations. Additionally, we put forward ALMITA, a manually curated dataset for evaluating AI agents in customer support, and use it to evaluate existing LLMs. Our results show that while tool-augmented LLMs perform well in single interactions, they often struggle to handle complete conversations. While our focus is on customer support, our method is general and capable of AI agents for different domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07768v1">Dialectical Behavior Therapy Approach to LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Large language models demonstrated state-of-the-art results on various reasoning tasks when applying the chain-of-thought (CoT) prompting technique. CoT prompting guides the model into breaking tasks into a few intermediate steps and provides step-by-step demonstrations. However, solving complex reasoning tasks remains a challenge. In this paper, we propose a novel prompting strategy inspired by Dialectical Behavioral Therapy (DBT). DBT, a form of cognitive-behavioral therapy, aims to help individuals cope with stress by developing a system of reasoning. We applied DBT's basic concepts of shaping dialog to construct prompts and conducted experiments on different datasets and LLMs with various numbers of parameters. Our results show that prompts crafted with DBT techniques significantly improve results on smaller models, achieving a 7% increase in accuracy on the StrategyQA, 4.8% on Aqua dataset using 8b parameters model, and a 16.2% increase on the StrategyQA, 5.3% on GSM8K dataset with 14b parameters model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07737v1">Plug-and-Play Performance Estimation for LLM Services without Relying on Labeled Data</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) services exhibit impressive capability on unlearned tasks leveraging only a few examples by in-context learning (ICL). However, the success of ICL varies depending on the task and context, leading to heterogeneous service quality. Directly estimating the performance of LLM services at each invocation can be laborious, especially requiring abundant labeled data or internal information within the LLM. This paper introduces a novel method to estimate the performance of LLM services across different tasks and contexts, which can be "plug-and-play" utilizing only a few unlabeled samples like ICL. Our findings suggest that the negative log-likelihood and perplexity derived from LLM service invocation can function as effective and significant features. Based on these features, we utilize four distinct meta-models to estimate the performance of LLM services. Our proposed method is compared against unlabeled estimation baselines across multiple LLM services and tasks. And it is experimentally applied to two scenarios, demonstrating its effectiveness in the selection and further optimization of LLM services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.10719v3">Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 16 pages, 2 figures, 14 tables
    </div>
    <details class="paper-abstract">
      Reinforcement Learning from Human Feedback (RLHF) is currently the most widely used method to align large language models (LLMs) with human preferences. Existing RLHF methods can be roughly categorized as either reward-based or reward-free. Novel applications such as ChatGPT and Claude leverage reward-based methods that first learn a reward model and apply actor-critic algorithms, such as Proximal Policy Optimization (PPO). However, in academic benchmarks, state-of-the-art results are often achieved via reward-free methods, such as Direct Preference Optimization (DPO). Is DPO truly superior to PPO? Why does PPO perform poorly on these benchmarks? In this paper, we first conduct both theoretical and empirical studies on the algorithmic properties of DPO and show that DPO may have fundamental limitations. Moreover, we also comprehensively examine PPO and reveal the key factors for the best performances of PPO in fine-tuning LLMs. Finally, we benchmark DPO and PPO across a collection of RLHF testbeds, ranging from dialogue to code generation. Experiment results demonstrate that PPO is able to surpass other alignment methods in all cases and achieve state-of-the-art results in challenging code competitions. Our code is publicly available at https://github.com/openpsi-project/ReaLHF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19744v1">Towards Next-Generation LLM-based Recommender Systems: A Survey and Beyond</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have not only revolutionized the field of natural language processing (NLP) but also have the potential to bring a paradigm shift in many other fields due to their remarkable abilities of language understanding, as well as impressive generalization capabilities and reasoning skills. As a result, recent studies have actively attempted to harness the power of LLMs to improve recommender systems, and it is imperative to thoroughly review the recent advances and challenges of LLM-based recommender systems. Unlike existing work, this survey does not merely analyze the classifications of LLM-based recommendation systems according to the technical framework of LLMs. Instead, it investigates how LLMs can better serve recommendation tasks from the perspective of the recommender system community, thus enhancing the integration of large language models into the research of recommender system and its practical application. In addition, the long-standing gap between academic research and industrial applications related to recommender systems has not been well discussed, especially in the era of large language models. In this review, we introduce a novel taxonomy that originates from the intrinsic essence of recommendation, delving into the application of large language model-based recommendation systems and their industrial implementation. Specifically, we propose a three-tier structure that more accurately reflects the developmental progression of recommendation systems from research to practical implementation, including representing and understanding, scheming and utilizing, and industrial deployment. Furthermore, we discuss critical challenges and opportunities in this emerging field. A more up-to-date version of the papers is maintained at: https://github.com/jindongli-Ai/Next-Generation-LLM-based-Recommender-Systems-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07706v1">AgentBank: Towards Generalized LLM Agents via Fine-Tuning on 50000+ Interaction Trajectories</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 Findings of EMNLP 2024
    </div>
    <details class="paper-abstract">
      Fine-tuning on agent-environment interaction trajectory data holds significant promise for surfacing generalized agent capabilities in open-source large language models (LLMs). In this work, we introduce AgentBank, by far the largest trajectory tuning data collection featuring more than 50k diverse high-quality interaction trajectories which comprises 16 tasks covering five distinct agent skill dimensions. Leveraging a novel annotation pipeline, we are able to scale the annotated trajectories and generate a trajectory dataset with minimized difficulty bias. Furthermore, we fine-tune LLMs on AgentBank to get a series of agent models, Samoyed. Our comparative experiments demonstrate the effectiveness of scaling the interaction trajectory data to acquire generalized agent capabilities. Additional studies also reveal some key observations regarding trajectory tuning and agent skill generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.00942v3">Teaching Human Behavior Improves Content Understanding Abilities Of LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Communication is defined as "Who says what to whom with what effect". A message from a communicator generates downstream receiver effects, also known as behavior. Receiver behavior, being a downstream effect of the message, carries rich signals about it. Even after carrying signals about the message, the behavior data is often ignored while training large language models. We show that training LLMs on receiver behavior can actually help improve their content-understanding abilities. Specifically, we show that training LLMs to predict the receiver behavior of likes and comments improves the LLM's performance on a wide variety of downstream content understanding tasks. We show this performance increase over 46 video and image understanding tasks over 26 benchmark datasets across both 0-shot and fine-tuning settings, outperforming many supervised baselines. Moreover, since receiver behavior, such as likes and comments, is collected by default on the internet and does not need any human annotations to be useful, the performance improvement we get after training on this data is essentially free-lunch. We release the receiver behavior cleaned comments and likes of 750k images and videos collected from multiple platforms along with our instruction-tuning data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07677v1">Smart Audit System Empowered by LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Manufacturing quality audits are pivotal for ensuring high product standards in mass production environments. Traditional auditing processes, however, are labor-intensive and reliant on human expertise, posing challenges in maintaining transparency, accountability, and continuous improvement across complex global supply chains. To address these challenges, we propose a smart audit system empowered by large language models (LLMs). Our approach introduces three innovations: a dynamic risk assessment model that streamlines audit procedures and optimizes resource allocation; a manufacturing compliance copilot that enhances data processing, retrieval, and evaluation for a self-evolving manufacturing knowledge base; and a Re-act framework commonality analysis agent that provides real-time, customized analysis to empower engineers with insights for supplier improvement. These enhancements elevate audit efficiency and effectiveness, with testing scenarios demonstrating an improvement of over 24%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.06798v2">It Cannot Be Right If It Was Written by AI: On Lawyers' Preferences of Documents Perceived as Authored by an LLM vs a Human</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 40 pages, 12 figures. Accepted for publication with Artificial Intelligence and Law (Springer Nature)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enable a future in which certain types of legal documents may be generated automatically. This has a great potential to streamline legal processes, lower the cost of legal services, and dramatically increase access to justice. While many researchers focus on proposing and evaluating LLM-based applications supporting tasks in the legal domain, there is a notable lack of investigations into how legal professionals perceive content if they believe an LLM has generated it. Yet, this is a critical point as over-reliance or unfounded scepticism may influence whether such documents bring about appropriate legal consequences. This study is the necessary analysis of the ongoing transition towards mature generative AI systems. Specifically, we examined whether the perception of legal documents' by lawyers and law students (n=75) varies based on their assumed origin (human-crafted vs AI-generated). The participants evaluated the documents, focusing on their correctness and language quality. Our analysis revealed a clear preference for documents perceived as crafted by a human over those believed to be generated by AI. At the same time, most participants expect the future in which documents will be generated automatically. These findings could be leveraged by legal practitioners, policymakers, and legislators to implement and adopt legal document generation technology responsibly and to fuel the necessary discussions on how legal processes should be updated to reflect recent technological developments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02890v2">Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text. Within our framework, we characterize the universally minimum Type-II error, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07627v1">Automatic Curriculum Expert Iteration for Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Hallucinations (i.e., generating plausible but inaccurate content) and laziness (i.e. excessive refusals or defaulting to "I don't know") persist as major challenges in LLM reasoning. Current efforts to reduce hallucinations primarily focus on factual errors in knowledge-grounded tasks, often neglecting hallucinations related to faulty reasoning. Meanwhile, some approaches render LLMs overly conservative, limiting their problem-solving capabilities. To mitigate hallucination and laziness in reasoning tasks, we propose Automatic Curriculum Expert Iteration (Auto-CEI) to enhance LLM reasoning and align responses to the model's capabilities--assertively answering within its limits and declining when tasks exceed them. In our method, Expert Iteration explores the reasoning trajectories near the LLM policy, guiding incorrect paths back on track to reduce compounding errors and improve robustness; it also promotes appropriate "I don't know" responses after sufficient reasoning attempts. The curriculum automatically adjusts rewards, incentivizing extended reasoning before acknowledging incapability, thereby pushing the limits of LLM reasoning and aligning its behaviour with these limits. We compare Auto-CEI with various SOTA baselines across logical reasoning, mathematics, and planning tasks, where Auto-CEI achieves superior alignment by effectively balancing assertiveness and conservativeness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.17235v3">A Simple LLM Framework for Long-Range Video Question-Answering</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 EMNLP 2024 main
    </div>
    <details class="paper-abstract">
      We present LLoVi, a language-based framework for long-range video question-answering (LVQA). Unlike prior long-range video understanding methods, which are often costly and require specialized long-range video modeling design (e.g., memory queues, state-space layers, etc.), our approach uses a frame/clip-level visual captioner (e.g., BLIP2, LaViLa, LLaVA) coupled with a Large Language Model (GPT-3.5, GPT-4) leading to a simple yet surprisingly effective LVQA framework. Specifically, we decompose short and long-range modeling aspects of LVQA into two stages. First, we use a short-term visual captioner to generate textual descriptions of short video clips (0.5-8s in length) densely sampled from a long input video. Afterward, an LLM aggregates the densely extracted short-term captions to perform long-range temporal reasoning needed to understand the whole video and answer a question. To analyze what makes our simple framework so effective, we thoroughly evaluate various components of our system. Our empirical analysis reveals that the choice of the visual captioner and LLM is critical for good LVQA performance. Furthermore, we show that a specialized prompt that asks the LLM first to summarize the noisy short-term visual captions and then answer a given input question leads to a significant LVQA performance boost. On EgoSchema, which is best known as a very long-form video question-answering benchmark, our method achieves 50.3% accuracy, outperforming the previous best-performing approach by 18.1% (absolute gain). In addition, our approach outperforms the previous state-of-the-art by 4.1% and 3.1% on NeXT-QA and IntentQA. We also extend LLoVi to grounded LVQA and show that it outperforms all prior methods on the NeXT-GQA dataset. We will release our code at https://github.com/CeeZh/LLoVi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04793v2">SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Optimizing the Key-Value (KV) cache of the Large Language Model (LLM) has been considered critical to saving the cost of inference. Most of the existing KV-cache compression algorithms attempted to sparsify the sequence of tokens by taking advantage of the different importance of tokens. However, most of these methods treat all layers equally, allocating the same KV budget to each layer. This approach is suboptimal, as some layers may be less sensitive to input tokens yet still receive the same budget as others. In this work, we found that by identifying the importance of attention layers, we could optimize the KV-cache jointly from two dimensions, i.e., sequence-wise and layer-wise. Based on our observations regarding layer-wise importance in inference, we propose SqueezeAttention to precisely optimize the allocation of KV-cache budget among layers on-the-fly and then incorporate three representative sequence-wise algorithms to compress the KV-cache for each layer with its very own budget. Specifically, we first measure each layer's importance by calculating the cosine similarity of the input prompt differences before and after the self-attention layers. Based on this similarity, we then categorize the layers into two groups and adjust their KV budgets accordingly. By optimizing the KV-cache from both sequence's and layer's dimensions, SqueezeAttention achieves around 30% to 70% of the memory reductions and up to 2.2 times of throughput improvements in a wide range of LLMs and benchmarks. The code is available at https://github.com/hetailang/SqueezeAttention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07589v1">No Free Lunch: Retrieval-Augmented Generation Undermines Fairness in LLMs, Even for Vigilant Users</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) is widely adopted for its effectiveness and cost-efficiency in mitigating hallucinations and enhancing the domain-specific generation capabilities of large language models (LLMs). However, is this effectiveness and cost-efficiency truly a free lunch? In this study, we comprehensively investigate the fairness costs associated with RAG by proposing a practical three-level threat model from the perspective of user awareness of fairness. Specifically, varying levels of user fairness awareness result in different degrees of fairness censorship on the external dataset. We examine the fairness implications of RAG using uncensored, partially censored, and fully censored datasets. Our experiments demonstrate that fairness alignment can be easily undermined through RAG without the need for fine-tuning or retraining. Even with fully censored and supposedly unbiased external datasets, RAG can lead to biased outputs. Our findings underscore the limitations of current alignment methods in the context of RAG-based LLMs and highlight the urgent need for new strategies to ensure fairness. We propose potential mitigations and call for further research to develop robust fairness safeguards in RAG-based LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05684v2">Copiloting Diagnosis of Autism in Real Clinical Scenarios via LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Autism spectrum disorder(ASD) is a pervasive developmental disorder that significantly impacts the daily functioning and social participation of individuals. Despite the abundance of research focused on supporting the clinical diagnosis of ASD, there is still a lack of systematic and comprehensive exploration in the field of methods based on Large Language Models (LLMs), particularly regarding the real-world clinical diagnostic scenarios based on Autism Diagnostic Observation Schedule, Second Edition (ADOS-2). Therefore, we have proposed a framework called ADOS-Copilot, which strikes a balance between scoring and explanation and explored the factors that influence the performance of LLMs in this task. The experimental results indicate that our proposed framework is competitive with the diagnostic results of clinicians, with a minimum MAE of 0.4643, binary classification F1-score of 81.79\%, and ternary classification F1-score of 78.37\%. Furthermore, we have systematically elucidated the strengths and limitations of current LLMs in this task from the perspectives of ADOS-2, LLMs' capabilities, language, and model scale aiming to inspire and guide the future application of LLMs in a broader fields of mental health disorders. We hope for more research to be transferred into real clinical practice, opening a window of kindness to the world for eccentric children.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07573v1">RealVul: Can We Detect Vulnerabilities in Web Applications with LLM?</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      The latest advancements in large language models (LLMs) have sparked interest in their potential for software vulnerability detection. However, there is currently a lack of research specifically focused on vulnerabilities in the PHP language, and challenges in extracting samples and processing persist, hindering the model's ability to effectively capture the characteristics of specific vulnerabilities. In this paper, we present RealVul, the first LLM-based framework designed for PHP vulnerability detection, addressing these issues. By vulnerability candidate detection methods and employing techniques such as normalization, we can isolate potential vulnerability triggers while streamlining the code and eliminating unnecessary semantic information, enabling the model to better understand and learn from the generated vulnerability samples. We also address the issue of insufficient PHP vulnerability samples by improving data synthesis methods. To evaluate RealVul's performance, we conduct an extensive analysis using five distinct code LLMs on vulnerability data from 180 PHP projects. The results demonstrate a significant improvement in both effectiveness and generalization compared to existing methods, effectively boosting the vulnerability detection capabilities of these models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07551v1">KRAG Framework for Enhancing LLMs in the Legal Domain</a></div>
    <div class="paper-meta">
      📅 2024-10-10
      | 💬 Presented at NeLaMKRR@KR, 2024 (arXiv:2410.05339)
    </div>
    <details class="paper-abstract">
      This paper introduces Knowledge Representation Augmented Generation (KRAG), a novel framework designed to enhance the capabilities of Large Language Models (LLMs) within domain-specific applications. KRAG points to the strategic inclusion of critical knowledge entities and relationships that are typically absent in standard data sets and which LLMs do not inherently learn. In the context of legal applications, we present Soft PROLEG, an implementation model under KRAG, which uses inference graphs to aid LLMs in delivering structured legal reasoning, argumentation, and explanations tailored to user inquiries. The integration of KRAG, either as a standalone framework or in tandem with retrieval augmented generation (RAG), markedly improves the ability of language models to navigate and solve the intricate challenges posed by legal texts and terminologies. This paper details KRAG's methodology, its implementation through Soft PROLEG, and potential broader applications, underscoring its significant role in advancing natural language understanding and processing in specialized knowledge domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07516v1">Exploring and Lifting the Robustness of LLM-powered Automated Program Repair with Metamorphic Testing</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      In recent years, Large language model-powered Automated Program Repair (LAPR) techniques have achieved state-of-the-art bug-fixing performance and have been pervasively applied and studied in both industry and academia. Nonetheless, LLMs were proved to be highly sensitive to input prompts, with slight differences in the expressions of semantically equivalent programs potentially causing repair failures. Therefore, it is crucial to conduct robustness testing on LAPR techniques before their practical deployment. However, related research is scarce. To this end, we propose MT-LAPR, a Metamorphic Testing framework exclusively for LAPR techniques, which summarizes nine widely-recognized Metamorphic Relations (MRs) by developers across three perturbation levels: token, statement, and block. Afterward, our proposed MRs are applied to buggy codes to generate test cases, which are semantically equivalent yet to affect the inference of LAPR. Experiments are carried out on two extensively examined bug-fixing datasets, i.e., Defect4J and QuixBugs, and four bug-fixing abled LLMs released recently, demonstrating that 34.4% - 48.5% of the test cases expose the instability of LAPR techniques on average, showing the effectiveness of MT-LAPR and uncovering a positive correlation between code readability and the robustness of LAPR techniques. Inspired by the above findings, this paper uses the test cases generated by MT-LAPR as samples to train a CodeT5-based code editing model aiming at improving code readability and then embeds it into the LAPR workflow as a data preprocessing step. Extensive experiments demonstrate that this approach significantly enhances the robustness of LAPR by 49.32% at most.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07507v1">Thought2Text: Text Generation from EEG Signal using Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Decoding and expressing brain activity in a comprehensible form is a challenging frontier in AI. This paper presents Thought2Text, which uses instruction-tuned Large Language Models (LLMs) fine-tuned with EEG data to achieve this goal. The approach involves three stages: (1) training an EEG encoder for visual feature extraction, (2) fine-tuning LLMs on image and text data, enabling multimodal description generation, and (3) further fine-tuning on EEG embeddings to generate text directly from EEG during inference. Experiments on a public EEG dataset collected for six subjects with image stimuli demonstrate the efficacy of multimodal LLMs (LLaMa-v3, Mistral-v0.3, Qwen2.5), validated using traditional language generation evaluation metrics, GPT-4 based assessments, and evaluations by human expert. This approach marks a significant advancement towards portable, low-cost "thoughts-to-text" technology with potential applications in both neuroscience and natural language processing (NLP).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07504v1">Using LLMs to Discover Legal Factors</a></div>
    <div class="paper-meta">
      📅 2024-10-10
    </div>
    <details class="paper-abstract">
      Factors are a foundational component of legal analysis and computational models of legal reasoning. These factor-based representations enable lawyers, judges, and AI and Law researchers to reason about legal cases. In this paper, we introduce a methodology that leverages large language models (LLMs) to discover lists of factors that effectively represent a legal domain. Our method takes as input raw court opinions and produces a set of factors and associated definitions. We demonstrate that a semi-automated approach, incorporating minimal human involvement, produces factor representations that can predict case outcomes with moderate success, if not yet as well as expert-defined factors can.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05218v2">Density estimation with LLMs: a geometric investigation of in-context learning trajectories</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable emergent abilities to perform in-context learning across various tasks, including time series forecasting. This work investigates LLMs' ability to estimate probability density functions (PDFs) from data observed in-context; such density estimation (DE) is a fundamental task underlying many probabilistic modeling problems. We leverage the Intensive Principal Component Analysis (InPCA) to visualize and analyze the in-context learning dynamics of LLaMA-2 models. Our main finding is that these LLMs all follow similar learning trajectories in a low-dimensional InPCA space, which are distinct from those of traditional density estimation methods like histograms and Gaussian kernel density estimation (KDE). We interpret the LLaMA in-context DE process as a KDE with an adaptive kernel width and shape. This custom kernel model captures a significant portion of LLaMA's behavior despite having only two parameters. We further speculate on why LLaMA's kernel width and shape differs from classical algorithms, providing insights into the mechanism of in-context probabilistic reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07461v1">Is C4 Dataset Optimal for Pruning? An Investigation of Calibration Data for LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Network pruning has emerged as a potential solution to make LLMs cheaper to deploy. However, existing LLM pruning approaches universally rely on the C4 dataset as the calibration data for calculating pruning scores, leaving its optimality unexplored. In this study, we evaluate the choice of calibration data on LLM pruning, across a wide range of datasets that are most commonly used in LLM training and evaluation, including four pertaining datasets as well as three categories of downstream tasks encompassing nine datasets. Each downstream dataset is prompted with In-Context Learning (ICL) and Chain-of-Thought (CoT), respectively. Besides the already intriguing observation that the choice of calibration data significantly impacts the performance of pruned LLMs, our results also uncover several subtle and often unexpected findings, summarized as follows: (1) C4 is not the optimal choice for LLM pruning, even among commonly used pre-training datasets; (2) arithmetic datasets, when used as calibration data, performs on par or even better than pre-training datasets; (3) pruning with downstream datasets does not necessarily help the corresponding downstream task, compared to pre-training data; (4) ICL is widely beneficial to all data categories, whereas CoT is only useful on certain tasks. Our findings shed light on the importance of carefully selecting calibration data for LLM pruning and pave the way for more efficient deployment of these powerful models in real-world applications. We release our code at: https://github.com/abx393/llm-pruning-calibration-data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07395v1">LLM Embeddings Improve Test-time Adaptation to Tabular $Y|X$-Shifts</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      For tabular datasets, the change in the relationship between the label and covariates ($Y|X$-shifts) is common due to missing variables (a.k.a. confounders). Since it is impossible to generalize to a completely new and unknown domain, we study models that are easy to adapt to the target domain even with few labeled examples. We focus on building more informative representations of tabular data that can mitigate $Y|X$-shifts, and propose to leverage the prior world knowledge in LLMs by serializing (write down) the tabular data to encode it. We find LLM embeddings alone provide inconsistent improvements in robustness, but models trained on them can be well adapted/finetuned to the target domain even using 32 labeled observations. Our finding is based on a comprehensive and systematic study consisting of 7650 source-target pairs and benchmark against 261,000 model configurations trained by 22 algorithms. Our observation holds when ablating the size of accessible target data and different adaptation strategies. The code is available at https://github.com/namkoong-lab/LLM-Tabular-Shifts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07177v1">MM-Ego: Towards Building Egocentric Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      This research aims to comprehensively explore building a multimodal foundation model for egocentric video understanding. To achieve this goal, we work on three fronts. First, as there is a lack of QA data for egocentric video understanding, we develop a data engine that efficiently generates 7M high-quality QA samples for egocentric videos ranging from 30 seconds to one hour long, based on human-annotated data. This is currently the largest egocentric QA dataset. Second, we contribute a challenging egocentric QA benchmark with 629 videos and 7,026 questions to evaluate the models' ability in recognizing and memorizing visual details across videos of varying lengths. We introduce a new de-biasing evaluation method to help mitigate the unavoidable language bias present in the models being evaluated. Third, we propose a specialized multimodal architecture featuring a novel "Memory Pointer Prompting" mechanism. This design includes a global glimpse step to gain an overarching understanding of the entire video and identify key visual information, followed by a fallback step that utilizes the key visual information to generate responses. This enables the model to more effectively comprehend extended video content. With the data, benchmark, and model, we successfully build MM-Ego, an egocentric multimodal LLM that shows powerful performance on egocentric video understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07137v1">Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Automatic LLM benchmarks, such as AlpacaEval 2.0, Arena-Hard-Auto, and MT-Bench, have become popular for evaluating language models due to their cost-effectiveness and scalability compared to human evaluation. Achieving high win rates on these benchmarks can significantly boost the promotional impact of newly released language models. This promotional benefit may motivate tricks, such as manipulating model output length or style to game win rates, even though several mechanisms have been developed to control length and disentangle style to reduce gameability. Nonetheless, we show that even a "null model" that always outputs a constant response (irrelevant to input instructions) can cheat automatic benchmarks and achieve top-ranked win rates: an 86.5% LC win rate on AlpacaEval 2.0; an 83.0 score on Arena-Hard-Auto; and a 9.55 score on MT-Bench. Moreover, the crafted cheating outputs are transferable because we assume that the instructions of these benchmarks (e.g., 805 samples of AlpacaEval 2.0) are private and cannot be accessed. While our experiments are primarily proof-of-concept, an adversary could use LLMs to generate more imperceptible cheating responses, unethically benefiting from high win rates and promotional impact. Our findings call for the development of anti-cheating mechanisms for reliable automatic benchmarks. The code is available at https://github.com/sail-sg/Cheating-LLM-Benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07304v1">The Moral Turing Test: Evaluating Human-LLM Alignment in Moral Decision-Making</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into society, their alignment with human morals is crucial. To better understand this alignment, we created a large corpus of human- and LLM-generated responses to various moral scenarios. We found a misalignment between human and LLM moral assessments; although both LLMs and humans tended to reject morally complex utilitarian dilemmas, LLMs were more sensitive to personal framing. We then conducted a quantitative user study involving 230 participants (N=230), who evaluated these responses by determining whether they were AI-generated and assessed their agreement with the responses. Human evaluators preferred LLMs' assessments in moral scenarios, though a systematic anti-AI bias was observed: participants were less likely to agree with judgments they believed to be machine-generated. Statistical and NLP-based analyses revealed subtle linguistic differences in responses, influencing detection and agreement. Overall, our findings highlight the complexities of human-AI perception in morally charged decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12839v1">Capturing Bias Diversity in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 2nd International Conference on Foundation and Large Language Models (FLLM2024), 26-29 November, 2024 | Dubai, UAE
    </div>
    <details class="paper-abstract">
      This paper presents research on enhancements to Large Language Models (LLMs) through the addition of diversity in its generated outputs. Our study introduces a configuration of multiple LLMs which demonstrates the diversities capable with a single LLM. By developing multiple customised instances of a GPT model, each reflecting biases in specific demographic characteristics including gender, age, and race, we propose, develop and evaluate a framework for a more nuanced and representative AI dialogue which we call BiasGPT. The customised GPT models will ultimately collaborate, merging their diverse perspectives on a topic into an integrated response that captures a broad spectrum of human experiences and viewpoints. In this paper, through experiments, we demonstrate the capabilities of a GPT model to embed different biases which, when combined, can open the possibilities of more inclusive AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07054v1">Mitigating the Language Mismatch and Repetition Issues in LLM-based Machine Translation via Model Editing</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 20 pages, EMNLP'2024 Main Conference
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently revolutionized the NLP field, while they still fall short in some specific down-stream tasks. In the work, we focus on utilizing LLMs to perform machine translation, where we observe that two patterns of errors frequently occur and drastically affect the translation quality: language mismatch and repetition. The work sets out to explore the potential for mitigating these two issues by leveraging model editing methods, e.g., by locating Feed-Forward Network (FFN) neurons or something that are responsible for the errors and deactivating them in the inference time. We find that directly applying such methods either limited effect on the targeted errors or has significant negative side-effect on the general translation quality, indicating that the located components may also be crucial for ensuring machine translation with LLMs on the rails. To this end, we propose to refine the located components by fetching the intersection of the locating results under different language settings, filtering out the aforementioned information that is irrelevant to targeted errors. The experiment results empirically demonstrate that our methods can effectively reduce the language mismatch and repetition ratios and meanwhile enhance or keep the general translation quality in most cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07053v1">Robots in the Middle: Evaluating LLMs in Dispute Resolution</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Mediation is a dispute resolution method featuring a neutral third-party (mediator) who intervenes to help the individuals resolve their dispute. In this paper, we investigate to which extent large language models (LLMs) are able to act as mediators. We investigate whether LLMs are able to analyze dispute conversations, select suitable intervention types, and generate appropriate intervention messages. Using a novel, manually created dataset of 50 dispute scenarios, we conduct a blind evaluation comparing LLMs with human annotators across several key metrics. Overall, the LLMs showed strong performance, even outperforming our human annotators across dimensions. Specifically, in 62% of the cases, the LLMs chose intervention types that were rated as better than or equivalent to those chosen by humans. Moreover, in 84% of the cases, the intervention messages generated by the LLMs were rated as better than or equal to the intervention messages written by humans. LLMs likewise performed favourably on metrics such as impartiality, understanding and contextualization. Our results demonstrate the potential of integrating AI in online dispute resolution (ODR) platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07295v1">IterGen: Iterative Structured LLM Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for tasks such as natural language and code generation. Still, their outputs often suffer from issues like privacy violations, and semantically inaccurate code generation. Current libraries for LLM generation rely on left-to-right decoding without systematic support for backtracking, limiting the ability to correct or refine outputs mid-generation. To address this issue, we introduce IterGen, an intuitive framework for iterative, grammar-guided LLM generation that enables users to move both forward and backward within the generated output based on grammar symbols. By leveraging a symbol-to-position mapping, IterGen ensures efficient and structured generation while allowing for corrections during the process. We demonstrate IterGen's effectiveness in two important applications: reducing privacy leakage in LLM outputs and improving the accuracy of LLM-generated SQL queries. Our code is available at https://github.com/uiuc-arc/itergen
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07035v1">PositionID: LLMs can Control Lengths, Copy and Paste with Explicit Positional Awareness</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 39 pages. CP-Bench and LenCtrl-Bench are available in https://huggingface.co/datasets/ZenMoore/CP-Bench and https://huggingface.co/datasets/ZenMoore/LenCtrl-Bench
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities across various domains, including role-playing, creative writing, mathematical reasoning, and coding. Despite these advancements, LLMs still encounter challenges with length control, frequently failing to adhere to specific length constraints due to their token-level operations and insufficient training on data with strict length limitations. We identify this issue as stemming from a lack of positional awareness and propose novel approaches--PositionID Prompting and PositionID Fine-Tuning--to address it. These methods enhance the model's ability to continuously monitor and manage text length during generation. Additionally, we introduce PositionID CP Prompting to enable LLMs to perform copy and paste operations accurately. Furthermore, we develop two benchmarks for evaluating length control and copy-paste abilities. Our experiments demonstrate that our methods significantly improve the model's adherence to length constraints and copy-paste accuracy without compromising response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.00795v4">LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Pretrained large language models (LLMs) are surprisingly effective at performing zero-shot tasks, including time-series forecasting. However, understanding the mechanisms behind such capabilities remains highly challenging due to the complexity of the models. We study LLMs' ability to extrapolate the behavior of dynamical systems whose evolution is governed by principles of physical interest. Our results show that LLaMA 2, a language model trained primarily on texts, achieves accurate predictions of dynamical system time series without fine-tuning or prompt engineering. Moreover, the accuracy of the learned physical rules increases with the length of the input context window, revealing an in-context version of neural scaling law. Along the way, we present a flexible and efficient algorithm for extracting probability density functions of multi-digit numbers directly from LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06972v1">Diamond of Thought: A Design Thinking-Based Framework for LLMs in Wearable Design</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Wearable design is an interdisciplinary field that balances technological innovation, human factors, and human-computer interactions. Despite contributions from various disciplines, many projects lack stable interdisciplinary teams, which often leads to design failures. Large language models (LLMs) integrate diverse information and generate innovative solutions, making them a valuable tool for enhancing design processes. Thus, we have explored the use of LLMs in wearable design by combining design-thinking principles with LLM capabilities. We have developed the "Diamond of Thought" framework and analysed 1,603 prototypes and 1,129 products from a body-centric perspective to create a comprehensive database. We employed retrieval-augmented generation to input database details into the LLMs, ensuring applicability to wearable design challenges and integration of embodied cognition into the process. Our LLM-based methodology for wearables has been experimentally validated, demonstrating the potential of LLMs for the advancement of design practices. This study offers new tools and methods for future wearable designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06943v1">AutoFeedback: An LLM-based Framework for Efficient and Accurate API Request Generation</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) leverage external tools primarily through generating the API request to enhance task completion efficiency. The accuracy of API request generation significantly determines the capability of LLMs to accomplish tasks. Due to the inherent hallucinations within the LLM, it is difficult to efficiently and accurately generate the correct API request. Current research uses prompt-based feedback to facilitate the LLM-based API request generation. However, existing methods lack factual information and are insufficiently detailed. To address these issues, we propose AutoFeedback, an LLM-based framework for efficient and accurate API request generation, with a Static Scanning Component (SSC) and a Dynamic Analysis Component (DAC). SSC incorporates errors detected in the API requests as pseudo-facts into the feedback, enriching the factual information. DAC retrieves information from API documentation, enhancing the level of detail in feedback. Based on this two components, Autofeedback implementes two feedback loops during the process of generating API requests by the LLM. Extensive experiments demonstrate that it significantly improves accuracy of API request generation and reduces the interaction cost. AutoFeedback achieves an accuracy of 100.00\% on a real-world API dataset and reduces the cost of interaction with GPT-3.5 Turbo by 23.44\%, and GPT-4 Turbo by 11.85\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06916v1">SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Speculative decoding (SD) has emerged as a widely used paradigm to accelerate the inference of large language models (LLMs) without compromising generation quality. It works by first employing a compact model to draft multiple tokens efficiently and then using the target LLM to verify them in parallel. While this technique has achieved notable speedups, most existing approaches necessitate either additional parameters or extensive training to construct effective draft models, thereby restricting their applicability across different LLMs and tasks. To address this limitation, we explore a novel plug-and-play SD solution with layer-skipping, which skips intermediate layers of the target LLM as the compact draft model. Our analysis reveals that LLMs exhibit great potential for self-acceleration through layer sparsity and the task-specific nature of this sparsity. Building on these insights, we introduce SWIFT, an on-the-fly self-speculative decoding algorithm that adaptively selects intermediate layers of LLMs to skip during inference. SWIFT does not require auxiliary models or additional training, making it a plug-and-play solution for accelerating LLM inference across diverse input data streams. Our extensive experiments across a wide range of models and downstream tasks demonstrate that SWIFT can achieve over a 1.3x-1.6x speedup while preserving the original distribution of the generated text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.12139v2">DTLLM-VLT: Diverse Text Generation for Visual Language Tracking Based on LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 Accepted by CVPR Workshop 2024, Oral Presentation, Best Paper Honorable Mention Award
    </div>
    <details class="paper-abstract">
      Visual Language Tracking (VLT) enhances single object tracking (SOT) by integrating natural language descriptions from a video, for the precise tracking of a specified object. By leveraging high-level semantic information, VLT guides object tracking, alleviating the constraints associated with relying on a visual modality. Nevertheless, most VLT benchmarks are annotated in a single granularity and lack a coherent semantic framework to provide scientific guidance. Moreover, coordinating human annotators for high-quality annotations is laborious and time-consuming. To address these challenges, we introduce DTLLM-VLT, which automatically generates extensive and multi-granularity text to enhance environmental diversity. (1) DTLLM-VLT generates scientific and multi-granularity text descriptions using a cohesive prompt framework. Its succinct and highly adaptable design allows seamless integration into various visual tracking benchmarks. (2) We select three prominent benchmarks to deploy our approach: short-term tracking, long-term tracking, and global instance tracking. We offer four granularity combinations for these benchmarks, considering the extent and density of semantic information, thereby showcasing the practicality and versatility of DTLLM-VLT. (3) We conduct comparative experiments on VLT benchmarks with different text granularities, evaluating and analyzing the impact of diverse text on tracking performance. Conclusionally, this work leverages LLM to provide multi-granularity semantic information for VLT task from efficient and diverse perspectives, enabling fine-grained evaluation of multi-modal trackers. In the future, we believe this work can be extended to more datasets to support vision datasets understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02492v2">DTVLT: A Multi-modal Diverse Text Benchmark for Visual Language Tracking Based on LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 Preprint, Under Review
    </div>
    <details class="paper-abstract">
      Visual language tracking (VLT) has emerged as a cutting-edge research area, harnessing linguistic data to enhance algorithms with multi-modal inputs and broadening the scope of traditional single object tracking (SOT) to encompass video understanding applications. Despite this, most VLT benchmarks still depend on succinct, human-annotated text descriptions for each video. These descriptions often fall short in capturing the nuances of video content dynamics and lack stylistic variety in language, constrained by their uniform level of detail and a fixed annotation frequency. As a result, algorithms tend to default to a "memorize the answer" strategy, diverging from the core objective of achieving a deeper understanding of video content. Fortunately, the emergence of large language models (LLMs) has enabled the generation of diverse text. This work utilizes LLMs to generate varied semantic annotations (in terms of text lengths and granularities) for representative SOT benchmarks, thereby establishing a novel multi-modal benchmark. Specifically, we (1) propose a new visual language tracking benchmark with diverse texts, named DTVLT, based on five prominent VLT and SOT benchmarks, including three sub-tasks: short-term tracking, long-term tracking, and global instance tracking. (2) We offer four granularity texts in our benchmark, considering the extent and density of semantic information. We expect this multi-granular generation strategy to foster a favorable environment for VLT and video understanding research. (3) We conduct comprehensive experimental analyses on DTVLT, evaluating the impact of diverse text on tracking performance and hope the identified performance bottlenecks of existing algorithms can support further research in VLT and video understanding. The proposed benchmark, experimental results and toolkit will be released gradually on http://videocube.aitestunion.com/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09102v1">Instructional Segment Embedding: Improving LLM Safety with Instruction Hierarchy</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are susceptible to security and safety threats, such as prompt injection, prompt extraction, and harmful requests. One major cause of these vulnerabilities is the lack of an instruction hierarchy. Modern LLM architectures treat all inputs equally, failing to distinguish between and prioritize various types of instructions, such as system messages, user prompts, and data. As a result, lower-priority user prompts may override more critical system instructions, including safety protocols. Existing approaches to achieving instruction hierarchy, such as delimiters and instruction-based training, do not address this issue at the architectural level. We introduce the Instructional Segment Embedding (ISE) technique, inspired by BERT, to modern large language models, which embeds instruction priority information directly into the model. This approach enables models to explicitly differentiate and prioritize various instruction types, significantly improving safety against malicious prompts that attempt to override priority rules. Our experiments on the Structured Query and Instruction Hierarchy benchmarks demonstrate an average robust accuracy increase of up to 15.75% and 18.68%, respectively. Furthermore, we observe an improvement in instruction-following capability of up to 4.1% evaluated on AlpacaEval. Overall, our approach offers a promising direction for enhancing the safety and effectiveness of LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06809v1">Root Defence Strategies: Ensuring Safety of LLM at the Decoding Level</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 19 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated immense utility across various industries. However, as LLMs advance, the risk of harmful outputs increases due to incorrect or malicious instruction prompts. While current methods effectively address jailbreak risks, they share common limitations: 1) Judging harmful responses from the prefill-level lacks utilization of the model's decoding outputs, leading to relatively lower effectiveness and robustness. 2) Rejecting potentially harmful responses based on a single evaluation can significantly impair the model's helpfulness.This paper examines the LLMs' capability to recognize harmful outputs, revealing and quantifying their proficiency in assessing the danger of previous tokens. Motivated by pilot experiment results, we design a robust defense mechanism at the decoding level. Our novel decoder-oriented, step-by-step defense architecture corrects harmful queries directly rather than rejecting them outright. We introduce speculative decoding to enhance usability and facilitate deployment to boost secure decoding speed. Extensive experiments demonstrate that our approach improves model security without compromising reasoning speed. Notably, our method leverages the model's ability to discern hazardous information, maintaining its helpfulness compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05127v3">Towards Semantic Equivalence of Tokenization in Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 Technical Report. The project page: https://chocowu.github.io/SeTok-web/
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated exceptional capabilities in processing vision-language tasks. One of the crux of MLLMs lies in vision tokenization, which involves efficiently transforming input visual signals into feature representations that are most beneficial for LLMs. However, existing vision tokenizers, essential for semantic alignment between vision and language, remain problematic. Existing methods aggressively fragment visual input, corrupting the visual semantic integrity. To address this, this paper proposes a novel dynamic Semantic-Equivalent Vision Tokenizer (SeTok), which groups visual features into semantic units via a dynamic clustering algorithm, flexibly determining the number of tokens based on image complexity. The resulting vision tokens effectively preserve semantic integrity and capture both low-frequency and high-frequency visual features. The proposed MLLM (Setokim) equipped with SeTok significantly demonstrates superior performance across various tasks, as evidenced by our experimental results. The project page is at https://chocowu.github.io/SeTok-web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07283v1">Prompt Infection: LLM-to-LLM Prompt Injection within Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) grow increasingly powerful, multi-agent systems are becoming more prevalent in modern AI applications. Most safety research, however, has focused on vulnerabilities in single-agent LLMs. These include prompt injection attacks, where malicious prompts embedded in external content trick the LLM into executing unintended or harmful actions, compromising the victim's application. In this paper, we reveal a more dangerous vector: LLM-to-LLM prompt injection within multi-agent systems. We introduce Prompt Infection, a novel attack where malicious prompts self-replicate across interconnected agents, behaving much like a computer virus. This attack poses severe threats, including data theft, scams, misinformation, and system-wide disruption, all while propagating silently through the system. Our extensive experiments demonstrate that multi-agent systems are highly susceptible, even when agents do not publicly share all communications. To address this, we propose LLM Tagging, a defense mechanism that, when combined with existing safeguards, significantly mitigates infection spread. This work underscores the urgent need for advanced security measures as multi-agent LLM systems become more widely adopted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06733v1">Weak-eval-Strong: Evaluating and Eliciting Lateral Thinking of LLMs with Situation Puzzles</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 Accepted by NeurIPS 2024
    </div>
    <details class="paper-abstract">
      While advancements in NLP have significantly improved the performance of Large Language Models (LLMs) on tasks requiring vertical thinking, their lateral thinking capabilities remain under-explored and challenging to measure due to the complexity of assessing creative thought processes and the scarcity of relevant data. To address these challenges, we introduce SPLAT, a benchmark leveraging Situation Puzzles to evaluate and elicit LAteral Thinking of LLMs. This benchmark, containing 975 graded situation puzzles across three difficulty levels, employs a new multi-turn player-judge framework instead of the traditional model-based evaluation, which often necessitates a stronger evaluation model. This framework simulates an interactive game where the model (player) asks the evaluation model (judge) questions about an incomplete story to infer the full scenario. The judge answers based on a detailed reference scenario or evaluates if the player's predictions align with the reference one. This approach lessens dependence on more robust evaluation models, enabling the assessment of state-of-the-art LLMs. The experiments demonstrate that a robust evaluation model, such as WizardLM-2, closely matches human judgements in both intermediate question-answering and final scenario accuracy, achieving over 80% agreement-similar to the agreement levels among humans. Furthermore, applying data and reasoning processes from our benchmark to other lateral thinking-related benchmarks, e.g., RiddleSense and BrainTeaser, leads to performance enhancements. This suggests that our benchmark effectively evaluates and elicits the lateral thinking abilities of LLMs. Code is available at: https://github.com/chenqi008/LateralThinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10877v1">Improving Data Efficiency via Curating LLM-Driven Rating Systems</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Instruction tuning is critical for adapting large language models (LLMs) to downstream tasks, and recent studies have demonstrated that small amounts of human-curated data can outperform larger datasets, challenging traditional data scaling laws. While LLM-based data quality rating systems offer a cost-effective alternative to human annotation, they often suffer from inaccuracies and biases, even in powerful models like GPT-4. In this work, we introduce DS2, a Diversity-aware Score curation method for Data Selection. By systematically modeling error patterns through a score transition matrix, DS2 corrects LLM-based scores and promotes diversity in the selected data samples. Our approach shows that a curated subset (just 3.3% of the original dataset) outperforms full-scale datasets (300k samples) across various machine-alignment benchmarks, and matches or surpasses human-aligned datasets such as LIMA with the same sample size (1k samples). These findings challenge conventional data scaling assumptions, highlighting that redundant, low-quality samples can degrade performance and reaffirming that "more can be less."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06704v1">PII-Scope: A Benchmark for Training Data PII Leakage Assessment in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      In this work, we introduce PII-Scope, a comprehensive benchmark designed to evaluate state-of-the-art methodologies for PII extraction attacks targeting LLMs across diverse threat settings. Our study provides a deeper understanding of these attacks by uncovering several hyperparameters (e.g., demonstration selection) crucial to their effectiveness. Building on this understanding, we extend our study to more realistic attack scenarios, exploring PII attacks that employ advanced adversarial strategies, including repeated and diverse querying, and leveraging iterative learning for continual PII extraction. Through extensive experimentation, our results reveal a notable underestimation of PII leakage in existing single-query attacks. In fact, we show that with sophisticated adversarial capabilities and a limited query budget, PII extraction rates can increase by up to fivefold when targeting the pretrained model. Moreover, we evaluate PII leakage on finetuned models, showing that they are more vulnerable to leakage than pretrained models. Overall, our work establishes a rigorous empirical benchmark for PII extraction attacks in realistic threat scenarios and provides a strong foundation for developing effective mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04927v2">FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Federated sequential recommendation (FedSeqRec) has gained growing attention due to its ability to protect user privacy. Unfortunately, the performance of FedSeqRec is still unsatisfactory because the models used in FedSeqRec have to be lightweight to accommodate communication bandwidth and clients' on-device computational resource constraints. Recently, large language models (LLMs) have exhibited strong transferable and generalized language understanding abilities and therefore, in the NLP area, many downstream tasks now utilize LLMs as a service to achieve superior performance without constructing complex models. Inspired by this successful practice, we propose a generic FedSeqRec framework, FELLAS, which aims to enhance FedSeqRec by utilizing LLMs as an external service. Specifically, FELLAS employs an LLM server to provide both item-level and sequence-level representation assistance. The item-level representation service is queried by the central server to enrich the original ID-based item embedding with textual information, while the sequence-level representation service is accessed by each client. However, invoking the sequence-level representation service requires clients to send sequences to the external LLM server. To safeguard privacy, we implement dx-privacy satisfied sequence perturbation, which protects clients' sensitive data with guarantees. Additionally, a contrastive learning-based method is designed to transfer knowledge from the noisy sequence representation to clients' sequential recommendation models. Furthermore, to empirically validate the privacy protection capability of FELLAS, we propose two interacted item inference attacks. Extensive experiments conducted on three datasets with two widely used sequential recommendation models demonstrate the effectiveness and privacy-preserving capability of FELLAS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06550v1">Investigating Cost-Efficiency of LLM-Generated Training Data for Conversational Semantic Frame Analysis</a></div>
    <div class="paper-meta">
      📅 2024-10-09
      | 💬 12 pages including 4 pages of references and appendix. 7 figures
    </div>
    <details class="paper-abstract">
      Recent studies have demonstrated that few-shot learning allows LLMs to generate training data for supervised models at a low cost. However, the quality of LLM-generated data may not entirely match that of human-labeled data. This raises a crucial question: how should one balance the trade-off between the higher quality but more expensive human data and the lower quality yet substantially cheaper LLM-generated data? In this paper, we synthesized training data for conversational semantic frame analysis using GPT-4 and examined how to allocate budgets optimally to achieve the best performance. Our experiments, conducted across various budget levels, reveal that optimal cost-efficiency is achieved by combining both human and LLM-generated data across a wide range of budget levels. Notably, as the budget decreases, a higher proportion of LLM-generated data becomes more preferable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10876v1">FreqMark: Frequency-Based Watermark for Sentence-Level Detection of LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      The increasing use of Large Language Models (LLMs) for generating highly coherent and contextually relevant text introduces new risks, including misuse for unethical purposes such as disinformation or academic dishonesty. To address these challenges, we propose FreqMark, a novel watermarking technique that embeds detectable frequency-based watermarks in LLM-generated text during the token sampling process. The method leverages periodic signals to guide token selection, creating a watermark that can be detected with Short-Time Fourier Transform (STFT) analysis. This approach enables accurate identification of LLM-generated content, even in mixed-text scenarios with both human-authored and LLM-generated segments. Our experiments demonstrate the robustness and precision of FreqMark, showing strong detection capabilities against various attack scenarios such as paraphrasing and token substitution. Results show that FreqMark achieves an AUC improvement of up to 0.98, significantly outperforming existing detection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06508v1">Towards Self-Improvement of LLMs via MCTS: Leveraging Stepwise Knowledge with Curriculum Preference Learning</a></div>
    <div class="paper-meta">
      📅 2024-10-09
    </div>
    <details class="paper-abstract">
      Monte Carlo Tree Search (MCTS) has recently emerged as a powerful technique for enhancing the reasoning capabilities of LLMs. Techniques such as SFT or DPO have enabled LLMs to distill high-quality behaviors from MCTS, improving their reasoning performance. However, existing distillation methods underutilize the rich trajectory information generated by MCTS, limiting the potential for improvements in LLM reasoning. In this paper, we propose AlphaLLM-CPL, a novel pairwise training framework that enables LLMs to self-improve through MCTS behavior distillation. AlphaLLM-CPL efficiently leverages MCTS trajectories via two key innovations: (1) AlphaLLM-CPL constructs stepwise trajectory pairs from child nodes sharing the same parent in the search tree, providing step-level information for more effective MCTS behavior distillation. (2) AlphaLLM-CPL introduces curriculum preference learning, dynamically adjusting the training sequence of trajectory pairs in each offline training epoch to prioritize critical learning steps and mitigate overfitting. Experimental results on mathematical reasoning tasks demonstrate that AlphaLLM-CPL significantly outperforms previous MCTS behavior distillation methods, substantially boosting the reasoning capabilities of LLMs.
    </details>
</div>
