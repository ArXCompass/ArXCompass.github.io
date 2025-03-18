# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00096v2">BixBench: a Comprehensive Benchmark for LLM-based Agents in Computational Biology</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 8 main text pages, 5 main figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and LLM-based agents show great promise in accelerating scientific research. Existing benchmarks for measuring this potential and guiding future development continue to evolve from pure recall and rote knowledge tasks, towards more practical work such as literature review and experimental planning. Bioinformatics is a domain where fully autonomous AI-driven discovery may be near, but no extensive benchmarks for measuring progress have been introduced to date. We therefore present the Bioinformatics Benchmark (BixBench), a dataset comprising over 50 real-world scenarios of practical biological data analysis with nearly 300 associated open-answer questions designed to measure the ability of LLM-based agents to explore biological datasets, perform long, multi-step analytical trajectories, and interpret the nuanced results of those analyses. We evaluate the performance of two frontier LLMs (GPT-4o and Claude 3.5 Sonnet) using a custom agent framework we open source. We find that even the latest frontier models only achieve 17% accuracy in the open-answer regime, and no better than random in a multiple-choice setting. By exposing the current limitations of frontier models, we hope BixBench can spur the development of agents capable of conducting rigorous bioinformatic analysis and accelerate scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23252v3">Evaluating Cultural and Social Awareness of LLM Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) expand into performing as agents for real-world applications beyond traditional NLP tasks, evaluating their robustness becomes increasingly important. However, existing benchmarks often overlook critical dimensions like cultural and social awareness. To address these, we introduce CASA, a benchmark designed to assess LLM agents' sensitivity to cultural and social norms across two web-based tasks: online shopping and social discussion forums. Our approach evaluates LLM agents' ability to detect and appropriately respond to norm-violating user queries and observations. Furthermore, we propose a comprehensive evaluation framework that measures awareness coverage, helpfulness in managing user queries, and the violation rate when facing misleading web content. Experiments show that current LLMs perform significantly better in non-agent than in web-based agent environments, with agents achieving less than 10% awareness coverage and over 40% violation rates. To improve performance, we explore two methods: prompting and fine-tuning, and find that combining both methods can offer complementary advantages -- fine-tuning on culture-specific datasets significantly enhances the agents' ability to generalize across different regions, while prompting boosts the agents' ability to navigate complex tasks. These findings highlight the importance of constantly benchmarking LLM agents' cultural and social awareness during the development cycle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11721v2">Explain-Query-Test: Self-Evaluating LLMs Via Explanation and Comprehension Discrepancy</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Accepted to ICLR 2025, SSI-FM
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in generating detailed and coherent explanations of complex concepts. However, the extent to which these models truly comprehend the concepts they articulate remains unclear. To assess the level of comprehension of a model relative to the content it generates, we implemented a self-evaluation pipeline where models: (i) given a topic generate an excerpt with information about the topic, (ii) given an excerpt generate question-answer pairs, and finally (iii) given a question generate an answer. We refer to this self-evaluation approach as Explain-Query-Test (EQT). Interestingly, the accuracy on generated questions resulting from running the EQT pipeline correlates strongly with the model performance as verified by typical benchmarks such as MMLU-Pro. In other words, EQT's performance is predictive of MMLU-Pro's, and EQT can be used to rank models without the need for any external source of evaluation data other than lists of topics of interest. Moreover, our results reveal a disparity between the models' ability to produce detailed explanations and their performance on questions related to those explanations. This gap highlights fundamental limitations in the internal knowledge representation and reasoning abilities of current LLMs. We release the code at https://github.com/asgsaeid/EQT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06330v1">States of LLM-generated Texts and Phase Transitions between them</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Published as a conference paper at MathAI 2025
    </div>
    <details class="paper-abstract">
      It is known for some time that autocorrelations of words in human-written texts decay according to a power law. Recent works have also shown that the autocorrelations decay in texts generated by LLMs is qualitatively different from the literary texts. Solid state physics tie the autocorrelations decay laws to the states of matter. In this work, we empirically demonstrate that, depending on the temperature parameter, LLMs can generate text that can be classified as solid, critical state or gas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06327v1">Unveiling Inefficiencies in LLM-Generated Code: Toward a Comprehensive Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely adopted for automated code generation with promising results. Although prior research has assessed LLM-generated code and identified various quality issues -- such as redundancy, poor maintainability, and sub-optimal performance a systematic understanding and categorization of these inefficiencies remain unexplored. Without such knowledge, practitioners struggle to optimize LLM-generated code for real-world applications, limiting its adoption. This study can also guide improving code LLMs, enhancing the quality and efficiency of code generation. Therefore, in this study, we empirically investigate inefficiencies in LLM-generated code by state-of-the-art models, i.e., CodeLlama, DeepSeek-Coder, and CodeGemma. To do so, we analyze 492 generated code snippets in the HumanEval++ dataset. We then construct a taxonomy of inefficiencies in LLM-generated code that includes 5 categories General Logic, Performance, Readability, Maintainability, and Errors) and 19 subcategories of inefficiencies. We then validate the proposed taxonomy through an online survey with 58 LLM practitioners and researchers. Our study indicates that logic and performance-related inefficiencies are the most popular, relevant, and frequently co-occur and impact overall code quality inefficiency. Our taxonomy provides a structured basis for evaluating the quality LLM-generated code and guiding future research to improve code generation efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06313v1">Advancing Autonomous Vehicle Intelligence: Deep Learning and Multimodal LLM for Traffic Sign Recognition and Robust Lane Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 11 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Autonomous vehicles (AVs) require reliable traffic sign recognition and robust lane detection capabilities to ensure safe navigation in complex and dynamic environments. This paper introduces an integrated approach combining advanced deep learning techniques and Multimodal Large Language Models (MLLMs) for comprehensive road perception. For traffic sign recognition, we systematically evaluate ResNet-50, YOLOv8, and RT-DETR, achieving state-of-the-art performance of 99.8% with ResNet-50, 98.0% accuracy with YOLOv8, and achieved 96.6% accuracy in RT-DETR despite its higher computational complexity. For lane detection, we propose a CNN-based segmentation method enhanced by polynomial curve fitting, which delivers high accuracy under favorable conditions. Furthermore, we introduce a lightweight, Multimodal, LLM-based framework that directly undergoes instruction tuning using small yet diverse datasets, eliminating the need for initial pretraining. This framework effectively handles various lane types, complex intersections, and merging zones, significantly enhancing lane detection reliability by reasoning under adverse conditions. Despite constraints in available training resources, our multimodal approach demonstrates advanced reasoning capabilities, achieving a Frame Overall Accuracy (FRM) of 53.87%, a Question Overall Accuracy (QNS) of 82.83%, lane detection accuracies of 99.6% in clear conditions and 93.0% at night, and robust performance in reasoning about lane invisibility due to rain (88.4%) or road degradation (95.6%). The proposed comprehensive framework markedly enhances AV perception reliability, thus contributing significantly to safer autonomous driving across diverse and challenging road scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07923v2">Asking Again and Again: Exploring LLM Robustness to Repeated Questions</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      This study investigates whether repeating questions within prompts influences the performance of large language models (LLMs). We hypothesize that reiterating a question within a single prompt might enhance the model's focus on key elements of the query. We evaluate five recent LLMs -- including GPT-4o-mini, DeepSeek-V3, and smaller open-source models -- on three reading comprehension datasets under different prompt settings, varying question repetition levels (1, 3, or 5 times per prompt). Our results demonstrate that question repetition can increase models' accuracy by up to $6\%$. However, across all models, settings, and datasets, we do not find the result statistically significant. These findings provide insights into prompt design and LLM behavior, suggesting that repetition alone does not significantly impact output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06273v1">Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      We explore a novel zero-shot Audio-Visual Speech Recognition (AVSR) framework, dubbed Zero-AVSR, which enables speech recognition in target languages without requiring any audio-visual speech data in those languages. Specifically, we introduce the Audio-Visual Speech Romanizer (AV-Romanizer), which learns language-agnostic speech representations by predicting Roman text. Then, by leveraging the strong multilingual modeling capabilities of Large Language Models (LLMs), we propose converting the predicted Roman text into language-specific graphemes, forming the proposed Cascaded Zero-AVSR. Taking it a step further, we explore a unified Zero-AVSR approach by directly integrating the audio-visual speech representations encoded by the AV-Romanizer into the LLM. This is achieved through finetuning the adapter and the LLM using our proposed multi-task learning scheme. To capture the wide spectrum of phonetic and linguistic diversity, we also introduce a Multilingual Audio-Visual Romanized Corpus (MARC) consisting of 2,916 hours of audio-visual speech data across 82 languages, along with transcriptions in both language-specific graphemes and Roman text. Extensive analysis and experiments confirm that the proposed Zero-AVSR framework has the potential to expand language support beyond the languages seen during the training of the AV-Romanizer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06253v1">MAD-MAX: Modular And Diverse Malicious Attack MiXtures for Automated LLM Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      With LLM usage rapidly increasing, their vulnerability to jailbreaks that create harmful outputs are a major security risk. As new jailbreaking strategies emerge and models are changed by fine-tuning, continuous testing for security vulnerabilities is necessary. Existing Red Teaming methods fall short in cost efficiency, attack success rate, attack diversity, or extensibility as new attack types emerge. We address these challenges with Modular And Diverse Malicious Attack MiXtures (MAD-MAX) for Automated LLM Red Teaming. MAD-MAX uses automatic assignment of attack strategies into relevant attack clusters, chooses the most relevant clusters for a malicious goal, and then combines strategies from the selected clusters to achieve diverse novel attacks with high attack success rates. MAD-MAX further merges promising attacks together at each iteration of Red Teaming to boost performance and introduces a similarity filter to prune out similar attacks for increased cost efficiency. The MAD-MAX approach is designed to be easily extensible with newly discovered attack strategies and outperforms the prominent Red Teaming method Tree of Attacks with Pruning (TAP) significantly in terms of Attack Success Rate (ASR) and queries needed to achieve jailbreaks. MAD-MAX jailbreaks 97% of malicious goals in our benchmarks on GPT-4o and Gemini-Pro compared to TAP with 66%. MAD-MAX does so with only 10.9 average queries to the target LLM compared to TAP with 23.3. WARNING: This paper contains contents which are offensive in nature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06705v2">LLMs can Find Mathematical Reasoning Mistakes by Pedagogical Chain-of-Thought</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Accepted by IJCAI 2024
    </div>
    <details class="paper-abstract">
      Self-correction is emerging as a promising approach to mitigate the issue of hallucination in Large Language Models (LLMs). To facilitate effective self-correction, recent research has proposed mistake detection as its initial step. However, current literature suggests that LLMs often struggle with reliably identifying reasoning mistakes when using simplistic prompting strategies. To address this challenge, we introduce a unique prompting strategy, termed the Pedagogical Chain-of-Thought (PedCoT), which is specifically designed to guide the identification of reasoning mistakes, particularly mathematical reasoning mistakes. PedCoT consists of pedagogical principles for prompts (PPP) design, two-stage interaction process (TIP) and grounded PedCoT prompts, all inspired by the educational theory of the Bloom Cognitive Model (BCM). We evaluate our approach on two public datasets featuring math problems of varying difficulty levels. The experiments demonstrate that our zero-shot prompting strategy significantly outperforms strong baselines. The proposed method can achieve the goal of reliable mathematical mistake identification and provide a foundation for automatic math answer grading. The results underscore the significance of educational theory, serving as domain knowledge, in guiding prompting strategy design for addressing challenging tasks with LLMs effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06203v1">Generation of Optimized Solidity Code for Machine Learning Models using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      While a plethora of machine learning (ML) models are currently available, along with their implementation on disparate platforms, there is hardly any verifiable ML code which can be executed on public blockchains. We propose a novel approach named LMST that enables conversion of the inferencing path of an ML model as well as its weights trained off-chain into Solidity code using Large Language Models (LLMs). Extensive prompt engineering is done to achieve gas cost optimization beyond mere correctness of the produced code, while taking into consideration the capabilities and limitations of the Ethereum Virtual Machine. We have also developed a proof of concept decentralized application using the code so generated for verifying the accuracy claims of the underlying ML model. An extensive set of experiments demonstrate the feasibility of deploying ML models on blockchains through automated code translation using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05673v5">Flow of Reasoning:Training LLMs for Divergent Problem Solving with Minimal Examples</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      The ability to generate diverse solutions to a given problem is a hallmark of human creativity. This divergent reasoning is also crucial for machines, enhancing their robustness and enabling them to assist humans in many applications such as scientific discovery. However, existing approaches to multi-step reasoning with large language models (LLMs) have mostly focused only on reasoning accuracy, without further discovering more diverse valid solutions. For example, supervised fine-tuning can improve LLM reasoning quality, but requires extensive supervised data to capture the full range of possible solutions. Reward-maximization reinforcement learning aims to find limited highest-reward solutions while neglecting the solution diversity. To fill this gap, we propose Flow of Reasoning (FoR), an efficient diversity-seeking LLM finetuning method aimed at improving reasoning quality and diversity with minimal data. FoR formulates multi-step LLM reasoning as a Markovian flow on a DAG-structured reasoning graph. This formulation allows us to incorporate and adapt principled GFlowNet approaches, for finetuning LLMs to sample divergent paths with probabilities proportional to the (unnormalized) reward of target problems. Extensive experiments show that, with limited training examples (e.g., 15 examples), FoR enables the discovery of diverse, creative, high-quality solutions, greatly outperforming a wide range of existing inference and training methods across six challenging reasoning tasks, including BlocksWorld (embodied reasoning), Game24 (math puzzle solving), Rubik's Cube (spatial reasoning), 1D-ARC (abstraction reasoning), GSM8k (math reasoning), and ProntoQA (logical reasoning). Code is available at https://github.com/Yu-Fangxu/FoR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08468v2">Multi-GraspLLM: A Multimodal LLM for Multi-Hand Semantic Guided Grasp Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Multi-hand semantic grasp generation aims to generate feasible and semantically appropriate grasp poses for different robotic hands based on natural language instructions. Although the task is highly valuable, due to the lack of multihand grasp datasets with fine-grained contact description between robotic hands and objects, it is still a long-standing difficult task. In this paper, we present Multi-GraspSet, the first large-scale multi-hand grasp dataset with automatically contact annotations. Based on Multi-GraspSet, we propose Multi-GraspLLM, a unified language-guided grasp generation framework, which leverages large language models (LLM) to handle variable-length sequences, generating grasp poses for diverse robotic hands in a single unified architecture. Multi-GraspLLM first aligns the encoded point cloud features and text features into a unified semantic space. It then generates grasp bin tokens that are subsequently converted into grasp pose for each robotic hand via hand-aware linear mapping. The experimental results demonstrate that our approach significantly outperforms existing methods in both real-world experiments and simulator. More information can be found on our project page https://multi-graspllm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06139v1">GRP: Goal-Reversed Prompting for Zero-Shot Evaluation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Ongoing Work
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) to evaluate and compare two answers from different models typically involves having LLM-based judges select the better answer. However, humans often approach problem-solving from a reverse perspective, for instance, by choosing the worse option instead of the better one in a pairwise comparison. Generally, this kind of reverse thinking plays a crucial role in human reasoning and decision-making and can further test the difference between original and reverse thought processes simultaneously. To address the above issue, in this paper, we propose a Goal-Reversed Prompting (GRP) approach for pairwise evaluation that shifts the original task from selecting the better answer to choosing the worse one. We encourage LLMs to think in reverse by prompting LLMs to identify the worse response. Experiments on closed-source models demonstrate that GRP significantly enhances evaluation capabilities, outperforming the prompt template with the original goal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06119v1">Unlocking Pretrained LLMs for Motion-Related Multimodal Generation: A Fine-Tuning Approach to Unify Diffusion and Next-Token Prediction</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      In this paper, we propose a unified framework that leverages a single pretrained LLM for Motion-related Multimodal Generation, referred to as MoMug. MoMug integrates diffusion-based continuous motion generation with the model's inherent autoregressive discrete text prediction capabilities by fine-tuning a pretrained LLM. This enables seamless switching between continuous motion output and discrete text token prediction within a single model architecture, effectively combining the strengths of both diffusion- and LLM-based approaches. Experimental results show that, compared to the most recent LLM-based baseline, MoMug improves FID by 38% and mean accuracy across seven metrics by 16.61% on the text-to-motion task. Additionally, it improves mean accuracy across eight metrics by 8.44% on the text-to-motion task. To the best of our knowledge, this is the first approach to integrate diffusion- and LLM-based generation within a single model for motion-related multimodal tasks while maintaining low training costs. This establishes a foundation for future advancements in motion-related generation, paving the way for high-quality yet cost-efficient motion synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01600v3">Reinforcement Learning for Long-Horizon Interactive LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Interactive digital agents (IDAs) leverage APIs of stateful digital environments to perform tasks in response to user requests. While IDAs powered by instruction-tuned large language models (LLMs) can react to feedback from interface invocations in multi-step exchanges, they have not been trained in their respective digital environments. Prior methods accomplish less than half of tasks in sophisticated benchmarks such as AppWorld. We present a reinforcement learning (RL) approach that trains IDAs directly in their target environments. We formalize this training as a partially observable Markov decision process and derive LOOP, a data- and memory-efficient variant of proximal policy optimization. LOOP uses no value network and maintains exactly one copy of the underlying LLM in memory, making its implementation straightforward and as memory-efficient as fine-tuning a single LLM. A 32-billion-parameter agent trained with LOOP in the AppWorld environment outperforms the much larger OpenAI o1 agent by 9 percentage points (15% relative). To our knowledge, this is the first reported application of RL to IDAs that interact with a stateful, multi-domain, multi-app environment via direct API calls. Our analysis sheds light on the effectiveness of RL in this area, showing that the agent learns to consult the API documentation, avoid unwarranted assumptions, minimize confabulation, and recover from setbacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02644v2">Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10868v3">NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai Legal Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the legal domain holds significant potential for information retrieval and question answering, yet Thai legal QA systems face challenges due to a lack of standardized evaluation benchmarks and the complexity of Thai legal structures. This paper introduces NitiBench, a benchmark comprising two datasets: the NitiBench-CCL, covering general Thai financial law, and the NitiBench-Tax, which includes real-world tax law cases requiring advanced legal reasoning. We evaluate retrieval-augmented generation (RAG) and long-context LLM-based approaches to address three key research questions: the impact of domain-specific components like section-based chunking and cross-referencing, the comparative performance of different retrievers and LLMs, and the viability of long-context LLMs as an alternative to RAG. Our results show that section-based chunking significantly improves retrieval and end-to-end performance, current retrievers struggle with complex queries, and long-context LLMs still underperform RAG-based systems in Thai legal QA. To support fair evaluation, we propose tailored multi-label retrieval metrics and the use of an LLM-as-judge for coverage and contradiction detection method. These findings highlight the limitations of current Thai legal NLP solutions and provide a foundation for future research in the field. We also open-sourced our codes and dataset to available publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06054v1">Fine-Grained Bias Detection in LLM: Enhancing detection mechanisms for nuanced biases</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 Bias detection, Large Language Models, nuanced biases, fine-grained mechanisms, model transparency, ethical AI
    </div>
    <details class="paper-abstract">
      Recent advancements in Artificial Intelligence, particularly in Large Language Models (LLMs), have transformed natural language processing by improving generative capabilities. However, detecting biases embedded within these models remains a challenge. Subtle biases can propagate misinformation, influence decision-making, and reinforce stereotypes, raising ethical concerns. This study presents a detection framework to identify nuanced biases in LLMs. The approach integrates contextual analysis, interpretability via attention mechanisms, and counterfactual data augmentation to capture hidden biases across linguistic contexts. The methodology employs contrastive prompts and synthetic datasets to analyze model behaviour across cultural, ideological, and demographic scenarios. Quantitative analysis using benchmark datasets and qualitative assessments through expert reviews validate the effectiveness of the framework. Results show improvements in detecting subtle biases compared to conventional methods, which often fail to highlight disparities in model responses to race, gender, and socio-political contexts. The framework also identifies biases arising from imbalances in training data and model architectures. Continuous user feedback ensures adaptability and refinement. This research underscores the importance of proactive bias mitigation strategies and calls for collaboration between policymakers, AI developers, and regulators. The proposed detection mechanisms enhance model transparency and support responsible LLM deployment in sensitive applications such as education, legal systems, and healthcare. Future work will focus on real-time bias monitoring and cross-linguistic generalization to improve fairness and inclusivity in AI-driven communication tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06047v1">DSGBench: A Diverse Strategic Game Benchmark for Evaluating LLM-based Agents in Complex Decision-Making Environments</a></div>
    <div class="paper-meta">
      📅 2025-03-08
      | 💬 43 pages, 5 figures, conference
    </div>
    <details class="paper-abstract">
      Large Language Model~(LLM) based agents have been increasingly popular in solving complex and dynamic tasks, which requires proper evaluation systems to assess their capabilities. Nevertheless, existing benchmarks usually either focus on single-objective tasks or use overly broad assessing metrics, failing to provide a comprehensive inspection of the actual capabilities of LLM-based agents in complicated decision-making tasks. To address these issues, we introduce DSGBench, a more rigorous evaluation platform for strategic decision-making. Firstly, it incorporates six complex strategic games which serve as ideal testbeds due to their long-term and multi-dimensional decision-making demands and flexibility in customizing tasks of various difficulty levels or multiple targets. Secondly, DSGBench employs a fine-grained evaluation scoring system which examines the decision-making capabilities by looking into the performance in five specific dimensions and offering a comprehensive assessment in a well-designed way. Furthermore, DSGBench also incorporates an automated decision-tracking mechanism which enables in-depth analysis of agent behaviour patterns and the changes in their strategies. We demonstrate the advances of DSGBench by applying it to multiple popular LLM-based agents and our results suggest that DSGBench provides valuable insights in choosing LLM-based agents as well as improving their future development. DSGBench is available at https://github.com/DeciBrain-Group/DSGBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06040v1">Mitigating Memorization in LLMs using Activation Steering</a></div>
    <div class="paper-meta">
      📅 2025-03-08
    </div>
    <details class="paper-abstract">
      The memorization of training data by Large Language Models (LLMs) poses significant risks, including privacy leaks and the regurgitation of copyrighted content. Activation steering, a technique that directly intervenes in model activations, has emerged as a promising approach for manipulating LLMs. In this work, we explore the effectiveness of activation steering in reducing memorization while preserving generalization capabilities. We conduct empirical evaluations using a controlled memorization benchmark of literary material and demonstrate that our method successfully suppresses memorized content with minimal degradation in model performance in Gemma. Additionally, we analyze the trade-offs between suppression effectiveness and linguistic fluency, highlighting the advantages and limitations of activation-based interventions. Our findings contribute to ongoing efforts in developing safer and more privacy-preserving LLMs by providing a practical and efficient mechanism to mitigate unintended memorization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v4">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve strong performances on various benchmark datasets without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05965v1">Validating LLM-as-a-Judge Systems in the Absence of Gold Labels</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm, in which a judge LLM system replaces human raters in rating the outputs of other generative AI (GenAI) systems, has come to play a critical role in scaling and standardizing GenAI evaluations. To validate judge systems, evaluators collect multiple human ratings for each item in a validation corpus, and then aggregate the ratings into a single, per-item gold label rating. High agreement rates between these gold labels and judge system ratings are then taken as a sign of good judge system performance. In many cases, however, items or rating criteria may be ambiguous, or there may be principled disagreement among human raters. In such settings, gold labels may not exist for many of the items. In this paper, we introduce a framework for LLM-as-a-judge validation in the absence of gold labels. We present a theoretical analysis drawing connections between different measures of judge system performance under different rating elicitation and aggregation schemes. We also demonstrate empirically that existing validation approaches can select judge systems that are highly suboptimal, performing as much as 34% worse than the systems selected by alternative approaches that we describe. Based on our findings, we provide concrete recommendations for developing more reliable approaches to LLM-as-a-judge validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05951v1">TPU-Gen: LLM-Driven Custom Tensor Processing Unit Generator</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 8 Pages, 9 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      The increasing complexity and scale of Deep Neural Networks (DNNs) necessitate specialized tensor accelerators, such as Tensor Processing Units (TPUs), to meet various computational and energy efficiency requirements. Nevertheless, designing optimal TPU remains challenging due to the high domain expertise level, considerable manual design time, and lack of high-quality, domain-specific datasets. This paper introduces TPU-Gen, the first Large Language Model (LLM) based framework designed to automate the exact and approximate TPU generation process, focusing on systolic array architectures. TPU-Gen is supported with a meticulously curated, comprehensive, and open-source dataset that covers a wide range of spatial array designs and approximate multiply-and-accumulate units, enabling design reuse, adaptation, and customization for different DNN workloads. The proposed framework leverages Retrieval-Augmented Generation (RAG) as an effective solution for a data-scare hardware domain in building LLMs, addressing the most intriguing issue, hallucinations. TPU-Gen transforms high-level architectural specifications into optimized low-level implementations through an effective hardware generation pipeline. Our extensive experimental evaluations demonstrate superior performance, power, and area efficiency, with an average reduction in area and power of 92\% and 96\% from the manual optimization reference values. These results set new standards for driving advancements in next-generation design automation tools powered by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19865v2">Reverse Thinking Makes LLMs Stronger Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted to NAACL 2025
    </div>
    <details class="paper-abstract">
      Reverse thinking plays a crucial role in human reasoning. Humans can reason not only from a problem to a solution but also in reverse, i.e., start from the solution and reason towards the problem. This often enhances overall reasoning performance as it enables consistency checks between their forward and backward thinking. To enable Large Language Models (LLMs) to perform reverse thinking, we introduce Reverse-Enhanced Thinking (RevThink), a framework composed of data augmentation and learning objectives. In RevThink, we augment the dataset by collecting structured forward-backward reasoning from a teacher model, consisting of: (1) the original question, (2) forward reasoning, (3) backward question, and (4) backward reasoning. We then employ three objectives to train a smaller student model in a multi-task learning fashion: (a) generate forward reasoning from a question, (b) generate a backward question from a question, and (c) generate backward reasoning from the backward question. Experiments across 12 datasets covering commonsense, math, and logical reasoning show an average 13.53% improvement over the student model's zero-shot performance and a 6.84% improvement over the strongest knowledge distillation baselines. Moreover, our method demonstrates sample efficiency -- using only 10% of the correct forward reasoning from the training data, it outperforms a standard fine-tuning method trained on 10x more forward reasoning. RevThink also exhibits strong generalization to out-of-distribution held-out datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.08291v3">CleanAgent: Automating Data Standardization with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Data standardization is a crucial part of the data science life cycle. While tools like Pandas offer robust functionalities, their complexity and the manual effort required for customizing code to diverse column types pose significant challenges. Although large language models (LLMs) like ChatGPT have shown promise in automating this process through natural language understanding and code generation, it still demands expert-level programming knowledge and continuous interaction for prompt refinement. To solve these challenges, our key idea is to propose a Python library with declarative, unified APIs for standardizing different column types, simplifying the LLM's code generation with concise API calls. We first propose Dataprep.Clean, a component of the Dataprep Python Library, significantly reduces the coding complexity by enabling the standardization of specific column types with a single line of code. Then, we introduce the CleanAgent framework integrating Dataprep.Clean and LLM-based agents to automate the data standardization process. With CleanAgent, data scientists only need to provide their requirements once, allowing for a hands-free process. To demonstrate the practical utility of CleanAgent, we developed a user-friendly web application, allowing attendees to interact with it using real-world datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05856v1">This Is Your Doge, If It Please You: Exploring Deception and Robustness in Mixture of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 35 pages, 9 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Mixture of large language model (LLMs) Agents (MoA) architectures achieve state-of-the-art performance on prominent benchmarks like AlpacaEval 2.0 by leveraging the collaboration of multiple LLMs at inference time. Despite these successes, an evaluation of the safety and reliability of MoA is missing. We present the first comprehensive study of MoA's robustness against deceptive LLM agents that deliberately provide misleading responses. We examine factors like the propagation of deceptive information, model size, and information availability, and uncover critical vulnerabilities. On AlpacaEval 2.0, the popular LLaMA 3.1-70B model achieves a length-controlled Win Rate (LC WR) of 49.2% when coupled with 3-layer MoA (6 LLM agents). However, we demonstrate that introducing only a $\textit{single}$ carefully-instructed deceptive agent into the MoA can reduce performance to 37.9%, effectively nullifying all MoA gains. On QuALITY, a multiple-choice comprehension task, the impact is also severe, with accuracy plummeting by a staggering 48.5%. Inspired in part by the historical Doge of Venice voting process, designed to minimize influence and deception, we propose a range of unsupervised defense mechanisms that recover most of the lost performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05854v1">Accelerating Earth Science Discovery via Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 10 pages, 1 figure. Perspective article
    </div>
    <details class="paper-abstract">
      This Perspective explores the transformative potential of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) in the geosciences. Users of geoscientific data repositories face challenges due to the complexity and diversity of data formats, inconsistent metadata practices, and a considerable number of unprocessed datasets. MAS possesses transformative potential for improving scientists' interaction with geoscientific data by enabling intelligent data processing, natural language interfaces, and collaborative problem-solving capabilities. We illustrate this approach with "PANGAEA GPT", a specialized MAS pipeline integrated with the diverse PANGAEA database for Earth and Environmental Science, demonstrating how MAS-driven workflows can effectively manage complex datasets and accelerate scientific discovery. We discuss how MAS can address current data challenges in geosciences, highlight advancements in other scientific fields, and propose future directions for integrating MAS into geoscientific data processing pipelines. In this Perspective, we show how MAS can fundamentally improve data accessibility, promote cross-disciplinary collaboration, and accelerate geoscientific discoveries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05846v1">Extracting and Emulsifying Cultural Explanation to Improve Multilingual Capability of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 under review, 18pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success, but their English-centric training data limits performance in non-English languages, highlighting the need for enhancements in their multilingual capabilities. While some work on multilingual prompting methods handles non-English queries by utilizing English translations or restructuring them to more closely align with LLM reasoning patterns, these works often overlook the importance of cultural context, limiting their effectiveness. To address this limitation, we propose EMCEI, a simple yet effective approach that improves LLMs' multilingual capabilities by incorporating cultural context for more accurate and appropriate responses. Specifically, EMCEI follows a two-step process that first extracts relevant cultural context from the LLM's parametric knowledge via prompting. Then, EMCEI employs an LLM-as-Judge mechanism to select the most appropriate response by balancing cultural relevance and reasoning ability. Experiments on diverse multilingual benchmarks show that EMCEI outperforms existing baselines, demonstrating its effectiveness in handling multilingual queries with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07657v1">SplitQuantV2: Enhancing Low-Bit Quantization of LLMs Without GPUs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The quantization of large language models (LLMs) is crucial for deploying them on devices with limited computational resources. While advanced quantization algorithms offer improved performance compared to the basic linear quantization, they typically require high-end graphics processing units (GPUs), are often restricted to specific deep neural network (DNN) frameworks, and require calibration datasets. This limitation poses challenges for using such algorithms on various neural processing units (NPUs) and edge AI devices, which have diverse model formats and frameworks. In this paper, we show SplitQuantV2, an innovative algorithm designed to enhance low-bit linear quantization of LLMs, can achieve results comparable to those of advanced algorithms. SplitQuantV2 preprocesses models by splitting linear and convolution layers into functionally equivalent, quantization-friendly structures. The algorithm's platform-agnostic, concise, and efficient nature allows for implementation without the need for GPUs. Our evaluation on the Llama 3.2 1B Instruct model using the AI2's Reasoning Challenge (ARC) dataset demonstrates that SplitQuantV2 improves the accuracy of the INT4 quantization model by 11.76%p, matching the performance of the original floating-point model. Remarkably, SplitQuantV2 took only 2 minutes 6 seconds to preprocess the 1B model and perform linear INT4 quantization using only an Apple M4 CPU. SplitQuantV2 provides a practical solution for low-bit quantization on LLMs, especially when complex, computation-intensive algorithms are inaccessible due to hardware limitations or framework incompatibilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10351v4">Bias Unveiled: Investigating Social Bias in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 accepted for publication in the Association for the Advancement of Artificial Intelligence (AAAI), 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced the field of automated code generation. However, a notable research gap exists in evaluating social biases that may be present in the code produced by LLMs. To solve this issue, we propose a novel fairness framework, i.e., Solar, to assess and mitigate the social biases of LLM-generated code. Specifically, Solar can automatically generate test cases for quantitatively uncovering social biases of the auto-generated code by LLMs. To quantify the severity of social biases in generated code, we develop a dataset that covers a diverse set of social problems. We applied Solar and the crafted dataset to four state-of-the-art LLMs for code generation. Our evaluation reveals severe bias in the LLM-generated code from all the subject LLMs. Furthermore, we explore several prompting strategies for mitigating bias, including Chain-of-Thought (CoT) prompting, combining positive role-playing with CoT prompting and dialogue with Solar. Our experiments show that dialogue with Solar can effectively reduce social bias in LLM-generated code by up to 90%. Last, we make the code and data publicly available is highly extensible to evaluate new social problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.00242v4">DeFT: Decoding with Flash Tree-attention for Efficient Tree-structured LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Update DeFT-v4, accepted by ICLR'25 (https://openreview.net/forum?id=2c7pfOqu9k). Our code is available at https://github.com/LINs-lab/DeFT
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly employed for complex tasks that process multiple generation calls in a tree structure with shared prefixes of tokens, including few-shot prompting, multi-step reasoning, speculative decoding, etc. However, existing inference systems for tree-based applications are inefficient due to improper partitioning of queries and KV cache during attention calculation. This leads to two main issues: (1) a lack of memory access (IO) reuse for KV cache of shared prefixes, and (2) poor load balancing.As a result, there is redundant KV cache IO between GPU global memory and shared memory, along with low GPU utilization. To address these challenges, we propose DeFT(Decoding with Flash Tree-Attention), a hardware-efficient attention algorithm with prefix-aware and load-balanced KV cache partitions. DeFT reduces the number of read/write operations of KV cache during attention calculation through KV-Guided Grouping, a method that avoids repeatedly loading KV cache of shared prefixes in attention computation. Additionally, we propose Flattened Tree KV Splitting, a mechanism that ensures even distribution of the KV cache across partitions with little computation redundancy, enhancing GPU utilization during attention computations. By reducing 73-99% KV cache IO and nearly 100% IO for partial results during attention calculation, DeFT achieves up to 2.23/3.59x speedup in the end-to-end/attention latency across three practical tree-based workloads compared to state-of-the-art attention algorithms. Our code is available at https://github.com/LINs-lab/DeFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05620v1">Learning LLM Preference over Intra-Dialogue Pairs: A Framework for Utterance-level Understandings</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in handling complex dialogue tasks without requiring use case-specific fine-tuning. However, analyzing live dialogues in real-time necessitates low-latency processing systems, making it impractical to deploy models with billions of parameters due to latency constraints. As a result, practitioners often prefer smaller models with millions of parameters, trained on high-quality, human-annotated datasets. Yet, curating such datasets is both time-consuming and costly. Consequently, there is a growing need to combine the scalability of LLM-generated labels with the precision of human annotations, enabling fine-tuned smaller models to achieve both higher speed and accuracy comparable to larger models. In this paper, we introduce a simple yet effective framework to address this challenge. Our approach is specifically designed for per-utterance classification problems, which encompass tasks such as intent detection, dialogue state tracking, and more. To mitigate the impact of labeling errors from LLMs -- the primary source of inaccuracies in student models -- we propose a noise-reduced preference learning loss. Experimental results demonstrate that our method significantly improves accuracy across utterance-level dialogue tasks, including sentiment detection (over $2\%$), dialogue act classification (over $1.5\%$), etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05592v1">R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Existing Large Reasoning Models (LRMs) have shown the potential of reinforcement learning (RL) to enhance the complex reasoning capabilities of Large Language Models~(LLMs). While they achieve remarkable performance on challenging tasks such as mathematics and coding, they often rely on their internal knowledge to solve problems, which can be inadequate for time-sensitive or knowledge-intensive questions, leading to inaccuracies and hallucinations. To address this, we propose \textbf{R1-Searcher}, a novel two-stage outcome-based RL approach designed to enhance the search capabilities of LLMs. This method allows LLMs to autonomously invoke external search systems to access additional knowledge during the reasoning process. Our framework relies exclusively on RL, without requiring process rewards or distillation for a cold start. % effectively generalizing to out-of-domain datasets and supporting both Base and Instruct models. Our experiments demonstrate that our method significantly outperforms previous strong RAG methods, even when compared to the closed-source GPT-4o-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17975v3">SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It)</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML 2025)
    </div>
    <details class="paper-abstract">
      Whether LLMs memorize their training data and what this means, from measuring privacy leakage to detecting copyright violations, has become a rapidly growing area of research. In the last few months, more than 10 new methods have been proposed to perform Membership Inference Attacks (MIAs) against LLMs. Contrary to traditional MIAs which rely on fixed-but randomized-records or models, these methods are mostly trained and tested on datasets collected post-hoc. Sets of members and non-members, used to evaluate the MIA, are constructed using informed guesses after the release of a model. This lack of randomization raises concerns of a distribution shift between members and non-members. In this work, we first extensively review the literature on MIAs against LLMs and show that, while most work focuses on sequence-level MIAs evaluated in post-hoc setups, a range of target models, motivations and units of interest are considered. We then quantify distribution shifts present in 6 datasets used in the literature using a model-less bag of word classifier and show that all datasets constructed post-hoc suffer from strong distribution shifts. These shifts invalidate the claims of LLMs memorizing strongly in real-world scenarios and, potentially, also the methodological contributions of the recent papers based on these datasets. Yet, all hope might not be lost. We introduce important considerations to properly evaluate MIAs against LLMs and discuss, in turn, potential ways forwards: randomized test splits, injections of randomized (unique) sequences, randomized fine-tuning, and several post-hoc control methods. While each option comes with its advantages and limitations, we believe they collectively provide solid grounds to guide MIA development and study LLM memorization. We conclude with an overview of recommended approaches to benchmark sequence-level and document-level MIAs against LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05529v1">PoSSUM: A Protocol for Surveying Social-media Users with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      This paper introduces PoSSUM, an open-source protocol for unobtrusive polling of social-media users via multimodal Large Language Models (LLMs). PoSSUM leverages users' real-time posts, images, and other digital traces to create silicon samples that capture information not present in the LLM's training data. To obtain representative estimates, PoSSUM employs Multilevel Regression and Post-Stratification (MrP) with structured priors to counteract the observable selection biases of social-media platforms. The protocol is validated during the 2024 U.S. Presidential Election, for which five PoSSUM polls were conducted and published on GitHub and X. In the final poll, fielded October 17-26 with a synthetic sample of 1,054 X users, PoSSUM accurately predicted the outcomes in 50 of 51 states and assigned the Republican candidate a win probability of 0.65. Notably, it also exhibited lower state-level bias than most established pollsters. These results demonstrate PoSSUM's potential as a fully automated, unobtrusive alternative to traditional survey methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05507v1">Grammar-Based Code Representation: Is It a Worthy Pursuit for LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Grammar serves as a cornerstone in programming languages and software engineering, providing frameworks to define the syntactic space and program structure. Existing research demonstrates the effectiveness of grammar-based code representations in small-scale models, showing their ability to reduce syntax errors and enhance performance. However, as language models scale to the billion level or beyond, syntax-level errors become rare, making it unclear whether grammar information still provides performance benefits. To explore this, we develop a series of billion-scale GrammarCoder models, incorporating grammar rules in the code generation process. Experiments on HumanEval (+) and MBPP (+) demonstrate a notable improvement in code generation accuracy. Further analysis shows that grammar-based representations enhance LLMs' ability to discern subtle code differences, reducing semantic errors caused by minor variations. These findings suggest that grammar-based code representations remain valuable even in billion-scale models, not only by maintaining syntax correctness but also by improving semantic differentiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05493v1">Benchmarking LLMs in Recommendation Tasks: A Comparative Evaluation with Conventional Recommenders</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      In recent years, integrating large language models (LLMs) into recommender systems has created new opportunities for improving recommendation quality. However, a comprehensive benchmark is needed to thoroughly evaluate and compare the recommendation capabilities of LLMs with traditional recommender systems. In this paper, we introduce RecBench, which systematically investigates various item representation forms (including unique identifier, text, semantic embedding, and semantic identifier) and evaluates two primary recommendation tasks, i.e., click-through rate prediction (CTR) and sequential recommendation (SeqRec). Our extensive experiments cover up to 17 large models and are conducted across five diverse datasets from fashion, news, video, books, and music domains. Our findings indicate that LLM-based recommenders outperform conventional recommenders, achieving up to a 5% AUC improvement in the CTR scenario and up to a 170% NDCG@10 improvement in the SeqRec scenario. However, these substantial performance gains come at the expense of significantly reduced inference efficiency, rendering the LLM-as-RS paradigm impractical for real-time recommendation environments. We aim for our findings to inspire future research, including recommendation-specific model acceleration methods. We will release our code, data, configurations, and platform to enable other researchers to reproduce and build upon our experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02694v4">MeanCache: User-Centric Semantic Caching for LLM Web Services</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted at 2025 IEEE 39th International Parallel and Distributed Processing Symposium (IPDPS)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) like ChatGPT and Llama have revolutionized natural language processing and search engine dynamics. However, these models incur exceptionally high computational costs. For instance, GPT-3 consists of 175 billion parameters, where inference demands billions of floating-point operations. Caching is a natural solution to reduce LLM inference costs on repeated queries, which constitute about 31% of the total queries. However, existing caching methods are incapable of finding semantic similarities among LLM queries nor do they operate on contextual queries, leading to unacceptable false hit-and-miss rates. This paper introduces MeanCache, a user-centric semantic cache for LLM-based services that identifies semantically similar queries to determine cache hit or miss. Using MeanCache, the response to a user's semantically similar query can be retrieved from a local cache rather than re-querying the LLM, thus reducing costs, service provider load, and environmental impact. MeanCache leverages Federated Learning (FL) to collaboratively train a query similarity model without violating user privacy. By placing a local cache in each user's device and using FL, MeanCache reduces the latency and costs and enhances model performance, resulting in lower false hit rates. MeanCache also encodes context chains for every cached query, offering a simple yet highly effective mechanism to discern contextual query responses from standalone. Our experiments benchmarked against the state-of-the-art caching method, reveal that MeanCache attains an approximately 17% higher F-score and a 20% increase in precision during semantic cache hit-and-miss decisions while performing even better on contextual queries. It also reduces the storage requirement by 83% and accelerates semantic cache hit-and-miss decisions by 11%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01161v2">GazeNoter: Co-Piloted AR Note-Taking via Gaze Selection of LLM Suggestions to Match Users' Intentions</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 22 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Note-taking is critical during speeches and discussions, serving not only for later summarization and organization but also for real-time question and opinion reminding in question-and-answer sessions or timely contributions in discussions. Manually typing on smartphones for note-taking could be distracting and increase cognitive load for users. While large language models (LLMs) are used to automatically generate summaries and highlights, the content generated by artificial intelligence (AI) may not match users' intentions without user input or interaction. Therefore, we propose an AI-copiloted augmented reality (AR) system, GazeNoter, to allow users to swiftly select diverse LLM-generated suggestions via gaze on an AR headset for real-time note-taking. GazeNoter leverages an AR headset as a medium for users to swiftly adjust the LLM output to match their intentions, forming a user-in-the-loop AI system for both within-context and beyond-context notes. We conducted two user studies to verify the usability of GazeNoter in attending speeches in a static sitting condition and walking meetings and discussions in a mobile walking condition, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05449v1">LLM-based Iterative Approach to Metamodeling in Automotive</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      In this paper, we introduce an automated approach to domain-specific metamodel construction relying on Large Language Model (LLM). The main focus is adoption in automotive domain. As outcome, a prototype was implemented as web service using Python programming language, while OpenAI's GPT-4o was used as the underlying LLM. Based on the initial experiments, this approach successfully constructs Ecore metamodel based on set of automotive requirements and visualizes it making use of PlantUML notation, so human experts can provide feedback in order to refine the result. Finally, locally deployable solution is also considered, including the limitations and additional steps required.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05445v1">Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05439v1">An Empirical Study of Conformal Prediction in LLM with ASP Scaffolds for Robust Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      In this paper, we examine the use of Conformal Language Modelling (CLM) alongside Answer Set Programming (ASP) to enhance the performance of standard open-weight LLMs on complex multi-step reasoning tasks. Using the StepGame dataset, which requires spatial reasoning, we apply CLM to generate sets of ASP programs from an LLM, providing statistical guarantees on the correctness of the outputs. Experimental results show that CLM significantly outperforms baseline models that use standard sampling methods, achieving substantial accuracy improvements across different levels of reasoning complexity. Additionally, the LLM-as-Judge metric enhances CLM's performance, especially in assessing structurally and logically correct ASP outputs. However, calibrating CLM with diverse calibration sets did not improve generalizability for tasks requiring much longer reasoning steps, indicating limitations in handling more complex tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20576v2">ECCOS: Efficient Capability and Cost Coordinated Scheduling for Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as service endpoints in systems, the surge in query volume creates significant scheduling challenges. Existing scheduling frameworks mainly target at latency optimization while neglecting the capability of LLMs to serve different level of queries, which could lead to computational resource waste. This paper addresses this challenge by proposing a capability-cost coordinated scheduling framework, ECCOS, for multi-LLM serving, which explicitly constrains response quality and workload to optimize LLM inference cost. Specifically, it introduces the two-stage scheduling by designing a multi-objective predictor and a constrained optimizer. The predictor estimates both model capabilities and computational costs through training-based and retrieval-based approaches, while the optimizer determines cost-optimal assignments under quality and workload constraints. It also introduces QAServe, a dataset collected for sample-wise response quality and costs by zero-shot prompting different LLMs on knowledge QA and mathematical reasoning. Extensive experiments demonstrate that ECCOS improves success rates by 6.30% while reducing costs by 10.15% compared to existing methods, consuming less than 0.5% of LLM response time. The code is available at: https://github.com/agiresearch/ECCOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05394v1">Static Program Analysis Guided LLM Based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      We describe a novel approach to automating unit test generation for Java methods using large language models (LLMs). Existing LLM-based approaches rely on sample usage(s) of the method to test (focal method) and/or provide the entire class of the focal method as input prompt and context. The former approach is often not viable due to the lack of sample usages, especially for newly written focal methods. The latter approach does not scale well enough; the bigger the complexity of the focal method and larger associated class, the harder it is to produce adequate test code (due to factors such as exceeding the prompt and context lengths of the underlying LLM). We show that augmenting prompts with \emph{concise} and \emph{precise} context information obtained by program analysis %of the focal method increases the effectiveness of generating unit test code through LLMs. We validate our approach on a large commercial Java project and a popular open-source Java project.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05371v1">Shifting Perspectives: Steering Vector Ensembles for Robust Bias Mitigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      We present a novel approach to bias mitigation in large language models (LLMs) by applying steering vectors to modify model activations in forward passes. We employ Bayesian optimization to systematically identify effective contrastive pair datasets across nine bias axes. When optimized on the BBQ dataset, our individually tuned steering vectors achieve average improvements of 12.2%, 4.7%, and 3.2% over the baseline for Mistral, Llama, and Qwen, respectively. Building on these promising results, we introduce Steering Vector Ensembles (SVE), a method that averages multiple individually optimized steering vectors, each targeting a specific bias axis such as age, race, or gender. By leveraging their collective strength, SVE outperforms individual steering vectors in both bias reduction and maintaining model performance. The work presents the first systematic investigation of steering vectors for bias mitigation, and we demonstrate that SVE is a powerful and computationally efficient strategy for reducing bias in LLMs, with broader implications for enhancing AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05346v1">AutoIOT: LLM-Driven Automated Natural Language Programming for AIoT Applications</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has profoundly transformed our lives, revolutionizing interactions with AI and lowering the barrier to AI usage. While LLMs are primarily designed for natural language interaction, the extensive embedded knowledge empowers them to comprehend digital sensor data. This capability enables LLMs to engage with the physical world through IoT sensors and actuators, performing a myriad of AIoT tasks. Consequently, this evolution triggers a paradigm shift in conventional AIoT application development, democratizing its accessibility to all by facilitating the design and development of AIoT applications via natural language. However, some limitations need to be addressed to unlock the full potential of LLMs in AIoT application development. First, existing solutions often require transferring raw sensor data to LLM servers, which raises privacy concerns, incurs high query fees, and is limited by token size. Moreover, the reasoning processes of LLMs are opaque to users, making it difficult to verify the robustness and correctness of inference results. This paper introduces AutoIOT, an LLM-based automated program generator for AIoT applications. AutoIOT enables users to specify their requirements using natural language (input) and automatically synthesizes interpretable programs with documentation (output). AutoIOT automates the iterative optimization to enhance the quality of generated code with minimum user involvement. AutoIOT not only makes the execution of AIoT tasks more explainable but also mitigates privacy concerns and reduces token costs with local execution of synthesized programs. Extensive experiments and user studies demonstrate AutoIOT's remarkable capability in program synthesis for various AIoT tasks. The synthesized programs can match and even outperform some representative baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10444v2">LLM-as-BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 7 pages. Accepted to ICRA 2025
    </div>
    <details class="paper-abstract">
      Robotic assembly tasks remain an open challenge due to their long horizon nature and complex part relations. Behavior trees (BTs) are increasingly used in robot task planning for their modularity and flexibility, but creating them manually can be effort-intensive. Large language models (LLMs) have recently been applied to robotic task planning for generating action sequences, yet their ability to generate BTs has not been fully investigated. To this end, we propose LLM-as-BT-Planner, a novel framework that leverages LLMs for BT generation in robotic assembly task planning. Four in-context learning methods are introduced to utilize the natural language processing and inference capabilities of LLMs for producing task plans in BT format, reducing manual effort while ensuring robustness and comprehensibility. Additionally, we evaluate the performance of fine-tuned smaller LLMs on the same tasks. Experiments in both simulated and real-world settings demonstrate that our framework enhances LLMs' ability to generate BTs, improving success rate through in-context learning and supervised fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05281v1">Similarity-Based Domain Adaptation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Unsupervised domain adaptation leverages abundant labeled data from various source domains to generalize onto unlabeled target data. Prior research has primarily focused on learning domain-invariant features across the source and target domains. However, these methods often require training a model using source domain data, which is time-consuming and can limit model usage for applications with different source data. This paper introduces a simple framework that utilizes the impressive generalization capabilities of Large Language Models (LLMs) for target data annotation without the need of source model training, followed by a novel similarity-based knowledge distillation loss. Our extensive experiments on cross-domain text classification reveal that our framework achieves impressive performance, specifically, 2.44\% accuracy improvement when compared to the SOTA method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23746v2">DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)
    </div>
    <details class="paper-abstract">
      Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05248v1">Optimizing LLM Inference Throughput via Memory-aware and SLA-constrained Dynamic Batching</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      The increasing adoption of large language models (LLMs) necessitates inference serving systems that can deliver both high throughput and low latency. Deploying LLMs with hundreds of billions of parameters on memory-constrained GPUs exposes significant limitations in static batching methods. Current inference serving systems often treat batch sizes as fixed hyper-parameters, hindering real-time adaptation to varying system conditions. In this paper, we propose a dynamic batching method that continuously monitors memory utilization and adheres to service-level agreements (SLAs) to enable real-time batch size configuration adjustment. The method comprises two core components: a memory-aware batch scheduler that dynamically allocates GPU resources and a latency feedback mechanism that optimizes decoding processes under SLA constraints. The numerical experiments demonstrate throughput gains of 8% to 28% and capacity improvements of 22% compared to traditional static batching methods, while maintaining full compatibility with existing inference infrastructure. These results highlight the effectiveness of dynamic batching in balancing computational efficiency and quality-of-service requirements for contemporary LLM deployment scenarios. The source code of this work is publicly available at https://github.com/KevinLee1110/dynamic-batching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05200v1">ORANSight-2.0: Foundational LLMs for O-RAN</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Despite the transformative impact of Large Language Models (LLMs) across critical domains such as healthcare, customer service, and business marketing, their integration into Open Radio Access Networks (O-RAN) remains limited. This gap is primarily due to the absence of domain-specific foundational models, with existing solutions often relying on general-purpose LLMs that fail to address the unique challenges and technical intricacies of O-RAN. To bridge this gap, we introduce ORANSight-2.0 (O-RAN Insights), a pioneering initiative aimed at developing specialized foundational LLMs tailored for O-RAN. Built on 18 LLMs spanning five open-source LLM frameworks, ORANSight-2.0 fine-tunes models ranging from 1 to 70B parameters, significantly reducing reliance on proprietary, closed-source models while enhancing performance for O-RAN. At the core of ORANSight-2.0 is RANSTRUCT, a novel Retrieval-Augmented Generation (RAG) based instruction-tuning framework that employs two LLM agents to create high-quality instruction-tuning datasets. The generated dataset is then used to fine-tune the 18 pre-trained open-source LLMs via QLoRA. To evaluate ORANSight-2.0, we introduce srsRANBench, a novel benchmark designed for code generation and codebase understanding in the context of srsRAN, a widely used 5G O-RAN stack. We also leverage ORANBench13K, an existing benchmark for assessing O-RAN-specific knowledge. Our comprehensive evaluations demonstrate that ORANSight-2.0 models outperform general-purpose and closed-source models, such as ChatGPT-4o and Gemini, by 5.421% on ORANBench and 18.465% on srsRANBench, achieving superior performance while maintaining lower computational and energy costs. We also experiment with RAG-augmented variants of ORANSight-2.0 LLMs and thoroughly evaluate their energy characteristics, demonstrating costs for training, standard inference, and RAG-augmented inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05193v1">Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable performance on knowledge graph question answering (KGQA) tasks by planning and interacting with knowledge graphs. However, existing methods often confuse tool utilization with knowledge reasoning, harming readability of model outputs and giving rise to hallucinatory tool invocations, which hinder the advancement of KGQA. To address this issue, we propose Memory-augmented Query Reconstruction for LLM-based Knowledge Graph Reasoning (MemQ) to decouple LLM from tool invocation tasks using LLM-built query memory. By establishing a memory module with explicit descriptions of query statements, the proposed MemQ facilitates the KGQA process with natural language reasoning and memory-augmented query reconstruction. Meanwhile, we design an effective and readable reasoning to enhance the LLM's reasoning capability in KGQA. Experimental results that MemQ achieves state-of-the-art performance on widely used benchmarks WebQSP and CWQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05188v1">Rewarding Curse: Analyze and Mitigate Reward Modeling Issues for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 18 pages, 21 figures
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) prompting demonstrates varying performance under different reasoning tasks. Previous work attempts to evaluate it but falls short in providing an in-depth analysis of patterns that influence the CoT. In this paper, we study the CoT performance from the perspective of effectiveness and faithfulness. For the former, we identify key factors that influence CoT effectiveness on performance improvement, including problem difficulty, information gain, and information flow. For the latter, we interpret the unfaithful CoT issue by conducting a joint analysis of the information interaction among the question, CoT, and answer. The result demonstrates that, when the LLM predicts answers, it can recall correct information missing in the CoT from the question, leading to the problem. Finally, we propose a novel algorithm to mitigate this issue, in which we recall extra information from the question to enhance the CoT generation and evaluate CoTs based on their information gain. Extensive experiments demonstrate that our approach enhances both the faithfulness and effectiveness of CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12643v2">LLM-based Discriminative Reasoning for Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based on generative pre-trained Transformer have achieved remarkable performance on knowledge graph question-answering (KGQA) tasks. However, LLMs often produce ungrounded subgraph planning or reasoning results in KGQA due to the hallucinatory behavior brought by the generative paradigm. To tackle this issue, we propose READS to reformulate the KGQA process into discriminative subtasks, which simplifies the search space for each subtasks. Based on the subtasks, we design a new corresponding discriminative inference strategy to conduct the reasoning for KGQA, thereby alleviating hallucination and ungrounded reasoning issues in LLMs. Experimental results show that the proposed approach outperforms multiple strong comparison methods, along with achieving state-of-the-art performance on widely used benchmarks WebQSP and CWQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11142v3">NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05164v1">A Comprehensive LLM-powered Framework for Driving Intelligence Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 8 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Evaluation methods for autonomous driving are crucial for algorithm optimization. However, due to the complexity of driving intelligence, there is currently no comprehensive evaluation method for the level of autonomous driving intelligence. In this paper, we propose an evaluation framework for driving behavior intelligence in complex traffic environments, aiming to fill this gap. We constructed a natural language evaluation dataset of human professional drivers and passengers through naturalistic driving experiments and post-driving behavior evaluation interviews. Based on this dataset, we developed an LLM-powered driving evaluation framework. The effectiveness of this framework was validated through simulated experiments in the CARLA urban traffic simulator and further corroborated by human assessment. Our research provides valuable insights for evaluating and designing more intelligent, human-like autonomous driving agents. The implementation details of the framework and detailed information about the dataset can be found at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05142v1">RocketEval: Efficient Automated LLM Evaluation via Grading Checklist</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Accepted by ICLR 2025: https://openreview.net/forum?id=zJjzNj6QUe
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in diverse and challenging scenarios is essential to align them with human preferences. To mitigate the prohibitive costs associated with human evaluations, utilizing a powerful LLM as a judge has emerged as a favored approach. Nevertheless, this methodology encounters several challenges, including substantial expenses, concerns regarding privacy and security, and reproducibility. In this paper, we propose a straightforward, replicable, and accurate automated evaluation method by leveraging a lightweight LLM as the judge, named RocketEval. Initially, we identify that the performance disparity between lightweight and powerful LLMs in evaluation tasks primarily stems from their ability to conduct comprehensive analyses, which is not easily enhanced through techniques such as chain-of-thought reasoning. By reframing the evaluation task as a multi-faceted Q&A using an instance-specific checklist, we demonstrate that the limited judgment accuracy of lightweight LLMs is largely attributes to high uncertainty and positional bias. To address these challenges, we introduce an automated evaluation process grounded in checklist grading, which is designed to accommodate a variety of scenarios and questions. This process encompasses the creation of checklists, the grading of these checklists by lightweight LLMs, and the reweighting of checklist items to align with the supervised annotations. Our experiments carried out on the automated evaluation benchmarks, MT-Bench and WildBench datasets, reveal that RocketEval, when using Gemma-2-2B as the judge, achieves a high correlation (0.965) with human preferences, which is comparable to GPT-4o. Moreover, RocketEval provides a cost reduction exceeding 50-fold for large-scale evaluation and comparison scenarios. Our code is available at https://github.com/Joinn99/RocketEval-ICLR .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05139v1">Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 34 pages
    </div>
    <details class="paper-abstract">
      In this technical report, we tackle the challenges of training large-scale Mixture of Experts (MoE) models, focusing on overcoming cost inefficiency and resource limitations prevalent in such systems. To address these issues, we present two differently sized MoE large language models (LLMs), namely Ling-Lite and Ling-Plus (referred to as "Bailing" in Chinese, spelled B\v{a}il\'ing in Pinyin). Ling-Lite contains 16.8 billion parameters with 2.75 billion activated parameters, while Ling-Plus boasts 290 billion parameters with 28.8 billion activated parameters. Both models exhibit comparable performance to leading industry benchmarks. This report offers actionable insights to improve the efficiency and accessibility of AI development in resource-constrained settings, promoting more scalable and sustainable technologies. Specifically, to reduce training costs for large-scale MoE models, we propose innovative methods for (1) optimization of model architecture and training processes, (2) refinement of training anomaly handling, and (3) enhancement of model evaluation efficiency. Additionally, leveraging high-quality data generated from knowledge graphs, our models demonstrate superior capabilities in tool use compared to other models. Ultimately, our experimental findings demonstrate that a 300B MoE LLM can be effectively trained on lower-performance devices while achieving comparable performance to models of a similar scale, including dense and MoE models. Compared to high-performance devices, utilizing a lower-specification hardware system during the pre-training phase demonstrates significant cost savings, reducing computing costs by approximately 20%. The models can be accessed at https://huggingface.co/inclusionAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.11521v2">Preempting Text Sanitization Utility in Resource-Constrained Privacy-Preserving LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Interactions with online Large Language Models raise privacy issues where providers can gather sensitive information about users and their companies from the prompts. While Differential Privacy can be applied on textual prompts through the Multidimensional Laplace Mechanism, we show that it is difficult to anticipate the utility of such sanitized prompt. Poor utility has clear monetary consequences for LLM services charging on a pay-per-use model as well as great amount of computing resources wasted. To this end, we propose an architecture to predict the utility of a given sanitized prompt before it is sent to the LLM. We experimentally show that our architecture helps prevent such resource waste for up to 12% of the prompts. We also reproduce experiments from one of the most cited paper on distance-based DP for text sanitization and show that a potential performance-driven implementation choice completely changes the output while not being explicitly defined in the paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10340v5">On the Vulnerability of LLM/VLM-Controlled Robotics</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      In this work, we highlight vulnerabilities in robotic systems integrating large language models (LLMs) and vision-language models (VLMs) due to input modality sensitivities. While LLM/VLM-controlled robots show impressive performance across various tasks, their reliability under slight input variations remains underexplored yet critical. These models are highly sensitive to instruction or perceptual input changes, which can trigger misalignment issues, leading to execution failures with severe real-world consequences. To study this issue, we analyze the misalignment-induced vulnerabilities within LLM/VLM-controlled robotic systems and present a mathematical formulation for failure modes arising from variations in input modalities. We propose empirical perturbation strategies to expose these vulnerabilities and validate their effectiveness through experiments on multiple robot manipulation tasks. Our results show that simple input perturbations reduce task execution success rates by 22.2% and 14.6% in two representative LLM/VLM-controlled robotic systems. These findings underscore the importance of input modality robustness and motivate further research to ensure the safe and reliable deployment of advanced LLM/VLM-controlled robotic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03704v2">A Practical Memory Injection Attack against LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Agents based on large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, that enables the injection of malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps leading to undesirable agent actions when executing the victim user's query. Specifically, we introduce a sequence of bridging steps to link the victim query to the malicious reasoning steps. During the injection of the malicious record, we propose an indication prompt to guide the agent to autonomously generate our designed bridging steps. We also propose a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing the victim query comes after. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting practical risks of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04723v2">Shifting Long-Context LLMs Research from Input to Output</a></div>
    <div class="paper-meta">
      📅 2025-03-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advancements in long-context Large Language Models (LLMs) have primarily concentrated on processing extended input contexts, resulting in significant strides in long-context comprehension. However, the equally critical aspect of generating long-form outputs has received comparatively less attention. This paper advocates for a paradigm shift in NLP research toward addressing the challenges of long-output generation. Tasks such as novel writing, long-term planning, and complex reasoning require models to understand extensive contexts and produce coherent, contextually rich, and logically consistent extended text. These demands highlight a critical gap in current LLM capabilities. We underscore the importance of this under-explored domain and call for focused efforts to develop foundational LLMs tailored for generating high-quality, long-form outputs, which hold immense potential for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05061v1">No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge is a framework that uses an LLM (large language model) to evaluate the quality of natural language text - typically text that is also generated by an LLM. This framework holds great promise due to its relative low-cost, ease of use, and strong correlations with human stylistic preferences. However, LLM Judges have been shown to exhibit biases that can distort their judgments. We evaluate how well LLM Judges can grade whether a given response to a conversational question is correct, an ability crucial to soundly estimating the overall response quality. To do so, we create and publicly release a human-annotated dataset with labels of correctness for 1,200 LLM responses. We source questions from a combination of existing datasets and a novel, challenging benchmark (BFF-Bench) created for this analysis. We demonstrate a strong connection between an LLM's ability to correctly answer a question and grade responses to that question. Although aggregate level statistics might imply a judge has high agreement with human annotators, it will struggle on the subset of questions it could not answer. To address this issue, we recommend a simple solution: provide the judge with a correct, human-written reference answer. We perform an in-depth analysis on how reference quality can affect the performance of an LLM Judge. We show that providing a weaker judge (e.g. Qwen 2.5 7B) with higher quality references reaches better agreement with human annotators than a stronger judge (e.g. GPT-4o) with synthetic references.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05980v1">SINdex: Semantic INconsistency Index for Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed across diverse domains, yet they are prone to generating factually incorrect outputs - commonly known as "hallucinations." Among existing mitigation strategies, uncertainty-based methods are particularly attractive due to their ease of implementation, independence from external data, and compatibility with standard LLMs. In this work, we introduce a novel and scalable uncertainty-based semantic clustering framework for automated hallucination detection. Our approach leverages sentence embeddings and hierarchical clustering alongside a newly proposed inconsistency measure, SINdex, to yield more homogeneous clusters and more accurate detection of hallucination phenomena across various LLMs. Evaluations on prominent open- and closed-book QA datasets demonstrate that our method achieves AUROC improvements of up to 9.3% over state-of-the-art techniques. Extensive ablation studies further validate the effectiveness of each component in our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04104v1">LLMs Can Generate a Better Answer by Aggregating Their Own Responses</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities across tasks, yet they often require additional prompting techniques when facing complex problems. While approaches like self-correction and response selection have emerged as popular solutions, recent studies have shown these methods perform poorly when relying on the LLM itself to provide feedback or selection criteria. We argue this limitation stems from the fact that common LLM post-training procedures lack explicit supervision for discriminative judgment tasks. In this paper, we propose Generative Self-Aggregation (GSA), a novel prompting method that improves answer quality without requiring the model's discriminative capabilities. GSA first samples multiple diverse responses from the LLM, then aggregates them to obtain an improved solution. Unlike previous approaches, our method does not require the LLM to correct errors or compare response quality; instead, it leverages the model's generative abilities to synthesize a new response based on the context of multiple samples. While GSA shares similarities with the self-consistency (SC) approach for response aggregation, SC requires specific verifiable tokens to enable majority voting. In contrast, our approach is more general and can be applied to open-ended tasks. Empirical evaluation demonstrates that GSA effectively improves response quality across various tasks, including mathematical reasoning, knowledge-based problems, and open-ended generation tasks such as code synthesis and conversational responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10069v3">A Survey on LLM Test-Time Compute via Search: Tasks, LLM Profiling, Search Algorithms, and Relevant Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      LLM test-time compute (or LLM inference) via search has emerged as a promising research area with rapid developments. However, current frameworks often adopt distinct perspectives on three key aspects (task definition, LLM profiling, and search procedures), making direct comparisons challenging. Moreover, the search algorithms employed often diverge from standard implementations, and their specific characteristics are not thoroughly specified. In this survey, we provide a comprehensive technical review that unifies task definitions and provides modular definitions of LLM profiling and search procedures. The definitions enable precise comparisons of various LLM inference frameworks while highlighting their departures from conventional search algorithms. We also discuss the applicability, performance, and efficiency of these methods. We have updated our content to include the latest papers, and the differences between versions are highlighted in the appendix. For further details and ongoing updates, please refer to our GitHub repository: https://github.com/xinzhel/LLM-Agent-Survey/blob/main/search.md
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04099v1">Disparities in LLM Reasoning Accuracy and Explanations: A Case Study on African American English</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 ARR Under Review, First two authors contribute equally
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning tasks, leading to their widespread deployment. However, recent studies have highlighted concerning biases in these models, particularly in their handling of dialectal variations like African American English (AAE). In this work, we systematically investigate dialectal disparities in LLM reasoning tasks. We develop an experimental framework comparing LLM performance given Standard American English (SAE) and AAE prompts, combining LLM-based dialect conversion with established linguistic analyses. We find that LLMs consistently produce less accurate responses and simpler reasoning chains and explanations for AAE inputs compared to equivalent SAE questions, with disparities most pronounced in social science and humanities domains. These findings highlight systematic differences in how LLMs process and reason about different language varieties, raising important questions about the development and deployment of these systems in our multilingual and multidialectal world. Our code repository is publicly available at https://github.com/Runtaozhou/dialect_bias_eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17399v2">MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      We present MultiChallenge, a pioneering benchmark evaluating large language models (LLMs) on conducting multi-turn conversations with human users, a crucial yet underexamined capability for their applications. MultiChallenge identifies four categories of challenges in multi-turn conversations that are not only common and realistic among current human-LLM interactions, but are also challenging to all current frontier LLMs. All 4 challenges require accurate instruction-following, context allocation, and in-context reasoning at the same time. We also develop LLM as judge with instance-level rubrics to facilitate an automatic evaluation method with fair agreement with experienced human raters. Despite achieving near-perfect scores on existing multi-turn evaluation benchmarks, all frontier models have less than 50% accuracy on MultiChallenge, with the top-performing Claude 3.5 Sonnet (June 2024) achieving just a 41.4% average accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04076v1">Beyond Memorization: Evaluating the True Type Inference Capabilities of LLMs for Java Code Snippets</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Type inference is a crucial task for reusing online code snippets, often found on platforms like StackOverflow, which frequently lack essential type information such as fully qualified names (FQNs) and required libraries. Recent studies have leveraged Large Language Models (LLMs) for type inference on code snippets, showing promising results. However, these results are potentially affected by data leakage, as the benchmark suite (StatType-SO) has been public on GitHub since 2017 (full suite in 2023). Thus, it is uncertain whether LLMs' strong performance reflects genuine code semantics understanding or a mere retrieval of ground truth from training data. To comprehensively assess LLMs' type inference capabilities on Java code snippets, we conducted a three-pronged evaluation. First, utilizing Thalia, a program synthesis technique, we created ThaliaType--a new, unseen dataset for type inference evaluation. On unseen snippets, LLM performance dropped significantly, with up to a 59% decrease in precision and 72% in recall. Second, we developed semantic-preserving transformations that significantly degraded LLMs' type inference performance, revealing weaknesses in understanding code semantics. Third, we used delta debugging to identify the minimal syntax elements sufficient for LLM inference. While type inference primarily involves inferring FQNs for types in the code snippet, LLMs correctly infer FQNs even when the types were absent from the snippets, suggesting a reliance on knowledge from training instead of thoroughly analyzing the snippets. Our findings indicate that LLMs' strong past performance likely stemmed from data leakage, rather than a genuine understanding of the semantics of code snippets. Our findings highlight the crucial need for carefully designed benchmarks using unseen code snippets to assess the true capabilities of LLMs for type inference tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03459v2">Unified Mind Model: Reimagining Autonomous Agents in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated remarkable capabilities across domains, tasks, and languages (e.g., ChatGPT and GPT-4), reviving the research of general autonomous agents with human-like cognitive abilities. Such human-level agents require semantic comprehension and instruction-following capabilities, which exactly fall into the strengths of LLMs. Although there have been several initial attempts to build human-level agents based on LLMs, the theoretical foundation remains a challenging open problem. In this paper, we propose a novel theoretical cognitive architecture, the Unified Mind Model (UMM), which offers guidance to facilitate the rapid creation of autonomous agents with human-level cognitive abilities. Specifically, our UMM starts with the global workspace theory and further leverage LLMs to enable the agent with various cognitive abilities, such as multi-modal perception, planning, reasoning, tool use, learning, memory, reflection and motivation. Building upon UMM, we then develop an agent-building engine, MindOS, which allows users to quickly create domain-/task-specific autonomous agents without any programming effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16466v2">On the Feasibility of Using LLMs to Execute Multistage Network Attacks</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 16 pages, 14 figures
    </div>
    <details class="paper-abstract">
      LLMs have shown preliminary promise in some security tasks and CTF challenges. However, it is unclear whether LLMs are able to realize multistage network attacks, which involve executing a wide variety of actions across multiple hosts such as conducting reconnaissance, exploiting vulnerabilities to gain initial access, leveraging internal hosts to move laterally, and using multiple compromised hosts to exfiltrate data. We evaluate LLMs across 10 multistage networks and find that popular LLMs are unable to realize these attacks. To enable LLMs to realize these attacks, we introduce Incalmo, an LLM-agnostic high-level attack abstraction layer that sits between an LLM and the environment. Rather than LLMs issuing low-level command-line instructions, which can lead to incorrect implementations, Incalmo allows LLMs to specify high-level tasks (e.g., infect a host, scan a network), which are then carried out by Incalmo. Incalmo realizes these tasks by translating them into low-level primitives (e.g., commands to exploit tools). Incalmo also provides an environment state service and an attack graph service to provide structure to LLMs in selecting actions relevant to a multistage attack. Across 9 out of 10 realistic emulated networks (from 25 to 50 hosts), LLMs using Incalmo can successfully autonomously execute multistage attacks. We also conduct an ablation analysis to show the key role the high-level abstractions play. For instance, we find that both Incalmo's high-level tasks and services are crucial. Furthermore, even smaller-parameter LLMs with Incalmo can fully succeed in 5 of 10 environments, while larger-parameter LLMs without Incalmo do not fully succeed in any.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06916v2">SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 ICLR 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Speculative decoding (SD) has emerged as a widely used paradigm to accelerate LLM inference without compromising quality. It works by first employing a compact model to draft multiple tokens efficiently and then using the target LLM to verify them in parallel. While this technique has achieved notable speedups, most existing approaches necessitate either additional parameters or extensive training to construct effective draft models, thereby restricting their applicability across different LLMs and tasks. To address this limitation, we explore a novel plug-and-play SD solution with layer-skipping, which skips intermediate layers of the target LLM as the compact draft model. Our analysis reveals that LLMs exhibit great potential for self-acceleration through layer sparsity and the task-specific nature of this sparsity. Building on these insights, we introduce SWIFT, an on-the-fly self-speculative decoding algorithm that adaptively selects intermediate layers of LLMs to skip during inference. SWIFT does not require auxiliary models or additional training, making it a plug-and-play solution for accelerating LLM inference across diverse input data streams. Our extensive experiments across a wide range of models and downstream tasks demonstrate that SWIFT can achieve over a 1.3x-1.6x speedup while preserving the original distribution of the generated text. We release our code in https://github.com/hemingkx/SWIFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03455v2">Watson: A Cognitive Observability Framework for the Reasoning of LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      As foundation models (FMs) play an increasingly prominent role in complex software systems, such as agentic software, they introduce significant observability and debuggability challenges. Although recent Large Reasoning Models (LRMs) generate their thought processes as part of the output, in many scenarios fast-thinking Large Language Models (LLMs) are still preferred due to latency constraints. LLM-powered agents operate autonomously with opaque implicit reasoning, making it difficult to debug their unexpected behaviors or errors. In this paper, we introduce Watson, a novel framework that provides reasoning observability into the implicit reasoning processes of agents driven by fast-thinking LLMs, allowing the identification and localization of errors and guidance for corrections. We demonstrate the accuracy of the recovered implicit reasoning trace by Watson and its usefulness through debugging and improving the performance of LLM-powered agents in two scenarios: Massive Multitask Language Understanding (MMLU) benchmark and SWE-bench-lite. Using Watson, we were able to observe and identify the implicit reasoning errors, and automatically provide targeted corrections at runtime that improve the Pass@1 of agents on MMLU and SWE-bench-lite by 7.58 (13.45% relative improvement) and 7.76 (12.31% relative improvement) percentage points, respectively, without updates to models or the cognitive architecture of the agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04006v1">DSV-LFS: Unifying LLM-Driven Semantic Cues with Visual Features for Robust Few-Shot Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Few-shot semantic segmentation (FSS) aims to enable models to segment novel/unseen object classes using only a limited number of labeled examples. However, current FSS methods frequently struggle with generalization due to incomplete and biased feature representations, especially when support images do not capture the full appearance variability of the target class. To improve the FSS pipeline, we propose a novel framework that utilizes large language models (LLMs) to adapt general class semantic information to the query image. Furthermore, the framework employs dense pixel-wise matching to identify similarities between query and support images, resulting in enhanced FSS performance. Inspired by reasoning-based segmentation frameworks, our method, named DSV-LFS, introduces an additional token into the LLM vocabulary, allowing a multimodal LLM to generate a "semantic prompt" from class descriptions. In parallel, a dense matching module identifies visual similarities between the query and support images, generating a "visual prompt". These prompts are then jointly employed to guide the prompt-based decoder for accurate segmentation of the query image. Comprehensive experiments on the benchmark datasets Pascal-$5^{i}$ and COCO-$20^{i}$ demonstrate that our framework achieves state-of-the-art performance-by a significant margin-demonstrating superior generalization to novel classes and robustness across diverse scenarios. The source code is available at \href{https://github.com/aminpdik/DSV-LFS}{https://github.com/aminpdik/DSV-LFS}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08177v2">SycEval: Evaluating LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in educational, clinical, and professional settings, but their tendency for sycophancy -- prioritizing user agreement over independent reasoning -- poses risks to reliability. This study introduces a framework to evaluate sycophantic behavior in ChatGPT-4o, Claude-Sonnet, and Gemini-1.5-Pro across AMPS (mathematics) and MedQuad (medical advice) datasets. Sycophantic behavior was observed in 58.19% of cases, with Gemini exhibiting the highest rate (62.47%) and ChatGPT the lowest (56.71%). Progressive sycophancy, leading to correct answers, occurred in 43.52% of cases, while regressive sycophancy, leading to incorrect answers, was observed in 14.66%. Preemptive rebuttals demonstrated significantly higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%, $Z=5.87$, $p<0.001$), particularly in computational tasks, where regressive sycophancy increased significantly (preemptive: 8.13%, in-context: 3.54%, $p<0.001$). Simple rebuttals maximized progressive sycophancy ($Z=6.59$, $p<0.001$), while citation-based rebuttals exhibited the highest regressive rates ($Z=6.59$, $p<0.001$). Sycophantic behavior showed high persistence (78.5%, 95% CI: [77.2%, 79.8%]) regardless of context or model. These findings emphasize the risks and opportunities of deploying LLMs in structured and dynamic domains, offering insights into prompt programming and model optimization for safer AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16457v2">Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Materials synthesis is vital for innovations such as energy storage, catalysis, electronics, and biomedical devices. Yet, the process relies heavily on empirical, trial-and-error methods guided by expert intuition. Our work aims to support the materials science community by providing a practical, data-driven resource. We have curated a comprehensive dataset of 17K expert-verified synthesis recipes from open-access literature, which forms the basis of our newly developed benchmark, AlchemyBench. AlchemyBench offers an end-to-end framework that supports research in large language models applied to synthesis prediction. It encompasses key tasks, including raw materials and equipment prediction, synthesis procedure generation, and characterization outcome forecasting. We propose an LLM-as-a-Judge framework that leverages large language models for automated evaluation, demonstrating strong statistical agreement with expert assessments. Overall, our contributions offer a supportive foundation for exploring the capabilities of LLMs in predicting and guiding materials synthesis, ultimately paving the way for more efficient experimental design and accelerated innovation in materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03993v3">TR-LLM: Integrating Trajectory Data for Scene-Aware LLM-Based Human Action Prediction</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Accurate prediction of human behavior is crucial for AI systems to effectively support real-world applications, such as autonomous robots anticipating and assisting with human tasks. Real-world scenarios frequently present challenges such as occlusions and incomplete scene observations, which can compromise predictive accuracy. Thus, traditional video-based methods often struggle due to limited temporal and spatial perspectives. Large Language Models (LLMs) offer a promising alternative. Having been trained on a large text corpus describing human behaviors, LLMs likely encode plausible sequences of human actions in a home environment. However, LLMs, trained primarily on text data, lack inherent spatial awareness and real-time environmental perception. They struggle with understanding physical constraints and spatial geometry. Therefore, to be effective in a real-world spatial scenario, we propose a multimodal prediction framework that enhances LLM-based action prediction by integrating physical constraints derived from human trajectories. Our experiments demonstrate that combining LLM predictions with trajectory data significantly improves overall prediction performance. This enhancement is particularly notable in situations where the LLM receives limited scene information, highlighting the complementary nature of linguistic knowledge and physical constraints in understanding and anticipating human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05050v1">A Unified Framework with Novel Metrics for Evaluating the Effectiveness of XAI Techniques in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 arXiv admin note: substantial text overlap with arXiv:2501.15374
    </div>
    <details class="paper-abstract">
      The increasing complexity of LLMs presents significant challenges to their transparency and interpretability, necessitating the use of eXplainable AI (XAI) techniques to enhance trustworthiness and usability. This study introduces a comprehensive evaluation framework with four novel metrics for assessing the effectiveness of five XAI techniques across five LLMs and two downstream tasks. We apply this framework to evaluate several XAI techniques LIME, SHAP, Integrated Gradients, Layer-wise Relevance Propagation (LRP), and Attention Mechanism Visualization (AMV) using the IMDB Movie Reviews and Tweet Sentiment Extraction datasets. The evaluation focuses on four key metrics: Human-reasoning Agreement (HA), Robustness, Consistency, and Contrastivity. Our results show that LIME consistently achieves high scores across multiple LLMs and evaluation metrics, while AMV demonstrates superior Robustness and near-perfect Consistency. LRP excels in Contrastivity, particularly with more complex models. Our findings provide valuable insights into the strengths and limitations of different XAI methods, offering guidance for developing and selecting appropriate XAI techniques for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05021v1">Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05012v1">LLMs' Reshaping of People, Processes, Products, and Society in Software Development: A Comprehensive Exploration with Early Adopters</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like OpenAI ChatGPT, Google Gemini, and GitHub Copilot are rapidly gaining traction in the software industry, but their full impact on software engineering remains insufficiently explored. Despite their growing adoption, there is a notable lack of formal, qualitative assessments of how LLMs are applied in real-world software development contexts. To fill this gap, we conducted semi-structured interviews with sixteen early-adopter professional developers to explore their use of LLMs throughout various stages of the software development life cycle. Our investigation examines four dimensions: people - how LLMs affect individual developers and teams; process - how LLMs alter software engineering workflows; product - LLM impact on software quality and innovation; and society - the broader socioeconomic and ethical implications of LLM adoption. Thematic analysis of our data reveals that while LLMs have not fundamentally revolutionized the development process, they have substantially enhanced routine coding tasks, including code generation, refactoring, and debugging. Developers reported the most effective outcomes when providing LLMs with clear, well-defined problem statements, indicating that LLMs excel with decomposed problems and specific requirements. Furthermore, these early-adopters identified that LLMs offer significant value for personal and professional development, aiding in learning new languages and concepts. Early-adopters, highly skilled in software engineering and how LLMs work, identified early and persisting challenges for software engineering, such as inaccuracies in generated content and the need for careful manual review before integrating LLM outputs into production environments. Our study provides a nuanced understanding of how LLMs are shaping the landscape of software development, with their benefits, limitations, and ongoing implications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05010v1">Leveraging Domain Knowledge at Inference Time for LLM Translation: Retrieval versus Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have been increasingly adopted for machine translation (MT), their performance for specialist domains such as medicine and law remains an open challenge. Prior work has shown that LLMs can be domain-adapted at test-time by retrieving targeted few-shot demonstrations or terminologies for inclusion in the prompt. Meanwhile, for general-purpose LLM MT, recent studies have found some success in generating similarly useful domain knowledge from an LLM itself, prior to translation. Our work studies domain-adapted MT with LLMs through a careful prompting setup, finding that demonstrations consistently outperform terminology, and retrieval consistently outperforms generation. We find that generating demonstrations with weaker models can close the gap with larger model's zero-shot performance. Given the effectiveness of demonstrations, we perform detailed analyses to understand their value. We find that domain-specificity is particularly important, and that the popular multi-domain benchmark is testing adaptation to a particular writing style more so than to a specific domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19320v2">Shh, don't say that! Domain Certification in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 10 pages, includes appendix Published in International Conference on Learning Representations (ICLR) 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often deployed to perform constrained tasks, with narrow domains. For example, customer support bots can be built on top of LLMs, relying on their broad language understanding and capabilities to enhance performance. However, these LLMs are adversarially susceptible, potentially generating outputs outside the intended domain. To formalize, assess, and mitigate this risk, we introduce domain certification; a guarantee that accurately characterizes the out-of-domain behavior of language models. We then propose a simple yet effective approach, which we call VALID that provides adversarial bounds as a certificate. Finally, we evaluate our method across a diverse set of datasets, demonstrating that it yields meaningful certificates, which bound the probability of out-of-domain samples tightly with minimum penalty to refusal behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07942v2">Symbiotic Cooperation for Web Agents: Harnessing Complementary Strengths of Large and Small LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Web browsing agents powered by large language models (LLMs) have shown tremendous potential in automating complex web-based tasks. Existing approaches typically rely on large LLMs (e.g., GPT-4o) to explore web environments and generate trajectory data, which is then used either for demonstration retrieval (for large LLMs) or to distill small LLMs (e.g., Llama3) in a process that remains decoupled from the exploration. In this paper, we propose AgentSymbiotic, an iterative framework that couples data synthesis with task-performance, yielding a "symbiotic improvement" for both large and small LLMs. Our study uncovers a complementary dynamic between LLM types: while large LLMs excel at generating high-quality trajectories for distillation, the distilled small LLMs-owing to their distinct reasoning capabilities-often choose actions that diverge from those of their larger counterparts. This divergence drives the exploration of novel trajectories, thereby enriching the synthesized data. However, we also observe that the performance of small LLMs becomes a bottleneck in this iterative enhancement process. To address this, we propose two innovations in LLM distillation: a speculative data synthesis strategy that mitigates off-policy bias, and a multi-task learning approach designed to boost the reasoning capabilities of the student LLM. Furthermore, we introduce a Hybrid Mode for Privacy Preservation to address user privacy concerns. Evaluated on the WEBARENA benchmark, AgentSymbiotic achieves SOTA performance with both LLM types. Our best Large LLM agent reaches 52%, surpassing the previous best of 45%, while our 8B distilled model demonstrates a competitive 49%, exceeding the prior best of 28%. Code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04874v1">Memory Is All You Need: Testing How Model Memory Affects LLM Performance in Annotation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Generative Large Language Models (LLMs) have shown promising results in text annotation using zero-shot and few-shot learning. Yet these approaches do not allow the model to retain information from previous annotations, making each response independent from the preceding ones. This raises the question of whether model memory -- the LLM having knowledge about its own previous annotations in the same task -- affects performance. In this article, using OpenAI's GPT-4o and Meta's Llama 3.1 on two political science datasets, we demonstrate that allowing the model to retain information about its own previous classifications yields significant performance improvements: between 5 and 25\% when compared to zero-shot and few-shot learning. Moreover, memory reinforcement, a novel approach we propose that combines model memory and reinforcement learning, yields additional performance gains in three out of our four tests. These findings have important implications for applied researchers looking to improve performance and efficiency in LLM annotation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04856v1">One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04724v1">LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Recent advancements in speech-to-speech dialogue systems leverage LLMs for multimodal interactions, yet they remain hindered by fine-tuning requirements, high computational overhead, and text-speech misalignment. Existing speech-enabled LLMs often degrade conversational quality by modifying the LLM, thereby compromising its linguistic capabilities. In contrast, we propose LLMVoX, a lightweight 30M-parameter, LLM-agnostic, autoregressive streaming TTS system that generates high-quality speech with low latency, while fully preserving the capabilities of the base LLM. Our approach achieves a significantly lower Word Error Rate compared to speech-enabled LLMs, while operating at comparable latency and UTMOS score. By decoupling speech synthesis from LLM processing via a multi-queue token streaming system, LLMVoX supports seamless, infinite-length dialogues. Its plug-and-play design also facilitates extension to various tasks with different backbones. Furthermore, LLMVoX generalizes to new languages with only dataset adaptation, attaining a low Character Error Rate on an Arabic speech task. Additionally, we have integrated LLMVoX with a Vision-Language Model to create an omni-model with speech, text, and vision capabilities, without requiring additional multimodal training. Our code base and project page is available at https://mbzuai-oryx.github.io/LLMVoX .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04723v1">Shifting Long-Context LLMs Research from Input to Output</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advancements in long-context Large Language Models (LLMs) have primarily concentrated on processing extended input contexts, resulting in significant strides in long-context comprehension. However, the equally critical aspect of generating long-form outputs has received comparatively less attention. This paper advocates for a paradigm shift in NLP research toward addressing the challenges of long-output generation. Tasks such as novel writing, long-term planning, and complex reasoning require models to understand extensive contexts and produce coherent, contextually rich, and logically consistent extended text. These demands highlight a critical gap in current LLM capabilities. We underscore the importance of this under-explored domain and call for focused efforts to develop foundational LLMs tailored for generating high-quality, long-form outputs, which hold immense potential for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04722v1">Enough Coin Flips Can Make LLMs Act Bayesian</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit the ability to generalize given few-shot examples in their input prompt, an emergent capability known as in-context learning (ICL). We investigate whether LLMs utilize ICL to perform structured reasoning in ways that are consistent with a Bayesian framework or rely on pattern matching. Using a controlled setting of biased coin flips, we find that: (1) LLMs often possess biased priors, causing initial divergence in zero-shot settings, (2) in-context evidence outweighs explicit bias instructions, (3) LLMs broadly follow Bayesian posterior updates, with deviations primarily due to miscalibrated priors rather than flawed updates, and (4) attention magnitude has negligible effect on Bayesian inference. With sufficient demonstrations of biased coin flips via ICL, LLMs update their priors in a Bayesian manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.11807v7">How Far Are We on the Decision-Making of LLMs? Evaluating LLMs' Gaming Ability in Multi-Agent Environments</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted to ICLR 2025; 11 pages of main text; 26 pages of appendices; Included models: GPT-3.5-{0613, 1106, 0125}, GPT-4-0125, GPT-4o-0806, Gemini-{1.0, 1.5)-Pro, LLaMA-3.1-{7, 70, 405}B, Mixtral-8x{7, 22}B, Qwen-2-72B
    </div>
    <details class="paper-abstract">
      Decision-making is a complex process requiring diverse abilities, making it an excellent framework for evaluating Large Language Models (LLMs). Researchers have examined LLMs' decision-making through the lens of Game Theory. However, existing evaluation mainly focus on two-player scenarios where an LLM competes against another. Additionally, previous benchmarks suffer from test set leakage due to their static design. We introduce GAMA($\gamma$)-Bench, a new framework for evaluating LLMs' Gaming Ability in Multi-Agent environments. It includes eight classical game theory scenarios and a dynamic scoring scheme specially designed to quantitatively assess LLMs' performance. $\gamma$-Bench allows flexible game settings and adapts the scoring system to different game parameters, enabling comprehensive evaluation of robustness, generalizability, and strategies for improvement. Our results indicate that GPT-3.5 demonstrates strong robustness but limited generalizability, which can be enhanced using methods like Chain-of-Thought. We also evaluate 13 LLMs from 6 model families, including GPT-3.5, GPT-4, Gemini, LLaMA-3.1, Mixtral, and Qwen-2. Gemini-1.5-Pro outperforms others, scoring of $69.8$ out of $100$, followed by LLaMA-3.1-70B ($65.9$) and Mixtral-8x22B ($62.4$). Our code and experimental results are publicly available at https://github.com/CUHK-ARISE/GAMABench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04693v1">UIPE: Enhancing LLM Unlearning by Removing Knowledge Related to Forgetting Targets</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) inevitably acquire harmful information during training on massive datasets. LLM unlearning aims to eliminate the influence of such harmful information while maintaining the model's overall performance. Existing unlearning methods, represented by gradient ascent-based approaches, primarily focus on forgetting target data while overlooking the crucial impact of logically related knowledge on the effectiveness of unlearning. In this paper, through both theoretical and experimental analyses, we first demonstrate that a key reason for the suboptimal unlearning performance is that models can reconstruct the target content through reasoning with logically related knowledge. To address this issue, we propose Unlearning Improvement via Parameter Extrapolation (UIPE), a method that removes knowledge highly correlated with the forgetting targets. Experimental results show that UIPE significantly enhances the performance of various mainstream LLM unlearning methods on the TOFU benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04691v1">Quantifying the Reasoning Abilities of LLMs on Real-world Clinical Cases</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      The latest reasoning-enhanced large language models (reasoning LLMs), such as DeepSeek-R1 and OpenAI-o3, have demonstrated remarkable success. However, the application of such reasoning enhancements to the highly professional medical domain has not been clearly evaluated, particularly regarding with not only assessing the final generation but also examining the quality of their reasoning processes. In this study, we present MedR-Bench, a reasoning-focused medical evaluation benchmark comprising 1,453 structured patient cases with reasoning references mined from case reports. Our benchmark spans 13 body systems and 10 specialty disorders, encompassing both common and rare diseases. In our evaluation, we introduce a versatile framework consisting of three critical clinical stages: assessment recommendation, diagnostic decision-making, and treatment planning, comprehensively capturing the LLMs' performance across the entire patient journey in healthcare. For metrics, we propose a novel agentic system, Reasoning Evaluator, designed to automate and objectively quantify free-text reasoning responses in a scalable manner from the perspectives of efficiency, factuality, and completeness by dynamically searching and performing cross-referencing checks. As a result, we assess five state-of-the-art reasoning LLMs, including DeepSeek-R1, OpenAI-o3-mini, and others. Our results reveal that current LLMs can handle relatively simple diagnostic tasks with sufficient critical assessment results, achieving accuracy generally over 85%. However, they still struggle with more complex tasks, such as assessment recommendation and treatment planning. In reasoning, their reasoning processes are generally reliable, with factuality scores exceeding 90%, though they often omit critical reasoning steps. Our study clearly reveals further development directions for current clinical LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02800v2">RAAD-LLM: Adaptive Anomaly Detection Using LLMs and RAG Integration</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.00914
    </div>
    <details class="paper-abstract">
      Anomaly detection in complex industrial environments poses unique challenges, particularly in contexts characterized by data sparsity and evolving operational conditions. Predictive maintenance (PdM) in such settings demands methodologies that are adaptive, transferable, and capable of integrating domain-specific knowledge. In this paper, we present RAAD-LLM, a novel framework for adaptive anomaly detection, leveraging large language models (LLMs) integrated with Retrieval-Augmented Generation (RAG). This approach addresses the aforementioned PdM challenges. By effectively utilizing domain-specific knowledge, RAAD-LLM enhances the detection of anomalies in time series data without requiring fine-tuning on specific datasets. The framework's adaptability mechanism enables it to adjust its understanding of normal operating conditions dynamically, thus increasing detection accuracy. We validate this methodology through a real-world application for a plastics manufacturing plant and the Skoltech Anomaly Benchmark (SKAB). Results show significant improvements over our previous model with an accuracy increase from 70.7% to 89.1% on the real-world dataset. By allowing for the enriching of input series data with semantics, RAAD-LLM incorporates multimodal capabilities that facilitate more collaborative decision-making between the model and plant operators. Overall, our findings support RAAD-LLM's ability to revolutionize anomaly detection methodologies in PdM, potentially leading to a paradigm shift in how anomaly detection is implemented across various industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04675v1">LLM-guided Plan and Retrieval: A Strategic Alignment for Interpretable User Satisfaction Estimation in Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted by NAACL 2025
    </div>
    <details class="paper-abstract">
      Understanding user satisfaction with conversational systems, known as User Satisfaction Estimation (USE), is essential for assessing dialogue quality and enhancing user experiences. However, existing methods for USE face challenges due to limited understanding of underlying reasons for user dissatisfaction and the high costs of annotating user intentions. To address these challenges, we propose PRAISE (Plan and Retrieval Alignment for Interpretable Satisfaction Estimation), an interpretable framework for effective user satisfaction prediction. PRAISE operates through three key modules. The Strategy Planner develops strategies, which are natural language criteria for classifying user satisfaction. The Feature Retriever then incorporates knowledge on user satisfaction from Large Language Models (LLMs) and retrieves relevance features from utterances. Finally, the Score Analyzer evaluates strategy predictions and classifies user satisfaction. Experimental results demonstrate that PRAISE achieves state-of-the-art performance on three benchmarks for the USE task. Beyond its superior performance, PRAISE offers additional benefits. It enhances interpretability by providing instance-level explanations through effective alignment of utterances with strategies. Moreover, PRAISE operates more efficiently than existing approaches by eliminating the need for LLMs during the inference phase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02067v2">AdaptBot: Combining LLM with Knowledge Graphs and Human Input for Generic-to-Specific Task Decomposition and Knowledge Refinement</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025
    </div>
    <details class="paper-abstract">
      An embodied agent assisting humans is often asked to complete new tasks, and there may not be sufficient time or labeled examples to train the agent to perform these new tasks. Large Language Models (LLMs) trained on considerable knowledge across many domains can be used to predict a sequence of abstract actions for completing such tasks, although the agent may not be able to execute this sequence due to task-, agent-, or domain-specific constraints. Our framework addresses these challenges by leveraging the generic predictions provided by LLM and the prior domain knowledge encoded in a Knowledge Graph (KG), enabling an agent to quickly adapt to new tasks. The robot also solicits and uses human input as needed to refine its existing knowledge. Based on experimental evaluation in the context of cooking and cleaning tasks in simulation domains, we demonstrate that the interplay between LLM, KG, and human input leads to substantial performance gains compared with just using the LLM. Project website{\S}: https://sssshivvvv.github.io/adaptbot/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00799v6">Get my drift? Catching LLM Task Drift with Activation Deltas</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 SaTML 2025
    </div>
    <details class="paper-abstract">
      LLMs are commonly used in retrieval-augmented applications to execute user instructions based on data from external sources. For example, modern search engines use LLMs to answer queries based on relevant search results; email plugins summarize emails by processing their content through an LLM. However, the potentially untrusted provenance of these data sources can lead to prompt injection attacks, where the LLM is manipulated by natural language instructions embedded in the external data, causing it to deviate from the user's original instruction(s). We define this deviation as task drift. Task drift is a significant concern as it allows attackers to exfiltrate data or influence the LLM's output for other users. We study LLM activations as a solution to detect task drift, showing that activation deltas - the difference in activations before and after processing external data - are strongly correlated with this phenomenon. Through two probing methods, we demonstrate that a simple linear classifier can detect drift with near-perfect ROC AUC on an out-of-distribution test set. We evaluate these methods by making minimal assumptions about how users' tasks, system prompts, and attacks can be phrased. We observe that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Interestingly, the fact that this solution does not require any modifications to the LLM (e.g., fine-tuning), as well as its compatibility with existing meta-prompting solutions, makes it cost-efficient and easy to deploy. To encourage further research on activation-based task inspection, decoding, and interpretability, we release our large-scale TaskTracker toolkit, featuring a dataset of over 500K instances, representations from six SoTA language models, and a suite of inspection tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04636v1">Mark Your LLM: Detecting the Misuse of Open-Source Large Language Models via Watermarking</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 Accepted by the 1st Workshop on GenAI Watermarking, collocated with ICLR 2025
    </div>
    <details class="paper-abstract">
      As open-source large language models (LLMs) like Llama3 become more capable, it is crucial to develop watermarking techniques to detect their potential misuse. Existing watermarking methods either add watermarks during LLM inference, which is unsuitable for open-source LLMs, or primarily target classification LLMs rather than recent generative LLMs. Adapting these watermarks to open-source LLMs for misuse detection remains an open challenge. This work defines two misuse scenarios for open-source LLMs: intellectual property (IP) violation and LLM Usage Violation. Then, we explore the application of inference-time watermark distillation and backdoor watermarking in these contexts. We propose comprehensive evaluation methods to assess the impact of various real-world further fine-tuning scenarios on watermarks and the effect of these watermarks on LLM performance. Our experiments reveal that backdoor watermarking could effectively detect IP Violation, while inference-time watermark distillation is applicable in both scenarios but less robust to further fine-tuning and has a more significant impact on LLM performance compared to backdoor watermarking. Exploring more advanced watermarking methods for open-source LLMs to detect their misuse should be an important future direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04619v1">SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 18 pages, 17 figures
    </div>
    <details class="paper-abstract">
      User reviews on e-commerce platforms exhibit dynamic sentiment patterns driven by temporal and contextual factors. Traditional sentiment analysis methods focus on static reviews, failing to capture the evolving temporal relationship between user sentiment rating and textual content. Sentiment analysis on streaming reviews addresses this limitation by modeling and predicting the temporal evolution of user sentiments. However, it suffers from data sparsity, manifesting in temporal, spatial, and combined forms. In this paper, we introduce SynGraph, a novel framework designed to address data sparsity in sentiment analysis on streaming reviews. SynGraph alleviates data sparsity by categorizing users into mid-tail, long-tail, and extreme scenarios and incorporating LLM-augmented enhancements within a dynamic graph-based structure. Experiments on real-world datasets demonstrate its effectiveness in addressing sparsity and improving sentiment modeling in streaming reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04615v1">HalluCounter: Reference-free LLM Hallucination Detection in the Wild!</a></div>
    <div class="paper-meta">
      📅 2025-03-06
      | 💬 30 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Response consistency-based, reference-free hallucination detection (RFHD) methods do not depend on internal model states, such as generation probabilities or gradients, which Grey-box models typically rely on but are inaccessible in closed-source LLMs. However, their inability to capture query-response alignment patterns often results in lower detection accuracy. Additionally, the lack of large-scale benchmark datasets spanning diverse domains remains a challenge, as most existing datasets are limited in size and scope. To this end, we propose HalluCounter, a novel reference-free hallucination detection method that utilizes both response-response and query-response consistency and alignment patterns. This enables the training of a classifier that detects hallucinations and provides a confidence score and an optimal response for user queries. Furthermore, we introduce HalluCounterEval, a benchmark dataset comprising both synthetically generated and human-curated samples across multiple domains. Our method outperforms state-of-the-art approaches by a significant margin, achieving over 90\% average confidence in hallucination detection across datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04596v1">The Next Frontier of LLM Applications: Open Ecosystems and Hardware Synergy</a></div>
    <div class="paper-meta">
      📅 2025-03-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) applications, including LLM app stores and autonomous agents, are shaping the future of AI ecosystems. However, platform silos, fragmented hardware integration, and the absence of standardized interfaces limit scalability, interoperability, and resource efficiency. While LLM app stores democratize AI, their closed ecosystems restrict modular AI reuse and cross-platform portability. Meanwhile, agent-based frameworks offer flexibility but often lack seamless integration across diverse environments. This paper envisions the future of LLM applications and proposes a three-layer decoupled architecture grounded in software engineering principles such as layered system design, service-oriented architectures, and hardware-software co-design. This architecture separates application logic, communication protocols, and hardware execution, enhancing modularity, efficiency, and cross-platform compatibility. Beyond architecture, we highlight key security and privacy challenges for safe, scalable AI deployment and outline research directions in software and security engineering. This vision aims to foster open, secure, and interoperable LLM ecosystems, guiding future advancements in AI applications.
    </details>
</div>
