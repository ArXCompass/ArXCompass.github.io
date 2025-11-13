# llm - 2024_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01461v4">Will the Real Linda Please Stand up...to Large Language Models? Examining the Representativeness Heuristic in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-23
      | 💬 Published as a conference paper at COLM 2024
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have demonstrated remarkable proficiency in modeling text and generating human-like text, they may exhibit biases acquired from training data in doing so. Specifically, LLMs may be susceptible to a common cognitive trap in human decision-making called the representativeness heuristic. This is a concept in psychology that refers to judging the likelihood of an event based on how closely it resembles a well-known prototype or typical example, versus considering broader facts or statistical evidence. This research investigates the impact of the representativeness heuristic on LLM reasoning. We created ReHeAT (Representativeness Heuristic AI Testing), a dataset containing a series of problems spanning six common types of representativeness heuristics. Experiments reveal that four LLMs applied to ReHeAT all exhibited representativeness heuristic biases. We further identify that the model's reasoning steps are often incorrectly based on a stereotype rather than on the problem's description. Interestingly, the performance improves when adding a hint in the prompt to remind the model to use its knowledge. This suggests the uniqueness of the representativeness heuristic compared to traditional biases. It can occur even when LLMs possess the correct knowledge while falling into a cognitive trap. This highlights the importance of future research focusing on the representativeness heuristic in model reasoning and decision-making and on developing solutions to address it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15788v1">Extracting Structured Insights from Financial News: An Augmented LLM Driven Approach</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 7 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Financial news plays a crucial role in decision-making processes across the financial sector, yet the efficient processing of this information into a structured format remains challenging. This paper presents a novel approach to financial news processing that leverages Large Language Models (LLMs) to overcome limitations that previously prevented the extraction of structured data from unstructured financial news. We introduce a system that extracts relevant company tickers from raw news article content, performs sentiment analysis at the company level, and generates summaries, all without relying on pre-structured data feeds. Our methodology combines the generative capabilities of LLMs, and recent prompting techniques, with a robust validation framework that uses a tailored string similarity approach. Evaluation on a dataset of 5530 financial news articles demonstrates the effectiveness of our approach, with 90% of articles not missing any tickers compared with current data providers, and 22% of articles having additional relevant tickers. In addition to this paper, the methodology has been implemented at scale with the resulting processed data made available through a live API endpoint, which is updated in real-time with the latest news. To the best of our knowledge, we are the first data provider to offer granular, per-company sentiment analysis from news articles, enhancing the depth of information available to market participants. We also release the evaluation dataset of 5530 processed articles as a static file, which we hope will facilitate further research leveraging financial news.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15695v1">Supporting the Digital Autonomy of Elders Through LLM Assistance</a></div>
    <div class="paper-meta">
      📅 2024-07-22
    </div>
    <details class="paper-abstract">
      The internet offers tremendous access to services, social connections, and needed products. However, to those without sufficient experience, engaging with businesses and friends across the internet can be daunting due to the ever present danger of scammers and thieves, to say nothing of the myriad of potential computer viruses. Like a forest rich with both edible and poisonous plants, those familiar with the norms inhabit it safely with ease while newcomers need a guide. However, reliance on a human digital guide can be taxing and often impractical. We propose and pilot a simple but unexplored idea: could an LLM provide the necessary support to help the elderly who are separated by the digital divide safely achieve digital autonomy?
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15309v1">vTensor: Flexible Virtual Tensor Management for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 16 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used across various domains, processing millions of daily requests. This surge in demand poses significant challenges in optimizing throughput and latency while keeping costs manageable. The Key-Value (KV) cache, a standard method for retaining previous computations, makes LLM inference highly bounded by memory. While batching strategies can enhance performance, they frequently lead to significant memory fragmentation. Even though cutting-edge systems like vLLM mitigate KV cache fragmentation using paged Attention mechanisms, they still suffer from inefficient memory and computational operations due to the tightly coupled page management and computation kernels. This study introduces the vTensor, an innovative tensor structure for LLM inference based on GPU virtual memory management (VMM). vTensor addresses existing limitations by decoupling computation from memory defragmentation and offering dynamic extensibility. Our framework employs a CPU-GPU heterogeneous approach, ensuring efficient, fragmentation-free memory management while accommodating various computation kernels across different LLM architectures. Experimental results indicate that vTensor achieves an average speedup of 1.86x across different models, with up to 2.42x in multi-turn chat scenarios. Additionally, vTensor provides average speedups of 2.12x and 3.15x in kernel evaluation, reaching up to 3.92x and 3.27x compared to SGLang Triton prefix-prefilling kernels and vLLM paged Attention kernel, respectively. Furthermore, it frees approximately 71.25% (57GB) of memory on the NVIDIA A100 GPU compared to vLLM, enabling more memory-intensive workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.06613v2">GameBench: Evaluating Strategic Reasoning Abilities of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-07-22
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated remarkable few-shot performance on many natural language understanding tasks. Despite several demonstrations of using large language models in complex, strategic scenarios, there lacks a comprehensive framework for evaluating agents' performance across various types of reasoning found in games. To address this gap, we introduce GameBench, a cross-domain benchmark for evaluating strategic reasoning abilities of LLM agents. We focus on 9 different game environments, where each covers at least one axis of key reasoning skill identified in strategy games, and select games for which strategy explanations are unlikely to form a significant portion of models' pretraining corpuses. Our evaluations use GPT-3 and GPT-4 in their base form along with two scaffolding frameworks designed to enhance strategic reasoning ability: Chain-of-Thought (CoT) prompting and Reasoning Via Planning (RAP). Our results show that none of the tested models match human performance, and at worst GPT-4 performs worse than random action. CoT and RAP both improve scores but not comparable to human levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18092v2">LLM experiments with simulation: Large Language Model Multi-Agent System for Simulation Model Parametrization in Digital Twins</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 Submitted to IEEE-ETFA2024, under peer-review
    </div>
    <details class="paper-abstract">
      This paper presents a novel design of a multi-agent system framework that applies large language models (LLMs) to automate the parametrization of simulation models in digital twins. This framework features specialized LLM agents tasked with observing, reasoning, decision-making, and summarizing, enabling them to dynamically interact with digital twin simulations to explore parametrization possibilities and determine feasible parameter settings to achieve an objective. The proposed approach enhances the usability of simulation model by infusing it with knowledge heuristics from LLM and enables autonomous search for feasible parametrization to solve a user task. Furthermore, the system has the potential to increase user-friendliness and reduce the cognitive load on human users by assisting in complex decision-making processes. The effectiveness and functionality of the system are demonstrated through a case study, and the visualized demos and codes are available at a GitHub Repository: https://github.com/YuchenXia/LLMDrivenSimulation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.19379v6">Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 20 pages; 13 visualizations (nine figures, four tables)
    </div>
    <details class="paper-abstract">
      Human forecasting accuracy in practice relies on the 'wisdom of the crowd' effect, in which predictions about future events are significantly improved by aggregating across a crowd of individual forecasters. Past work on the forecasting ability of large language models (LLMs) suggests that frontier LLMs, as individual forecasters, underperform compared to the gold standard of a human crowd forecasting tournament aggregate. In Study 1, we expand this research by using an LLM ensemble approach consisting of a crowd of twelve LLMs. We compare the aggregated LLM predictions on 31 binary questions to that of a crowd of 925 human forecasters from a three-month forecasting tournament. Our preregistered main analysis shows that the LLM crowd outperforms a simple no-information benchmark and is not statistically different from the human crowd. In exploratory analyses, we find that these two approaches are equivalent with respect to medium-effect-size equivalence bounds. We also observe an acquiescence effect, with mean model predictions being significantly above 50%, despite an almost even split of positive and negative resolutions. Moreover, in Study 2, we test whether LLM predictions (of GPT-4 and Claude 2) can be improved by drawing on human cognitive output. We find that both models' forecasting accuracy benefits from exposure to the median human prediction as information, improving accuracy by between 17% and 28%: though this leads to less accurate predictions than simply averaging human and machine forecasts. Our results suggest that LLMs can achieve forecasting accuracy rivaling that of human crowd forecasting tournaments: via the simple, practically applicable method of forecast aggregation. This replicates the 'wisdom of the crowd' effect for LLMs, and opens up their use for a variety of applications throughout society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14283v4">Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning</a></div>
    <div class="paper-meta">
      📅 2024-07-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capability in many natural language tasks. However, the auto-regressive generation process makes LLMs prone to produce errors, hallucinations and inconsistent statements when performing multi-step reasoning. In this paper, by casting multi-step reasoning of LLMs as a heuristic search problem, we aim to alleviate the pathology by introducing Q*, a general, versatile and agile framework for guiding LLMs decoding process with deliberative planning. By learning a plug-and-play Q-value model as heuristic function for estimating expected future rewards, our Q* can effectively guide LLMs to select the most promising next reasoning step without fine-tuning LLMs for the current task, which avoids the significant computational overhead and potential risk of performance degeneration on other tasks. Extensive experiments on GSM8K, MATH and MBPP demonstrate the superiority of our method, contributing to improving the reasoning performance of existing open-source LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.07584v3">UltraEval: A Lightweight Platform for Flexible and Comprehensive Evaluation for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 Accepted by ACL 2024 System Demostration Track, update
    </div>
    <details class="paper-abstract">
      Evaluation is pivotal for refining Large Language Models (LLMs), pinpointing their capabilities, and guiding enhancements. The rapid development of LLMs calls for a lightweight and easy-to-use framework for swift evaluation deployment. However, considering various implementation details, developing a comprehensive evaluation platform is never easy. Existing platforms are often complex and poorly modularized, hindering seamless incorporation into research workflows. This paper introduces UltraEval, a user-friendly evaluation framework characterized by its lightweight nature, comprehensiveness, modularity, and efficiency. We identify and reimplement three core components of model evaluation (models, data, and metrics). The resulting composability allows for the free combination of different models, tasks, prompts, benchmarks, and metrics within a unified evaluation workflow. Additionally, UltraEval supports diverse models owing to a unified HTTP service and provides sufficient inference acceleration. UltraEval is now available for researchers publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00908v3">FineSurE: Fine-grained Summarization Evaluation using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 Accepted at ACL 2024 (main, long)
    </div>
    <details class="paper-abstract">
      Automated evaluation is crucial for streamlining text summarization benchmarking and model development, given the costly and time-consuming nature of human evaluation. Traditional methods like ROUGE do not correlate well with human judgment, while recently proposed LLM-based metrics provide only summary-level assessment using Likert-scale scores. This limits deeper model analysis, e.g., we can only assign one hallucination score at the summary level, while at the sentence level, we can count sentences containing hallucinations. To remedy those limitations, we propose FineSurE, a fine-grained evaluator specifically tailored for the summarization task using large language models (LLMs). It also employs completeness and conciseness criteria, in addition to faithfulness, enabling multi-dimensional assessment. We compare various open-source and proprietary LLMs as backbones for FineSurE. In addition, we conduct extensive benchmarking of FineSurE against SOTA methods including NLI-, QA-, and LLM-based methods, showing improved performance especially on the completeness and conciseness dimensions. The code is available at https://github.com/DISL-Lab/FineSurE-ACL24.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15360v1">Dissecting Multiplication in Transformers: Insights into LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Transformer-based large language models have achieved remarkable performance across various natural language processing tasks. However, they often struggle with seemingly easy tasks like arithmetic despite their vast capabilities. This stark disparity raise human's concerns about their safe and ethical use, hinder their widespread adoption.In this paper, we focus on a typical arithmetic task, integer multiplication, to explore and explain the imperfection of transformers in this domain. We provide comprehensive analysis of a vanilla transformer trained to perform n-digit integer multiplication. Our observations indicate that the model decomposes multiplication task into multiple parallel subtasks, sequentially optimizing each subtask for each digit to complete the final multiplication. Based on observation and analysis, we infer the reasons of transformers deficiencies in multiplication tasks lies in their difficulty in calculating successive carryovers and caching intermediate results, and confirmed this inference through experiments. Guided by these findings, we propose improvements to enhance transformers performance on multiplication tasks. These enhancements are validated through rigorous testing and mathematical modeling, not only enhance transformer's interpretability, but also improve its performance, e.g., we achieve over 99.9% accuracy on 5-digit integer multiplication with a tiny transformer, outperform LLMs GPT-4. Our method contributes to the broader fields of model understanding and interpretability, paving the way for analyzing more complex tasks and Transformer models. This work underscores the importance of explainable AI, helping to build trust in large language models and promoting their adoption in critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.18327v2">$\forall$uto$\exists$val: Autonomous Assessment of LLMs in Formal Synthesis and Interpretation Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-22
    </div>
    <details class="paper-abstract">
      This paper presents $\forall$uto$\exists$val, a new approach for scaling LLM assessment in translating formal syntax -- such as first-order logic, regular expressions, etc -- to natural language (interpretation) or vice versa (compilation), thereby facilitating their use in applications such as generating/explaining logic and control flow for programs etc. Existing approaches for LLM assessment in these areas require labor-intensive ground-truth creation, the availability of which undermines the separation of training and test sets. Furthermore, such datasets typically include relatively few hand-coded test cases over which LLM accuracy is determined, thus making them inadequate for determining the safety or correctness of their generated outputs. We introduce a new approach that utilizes context-free grammars (CFGs) to generate out-of-distribution datasets on the fly and perform closed-loop testing of LLM capabilities using formal verifiers to guarantee the correctness of LLM outputs without any human intervention. We release our dataset and benchmark as open-source code at \url{https://github.com/AAIR-lab/auto-llm-assessment}. We also conduct an assessment of several SOTA closed and open-source LLMs to showcase the feasibility and scalability of this paradigm. Our experiments reveal that SOTA LLMs are unable to solve the formal translation task adequately.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13193v2">SurrealDriver: Designing LLM-powered Generative Driver Agent Framework based on Human Drivers' Driving-thinking Data</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Leveraging advanced reasoning capabilities and extensive world knowledge of large language models (LLMs) to construct generative agents for solving complex real-world problems is a major trend. However, LLMs inherently lack embodiment as humans, resulting in suboptimal performance in many embodied decision-making tasks. In this paper, we introduce a framework for building human-like generative driving agents using post-driving self-report driving-thinking data from human drivers as both demonstration and feedback. To capture high-quality, natural language data from drivers, we conducted urban driving experiments, recording drivers' verbalized thoughts under various conditions to serve as chain-of-thought prompts and demonstration examples for the LLM-Agent. The framework's effectiveness was evaluated through simulations and human assessments. Results indicate that incorporating expert demonstration data significantly reduced collision rates by 81.04\% and increased human likeness by 50\% compared to a baseline LLM-based agent. Our study provides insights into using natural language-based human demonstration data for embodied tasks. The driving-thinking dataset is available at \url{https://github.com/AIR-DISCOVER/Driving-Thinking-Dataset}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20256v1">Making LLMs Work for Enterprise Data Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 Poster at North East Database Day 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) know little about enterprise database tables in the private data ecosystem, which substantially differ from web text in structure and content. As LLMs' performance is tied to their training data, a crucial question is how useful they can be in improving enterprise database management and analysis tasks. To address this, we contribute experimental results on LLMs' performance for text-to-SQL and semantic column-type detection tasks on enterprise datasets. The performance of LLMs on enterprise data is significantly lower than on benchmark datasets commonly used. Informed by our findings and feedback from industry practitioners, we identify three fundamental challenges -- latency, cost, and quality -- and propose potential solutions to use LLMs in enterprise data workflows effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.00802v1">Leveraging LLM Reasoning Enhances Personalized Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 To be published at ACL 2024
    </div>
    <details class="paper-abstract">
      Recent advancements have showcased the potential of Large Language Models (LLMs) in executing reasoning tasks, particularly facilitated by Chain-of-Thought (CoT) prompting. While tasks like arithmetic reasoning involve clear, definitive answers and logical chains of thought, the application of LLM reasoning in recommendation systems (RecSys) presents a distinct challenge. RecSys tasks revolve around subjectivity and personalized preferences, an under-explored domain in utilizing LLMs' reasoning capabilities. Our study explores several aspects to better understand reasoning for RecSys and demonstrate how task quality improves by utilizing LLM reasoning in both zero-shot and finetuning settings. Additionally, we propose RecSAVER (Recommender Systems Automatic Verification and Evaluation of Reasoning) to automatically assess the quality of LLM reasoning responses without the requirement of curated gold references or human raters. We show that our framework aligns with real human judgment on the coherence and faithfulness of reasoning responses. Overall, our work shows that incorporating reasoning into RecSys can improve personalized tasks, paving the way for further advancements in recommender system methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16030v1">Enhancing Temporal Understanding in LLMs for Semi-structured Tables</a></div>
    <div class="paper-meta">
      📅 2024-07-22
      | 💬 Total Pages 18, Total Tables 6, Total figures 7
    </div>
    <details class="paper-abstract">
      Temporal reasoning over tabular data presents substantial challenges for large language models (LLMs), as evidenced by recent research. In this study, we conduct a comprehensive analysis of temporal datasets to pinpoint the specific limitations of LLMs. Our investigation leads to enhancements in TempTabQA, a dataset specifically designed for tabular temporal question answering. We provide critical insights for improving LLM performance in temporal reasoning tasks with tabular data. Furthermore, we introduce a novel approach, C.L.E.A.R to strengthen LLM capabilities in this domain. Our findings demonstrate that our method significantly improves evidence-based reasoning across various models. Additionally, our experimental results reveal that indirect supervision with auxiliary data substantially boosts model performance in these tasks. This work contributes to a deeper understanding of LLMs' temporal reasoning abilities over tabular data and promotes advancements in their application across diverse fields.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.17663v2">LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic</a></div>
    <div class="paper-meta">
      📅 2024-07-22
    </div>
    <details class="paper-abstract">
      We introduce LLM-ARC, a neuro-symbolic framework designed to enhance the logical reasoning capabilities of Large Language Models (LLMs), by combining them with an Automated Reasoning Critic (ARC). LLM-ARC employs an Actor-Critic method where the LLM Actor generates declarative logic programs along with tests for semantic correctness, while the Automated Reasoning Critic evaluates the code, runs the tests and provides feedback on test failures for iterative refinement. Implemented using Answer Set Programming (ASP), LLM-ARC achieves a new state-of-the-art accuracy of 88.32% on the FOLIO benchmark which tests complex logical reasoning capabilities. Our experiments demonstrate significant improvements over LLM-only baselines, highlighting the importance of logic test generation and iterative self-refinement. We achieve our best result using a fully automated self-supervised training loop where the Actor is trained on end-to-end dialog traces with Critic feedback. We discuss potential enhancements and provide a detailed error analysis, showcasing the robustness and efficacy of LLM-ARC for complex natural language reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15248v1">XAI meets LLMs: A Survey of the Relation between Explainable AI and Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-07-21
    </div>
    <details class="paper-abstract">
      In this survey, we address the key challenges in Large Language Models (LLM) research, focusing on the importance of interpretability. Driven by increasing interest from AI and business sectors, we highlight the need for transparency in LLMs. We examine the dual paths in current LLM research and eXplainable Artificial Intelligence (XAI): enhancing performance through XAI and the emerging focus on model interpretability. Our paper advocates for a balanced approach that values interpretability equally with functional advancements. Recognizing the rapid development in LLM research, our survey includes both peer-reviewed and preprint (arXiv) papers, offering a comprehensive overview of XAI's role in LLM research. We conclude by urging the research community to advance both LLM and XAI fields together.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16832v2">Cross-Modal Projection in Multimodal LLMs Doesn't Really Project Visual Attributes to Textual Space</a></div>
    <div class="paper-meta">
      📅 2024-07-21
      | 💬 Accepted at ACL 2024 (Main, Short)
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) like LLaVA and GPT-4(V) enable general-purpose conversations about images with the language modality. As off-the-shelf MLLMs may have limited capabilities on images from domains like dermatology and agriculture, they must be fine-tuned to unlock domain-specific applications. The prevalent architecture of current open-source MLLMs comprises two major modules: an image-language (cross-modal) projection network and a large language model. It is desirable to understand the roles of these two modules in modeling domain-specific visual attributes to inform the design of future models and streamline the interpretability efforts on the current models. To this end, via experiments on 4 datasets and under 2 fine-tuning settings, we find that as the MLLM is fine-tuned, it indeed gains domain-specific visual capabilities, but the updates do not lead to the projection extracting relevant domain-specific visual attributes. Our results indicate that the domain-specific visual attributes are modeled by the LLM, even when only the projection is fine-tuned. Through this study, we offer a potential reinterpretation of the role of cross-modal projections in MLLM architectures. Project webpage: https://claws-lab.github.io/projection-in-MLLMs/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18333v1">AutoVCoder: A Systematic Framework for Automated Verilog Code Generation using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-21
    </div>
    <details class="paper-abstract">
      Recently, the use of large language models (LLMs) for software code generation, e.g., C/C++ and Python, has proven a great success. However, LLMs still suffer from low syntactic and functional correctness when it comes to the generation of register-transfer level (RTL) code, such as Verilog. To address this issue, in this paper, we develop AutoVCoder, a systematic open-source framework that significantly improves the LLMs' correctness of generating Verilog code and enhances the quality of its output at the same time. Our framework integrates three novel techniques, including a high-quality hardware dataset generation approach, a two-round LLM fine-tuning method and a domain-specific retrieval-augmented generation (RAG) mechanism. Experimental results demonstrate that AutoVCoder outperforms both industrial and academic LLMs in Verilog code generation. Specifically, AutoVCoder shows a 0.5% and 2.2% improvement in functional correctness on the EvalMachine and EvalHuman benchmarks compared with BetterV, and also achieves a 3.4% increase in syntax correctness and a 3.4% increase in functional correctness on the RTLLM benchmark compared with RTLCoder.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15184v1">Decoding Multilingual Moral Preferences: Unveiling LLM's Biases Through the Moral Machine Experiment</a></div>
    <div class="paper-meta">
      📅 2024-07-21
      | 💬 to be published in AIES 2024 Proceedings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly find their way into the most diverse areas of our everyday lives. They indirectly influence people's decisions or opinions through their daily use. Therefore, understanding how and which moral judgements these LLMs make is crucial. However, morality is not universal and depends on the cultural background. This raises the question of whether these cultural preferences are also reflected in LLMs when prompted in different languages or whether moral decision-making is consistent across different languages. So far, most research has focused on investigating the inherent values of LLMs in English. While a few works conduct multilingual analyses of moral bias in LLMs in a multilingual setting, these analyses do not go beyond atomic actions. To the best of our knowledge, a multilingual analysis of moral bias in dilemmas has not yet been conducted. To address this, our paper builds on the moral machine experiment (MME) to investigate the moral preferences of five LLMs, Falcon, Gemini, Llama, GPT, and MPT, in a multilingual setting and compares them with the preferences collected from humans belonging to different cultures. To accomplish this, we generate 6500 scenarios of the MME and prompt the models in ten languages on which action to take. Our analysis reveals that all LLMs inhibit different moral biases to some degree and that they not only differ from the human preferences but also across multiple languages within the models themselves. Moreover, we find that almost all models, particularly Llama 3, divert greatly from human values and, for instance, prefer saving fewer people over saving more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15141v1">Text-Augmented Multimodal LLMs for Chemical Reaction Condition Recommendation</a></div>
    <div class="paper-meta">
      📅 2024-07-21
    </div>
    <details class="paper-abstract">
      High-throughput reaction condition (RC) screening is fundamental to chemical synthesis. However, current RC screening suffers from laborious and costly trial-and-error workflows. Traditional computer-aided synthesis planning (CASP) tools fail to find suitable RCs due to data sparsity and inadequate reaction representations. Nowadays, large language models (LLMs) are capable of tackling chemistry-related problems, such as molecule design, and chemical logic Q\&A tasks. However, LLMs have not yet achieved accurate predictions of chemical reaction conditions. Here, we present MM-RCR, a text-augmented multimodal LLM that learns a unified reaction representation from SMILES, reaction graphs, and textual corpus for chemical reaction recommendation (RCR). To train MM-RCR, we construct 1.2 million pair-wised Q\&A instruction datasets. Our experimental results demonstrate that MM-RCR achieves state-of-the-art performance on two open benchmark datasets and exhibits strong generalization capabilities on out-of-domain (OOD) and High-Throughput Experimentation (HTE) datasets. MM-RCR has the potential to accelerate high-throughput condition screening in chemical synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18665v3">RouteLLM: Learning to Route LLMs with Preference Data</a></div>
    <div class="paper-meta">
      📅 2024-07-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive capabilities across a wide range of tasks, yet the choice of which model to use often involves a trade-off between performance and cost. More powerful models, though effective, come with higher expenses, while less capable models are more cost-effective. To address this dilemma, we propose several efficient router models that dynamically select between a stronger and a weaker LLM during inference, aiming to optimize the balance between cost and response quality. We develop a training framework for these routers leveraging human preference data and data augmentation techniques to enhance performance. Our evaluation on widely-recognized benchmarks shows that our approach significantly reduces costs-by over 2 times in certain cases-without compromising the quality of responses. Interestingly, our router models also demonstrate significant transfer learning capabilities, maintaining their performance even when the strong and weak models are changed at test time. This highlights the potential of these routers to provide a cost-effective yet high-performance solution for deploying LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.15046v1">Audio-visual training for improved grounding in video-text LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-21
    </div>
    <details class="paper-abstract">
      Recent advances in multimodal LLMs, have led to several video-text models being proposed for critical video-related tasks. However, most of the previous works support visual input only, essentially muting the audio signal in the video. Few models that support both audio and visual input, are not explicitly trained on audio data. Hence, the effect of audio towards video understanding is largely unexplored. To this end, we propose a model architecture that handles audio-visual inputs explicitly. We train our model with both audio and visual data from a video instruction-tuning dataset. Comparison with vision-only baselines, and other audio-visual models showcase that training on audio data indeed leads to improved grounding of responses. For better evaluation of audio-visual models, we also release a human-annotated benchmark dataset, with audio-aware question-answer pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08454v2">Model Tells You Where to Merge: Adaptive KV Cache Merging for LLMs on Long-Context Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-21
    </div>
    <details class="paper-abstract">
      How to efficiently serve Large Language Models (LLMs) has become a pressing issue because of their huge computational cost in their autoregressive generation process. To mitigate computational costs, LLMs often employ the KV Cache technique to improve the generation speed. While improving the computational efficiency, the storage requirements of the KV cache are substantial, particularly in long-context scenarios, leading to significant memory consumption. Existing KV cache eviction methods often degrade the performance of LLMs in long-context scenarios due to the information loss introduced by eviction. In this paper, we propose a novel KV cache merging approach, called KVMerger, to achieve adaptive KV cache compression for long-context tasks without significant performance degradation under constrained memory budgets. Our approach is inspired by the intriguing observation that key states exhibit high similarity at the token level within a single sequence. To facilitate merging, we develop an effective yet straightforward merging set identification algorithm to identify suitable KV states for merging. Our merging set identification algorithm stimulates the second observation that KV cache sparsity, from similarity perspective, is independent of the dataset and remains persistent at the model level. Subsequently, we propose a Gaussian kernel weighted merging algorithm to selectively merge all states within each merging set. We conduct extensive experiments to demonstrate the effectiveness of KVMerger for long-context tasks under constrained memory budgets, applying it to models including Llama2-7B-chat and Llama2-13B-chat. Using the LongBench and ZeroScroll benchmarks, we compare our method with other KV cache compression techniques, including H2O and CaM, showing that our method achieves superior performance across tasks with both 50% and 35% KV cache budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14937v1">Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2024-07-20
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10424v4">CodeV: Empowering LLMs for Verilog Generation through Multi-Level Summarization</a></div>
    <div class="paper-meta">
      📅 2024-07-20
      | 💬 16 pages, 8 figures, conference
    </div>
    <details class="paper-abstract">
      The increasing complexity and high costs associated with modern processor design have led to a surge in demand for processor design automation. Instruction-tuned large language models (LLMs) have demonstrated remarkable performance in automatically generating code for general-purpose programming languages like Python. However, these methods fail on hardware description languages (HDLs) like Verilog due to the scarcity of high-quality instruction tuning data, as even advanced LLMs like GPT-3.5 exhibit limited performance on Verilog generation. Regarding this issue, we observe that (1) Verilog code collected from the real world has higher quality than those generated by LLMs. (2) LLMs like GPT-3.5 excel in summarizing Verilog code rather than generating it. Based on these observations, this paper introduces CodeV, a series of open-source instruction-tuned Verilog generation LLMs. Instead of generating descriptions first and then getting the corresponding code from advanced LLMs, we prompt the LLM with Verilog code and let the LLM generate the corresponding natural language description by multi-level summarization. Experimental results show that CodeV relatively surpasses the previous open-source SOTA by 14.4% (BetterV in VerilogEval) and 11.3% (RTLCoder in RTLLM) respectively, and also relatively outperforms previous commercial SOTA GPT-4 by 22.1% in VerilogEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10241v2">BiasAlert: A Plug-and-play Tool for Social Bias Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-20
    </div>
    <details class="paper-abstract">
      Evaluating the bias in Large Language Models (LLMs) becomes increasingly crucial with their rapid development. However, existing evaluation methods rely on fixed-form outputs and cannot adapt to the flexible open-text generation scenarios of LLMs (e.g., sentence completion and question answering). To address this, we introduce BiasAlert, a plug-and-play tool designed to detect social bias in open-text generations of LLMs. BiasAlert integrates external human knowledge with inherent reasoning capabilities to detect bias reliably. Extensive experiments demonstrate that BiasAlert significantly outperforms existing state-of-the-art methods like GPT4-as-A-Judge in detecting bias. Furthermore, through application studies, we demonstrate the utility of BiasAlert in reliable LLM bias evaluation and bias mitigation across various scenarios. Model and code will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.05811v2">Tuning LLMs with Contrastive Alignment Instructions for Machine Translation in Unseen, Low-resource Languages</a></div>
    <div class="paper-meta">
      📅 2024-07-20
      | 💬 Accepted to LoResMT 2024
    </div>
    <details class="paper-abstract">
      This article introduces contrastive alignment instructions (AlignInstruct) to address two challenges in machine translation (MT) on large language models (LLMs). One is the expansion of supported languages to previously unseen ones. The second relates to the lack of data in low-resource languages. Model fine-tuning through MT instructions (MTInstruct) is a straightforward approach to the first challenge. However, MTInstruct is limited by weak cross-lingual signals inherent in the second challenge. AlignInstruct emphasizes cross-lingual supervision via a cross-lingual discriminator built using statistical word alignments. Our results based on fine-tuning the BLOOMZ models (1b1, 3b, and 7b1) in up to 24 unseen languages showed that: (1) LLMs can effectively translate unseen languages using MTInstruct; (2) AlignInstruct led to consistent improvements in translation quality across 48 translation directions involving English; (3) Discriminator-based instructions outperformed their generative counterparts as cross-lingual instructions; (4) AlignInstruct improved performance in 30 zero-shot directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14834v1">Can VLMs be used on videos for action recognition? LLMs are Visual Reasoning Coordinators</a></div>
    <div class="paper-meta">
      📅 2024-07-20
      | 💬 LLMs, VLMs, Action Recognition
    </div>
    <details class="paper-abstract">
      Recent advancements have introduced multiple vision-language models (VLMs) demonstrating impressive commonsense reasoning across various domains. Despite their individual capabilities, the potential of synergizing these complementary VLMs remains underexplored. The Cola Framework addresses this by showcasing how a large language model (LLM) can efficiently coordinate multiple VLMs through natural language communication, leveraging their distinct strengths. We have verified this claim on the challenging A-OKVQA dataset, confirming the effectiveness of such coordination. Building on this, our study investigates whether the same methodology can be applied to surveillance videos for action recognition. Specifically, we explore if leveraging the combined knowledge base of VLMs and LLM can effectively deduce actions from a video when presented with only a few selectively important frames and minimal temporal information. Our experiments demonstrate that LLM, when coordinating different VLMs, can successfully recognize patterns and deduce actions in various scenarios despite the weak temporal signals. However, our findings suggest that to enhance this approach as a viable alternative solution, integrating a stronger temporal signal and exposing the models to slightly more frames would be beneficial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19883v2">From Words to Actions: Unveiling the Theoretical Underpinnings of LLM-Driven Autonomous Systems</a></div>
    <div class="paper-meta">
      📅 2024-07-20
      | 💬 47 pages, accepted by ICML 2024
    </div>
    <details class="paper-abstract">
      In this work, from a theoretical lens, we aim to understand why large language model (LLM) empowered agents are able to solve decision-making problems in the physical world. To this end, consider a hierarchical reinforcement learning (RL) model where the LLM Planner and the Actor perform high-level task planning and low-level execution, respectively. Under this model, the LLM Planner navigates a partially observable Markov decision process (POMDP) by iteratively generating language-based subgoals via prompting. Under proper assumptions on the pretraining data, we prove that the pretrained LLM Planner effectively performs Bayesian aggregated imitation learning (BAIL) through in-context learning. Additionally, we highlight the necessity for exploration beyond the subgoals derived from BAIL by proving that naively executing the subgoals returned by LLM leads to a linear regret. As a remedy, we introduce an $\epsilon$-greedy exploration strategy to BAIL, which is proven to incur sublinear regret when the pretraining error is small. Finally, we extend our theoretical framework to include scenarios where the LLM Planner serves as a world model for inferring the transition model of the environment and to multi-agent settings, enabling coordination among multiple Actors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11438v2">Trust No Bot: Discovering Personal Disclosures in Human-LLM Conversations in the Wild</a></div>
    <div class="paper-meta">
      📅 2024-07-20
    </div>
    <details class="paper-abstract">
      Measuring personal disclosures made in human-chatbot interactions can provide a better understanding of users' AI literacy and facilitate privacy research for large language models (LLMs). We run an extensive, fine-grained analysis on the personal disclosures made by real users to commercial GPT models, investigating the leakage of personally identifiable and sensitive information. To understand the contexts in which users disclose to chatbots, we develop a taxonomy of tasks and sensitive topics, based on qualitative and quantitative analysis of naturally occurring conversations. We discuss these potential privacy harms and observe that: (1) personally identifiable information (PII) appears in unexpected contexts such as in translation or code editing (48% and 16% of the time, respectively) and (2) PII detection alone is insufficient to capture the sensitive topics that are common in human-chatbot interactions, such as detailed sexual preferences or specific drug use habits. We believe that these high disclosure rates are of significant importance for researchers and data curators, and we call for the design of appropriate nudging mechanisms to help users moderate their interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00554v2">Guiding and Diversifying LLM-Based Story Generation via Answer Set Programming</a></div>
    <div class="paper-meta">
      📅 2024-07-19
      | 💬 Accepted to Wordplay @ ACL 2024
    </div>
    <details class="paper-abstract">
      Instruction-tuned large language models (LLMs) are capable of generating stories in response to open-ended user requests, but the resulting stories tend to be limited in their diversity. Older, symbolic approaches to story generation (such as planning) can generate substantially more diverse plot outlines, but are limited to producing stories that recombine a fixed set of hand-engineered character action templates. Can we combine the strengths of these approaches while mitigating their weaknesses? We propose to do so by using a higher-level and more abstract symbolic specification of high-level story structure -- implemented via answer set programming (ASP) -- to guide and diversify LLM-based story generation. Via semantic similarity analysis, we demonstrate that our approach produces more diverse stories than an unguided LLM, and via code excerpts, we demonstrate the improved compactness and flexibility of ASP-based outline generation over full-fledged narrative planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14640v1">CVE-LLM : Automatic vulnerability evaluation in medical device industry using large language models</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      The healthcare industry is currently experiencing an unprecedented wave of cybersecurity attacks, impacting millions of individuals. With the discovery of thousands of vulnerabilities each month, there is a pressing need to drive the automation of vulnerability assessment processes for medical devices, facilitating rapid mitigation efforts. Generative AI systems have revolutionized various industries, offering unparalleled opportunities for automation and increased efficiency. This paper presents a solution leveraging Large Language Models (LLMs) to learn from historical evaluations of vulnerabilities for the automatic assessment of vulnerabilities in the medical devices industry. This approach is applied within the portfolio of a single manufacturer, taking into account device characteristics, including existing security posture and controls. The primary contributions of this paper are threefold. Firstly, it provides a detailed examination of the best practices for training a vulnerability Language Model (LM) in an industrial context. Secondly, it presents a comprehensive comparison and insightful analysis of the effectiveness of Language Models in vulnerability assessment. Finally, it proposes a new human-in-the-loop framework to expedite vulnerability evaluation processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.11372v2">Democratizing LLMs for Low-Resource Languages by Leveraging their English Dominant Abilities with Linguistically-Diverse Prompts</a></div>
    <div class="paper-meta">
      📅 2024-07-19
      | 💬 ACL 2024 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to effectively perform tasks by simply observing few exemplars. However, in low-resource languages, obtaining such hand-picked exemplars can still be challenging, where unsupervised techniques may be necessary. Moreover, competent generative capabilities of LLMs are observed only in high-resource languages, while their performances among under-represented languages fall behind due to pre-training data imbalance. To elicit LLMs' ability onto low-resource languages without any supervised data, we propose to assemble synthetic exemplars from a diverse set of high-resource languages to prompt the LLMs to translate from any language into English. These prompts are then used to create intra-lingual exemplars to perform tasks in the target languages. Our unsupervised prompting method performs on par with supervised few-shot learning in LLMs of different sizes for translations between English and 13 Indic and 21 African low-resource languages. We also show that fine-tuning a 7B model on data generated from our method helps it perform competitively with a 175B model. In non-English translation tasks, our method even outperforms supervised prompting by up to 3 chrF++ in many low-resource languages. When evaluated on zero-shot multilingual summarization, our method surpasses other English-pivoting baselines by up to 4 ROUGE-L and is also favored by GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14402v1">The Vision of Autonomic Computing: Can LLMs Make It a Reality?</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      The Vision of Autonomic Computing (ACV), proposed over two decades ago, envisions computing systems that self-manage akin to biological organisms, adapting seamlessly to changing environments. Despite decades of research, achieving ACV remains challenging due to the dynamic and complex nature of modern computing systems. Recent advancements in Large Language Models (LLMs) offer promising solutions to these challenges by leveraging their extensive knowledge, language understanding, and task automation capabilities. This paper explores the feasibility of realizing ACV through an LLM-based multi-agent framework for microservice management. We introduce a five-level taxonomy for autonomous service maintenance and present an online evaluation benchmark based on the Sock Shop microservice demo project to assess our framework's performance. Our findings demonstrate significant progress towards achieving Level 3 autonomy, highlighting the effectiveness of LLMs in detecting and resolving issues within microservice architectures. This study contributes to advancing autonomic computing by pioneering the integration of LLMs into microservice management frameworks, paving the way for more adaptive and self-managing computing systems. The code will be made available at https://aka.ms/ACV-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14372v1">SCoPE: Evaluating LLMs for Software Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2024-07-19
      | 💬 10 pages, 3 figures, 1 table, published in DCAI 24 conference
    </div>
    <details class="paper-abstract">
      In recent years, code security has become increasingly important, especially with the rise of interconnected technologies. Detecting vulnerabilities early in the software development process has demonstrated numerous benefits. Consequently, the scientific community started using machine learning for automated detection of source code vulnerabilities. This work explores and refines the CVEFixes dataset, which is commonly used to train models for code-related tasks, specifically the C/C++ subset. To this purpose, the Source Code Processing Engine (SCoPE), a framework composed of strategized techniques that can be used to reduce the size and normalize C/C++ functions is presented. The output generated by SCoPE was used to create a new version of CVEFixes. This refined dataset was then employed in a feature representation analysis to assess the effectiveness of the tool's code processing techniques, consisting of fine-tuning three pre-trained LLMs for software vulnerability detection. The results show that SCoPE successfully helped to identify 905 duplicates within the evaluated subset. The LLM results corroborate with the literature regarding their suitability for software vulnerability detection, with the best model achieving 53% F1-score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17663v2">LLM-ARC: Enhancing LLMs with an Automated Reasoning Critic</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      We introduce LLM-ARC, a neuro-symbolic framework designed to enhance the logical reasoning capabilities of Large Language Models (LLMs), by combining them with an Automated Reasoning Critic (ARC). LLM-ARC employs an Actor-Critic method where the LLM Actor generates declarative logic programs along with tests for semantic correctness, while the Automated Reasoning Critic evaluates the code, runs the tests and provides feedback on test failures for iterative refinement. Implemented using Answer Set Programming (ASP), LLM-ARC achieves a new state-of-the-art accuracy of 88.32% on the FOLIO benchmark which tests complex logical reasoning capabilities. Our experiments demonstrate significant improvements over LLM-only baselines, highlighting the importance of logic test generation and iterative self-refinement. We achieve our best result using a fully automated self-supervised training loop where the Actor is trained on end-to-end dialog traces with Critic feedback. We discuss potential enhancements and provide a detailed error analysis, showcasing the robustness and efficacy of LLM-ARC for complex natural language reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01490v2">LLM See, LLM Do: Guiding Data Generation to Target Non-Differentiable Objectives</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      The widespread adoption of synthetic data raises new questions about how models generating the data can influence other large language models (LLMs) via distilled data. To start, our work exhaustively characterizes the impact of passive inheritance of model properties by systematically studying the consequences of synthetic data integration. We provide one of the most comprehensive studies to-date of how the source of synthetic data shapes models' internal biases, calibration and generations' textual attributes and preferences. We find that models are surprisingly sensitive towards certain attributes even when the synthetic data prompts appear "neutral". which invites the question whether this sensitivity can be exploited for good. Our findings invite the question can we explicitly steer the models towards the properties we want at test time by exploiting the data generation process? This would have historically been considered infeasible due to the cost of collecting data with a specific characteristic or objective in mind. However, improvement in the quality of synthetic data, as well as a shift towards general-purpose models designed to follow a diverse way of instructions, means this question is timely. We propose active inheritance as a term to describe intentionally constraining synthetic data according to a non-differentiable objective. We demonstrate how active inheritance can steer the generation profiles of models towards desirable non-differentiable attributes, e.g. high lexical diversity or low toxicity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10107v2">Enhancing Human-Centered Dynamic Scene Understanding via Multiple LLMs Collaborated Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      Human-centered dynamic scene understanding plays a pivotal role in enhancing the capability of robotic and autonomous systems, in which Video-based Human-Object Interaction (V-HOI) detection is a crucial task in semantic scene understanding, aimed at comprehensively understanding HOI relationships within a video to benefit the behavioral decisions of mobile robots and autonomous driving systems. Although previous V-HOI detection models have made significant strides in accurate detection on specific datasets, they still lack the general reasoning ability like human beings to effectively induce HOI relationships. In this study, we propose V-HOI Multi-LLMs Collaborated Reasoning (V-HOI MLCR), a novel framework consisting of a series of plug-and-play modules that could facilitate the performance of current V-HOI detection models by leveraging the strong reasoning ability of different off-the-shelf pre-trained large language models (LLMs). We design a two-stage collaboration system of different LLMs for the V-HOI task. Specifically, in the first stage, we design a Cross-Agents Reasoning scheme to leverage the LLM conduct reasoning from different aspects. In the second stage, we perform Multi-LLMs Debate to get the final reasoning answer based on the different knowledge in different LLMs. Additionally, we devise an auxiliary training strategy that utilizes CLIP, a large vision-language model to enhance the base V-HOI models' discriminative ability to better cooperate with LLMs. We validate the superiority of our design by demonstrating its effectiveness in improving the prediction accuracy of the base V-HOI model via reasoning from multiple perspectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14118v1">Beyond Code Generation: Assessing Code LLM Maturity with Postconditions</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      Most existing code Large Language Model (LLM) benchmarks, e.g., EvalPlus, focus on the code generation tasks. Namely, they contain a natural language description of a problem and ask the LLM to write code to solve the problem. We argue that they do not capture all capabilities needed to assess the quality of a code LLM. In this paper, we propose a code LLM maturity model, based on the postcondition generation problem, to access a more complete set of code LLM capabilities. We choose the postcondition generation problem as it requires the code LLM to understand the code including semantics, natural language, and also have the capability to generate unambiguous postconditions in programming languages (i.e., the generation capablity). Moreover, postconditions have various types, requiring different levels of these capabilities, making it suitable to evaluate the maturity of the code LLM. Based on our designed maturity model, we augment the EvalPlus dataset to a postcondition testing benchmark, and evaluated several open-sourced models. Our results highlight the necessary improvements needed for better LLMs for code. Code: https://github.com/MatureModel/PostcondGen
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08754v2">Exploiting Uncommon Text-Encoded Structures for Automated Jailbreaks in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-19
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in natural language processing but face the risk of jailbreak attacks that maliciously induce them to generate harmful content. Existing jailbreak attacks, including character-level and context-level attacks, mainly focus on the prompt of the plain text without specifically exploring the significant influence of its structure. In this paper, we focus on studying how prompt structure contributes to the jailbreak attack. We introduce a novel structure-level attack method based on tail structures that are rarely used during LLM training, which we refer to as Uncommon Text-Encoded Structure (UTES). We extensively study 12 UTESs templates and 6 obfuscation methods to build an effective automated jailbreak tool named StructuralSleight that contains three escalating attack strategies: Structural Attack, Structural and Character/Context Obfuscation Attack, and Fully Obfuscated Structural Attack. Extensive experiments on existing LLMs show that StructuralSleight significantly outperforms baseline methods. In particular, the attack success rate reaches 94.62\% on GPT-4o, which has not been addressed by state-of-the-art techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14088v1">Impact of Model Size on Fine-tuned LLM Performance in Data-to-Text Generation: A State-of-the-Art Investigation</a></div>
    <div class="paper-meta">
      📅 2024-07-19
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Data-to-text (D2T) generation aims to generate human-readable text from semi-structured data, such as tables and graphs. The recent success of D2T is largely attributed to advancements in LLMs. Despite the success of LLMs, no research has been conducted to illustrate the impact of model size on the performance of fine-tuned LLMs for D2T tasks. D2T model performance is typically assessed based on three key qualities: \textit{readability} (indicates fluency and coherence), \textit{informativeness} (measures content similarity), and \textit{faithfulness} (assesses consistency of factual information). It is currently uncertain whether increasing the size of LLMs effectively improves performance in D2T tasks across these three qualities. The objective of this study is to investigate the performance of fine-tuned LLMs in D2T tasks in terms of model size. Through extensive comparative analysis, we aim to elucidate both the advantages and limitations of scaling model sizes across five widely used D2T datasets (E2E, ViGGo, WikiTableText, DART, and WebNLG) and twelve state-of-the-art LLMs with varying sizes from five different LLM families (T5, BART, OPT, BLOOM, and Llama 2). To comprehensively cover all the three essential qualities of D2T models, we incorporate six widely recognized automatic metrics -- \textsc{BLEU}, \textsc{METEOR}, \textsc{BERTScore}, \textsc{MoverScore}, \textsc{Parent}, and \textsc{BARTScore}. We also provide an in-depth analysis of LLM performance concerning model size in the presence of source-reference divergence, a critical aspect of D2T tasks. Our investigation reveals that increasing LLM size enhances \textit{readability} and \textit{informativeness} in D2T tasks, but larger (in terms of size) LLMs may sacrifice \textit{faithfulness}. Moreover, small-sized LLMs show more resilience than larger ones when source-reference divergence is present.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.14568v1">SQLfuse: Enhancing Text-to-SQL Performance through Comprehensive LLM Synergy</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      Text-to-SQL conversion is a critical innovation, simplifying the transition from complex SQL to intuitive natural language queries, especially significant given SQL's prevalence in the job market across various roles. The rise of Large Language Models (LLMs) like GPT-3.5 and GPT-4 has greatly advanced this field, offering improved natural language understanding and the ability to generate nuanced SQL statements. However, the potential of open-source LLMs in Text-to-SQL applications remains underexplored, with many frameworks failing to leverage their full capabilities, particularly in handling complex database queries and incorporating feedback for iterative refinement. Addressing these limitations, this paper introduces SQLfuse, a robust system integrating open-source LLMs with a suite of tools to enhance Text-to-SQL translation's accuracy and usability. SQLfuse features four modules: schema mining, schema linking, SQL generation, and a SQL critic module, to not only generate but also continuously enhance SQL query quality. Demonstrated by its leading performance on the Spider Leaderboard and deployment by Ant Group, SQLfuse showcases the practical merits of open-source LLMs in diverse business contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.10276v2">Revisiting OPRO: The Limitations of Small-Scale LLMs as Optimizers</a></div>
    <div class="paper-meta">
      📅 2024-07-19
    </div>
    <details class="paper-abstract">
      Numerous recent works aim to enhance the efficacy of Large Language Models (LLMs) through strategic prompting. In particular, the Optimization by PROmpting (OPRO) approach provides state-of-the-art performance by leveraging LLMs as optimizers where the optimization task is to find instructions that maximize the task accuracy. In this paper, we revisit OPRO for automated prompting with relatively small-scale LLMs, such as LLaMa-2 family and Mistral 7B. Our investigation reveals that OPRO shows limited effectiveness in small-scale LLMs, with limited inference capabilities constraining optimization ability. We suggest future automatic prompting engineering to consider both model capabilities and computational costs. Additionally, for small-scale LLMs, we recommend direct instructions that clearly outline objectives and methodologies as robust prompt baselines, ensuring efficient and effective prompt engineering in ongoing research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13943v1">Werewolf Arena: A Case Study in LLM Evaluation via Social Deduction</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 13 pages, 10 figures
    </div>
    <details class="paper-abstract">
      This paper introduces Werewolf Arena, a novel framework for evaluating large language models (LLMs) through the lens of the classic social deduction game, Werewolf. In Werewolf Arena, LLMs compete against each other, navigating the game's complex dynamics of deception, deduction, and persuasion. The framework introduces a dynamic turn-taking system based on bidding, mirroring real-world discussions where individuals strategically choose when to speak. We demonstrate the framework's utility through an arena-style tournament featuring Gemini and GPT models. Our results reveal distinct strengths and weaknesses in the models' strategic reasoning and communication. These findings highlight Werewolf Arena's potential as a challenging and scalable LLM benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13900v1">Exploring the Evidence-Based Beliefs and Behaviors of LLM-Based Programming Assistants</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      Recent innovations in artificial intelligence (AI), primarily powered by large language models (LLMs), have transformed how programmers develop and maintain software -- leading to new frontiers in software engineering (SE). The advanced capabilities of LLM-based programming assistants to support software development tasks have led to a rise in the adoption of LLMs in SE. However, little is known about the evidenced-based practices, tools and processes verified by research findings, supported and adopted by AI programming assistants. To this end, our work conducts a preliminary evaluation exploring the beliefs and behaviors of LLM used to support software development tasks. We investigate 17 evidence-based claims posited by empirical SE research across five LLM-based programming assistants. Our findings show that LLM-based programming assistants have ambiguous beliefs regarding research claims, lack credible evidence to support responses, and are incapable of adopting practices demonstrated by empirical SE research to support development tasks. Based on our results, we provide implications for practitioners adopting LLM-based programming assistants in development contexts and shed light on future research directions to enhance the reliability and trustworthiness of LLMs -- aiming to increase awareness and adoption of evidence-based SE research findings in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.01433v1">Evaluating and Enhancing Trustworthiness of LLMs in Perception Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 Accepted in 27th IEEE International Conference on Intelligent Transportation Systems (ITSC) 2024
    </div>
    <details class="paper-abstract">
      Today's advanced driver assistance systems (ADAS), like adaptive cruise control or rear collision warning, are finding broader adoption across vehicle classes. Integrating such advanced, multimodal Large Language Models (LLMs) on board a vehicle, which are capable of processing text, images, audio, and other data types, may have the potential to greatly enhance passenger comfort. Yet, an LLM's hallucinations are still a major challenge to be addressed. In this paper, we systematically assessed potential hallucination detection strategies for such LLMs in the context of object detection in vision-based data on the example of pedestrian detection and localization. We evaluate three hallucination detection strategies applied to two state-of-the-art LLMs, the proprietary GPT-4V and the open LLaVA, on two datasets (Waymo/US and PREPER CITY/Sweden). Our results show that these LLMs can describe a traffic situation to an impressive level of detail but are still challenged for further analysis activities such as object localization. We evaluate and extend hallucination detection approaches when applying these LLMs to video sequences in the example of pedestrian detection. Our experiments show that, at the moment, the state-of-the-art proprietary LLM performs much better than the open LLM. Furthermore, consistency enhancement techniques based on voting, such as the Best-of-Three (BO3) method, do not effectively reduce hallucinations in LLMs that tend to exhibit high false negatives in detecting pedestrians. However, extending the hallucination detection by including information from the past helps to improve results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2306.00978v5">AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 MLSys 2024 Best Paper Award. Code available at: https://github.com/mit-han-lab/llm-awq
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have transformed numerous AI applications. On-device LLM is becoming increasingly important: running LLMs locally on edge devices can reduce the cloud computing cost and protect users' privacy. However, the astronomical model size and the limited hardware resource pose significant deployment challenges. We propose Activation-aware Weight Quantization (AWQ), a hardware-friendly approach for LLM low-bit weight-only quantization. AWQ finds that not all weights in an LLM are equally important. Protecting only 1% salient weights can greatly reduce quantization error. To identify salient weight channels, we should refer to the activation distribution, not weights. To avoid the hardware-inefficient mix-precision quantization, we mathematically derive that scaling up the salient channels can reduce the quantization error. AWQ employs an equivalent transformation to scale the salient weight channels to protect them. The scale is determined by collecting the activation statistics offline. AWQ does not rely on any backpropagation or reconstruction, so it generalizes to different domains and modalities without overfitting the calibration set. AWQ outperforms existing work on various language modeling and domain-specific benchmarks (coding and math). Thanks to better generalization, it achieves excellent quantization performance for instruction-tuned LMs and, for the first time, multi-modal LMs. Alongside AWQ, we implement TinyChat, an efficient and flexible inference framework tailored for 4-bit on-device LLM/VLMs. With kernel fusion and platform-aware weight packing, TinyChat offers more than 3x speedup over the Huggingface FP16 implementation on both desktop and mobile GPUs. It also democratizes the deployment of the 70B Llama-2 model on mobile GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13598v1">KNOWNET: Guided Health Information Seeking from LLMs via Knowledge Graph Integration</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 9 pages, 9 figures, accepted by IEEE VIS 2024
    </div>
    <details class="paper-abstract">
      The increasing reliance on Large Language Models (LLMs) for health information seeking can pose severe risks due to the potential for misinformation and the complexity of these topics. This paper introduces KNOWNET a visualization system that integrates LLMs with Knowledge Graphs (KG) to provide enhanced accuracy and structured exploration. Specifically, for enhanced accuracy, KNOWNET extracts triples (e.g., entities and their relations) from LLM outputs and maps them into the validated information and supported evidence in external KGs. For structured exploration, KNOWNET provides next-step recommendations based on the neighborhood of the currently explored entities in KGs, aiming to guide a comprehensive understanding without overlooking critical aspects. To enable reasoning with both the structured data in KGs and the unstructured outputs from LLMs, KNOWNET conceptualizes the understanding of a subject as the gradual construction of graph visualization. A progressive graph visualization is introduced to monitor past inquiries, and bridge the current query with the exploration history and next-step recommendations. We demonstrate the effectiveness of our system via use cases and expert interviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13561v1">Research on Tibetan Tourism Viewpoints information generation system based on LLM</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      Tibet, ensconced within China's territorial expanse, is distinguished by its labyrinthine and heterogeneous topography, a testament to its profound historical heritage, and the cradle of a unique religious ethos. The very essence of these attributes, however, has impeded the advancement of Tibet's tourism service infrastructure, rendering existing smart tourism services inadequate for the region's visitors. This study delves into the ramifications of informational disparities at tourist sites on Tibetan tourism and addresses the challenge of establishing the Large Language Model (LLM) evaluation criteria. It introduces an innovative approach, the DualGen Bridge AI system, employing supervised fine-tuning techniques to bolster model functionality and enhance optimization processes. Furthermore, it pioneers a multi-structured generative results assessment framework. Empirical validation confirms the efficacy of this framework. The study also explores the application of the supervised fine-tuning method within the proprietary DualGen Bridge AI, aimed at refining the generation of tourist site information. The study's findings offer valuable insights for optimizing system performance and provide support and inspiration for the application of LLM technology in Tibet's tourism services and beyond, potentially revolutionizing the smart tourism industry with advanced, tailored information generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13522v1">INDIC QA BENCHMARK: A Multilingual Benchmark to Evaluate Question Answering capability of LLMs for Indic Languages</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable zero-shot and few-shot capabilities in unseen tasks, including context-grounded question answering (QA) in English. However, the evaluation of LLMs' capabilities in non-English languages for context-based QA is limited by the scarcity of benchmarks in non-English languages. To address this gap, we introduce Indic-QA, the largest publicly available context-grounded question-answering dataset for 11 major Indian languages from two language families. The dataset comprises both extractive and abstractive question-answering tasks and includes existing datasets as well as English QA datasets translated into Indian languages. Additionally, we generate a synthetic dataset using the Gemini model to create question-answer pairs given a passage, which is then manually verified for quality assurance. We evaluate various multilingual Large Language Models and their instruction-fine-tuned variants on the benchmark and observe that their performance is subpar, particularly for low-resource languages. We hope that the release of this dataset will stimulate further research on the question-answering abilities of LLMs for low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13511v1">Can Open-Source LLMs Compete with Commercial Models? Exploring the Few-Shot Performance of Current GPT Models in Biomedical Tasks</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 Version as accepted at the BioASQ Lab at CLEF 2024
    </div>
    <details class="paper-abstract">
      Commercial large language models (LLMs), like OpenAI's GPT-4 powering ChatGPT and Anthropic's Claude 3 Opus, have dominated natural language processing (NLP) benchmarks across different domains. New competing Open-Source alternatives like Mixtral 8x7B or Llama 3 have emerged and seem to be closing the gap while often offering higher throughput and being less costly to use. Open-Source LLMs can also be self-hosted, which makes them interesting for enterprise and clinical use cases where sensitive data should not be processed by third parties. We participated in the 12th BioASQ challenge, which is a retrieval augmented generation (RAG) setting, and explored the performance of current GPT models Claude 3 Opus, GPT-3.5-turbo and Mixtral 8x7b with in-context learning (zero-shot, few-shot) and QLoRa fine-tuning. We also explored how additional relevant knowledge from Wikipedia added to the context-window of the LLM might improve their performance. Mixtral 8x7b was competitive in the 10-shot setting, both with and without fine-tuning, but failed to produce usable results in the zero-shot setting. QLoRa fine-tuning and Wikipedia context did not lead to measurable performance gains. Our results indicate that the performance gap between commercial and open-source models in RAG setups exists mainly in the zero-shot setting and can be closed by simply collecting few-shot examples for domain-specific use cases. The code needed to rerun these experiments is available through GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12468v2">Search Engines, LLMs or Both? Evaluating Information Seeking Strategies for Answering Health Questions</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      Search engines have traditionally served as primary tools for information seeking. However, the new Large Language Models (LLMs) have recently demonstrated remarkable capabilities in multiple tasks and, specifically, their adoption as question answering systems is becoming increasingly prevalent. It is expected that LLM-based conversational systems and traditional web engines will continue to coexist in the future, supporting end users in various ways. But there is a need for more scientific research on the effectiveness of both types of systems in facilitating accurate information seeking. In this study, we focus on their merits in answering health questions. We conducted an extensive study comparing different web search engines, LLMs and retrieval-augmented (RAG) approaches. Our research reveals intriguing conclusions. For example, we observed that the quality of webpages potentially responding to a health question does not decline as we navigate further down the ranked lists. However, according to our evaluation, web engines are less accurate than LLMs in finding correct answers to health questions. On the other hand, LLMs are quite sensitive to the input prompts, and we also found out that RAG leads to highly effective information seeking methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.17249v2">Assessing LLMs Suitability for Knowledge Graph Completion</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 Accepted at 18th International Conference on Neural-Symbolic Learning and Reasoning, NESY 2024. Evaluating Mixtral-8x7b-Instruct-v0.1, GPT-3.5-Turbo-0125 and GPT-4o for Knowledge Graph Completion task with prompts formatted according to the TELeR taxonomy
    </div>
    <details class="paper-abstract">
      Recent work has shown the capability of Large Language Models (LLMs) to solve tasks related to Knowledge Graphs, such as Knowledge Graph Completion, even in Zero- or Few-Shot paradigms. However, they are known to hallucinate answers, or output results in a non-deterministic manner, thus leading to wrongly reasoned responses, even if they satisfy the user's demands. To highlight opportunities and challenges in knowledge graphs-related tasks, we experiment with three distinguished LLMs, namely Mixtral-8x7b-Instruct-v0.1, GPT-3.5-Turbo-0125 and GPT-4o, on Knowledge Graph Completion for static knowledge graphs, using prompts constructed following the TELeR taxonomy, in Zero- and One-Shot contexts, on a Task-Oriented Dialogue system use case. When evaluated using both strict and flexible metrics measurement manners, our results show that LLMs could be fit for such a task if prompts encapsulate sufficient information and relevant examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.16374v3">LLM Factoscope: Uncovering LLMs' Factual Discernment through Inner States Analysis</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized various domains with extensive knowledge and creative capabilities. However, a critical issue with LLMs is their tendency to produce outputs that diverge from factual reality. This phenomenon is particularly concerning in sensitive applications such as medical consultation and legal advice, where accuracy is paramount. In this paper, we introduce the LLM factoscope, a novel Siamese network-based model that leverages the inner states of LLMs for factual detection. Our investigation reveals distinguishable patterns in LLMs' inner states when generating factual versus non-factual content. We demonstrate the LLM factoscope's effectiveness across various architectures, achieving over 96% accuracy in factual detection. Our work opens a new avenue for utilizing LLMs' inner states for factual detection and encourages further exploration into LLMs' inner workings for enhanced reliability and transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18676v2">Understand What LLM Needs: Dual Preference Alignment for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has demonstrated effectiveness in mitigating the hallucination problem of large language models (LLMs). However, the difficulty of aligning the retriever with the diverse LLMs' knowledge preferences inevitably poses an inevitable challenge in developing a reliable RAG system. To address this issue, we propose DPA-RAG, a universal framework designed to align diverse knowledge preferences within RAG systems. Specifically, we initially introduce a preference knowledge construction pipline and incorporate five novel query augmentation strategies to alleviate preference data scarcity. Based on preference data, DPA-RAG accomplishes both external and internal preference alignment: 1) It jointly integrate pair-wise, point-wise, and contrastive preference alignment abilities into the reranker, achieving external preference alignment among RAG components. 2) It further introduces a pre-aligned stage before vanilla Supervised Fine-tuning (SFT), enabling LLMs to implicitly capture knowledge aligned with their reasoning preferences, achieving LLMs' internal alignment. Experimental results across four knowledge-intensive QA datasets demonstrate that DPA-RAG outperforms all baselines and seamlessly integrates both black-box and open-sourced LLM readers. Further qualitative analysis and discussions also provide empirical guidance for achieving reliable RAG systems. Our code is publicly available at https://github.com/dongguanting/DPA-RAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13237v1">LLM-Empowered State Representation for Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      Conventional state representations in reinforcement learning often omit critical task-related details, presenting a significant challenge for value networks in establishing accurate mappings from states to task rewards. Traditional methods typically depend on extensive sample learning to enrich state representations with task-specific information, which leads to low sample efficiency and high time costs. Recently, surging knowledgeable large language models (LLM) have provided promising substitutes for prior injection with minimal human intervention. Motivated by this, we propose LLM-Empowered State Representation (LESR), a novel approach that utilizes LLM to autonomously generate task-related state representation codes which help to enhance the continuity of network mappings and facilitate efficient training. Experimental results demonstrate LESR exhibits high sample efficiency and outperforms state-of-the-art baselines by an average of 29% in accumulated reward in Mujoco tasks and 30% in success rates in Gym-Robotics tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.06391v3">Teaching Code LLMs to Use Autocompletion Tools in Repository-Level Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      Code large language models (LLMs) face limitations in repository-level code generation due to their lack of awareness of repository-level dependencies (e.g., user-defined attributes), resulting in dependency errors such as undefined-variable and no-member errors. In this work, we introduce ToolGen, an approach that integrates autocompletion tools into the code LLM generation process to address these dependencies. ToolGen comprises two main phases: Trigger Insertion and Model Fine-tuning (Offline), and Tool-integrated Code Generation (Online). During the offline phase, ToolGen augments functions within a given code corpus with a special mark token, indicating positions to trigger autocompletion tools. These augmented functions, along with their corresponding docstrings, are then used to fine-tune a selected code LLM. In the online phase, ToolGen iteratively generates functions by predicting tokens step-by-step using the fine-tuned LLM. Whenever a mark token is encountered, ToolGen invokes the autocompletion tool to suggest code completions and selects the most appropriate one. We conduct comprehensive experiments to evaluate ToolGen's effectiveness in repository-level code generation. To facilitate this evaluation, we create a benchmark comprising 671 real-world code repositories and introduce two new dependency-based metrics: Dependency Coverage and Static Validity Rate. The results demonstrate that ToolGen significantly improves Dependency Coverage by 31.4% to 39.1% and Static Validity Rate by 44.9% to 57.7% across the three LLMs, while maintaining competitive or improved performance in widely recognized similarity metrics such as BLEU-4, CodeBLEU, Edit Similarity, and Exact Match. On the CoderEval dataset, ToolGen achieves improvements of 40.0% and 25.0% in Pass@1 for CodeT5 and CodeLlama, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.19094v2">Learning From Correctness Without Prompting Makes LLM Efficient Reasoner</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 Accepted to COLM 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated outstanding performance across various tasks, yet they still exhibit limitations such as hallucination, unfaithful reasoning, and toxic content. One potential approach to mitigate these issues is learning from human or external feedback (e.g. tools). In this paper, we introduce an intrinsic self-correct reasoning framework for LLMs that eliminates the need for human feedback, external tools, and handcraft prompts. The proposed framework, based on a multi-step reasoning paradigm \textbf{Le}arning from \textbf{Co}rrectness (\textsc{LeCo}), improves reasoning performance without needing to learn from errors. This paradigm prioritizes learning from correct reasoning steps, and a unique method to measure confidence for each reasoning step based on generation logits. Experimental results across various multi-step reasoning tasks demonstrate the effectiveness of the framework in improving reasoning performance with reduced token consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17371v3">Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 2 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) underscore their potential for responding to inquiries in various domains. However, ensuring that generative agents provide accurate and reliable answers remains an ongoing challenge. In this context, multi-agent debate (MAD) has emerged as a promising strategy for enhancing the truthfulness of LLMs. We benchmark a range of debating and prompting strategies to explore the trade-offs between cost, time, and accuracy. Importantly, we find that multi-agent debating systems, in their current form, do not reliably outperform other proposed prompting strategies, such as self-consistency and ensembling using multiple reasoning paths. However, when performing hyperparameter tuning, several MAD systems, such as Multi-Persona, perform better. This suggests that MAD protocols might not be inherently worse than other approaches, but that they are more sensitive to different hyperparameter settings and difficult to optimize. We build on these results to offer insights into improving debating strategies, such as adjusting agent agreement levels, which can significantly enhance performance and even surpass all other non-debate protocols we evaluated. We provide an open-source repository to the community with several state-of-the-art protocols together with evaluation scripts to benchmark across popular research datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13166v1">Using LLMs to Investigate Correlations of Conversational Follow-up Queries with User Satisfaction</a></div>
    <div class="paper-meta">
      📅 2024-07-18
      | 💬 Accepted to LLM4Eval @ SIGIR 2024 - The First Workshop on Large Language Models (LLMs) for Evaluation in Information Retrieval
    </div>
    <details class="paper-abstract">
      With large language models (LLMs), conversational search engines shift how users retrieve information from the web by enabling natural conversations to express their search intents over multiple turns. Users' natural conversation embodies rich but implicit signals of users' search intents and evaluation of search results to understand user experience with the system. However, it is underexplored how and why users ask follow-up queries to continue conversations with conversational search engines and how the follow-up queries signal users' satisfaction. From qualitative analysis of 250 conversational turns from an in-lab user evaluation of Naver Cue:, a commercial conversational search engine, we propose a taxonomy of 18 users' follow-up query patterns from conversational search, comprising two major axes: (1) users' motivations behind continuing conversations (N = 7) and (2) actions of follow-up queries (N = 11). Compared to the existing literature on query reformulations, we uncovered a new set of motivations and actions behind follow-up queries, including asking for subjective opinions or providing natural language feedback on the engine's responses. To analyze conversational search logs with our taxonomy in a scalable and efficient manner, we built an LLM-powered classifier (73% accuracy). With our classifier, we analyzed 2,061 conversational tuples collected from real-world usage logs of Cue: and examined how the conversation patterns from our taxonomy correlates with satisfaction. Our initial findings suggest some signals of dissatisfactions, such as Clarifying Queries, Excluding Condition, and Substituting Condition with follow-up queries. We envision our approach could contribute to automated evaluation of conversation search experience by providing satisfaction signals and grounds for realistic user simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13093v1">Using LLMs to Automate Threat Intelligence Analysis Workflows in Security Operation Centers</a></div>
    <div class="paper-meta">
      📅 2024-07-18
    </div>
    <details class="paper-abstract">
      SIEM systems are prevalent and play a critical role in a variety of analyst workflows in Security Operation Centers. However, modern SIEMs face a big challenge: they still cannot relieve analysts from the repetitive tasks involved in analyzing CTI (Cyber Threat Intelligence) reports written in natural languages. This project aims to develop an AI agent to replace the labor intensive repetitive tasks involved in analyzing CTI reports. The agent exploits the revolutionary capabilities of LLMs (e.g., GPT-4), but it does not require any human intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20244v1">Steamroller Problems: An Evaluation of LLM Reasoning Capability with Automated Theorem Prover Strategies</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      This study presents the first examination of the ability of Large Language Models (LLMs) to follow reasoning strategies that are used to guide Automated Theorem Provers (ATPs). We evaluate the performance of GPT4, GPT3.5 Turbo and Google's recent Gemini model on problems from a steamroller domain. In addition to determining accuracy we make use of the Natural Language Processing library spaCy to explore new methods of investigating LLM's reasoning capabilities. This led to one alarming result, the low correlation between correct reasoning and correct answers for any of the tested models. We found that the models' performance when using the ATP reasoning strategies was comparable to one-shot chain of thought and observe that attention to uncertainty in the accuracy results is critical when drawing conclusions about model performance. Consistent with previous speculation we confirm that LLMs have a preference for, and are best able to follow, bottom up reasoning processes. However, the reasoning strategies can still be beneficial for deriving small and relevant sets of formulas for external processing by a trusted inference engine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.13803v1">Less is More: Sparse Watermarking in LLMs with Enhanced Text Quality</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs), concerns about potential misuse have emerged. To this end, watermarking has been adapted to LLM, enabling a simple and effective way to detect and monitor generated text. However, while the existing methods can differentiate between watermarked and unwatermarked text with high accuracy, they often face a trade-off between the quality of the generated text and the effectiveness of the watermarking process. In this work, we present a novel type of LLM watermark, Sparse Watermark, which aims to mitigate this trade-off by applying watermarks to a small subset of generated tokens distributed across the text. The key strategy involves anchoring watermarked tokens to words that have specific Part-of-Speech (POS) tags. Our experimental results demonstrate that the proposed watermarking scheme achieves high detectability while generating text that outperforms previous LLM watermarking methods in quality across various tasks
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12784v1">AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 22 pages, 13 figures, 7 tables
    </div>
    <details class="paper-abstract">
      LLM agents have demonstrated remarkable performance across various applications, primarily due to their advanced capabilities in reasoning, utilizing external knowledge and tools, calling APIs, and executing actions to interact with environments. Current agents typically utilize a memory module or a retrieval-augmented generation (RAG) mechanism, retrieving past knowledge and instances with similar embeddings from knowledge bases to inform task planning and execution. However, the reliance on unverified knowledge bases raises significant concerns about their safety and trustworthiness. To uncover such vulnerabilities, we propose a novel red teaming approach AgentPoison, the first backdoor attack targeting generic and RAG-based LLM agents by poisoning their long-term memory or RAG knowledge base. In particular, we form the trigger generation process as a constrained optimization to optimize backdoor triggers by mapping the triggered instances to a unique embedding space, so as to ensure that whenever a user instruction contains the optimized backdoor trigger, the malicious demonstrations are retrieved from the poisoned memory or knowledge base with high probability. In the meantime, benign instructions without the trigger will still maintain normal performance. Unlike conventional backdoor attacks, AgentPoison requires no additional model training or fine-tuning, and the optimized backdoor trigger exhibits superior transferability, in-context coherence, and stealthiness. Extensive experiments demonstrate AgentPoison's effectiveness in attacking three types of real-world LLM agents: RAG-based autonomous driving agent, knowledge-intensive QA agent, and healthcare EHRAgent. On each agent, AgentPoison achieves an average attack success rate higher than 80% with minimal impact on benign performance (less than 1%) with a poison rate less than 0.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12899v1">DreamStory: Open-Domain Story Visualization by LLM-Guided Multi-Subject Consistent Diffusion</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      Story visualization aims to create visually compelling images or videos corresponding to textual narratives. Despite recent advances in diffusion models yielding promising results, existing methods still struggle to create a coherent sequence of subject-consistent frames based solely on a story. To this end, we propose DreamStory, an automatic open-domain story visualization framework by leveraging the LLMs and a novel multi-subject consistent diffusion model. DreamStory consists of (1) an LLM acting as a story director and (2) an innovative Multi-Subject consistent Diffusion model (MSD) for generating consistent multi-subject across the images. First, DreamStory employs the LLM to generate descriptive prompts for subjects and scenes aligned with the story, annotating each scene's subjects for subsequent subject-consistent generation. Second, DreamStory utilizes these detailed subject descriptions to create portraits of the subjects, with these portraits and their corresponding textual information serving as multimodal anchors (guidance). Finally, the MSD uses these multimodal anchors to generate story scenes with consistent multi-subject. Specifically, the MSD includes Masked Mutual Self-Attention (MMSA) and Masked Mutual Cross-Attention (MMCA) modules. MMSA and MMCA modules ensure appearance and semantic consistency with reference images and text, respectively. Both modules employ masking mechanisms to prevent subject blending. To validate our approach and promote progress in story visualization, we established a benchmark, DS-500, which can assess the overall performance of the story visualization framework, subject-identification accuracy, and the consistency of the generation model. Extensive experiments validate the effectiveness of DreamStory in both subjective and objective evaluations. Please visit our project homepage at https://dream-xyz.github.io/dreamstory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12734v1">A LLM Benchmark based on the Minecraft Builder Dialog Agent Task</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      In this work we proposing adapting the Minecraft builder task into an LLM benchmark suitable for evaluating LLM ability in spatially orientated tasks, and informing builder agent design. Previous works have proposed corpora with varying complex structures, and human written instructions. We instead attempt to provide a comprehensive synthetic benchmark for testing builder agents over a series of distinct tasks that comprise of common building operations. We believe this approach allows us to probe specific strengths and weaknesses of different agents, and test the ability of LLMs in the challenging area of spatial reasoning and vector based math.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02543v2">To Believe or Not to Believe Your LLM</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      We explore uncertainty quantification in large language models (LLMs), with the goal to identify when uncertainty in responses given a query is large. We simultaneously consider both epistemic and aleatoric uncertainties, where the former comes from the lack of knowledge about the ground truth (such as about facts or the language), and the latter comes from irreducible randomness (such as multiple possible answers). In particular, we derive an information-theoretic metric that allows to reliably detect when only epistemic uncertainty is large, in which case the output of the model is unreliable. This condition can be computed based solely on the output of the model obtained simply by some special iterative prompting based on the previous responses. Such quantification, for instance, allows to detect hallucinations (cases when epistemic uncertainty is high) in both single- and multi-answer responses. This is in contrast to many standard uncertainty quantification strategies (such as thresholding the log-likelihood of a response) where hallucinations in the multi-answer case cannot be detected. We conduct a series of experiments which demonstrate the advantage of our formulation. Further, our investigations shed some light on how the probabilities assigned to a given output by an LLM can be amplified by iterative prompting, which might be of independent interest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12579v1">The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 Accepted by ECCV 2024
    </div>
    <details class="paper-abstract">
      In spite of recent advancements in text-to-image generation, limitations persist in handling complex and imaginative prompts due to the restricted diversity and complexity of training data. This work explores how diffusion models can generate images from prompts requiring artistic creativity or specialized knowledge. We introduce the Realistic-Fantasy Benchmark (RFBench), a novel evaluation framework blending realistic and fantastical scenarios. To address these challenges, we propose the Realistic-Fantasy Network (RFNet), a training-free approach integrating diffusion models with LLMs. Extensive human evaluations and GPT-based compositional assessments demonstrate our approach's superiority over state-of-the-art methods. Our code and dataset is available at https://leo81005.github.io/Reality-and-Fantasy/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.16844v3">Think Big, Generate Quick: LLM-to-SLM for Fast Autoregressive Decoding</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 Work presented at the ES-FoMo II Workshop at ICML 2024
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become ubiquitous in practice and are widely used for generation tasks such as translation, summarization and instruction following. However, their enormous size and reliance on autoregressive decoding increase deployment costs and complicate their use in latency-critical applications. In this work, we propose a hybrid approach that combines language models of different sizes to increase the efficiency of autoregressive decoding while maintaining high performance. Our method utilizes a pretrained frozen LLM that encodes all prompt tokens once in parallel, and uses the resulting representations to condition and guide a small language model (SLM), which then generates the response more efficiently. We investigate the combination of encoder-decoder LLMs with both encoder-decoder and decoder-only SLMs from different model families and only require fine-tuning of the SLM. Experiments with various benchmarks show substantial speedups of up to $4\times$, with minor performance penalties of $1-2\%$ for translation and summarization tasks compared to the LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.09308v2">Enabling Waypoint Generation for Collaborative Robots using LLMs and Mixed Reality</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 Published in VLMNM 2024 - Workshop, ICRA 2024
    </div>
    <details class="paper-abstract">
      Programming a robotic is a complex task, as it demands the user to have a good command of specific programming languages and awareness of the robot's physical constraints. We propose a framework that simplifies robot deployment by allowing direct communication using natural language. It uses large language models (LLM) for prompt processing, workspace understanding, and waypoint generation. It also employs Augmented Reality (AR) to provide visual feedback of the planned outcome. We showcase the effectiveness of our framework with a simple pick-and-place task, which we implement on a real robot. Moreover, we present an early concept of expressive robot behavior and skill generation that can be used to communicate with the user and learn new skills (e.g., object grasping).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07243v3">MBBQ: A Dataset for Cross-Lingual Comparison of Stereotypes in Generative LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 Accepted to COLM 2024
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have been shown to exhibit harmful biases and stereotypes. While safety fine-tuning typically takes place in English, if at all, these models are being used by speakers of many different languages. There is existing evidence that the performance of these models is inconsistent across languages and that they discriminate based on demographic factors of the user. Motivated by this, we investigate whether the social stereotypes exhibited by LLMs differ as a function of the language used to prompt them, while controlling for cultural differences and task accuracy. To this end, we present MBBQ (Multilingual Bias Benchmark for Question-answering), a carefully curated version of the English BBQ dataset extended to Dutch, Spanish, and Turkish, which measures stereotypes commonly held across these languages. We further complement MBBQ with a parallel control dataset to measure task performance on the question-answering task independently of bias. Our results based on several open-source and proprietary LLMs confirm that some non-English languages suffer from bias more than English, even when controlling for cultural shifts. Moreover, we observe significant cross-lingual differences in bias behaviour for all except the most accurate models. With the release of MBBQ, we hope to encourage further research on bias in multilingual settings. The dataset and code are available at https://github.com/Veranep/MBBQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12391v1">LLM Inference Serving: Survey of Recent Advances and Opportunities</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      This survey offers a comprehensive overview of recent advancements in Large Language Model (LLM) serving systems, focusing on research since the year 2023. We specifically examine system-level enhancements that improve performance and efficiency without altering the core LLM decoding mechanisms. By selecting and reviewing high-quality papers from prestigious ML and system venues, we highlight key innovations and practical considerations for deploying and scaling LLMs in real-world production environments. This survey serves as a valuable resource for LLM practitioners seeking to stay abreast of the latest developments in this rapidly evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10887v2">Hey, That's My Model! Introducing Chain & Hash, An LLM Fingerprinting Technique</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      Amid growing concerns over the ease of theft and misuse of Large Language Models (LLMs), the need for fingerprinting models has increased. Fingerprinting, in this context, means that the model owner can link a given model to their original version, thereby identifying if their model is being misused or has been completely stolen. In this paper, we first define a set five properties a successful fingerprint should satisfy; namely, the fingerprint should be Transparent, Efficient, Persistent, Robust, and Unforgeable. Next, we propose Chain & Hash, a new, simple fingerprinting approach that implements a fingerprint with a cryptographic flavor, achieving all these properties. Chain & Hash involves generating a set of questions (the fingerprints) along with a set of potential answers. These elements are hashed together using a secure hashing technique to select the value for each question, hence providing an unforgeability property-preventing adversaries from claiming false ownership. We evaluate the Chain & Hash technique on multiple models and demonstrate its robustness against benign transformations, such as fine-tuning on different datasets, and adversarial attempts to erase the fingerprint. Finally, our experiments demonstrate the efficiency of implementing Chain & Hash and its utility, where fingerprinted models achieve almost the same performance as non-fingerprinted ones across different benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01343v4">CHOPS: CHat with custOmer Profile Systems for Customer Service with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 Accepted by COLM 2024
    </div>
    <details class="paper-abstract">
      Businesses and software platforms are increasingly turning to Large Language Models (LLMs) such as GPT-3.5, GPT-4, GLM-3, and LLaMa-2 for chat assistance with file access or as reasoning agents for customer service. However, current LLM-based customer service models have limited integration with customer profiles and lack the operational capabilities necessary for effective service. Moreover, existing API integrations emphasize diversity over the precision and error avoidance essential in real-world customer service scenarios. To address these issues, we propose an LLM agent named CHOPS (CHat with custOmer Profile in existing System), designed to: (1) efficiently utilize existing databases or systems for accessing user information or interacting with these systems following existing guidelines; (2) provide accurate and reasonable responses or carry out required operations in the system while avoiding harmful operations; and (3) leverage a combination of small and large LLMs to achieve satisfying performance at a reasonable inference cost. We introduce a practical dataset, the CPHOS-dataset, which includes a database, guiding files, and QA pairs collected from CPHOS, an online platform that facilitates the organization of simulated Physics Olympiads for high school teachers and students. We have conducted extensive experiments to validate the performance of our proposed CHOPS architecture using the CPHOS-dataset, with the aim of demonstrating how LLMs can enhance or serve as alternatives to human customer service. Code for our proposed architecture and dataset can be found at {https://github.com/JingzheShi/CHOPS}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12352v1">SENTAUR: Security EnhaNced Trojan Assessment Using LLMs Against Undesirable Revisions</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      A globally distributed IC supply chain brings risks due to untrusted third parties. The risks span inadvertent use of hardware Trojan (HT), inserted Intellectual Property (3P-IP) or Electronic Design Automation (EDA) flows. HT can introduce stealthy HT behavior, prevent an IC work as intended, or leak sensitive data via side channels. To counter HTs, rapidly examining HT scenarios is a key requirement. While Trust-Hub benchmarks are a good starting point to assess defenses, they encompass a small subset of manually created HTs within the expanse of HT designs. Further, the HTs may disappear during synthesis. We propose a large language model (LLM) framework SENTAUR to generate a suite of legitimate HTs for a Register Transfer Level (RTL) design by learning its specifications, descriptions, and natural language descriptions of HT effects. Existing tools and benchmarks are limited; they need a learning period to construct an ML model to mimic the threat model and are difficult to reproduce. SENTAUR can swiftly produce HT instances by leveraging LLMs without any learning period and sanitizing the HTs facilitating their rapid assessment. Evaluation of SENTAUR involved generating effective, synthesizable, and practical HTs from TrustHub and elsewhere, investigating impacts of payloads/triggers at the RTL. While our evaluation focused on HT insertion, SENTAUR can generalize to automatically transform an RTL code to have defined functional modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12344v1">The Better Angels of Machine Personality: How Personality Relates to LLM Safety</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      Personality psychologists have analyzed the relationship between personality and safety behaviors in human society. Although Large Language Models (LLMs) demonstrate personality traits, the relationship between personality traits and safety abilities in LLMs still remains a mystery. In this paper, we discover that LLMs' personality traits are closely related to their safety abilities, i.e., toxicity, privacy, and fairness, based on the reliable MBTI-M scale. Meanwhile, the safety alignment generally increases various LLMs' Extraversion, Sensing, and Judging traits. According to such findings, we can edit LLMs' personality traits and improve their safety performance, e.g., inducing personality from ISTJ to ISTP resulted in a relative improvement of approximately 43% and 10% in privacy and fairness performance, respectively. Additionally, we find that LLMs with different personality traits are differentially susceptible to jailbreak. This study pioneers the investigation of LLM safety from a personality perspective, providing new insights into LLM safety enhancement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12341v1">LLM-based query paraphrasing for video search</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      Text-to-video retrieval answers user queries through search by concepts and embeddings. Limited by the size of the concept bank and the amount of training data, answering queries in the wild is not always effective due to the out-of-vocabulary problem. Furthermore, neither concept-based nor embedding-based search can perform reasoning to consolidate the search results for complex queries mixed with logical and spatial constraints. To address these problems, we leverage large language models (LLM) to paraphrase the query by text-to-text (T2T), text-to-image (T2I), and image-to-text (I2T) transformations. These transformations rephrase abstract concepts into simple words to address the out-of-vocabulary problem. Furthermore, the complex relationship in a query can be decoupled into simpler sub-queries, yielding better retrieval performance when fusing the search results of these sub-queries. To address the LLM hallucination problem, this paper also proposes a novel consistency-based verification strategy to filter the paraphrased queries that are factually incorrect. Extensive experiments are conducted for ad-hoc video search and known-item search on the TRECVid datasets. We provide empirical insights into how traditionally difficult-to-answer queries can be resolved by query paraphrasing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.13062v5">Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study</a></div>
    <div class="paper-meta">
      📅 2024-07-17
      | 💬 This paper has been accepted as a full paper at WSDM 2024. Explore the MS research blog of our work at https://www.microsoft.com/en-us/research/blog/improving-llm-understanding-of-structured-data-and-exploring-advanced-prompting-methods/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming attractive as few-shot reasoners to solve Natural Language (NL)-related tasks. However, the understanding of their capability to process structured data like tables remains an under-explored area. While tables can be serialized as input for LLMs, there is a lack of comprehensive studies on whether LLMs genuinely comprehend this data. In this paper, we try to understand this by designing a benchmark to evaluate the structural understanding capabilities of LLMs through seven distinct tasks, e.g., cell lookup, row retrieval and size detection. Specially, we perform a series of evaluations on the recent most advanced LLM models, GPT-3.5 and GPT-4 and observe that performance varied with different input choices, including table input format, content order, role prompting, and partition marks. Drawing from the insights gained through the benchmark evaluations, we propose $\textit{self-augmentation}$ for effective structural prompting, such as critical value / range identification using internal knowledge of LLMs. When combined with carefully chosen input choices, these structural prompting methods lead to promising improvements in LLM performance on a variety of tabular tasks, e.g., TabFact($\uparrow2.31\%$), HybridQA($\uparrow2.13\%$), SQA($\uparrow2.72\%$), Feverous($\uparrow0.84\%$), and ToTTo($\uparrow5.68\%$). We believe that our open source benchmark and proposed prompting methods can serve as a simple yet generic selection for future research. The code and data of this paper will be temporality released at https://anonymous.4open.science/r/StructuredLLM-76F3/README.md and will be replaced with an official one at https://github.com/microsoft/TableProvider later.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04066v2">VoicePilot: Harnessing LLMs as Speech Interfaces for Physically Assistive Robots</a></div>
    <div class="paper-meta">
      📅 2024-07-17
    </div>
    <details class="paper-abstract">
      Physically assistive robots present an opportunity to significantly increase the well-being and independence of individuals with motor impairments or other forms of disability who are unable to complete activities of daily living. Speech interfaces, especially ones that utilize Large Language Models (LLMs), can enable individuals to effectively and naturally communicate high-level commands and nuanced preferences to robots. Frameworks for integrating LLMs as interfaces to robots for high level task planning and code generation have been proposed, but fail to incorporate human-centric considerations which are essential while developing assistive interfaces. In this work, we present a framework for incorporating LLMs as speech interfaces for physically assistive robots, constructed iteratively with 3 stages of testing involving a feeding robot, culminating in an evaluation with 11 older adults at an independent living facility. We use both quantitative and qualitative data from the final study to validate our framework and additionally provide design guidelines for using LLMs as speech interfaces for assistive robots. Videos and supporting files are located on our project website: https://sites.google.com/andrew.cmu.edu/voicepilot/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12886v1">Whitening Not Recommended for Classification Tasks in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-16
    </div>
    <details class="paper-abstract">
      Sentence embedding is a cornerstone in NLP. Whitening has been claimed to be an effective operation to improve embedding quality obtained from Large Language Models (LLMs). However, we find that the efficacy of whitening is model-dependent and task-dependent. In particular, whitening degenerates embeddings for classification tasks. The conclusion is supported by extensive experiments. We also explored a variety of whitening operations, including PCA, ZCA, PCA-Cor, ZCA-Cor and Cholesky whitenings. A by-product of our research is embedding evaluation platform for LLMs called SentEval+.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11963v1">NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?</a></div>
    <div class="paper-meta">
      📅 2024-07-16
    </div>
    <details class="paper-abstract">
      In evaluating the long-context capabilities of large language models (LLMs), identifying content relevant to a user's query from original long documents is a crucial prerequisite for any LLM to answer questions based on long text. We present NeedleBench, a framework consisting of a series of progressively more challenging tasks for assessing bilingual long-context capabilities, spanning multiple length intervals (4k, 8k, 32k, 128k, 200k, 1000k, and beyond) and different depth ranges, allowing the strategic insertion of critical data points in different text depth zones to rigorously test the retrieval and reasoning capabilities of models in diverse contexts. We use the NeedleBench framework to assess how well the leading open-source models can identify key information relevant to the question and apply that information to reasoning in bilingual long texts. Furthermore, we propose the Ancestral Trace Challenge (ATC) to mimic the complexity of logical reasoning challenges that are likely to be present in real-world long-context tasks, providing a simple method for evaluating LLMs in dealing with complex long-context situations. Our results suggest that current LLMs have significant room for improvement in practical long-context applications, as they struggle with the complexity of logical reasoning challenges that are likely to be present in real-world long-context tasks. All codes and resources are available at OpenCompass: https://github.com/open-compass/opencompass.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11919v1">What's Wrong? Refining Meeting Summaries with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2024-07-16
    </div>
    <details class="paper-abstract">
      Meeting summarization has become a critical task since digital encounters have become a common practice. Large language models (LLMs) show great potential in summarization, offering enhanced coherence and context understanding compared to traditional methods. However, they still struggle to maintain relevance and avoid hallucination. We introduce a multi-LLM correction approach for meeting summarization using a two-phase process that mimics the human review process: mistake identification and summary refinement. We release QMSum Mistake, a dataset of 200 automatically generated meeting summaries annotated by humans on nine error types, including structural, omission, and irrelevance errors. Our experiments show that these errors can be identified with high accuracy by an LLM. We transform identified mistakes into actionable feedback to improve the quality of a given summary measured by relevance, informativeness, conciseness, and coherence. This post-hoc refinement effectively improves summary quality by leveraging multiple LLMs to validate output quality. Our multi-LLM approach for meeting summarization shows potential for similar complex text generation tasks requiring robustness, action planning, and discussion towards a goal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05506v2">Towards a Benchmark for Causal Business Process Reasoning with LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for boosting organizational efficiency and automating tasks. While not originally designed for complex cognitive processes, recent efforts have further extended to employ LLMs in activities such as reasoning, planning, and decision-making. In business processes, such abilities could be invaluable for leveraging on the massive corpora LLMs have been trained on for gaining deep understanding of such processes. In this work, we plant the seeds for the development of a benchmark to assess the ability of LLMs to reason about causal and process perspectives of business operations. We refer to this view as Causally-augmented Business Processes (BP^C). The core of the benchmark comprises a set of BP^C related situations, a set of questions about these situations, and a set of deductive rules employed to systematically resolve the ground truth answers to these questions. Also with the power of LLMs, the seed is then instantiated into a larger-scale set of domain-specific situations and questions. Reasoning on BP^C is of crucial importance for process interventions and process improvement. Our benchmark, accessible at https://huggingface.co/datasets/ibm/BPC, can be used in one of two possible modalities: testing the performance of any target LLM and training an LLM to advance its capability to reason about BP^C.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19802v3">Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models</a></div>
    <div class="paper-meta">
      📅 2024-07-16
    </div>
    <details class="paper-abstract">
      Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.08426v3">Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2024-07-16
    </div>
    <details class="paper-abstract">
      Generating accurate SQL from natural language questions (text-to-SQL) is a long-standing challenge due to the complexities in user question understanding, database schema comprehension, and SQL generation. Conventional text-to-SQL systems, comprising human engineering and deep neural networks, have made substantial progress. Subsequently, pre-trained language models (PLMs) have been developed and utilized for text-to-SQL tasks, achieving promising performance. As modern databases become more complex, the corresponding user questions also grow more challenging, causing PLMs with parameter constraints to produce incorrect SQL. This necessitates more sophisticated and tailored optimization methods, which, in turn, restricts the applications of PLM-based systems. Recently, large language models (LLMs) have demonstrated significant capabilities in natural language understanding as the model scale increases. Therefore, integrating LLM-based implementation can bring unique opportunities, improvements, and solutions to text-to-SQL research. In this survey, we present a comprehensive review of LLM-based text-to-SQL. Specifically, we propose a brief overview of the technical challenges and the evolutionary process of text-to-SQL. Then, we provide a detailed introduction to the datasets and metrics designed to evaluate text-to-SQL systems. After that, we present a systematic analysis of recent advances in LLM-based text-to-SQL. Finally, we discuss the remaining challenges in this field and propose expectations for future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11421v1">States Hidden in Hidden States: LLMs Emerge Discrete State Representations Implicitly</a></div>
    <div class="paper-meta">
      📅 2024-07-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit various emergent abilities. Among these abilities, some might reveal the internal working mechanisms of models. In this paper, we uncover a novel emergent capability in models: the intrinsic ability to perform extended sequences of calculations without relying on chain-of-thought step-by-step solutions. Remarkably, the most advanced models can directly output the results of two-digit number additions with lengths extending up to 15 addends. We hypothesize that the model emerges Implicit Discrete State Representations (IDSRs) within its hidden states and performs symbolic calculations internally. To test this hypothesis, we design a sequence of experiments that look into the hidden states. Specifically, we first confirm that IDSRs exist. Then, we provide interesting observations about the formation of IDSRs from layer, digit, and sequence perspectives. Finally, we confirm that models indeed use IDSRs to produce the final answers. However, we also discover that these state representations are far from lossless in current open-sourced models, leading to inaccuracies in their final performance. Our work presents a novel exploration of LLMs' symbolic calculation abilities and the underlying mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11368v1">Ancient Korean Archive Translation: Comparison Analysis on Statistical phrase alignment, LLM in-context learning, and inter-methodological approach</a></div>
    <div class="paper-meta">
      📅 2024-07-16
      | 💬 ACL2024 submitted
    </div>
    <details class="paper-abstract">
      This study aims to compare three methods for translating ancient texts with sparse corpora: (1) the traditional statistical translation method of phrase alignment, (2) in-context LLM learning, and (3) proposed inter methodological approach - statistical machine translation method using sentence piece tokens derived from unified set of source-target corpus. The performance of the proposed approach in this study is 36.71 in BLEU score, surpassing the scores of SOLAR-10.7B context learning and the best existing Seq2Seq model. Further analysis and discussion are presented.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11240v1">Making New Connections: LLMs as Puzzle Generators for The New York Times' Connections Word Game</a></div>
    <div class="paper-meta">
      📅 2024-07-15
    </div>
    <details class="paper-abstract">
      The Connections puzzle is a word association game published daily by The New York Times (NYT). In this game, players are asked to find groups of four words that are connected by a common theme. While solving a given Connections puzzle requires both semantic knowledge and abstract reasoning, generating novel puzzles additionally requires a form of metacognition: generators must be able to accurately model the downstream reasoning of potential solvers. In this paper, we investigate the ability of the GPT family of Large Language Models (LLMs) to generate challenging and creative word games for human players. We start with an analysis of the word game Connections and the unique challenges it poses as a Procedural Content Generation (PCG) domain. We then propose a method for generating Connections puzzles using LLMs by adapting a Tree of Thoughts (ToT) prompting approach. We evaluate this method by conducting a user study, asking human players to compare AI-generated puzzles against published Connections puzzles. Our findings show that LLMs are capable puzzle creators, and can generate diverse sets of enjoyable, challenging, and creative Connections puzzles as judged by human users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.11198v1">A Framework For Discussing LLMs as Tools for Qualitative Analysis</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 4 pages, 1 table. Presented at the "LLMs as Research Tools" workshop at CHI 2024 (https://dl.acm.org/doi/10.1145/3613905.3636301)
    </div>
    <details class="paper-abstract">
      We review discourses about the philosophy of science in qualitative research and evidence from cognitive linguistics in order to ground a framework for discussing the use of Large Language Models (LLMs) to support the qualitative analysis process. This framework involves asking two key questions: "is the LLM proposing or refuting a qualitative model?" and "is the human researcher checking the LLM's decision-making directly?". We then discuss an implication of this framework: that using LLMs to surface counter-examples for human review represents a promising space for the adoption of LLMs into the qualitative research process. This space is promising because it is a site of overlap between researchers working from a variety of philosophical assumptions, enabling productive cross-paradigm collaboration on tools and practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10899v1">Leveraging LLM-Respondents for Item Evaluation: a Psychometric Analysis</a></div>
    <div class="paper-meta">
      📅 2024-07-15
    </div>
    <details class="paper-abstract">
      Effective educational measurement relies heavily on the curation of well-designed item pools (i.e., possessing the right psychometric properties). However, item calibration is time-consuming and costly, requiring a sufficient number of respondents for the response process. We explore using six different LLMs (GPT-3.5, GPT-4, Llama 2, Llama 3, Gemini-Pro, and Cohere Command R Plus) and various combinations of them using sampling methods to produce responses with psychometric properties similar to human answers. Results show that some LLMs have comparable or higher proficiency in College Algebra than college students. No single LLM mimics human respondents due to narrow proficiency distributions, but an ensemble of LLMs can better resemble college students' ability distribution. The item parameters calibrated by LLM-Respondents have high correlations (e.g. > 0.8 for GPT-3.5) compared to their human calibrated counterparts, and closely resemble the parameters of the human subset (e.g. 0.02 Spearman correlation difference). Several augmentation strategies are evaluated for their relative performance, with resampling methods proving most effective, enhancing the Spearman correlation from 0.89 (human only) to 0.93 (augmented human).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.12128v2">DiagrammerGPT: Generating Open-Domain, Open-Platform Diagrams via LLM Planning</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 COLM 2024; Project page: https://diagrammerGPT.github.io/
    </div>
    <details class="paper-abstract">
      Text-to-image (T2I) generation has seen significant growth over the past few years. Despite this, there has been little work on generating diagrams with T2I models. A diagram is a symbolic/schematic representation that explains information using structurally rich and spatially complex visualizations (e.g., a dense combination of related objects, text labels, directional arrows/lines, etc.). Existing state-of-the-art T2I models often fail at diagram generation because they lack fine-grained object layout control when many objects are densely connected via complex relations such as arrows/lines, and also often fail to render comprehensible text labels. To address this gap, we present DiagrammerGPT, a novel two-stage text-to-diagram generation framework leveraging the layout guidance capabilities of LLMs to generate more accurate diagrams. In the first stage, we use LLMs to generate and iteratively refine 'diagram plans' (in a planner-auditor feedback loop). In the second stage, we use a diagram generator, DiagramGLIGEN, and a text label rendering module to generate diagrams (with clear text labels) following the diagram plans. To benchmark the text-to-diagram generation task, we introduce AI2D-Caption, a densely annotated diagram dataset built on top of the AI2D dataset. We show that our DiagrammerGPT framework produces more accurate diagrams, outperforming existing T2I models. We also provide comprehensive analysis, including open-domain diagram generation, multi-platform vector graphic diagram generation, human-in-the-loop editing, and multimodal planner/auditor LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10793v1">GraphEval: A Knowledge-Graph Based LLM Hallucination Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 12 pages, to be published at KiL'24: Workshop on Knowledge-infused Learning co-located with 30th ACM KDD Conference, August 26, 2024, Barcelona, Spain
    </div>
    <details class="paper-abstract">
      Methods to evaluate Large Language Model (LLM) responses and detect inconsistencies, also known as hallucinations, with respect to the provided knowledge, are becoming increasingly important for LLM applications. Current metrics fall short in their ability to provide explainable decisions, systematically check all pieces of information in the response, and are often too computationally expensive to be used in practice. We present GraphEval: a hallucination evaluation framework based on representing information in Knowledge Graph (KG) structures. Our method identifies the specific triples in the KG that are prone to hallucinations and hence provides more insight into where in the response a hallucination has occurred, if at all, than previous methods. Furthermore, using our approach in conjunction with state-of-the-art natural language inference (NLI) models leads to an improvement in balanced accuracy on various hallucination benchmarks, compared to using the raw NLI models. Lastly, we explore the use of GraphEval for hallucination correction by leveraging the structure of the KG, a method we name GraphCorrect, and demonstrate that the majority of hallucinations can indeed be rectified.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10725v1">CLAVE: An Adaptive Framework for Evaluating Values of LLM Generated Responses</a></div>
    <div class="paper-meta">
      📅 2024-07-15
    </div>
    <details class="paper-abstract">
      The rapid progress in Large Language Models (LLMs) poses potential risks such as generating unethical content. Assessing LLMs' values can help expose their misalignment, but relies on reference-free evaluators, e.g., fine-tuned LLMs or close-source ones like GPT-4, to identify values reflected in generated responses. Nevertheless, these evaluators face two challenges in open-ended value evaluation: they should align with changing human value definitions with minimal annotation, against their own bias (adaptability), and detect varying value expressions and scenarios robustly (generalizability). To handle these challenges, we introduce CLAVE, a novel framework which integrates two complementary LLMs, a large one to extract high-level value concepts from a few human labels, leveraging its extensive knowledge and generalizability, and a smaller one fine-tuned on such concepts to better align with human value understanding. This dual-model approach enables calibration with any value systems using <100 human-labeled samples per value type. Then we present ValEval, a comprehensive dataset comprising 13k+ (text,value,label) tuples across diverse domains, covering three major value systems. We benchmark the capabilities of 12+ popular LLM evaluators and analyze their strengths and weaknesses. Our findings reveal that combining fine-tuned small models and prompt-based large ones serves as a superior balance in value evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10701v1">DOCBENCH: A Benchmark for Evaluating LLM-based Document Reading Systems</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Recently, there has been a growing interest among large language model (LLM) developers in LLM-based document reading systems, which enable users to upload their own documents and pose questions related to the document contents, going beyond simple reading comprehension tasks. Consequently, these systems have been carefully designed to tackle challenges such as file parsing, metadata extraction, multi-modal information understanding and long-context reading. However, no current benchmark exists to evaluate their performance in such scenarios, where a raw file and questions are provided as input, and a corresponding response is expected as output. In this paper, we introduce DocBench, a new benchmark designed to evaluate LLM-based document reading systems. Our benchmark involves a meticulously crafted process, including the recruitment of human annotators and the generation of synthetic questions. It includes 229 real documents and 1,102 questions, spanning across five different domains and four major types of questions. We evaluate both proprietary LLM-based systems accessible via web interfaces or APIs, and a parse-then-read pipeline employing open-source LLMs. Our evaluations reveal noticeable gaps between existing LLM-based document reading systems and human performance, underscoring the challenges of developing proficient systems. To summarize, DocBench aims to establish a standardized benchmark for evaluating LLM-based document reading systems under diverse real-world scenarios, thereby guiding future advancements in this research area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.10707v2">Discovering Latent Themes in Social Media Messaging: A Machine-in-the-Loop Approach Integrating LLMs</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 Accepted at 19th International AAAI Conference on Web and Social Media (ICWSM-2025)
    </div>
    <details class="paper-abstract">
      Grasping the themes of social media content is key to understanding the narratives that influence public opinion and behavior. The thematic analysis goes beyond traditional topic-level analysis, which often captures only the broadest patterns, providing deeper insights into specific and actionable themes such as "public sentiment towards vaccination", "political discourse surrounding climate policies," etc. In this paper, we introduce a novel approach to uncovering latent themes in social media messaging. Recognizing the limitations of the traditional topic-level analysis, which tends to capture only overarching patterns, this study emphasizes the need for a finer-grained, theme-focused exploration. Traditional theme discovery methods typically involve manual processes and a human-in-the-loop approach. While valuable, these methods face challenges in scalability, consistency, and resource intensity in terms of time and cost. To address these challenges, we propose a machine-in-the-loop approach that leverages the advanced capabilities of Large Language Models (LLMs). To demonstrate our approach, we apply our framework to contentious topics, such as climate debate and vaccine debate. We use two publicly available datasets: (1) the climate campaigns dataset of 21k Facebook ads and (2) the COVID-19 vaccine campaigns dataset of 9k Facebook ads. Our quantitative and qualitative analysis shows that our methodology yields more accurate and interpretable results compared to the baselines. Our results not only demonstrate the effectiveness of our approach in uncovering latent themes but also illuminate how these themes are tailored for demographic targeting in social media contexts. Additionally, our work sheds light on the dynamic nature of social media, revealing the shifts in the thematic focus of messaging in response to real-world events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10652v1">Cutting Through the Clutter: The Potential of LLMs for Efficient Filtration in Systematic Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 5 pages, 5 figures, 1 table
    </div>
    <details class="paper-abstract">
      In academic research, systematic literature reviews are foundational and highly relevant, yet tedious to create due to the high volume of publications and labor-intensive processes involved. Systematic selection of relevant papers through conventional means like keyword-based filtering techniques can sometimes be inadequate, plagued by semantic ambiguities and inconsistent terminology, which can lead to sub-optimal outcomes. To mitigate the required extensive manual filtering, we explore and evaluate the potential of using Large Language Models (LLMs) to enhance the efficiency, speed, and precision of literature review filtering, reducing the amount of manual screening required. By using models as classification agents acting on a structured database only, we prevent common problems inherent in LLMs, such as hallucinations. We evaluate the real-world performance of such a setup during the construction of a recent literature survey paper with initially more than 8.3k potentially relevant articles under consideration and compare this with human performance on the same dataset. Our findings indicate that employing advanced LLMs like GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Flash, or Llama3 with simple prompting can significantly reduce the time required for literature filtering - from usually weeks of manual research to only a few minutes. Simultaneously, we crucially show that false negatives can indeed be controlled through a consensus scheme, achieving recalls >98.8% at or even beyond the typical human error threshold, thereby also providing for more accurate and relevant articles selected. Our research not only demonstrates a substantial improvement in the methodology of literature reviews but also sets the stage for further integration and extensive future applications of responsible AI in academic research practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10627v1">Arena Learning: Build Data Flywheel for LLMs Post-training via Simulated Chatbot Arena</a></div>
    <div class="paper-meta">
      📅 2024-07-15
    </div>
    <details class="paper-abstract">
      Assessing the effectiveness of large language models (LLMs) presents substantial challenges. The method of conducting human-annotated battles in an online Chatbot Arena is a highly effective evaluative technique. However, this approach is limited by the costs and time required for human annotation. In this paper, we introduce Arena Learning, an innovative offline strategy designed to simulate these arena battles using AI-driven annotations to evaluate battle outcomes, thus facilitating the continuous improvement of the target model through both supervised fine-tuning and reinforcement learning. Arena Learning comprises two key elements. First, it ensures precise evaluations and maintains consistency between offline simulations and online competitions via WizardArena, a pipeline developed to accurately predict the Elo rankings of various models using a meticulously designed offline test set. Our results demonstrate that WizardArena's predictions closely align with those from the online Arena. Second, it involves the continuous improvement of training data based on the battle results and the refined model. We establish a data flywheel to iteratively update the training data by highlighting the weaknesses of the target model based on its battle results, enabling it to learn from the strengths of multiple different models. We apply Arena Learning to train our target model, WizardLM-$\beta$, and demonstrate significant performance enhancements across various metrics. This fully automated training and evaluation pipeline sets the stage for continuous advancements in various LLMs via post-training. Notably, Arena Learning plays a pivotal role in the success of WizardLM-2, and this paper serves both as an exploration of its efficacy and a foundational study for future discussions related to WizardLM-2 and its derivatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02130v4">Using LLMs for the Extraction and Normalization of Product Attribute Values</a></div>
    <div class="paper-meta">
      📅 2024-07-15
      | 💬 The paper has been accepted at ADBIS2024
    </div>
    <details class="paper-abstract">
      Product offers on e-commerce websites often consist of a product title and a textual product description. In order to enable features such as faceted product search or to generate product comparison tables, it is necessary to extract structured attribute-value pairs from the unstructured product titles and descriptions and to normalize the extracted values to a single, unified scale for each attribute. This paper explores the potential of using large language models (LLMs), such as GPT-3.5 and GPT-4, to extract and normalize attribute values from product titles and descriptions. We experiment with different zero-shot and few-shot prompt templates for instructing LLMs to extract and normalize attribute-value pairs. We introduce the Web Data Commons - Product Attribute Value Extraction (WDC-PAVE) benchmark dataset for our experiments. WDC-PAVE consists of product offers from 59 different websites which provide schema.org annotations. The offers belong to five different product categories, each with a specific set of attributes. The dataset provides manually verified attribute-value pairs in two forms: (i) directly extracted values and (ii) normalized attribute values. The normalization of the attribute values requires systems to perform the following types of operations: name expansion, generalization, unit of measurement conversion, and string wrangling. Our experiments demonstrate that GPT-4 outperforms the PLM-based extraction methods SU-OpenTag, AVEQA, and MAVEQA by 10%, achieving an F1-score of 91%. For the extraction and normalization of product attribute values, GPT-4 achieves a similar performance to the extraction scenario, while being particularly strong at string wrangling and name expansion.
    </details>
</div>
