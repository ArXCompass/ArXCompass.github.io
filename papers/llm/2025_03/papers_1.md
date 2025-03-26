# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19855v1">Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs), such as OpenAI-o1 and DeepSeek-R1, have demonstrated the effectiveness of test-time scaling, where extended reasoning processes substantially enhance model performance. Despite this, current models are constrained by limitations in handling long texts and reinforcement learning (RL) training efficiency. To address these issues, we propose a simple yet effective test-time scaling approach Multi-round Thinking. This method iteratively refines model reasoning by leveraging previous answers as prompts for subsequent rounds. Extensive experiments across multiple models, including QwQ-32B and DeepSeek-R1, consistently show performance improvements on various benchmarks such as AIME 2024, MATH-500, GPQA-diamond, and LiveCodeBench. For instance, the accuracy of QwQ-32B improved from 80.3% (Round 1) to 82.1% (Round 2) on the AIME 2024 dataset, while DeepSeek-R1 showed a similar increase from 79.7% to 82.0%. These results confirm that Multi-round Thinking is a broadly applicable, straightforward approach to achieving stable enhancements in model performance, underscoring its potential for future developments in test-time scaling techniques. The key prompt: {Original question prompt} The assistant's previous answer is: <answer> {last round answer} </answer>, and please re-answer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19850v1">FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Information retrieval in hour-long videos presents a significant challenge, even for state-of-the-art Vision-Language Models (VLMs), particularly when the desired information is localized within a small subset of frames. Long video data presents challenges for VLMs due to context window limitations and the difficulty of pinpointing frames containing the answer. Our novel video agent, FALCONEye, combines a VLM and a Large Language Model (LLM) to search relevant information along the video, and locate the frames with the answer. FALCONEye novelty relies on 1) the proposed meta-architecture, which is better suited to tackle hour-long videos compared to short video approaches in the state-of-the-art; 2) a new efficient exploration algorithm to locate the information using short clips, captions and answer confidence; and 3) our state-of-the-art VLMs calibration analysis for the answer confidence. Our agent is built over a small-size VLM and a medium-size LLM being accessible to run on standard computational resources. We also release FALCON-Bench, a benchmark to evaluate long (average > 1 hour) Video Answer Search challenges, highlighting the need for open-ended question evaluation. Our experiments show FALCONEye's superior performance than the state-of-the-art in FALCON-Bench, and similar or better performance in related benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02341v2">UAVs Meet LLMs: Overviews and Perspectives Toward Agentic Low-Altitude Mobility</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Low-altitude mobility, exemplified by unmanned aerial vehicles (UAVs), has introduced transformative advancements across various domains, like transportation, logistics, and agriculture. Leveraging flexible perspectives and rapid maneuverability, UAVs extend traditional systems' perception and action capabilities, garnering widespread attention from academia and industry. However, current UAV operations primarily depend on human control, with only limited autonomy in simple scenarios, and lack the intelligence and adaptability needed for more complex environments and tasks. The emergence of large language models (LLMs) demonstrates remarkable problem-solving and generalization capabilities, offering a promising pathway for advancing UAV intelligence. This paper explores the integration of LLMs and UAVs, beginning with an overview of UAV systems' fundamental components and functionalities, followed by an overview of the state-of-the-art in LLM technology. Subsequently, it systematically highlights the multimodal data resources available for UAVs, which provide critical support for training and evaluation. Furthermore, it categorizes and analyzes key tasks and application scenarios where UAVs and LLMs converge. Finally, a reference roadmap towards agentic UAVs is proposed, aiming to enable UAVs to achieve agentic intelligence through autonomous perception, memory, reasoning, and tool utilization. Related resources are available at https://github.com/Hub-Tian/UAVs_Meet_LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11102v3">Empowering LLMs to Understand and Generate Complex Vector Graphics</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Accepted by CVPR 2025. Project Page: https://ximinng.github.io/LLM4SVGProject/
    </div>
    <details class="paper-abstract">
      The unprecedented advancements in Large Language Models (LLMs) have profoundly impacted natural language processing but have yet to fully embrace the realm of scalable vector graphics (SVG) generation. While LLMs encode partial knowledge of SVG data from web pages during training, recent findings suggest that semantically ambiguous and tokenized representations within LLMs may result in hallucinations in vector primitive predictions. Additionally, LLM training typically lacks modeling and understanding of the rendering sequence of vector paths, which can lead to occlusion between output vector primitives. In this paper, we present LLM4SVG, an initial yet substantial step toward bridging this gap by enabling LLMs to better understand and generate vector graphics. LLM4SVG facilitates a deeper understanding of SVG components through learnable semantic tokens, which precisely encode these tokens and their corresponding properties to generate semantically aligned SVG outputs. Using a series of learnable semantic tokens, a structured dataset for instruction following is developed to support comprehension and generation across two primary tasks. Our method introduces a modular architecture to existing large language models, integrating semantic tags, vector instruction encoders, fine-tuned commands, and powerful LLMs to tightly combine geometric, appearance, and language information. To overcome the scarcity of SVG-text instruction data, we developed an automated data generation pipeline that collected our SVGX-SFT Dataset, consisting of high-quality human-designed SVGs and 580k SVG instruction following data specifically crafted for LLM training, which facilitated the adoption of the supervised fine-tuning strategy popular in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.05215v3">DeltaZip: Efficient Serving of Multiple Full-Model-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 EuroSys 2025'
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) greatly improves model quality for downstream tasks. However, serving many fine-tuned LLMs concurrently is challenging due to the sporadic, bursty, and varying request patterns of different LLMs. To bridge this gap, we present DeltaZip, an LLM serving system that efficiently serves multiple full-parameter fine-tuned models concurrently by aggressively compressing model deltas by up to 10x while maintaining high model quality. The key insight behind this design is that fine-tuning results in small-magnitude changes to the pre-trained model. By co-designing the serving system with the compression algorithm, DeltaZip achieves 2x to 12x improvement in throughput compared to the state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17662v2">Enhancing Persona Consistency for LLMs' Role-Playing using Persona-Aware Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have achieved breakthrough progress in many dialogue generation tasks. However, their lack of emotion and fine-grained role awareness limits the model's ability to provide personalized and diverse interactions further. Current methods face high costs in collecting high-quality annotated data for scenarios such as role-playing, and traditional human alignment methods are difficult to deploy due to the inherent diversity of model behavior in role-playing scenarios. Inspired by the alignment of models for safety behaviors through RLHF (Reinforcement Learning from Human Feedback), in this paper, we revisit model role-playing behavior from the perspective of persona alignment and propose a novel annotation-free framework named \textbf{\underline{P}}ersona-Aware \textbf{\underline{C}}ontrastive \textbf{\underline{L}}earning (PCL) to align LLMs' behavior during role-playing, enhancing the model's role consistency. Specifically, we first design a role chain method to encourage the model to self-question based on the role characteristics and dialogue context to adjust personality consistency. Then, we further enhance the model's role-playing strategy through iterative contrastive learning between the use of role characteristics and not. Experiments on both black-box and white-box LLMs show that LLMs equipped with PCL significantly outperform vanilla LLMs under automatic evaluation methods (CharEval \& GPT-4) and human expert evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19693v1">AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive versatility as general purpose models. However, their broad applicability comes at a high-cost computational overhead, particularly in auto-regressive decoding where each step requires a forward pass. In domain-specific settings, general-purpose capabilities are unnecessary and can be exchanged for efficiency. In this work, we take a novel perspective on domain adaptation, reducing latency and computational costs by adapting the vocabulary to focused domains of interest. We introduce AdaptiVocab, an end-to-end approach for vocabulary adaptation, designed to enhance LLM efficiency in low-resource domains. AdaptiVocab can be applied to any tokenizer and architecture, modifying the vocabulary by replacing tokens with domain-specific n-gram-based tokens, thereby reducing the number of tokens required for both input processing and output generation. AdaptiVocab initializes new n-token embeddings using an exponentially weighted combination of existing embeddings and employs a lightweight fine-tuning phase that can be efficiently performed on a single GPU. We evaluate two 7B LLMs across three niche domains, assessing efficiency, generation quality, and end-task performance. Our results show that AdaptiVocab reduces token usage by over 25% without compromising performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19620v1">Optimization through In-Context Learning and Iterative LLM Prompting for Nuclear Engineering Design Problems</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Codes and data are available upon request
    </div>
    <details class="paper-abstract">
      The optimization of nuclear engineering designs, such as nuclear fuel assembly configurations, involves managing competing objectives like reactivity control and power distribution. This study explores the use of Optimization by Prompting, an iterative approach utilizing large language models (LLMs), to address these challenges. The method is straightforward to implement, requiring no hyperparameter tuning or complex mathematical formulations. Optimization problems can be described in plain English, with only an evaluator and a parsing script needed for execution. The in-context learning capabilities of LLMs enable them to understand problem nuances, therefore, they have the potential to surpass traditional metaheuristic optimization methods. This study demonstrates the application of LLMs as optimizers to Boiling Water Reactor (BWR) fuel lattice design, showing the capability of commercial LLMs to achieve superior optimization results compared to traditional methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03235v2">Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Accepted in ICLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19598v1">The Greatest Good Benchmark: Measuring LLMs' Alignment with Utilitarian Moral Dilemmas</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The question of how to make decisions that maximise the well-being of all persons is very relevant to design language models that are beneficial to humanity and free from harm. We introduce the Greatest Good Benchmark to evaluate the moral judgments of LLMs using utilitarian dilemmas. Our analysis across 15 diverse LLMs reveals consistently encoded moral preferences that diverge from established moral theories and lay population moral standards. Most LLMs have a marked preference for impartial beneficence and rejection of instrumental harm. These findings showcase the 'artificial moral compass' of LLMs, offering insights into their moral alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05374v3">Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.18710v2">To FP8 and Back Again: Quantifying Reduced Precision Effects on LLM Training Stability</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The massive computational costs associated with large language model (LLM) pretraining have spurred great interest in reduced-precision floating-point representations to accelerate the process. As a result, the BrainFloat16 (BF16) precision has become the de facto standard for LLM training, with hardware support included in recent generations of accelerators. This trend has gone even further in the latest processors, where FP8 has recently been introduced. However, prior experience with FP16, which was found to be less stable than BF16, raises concerns as to whether FP8, with even fewer bits than FP16, can be a cost-effective option for LLM training. We argue that reduced-precision training schemes must have similar training stability and hyperparameter sensitivities to their higher-precision counterparts in order to be cost-effective. However, we find that currently available methods for FP8 training are not robust enough to allow their use as economical replacements. This prompts us to investigate the stability of reduced-precision LLM training in terms of robustness across random seeds, learning rates, and datasets. To this end, we propose new evaluation techniques and a new metric for quantifying loss landscape sharpness in autoregressive language models. By simulating incremental bit reductions in floating-point representations, we analyze the relationship between representational power and training stability with the intent of aiding future research into the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17056v2">The HalluRAG Dataset: Detecting Closed-Domain Hallucinations in RAG Applications Using an LLM's Internal States</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 19 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Detecting hallucinations in large language models (LLMs) is critical for enhancing their reliability and trustworthiness. Most research focuses on hallucinations as deviations from information seen during training. However, the opaque nature of an LLM's parametric knowledge complicates the understanding of why generated texts appear ungrounded: The LLM might not have picked up the necessary knowledge from large and often inaccessible datasets, or the information might have been changed or contradicted during further training. Our focus is on hallucinations involving information not used in training, which we determine by using recency to ensure the information emerged after a cut-off date. This study investigates these hallucinations by detecting them at sentence level using different internal states of various LLMs. We present HalluRAG, a dataset designed to train classifiers on these hallucinations. Depending on the model and quantization, MLPs trained on HalluRAG detect hallucinations with test accuracies ranging up to 75 %, with Mistral-7B-Instruct-v0.1 achieving the highest test accuracies. Our results show that IAVs detect hallucinations as effectively as CEVs and reveal that answerable and unanswerable prompts are encoded differently as separate classifiers for these categories improved accuracy. However, HalluRAG showed some limited generalizability, advocating for more diversity in datasets on hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12355v2">DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Accepted by CVPR 25
    </div>
    <details class="paper-abstract">
      The challenge in LLM-based video understanding lies in preserving visual and semantic information in long videos while maintaining a memory-affordable token count. However, redundancy and correspondence in videos have hindered the performance potential of existing methods. Through statistical learning on current datasets, we observe that redundancy occurs in both repeated and answer-irrelevant frames, and the corresponding frames vary with different questions. This suggests the possibility of adopting dynamic encoding to balance detailed video information preservation with token budget reduction. To this end, we propose a dynamic cooperative network, DynFocus, for memory-efficient video encoding in this paper. Specifically, i) a Dynamic Event Prototype Estimation (DPE) module to dynamically select meaningful frames for question answering; (ii) a Compact Cooperative Encoding (CCE) module that encodes meaningful frames with detailed visual appearance and the remaining frames with sketchy perception separately. We evaluate our method on five publicly available benchmarks, and experimental results consistently demonstrate that our method achieves competitive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00088v2">T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 EuroSys 2025
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) on edge devices is increasingly important to enhance on-device intelligence. Weight quantization is crucial for reducing the memory footprint of LLMs on devices. However, low-bit LLMs necessitate mixed precision matrix multiplication (mpGEMM) of low precision weights and high precision activations during inference. Existing systems, lacking native support for mpGEMM, resort to dequantize weights for high precision computation. Such an indirect way can lead to a significant inference overhead. In this paper, we introduce T-MAC, an innovative lookup table(LUT)-based method designed for efficient low-bit LLM (i.e., weight-quantized LLM) inference on CPUs. T-MAC directly supports mpGEMM without dequantization, while simultaneously eliminating multiplications and reducing additions required. Specifically, T-MAC transforms the traditional data-type-centric multiplication to bit-wise table lookup, and enables a unified and scalable mpGEMM solution. Our LUT-based kernels scale linearly to the weight bit-width. Evaluated on low-bit Llama and BitNet models, T-MAC demonstrates up to 4x increase in throughput and 70% reduction in energy consumption compared to llama.cpp. For BitNet-b1.58-3B, T-MAC delivers a token generation throughput of 30 tokens/s with a single core and 71 tokens/s with eight cores on M2-Ultra, and 11 tokens/s on lower-end devices like Raspberry Pi 5, which significantly exceeds the adult average reading speed. T-MAC with LUT-based computing paradigm, paves the way for the practical deployment of low-bit LLMs on resource-constrained edge devices without compromising computational efficiency. The system is open-sourced at https://github.com/microsoft/T-MAC .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19470v1">ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in reasoning, exemplified by the success of OpenAI-o1 and DeepSeek-R1. However, integrating reasoning with external search processes remains challenging, especially for complex multi-hop questions requiring multiple retrieval steps. We propose ReSearch, a novel framework that trains LLMs to Reason with Search via reinforcement learning without using any supervised data on reasoning steps. Our approach treats search operations as integral components of the reasoning chain, where when and how to perform searches is guided by text-based thinking, and search results subsequently influence further reasoning. We train ReSearch on Qwen2.5-7B(-Instruct) and Qwen2.5-32B(-Instruct) models and conduct extensive experiments. Despite being trained on only one dataset, our models demonstrate strong generalizability across various benchmarks. Analysis reveals that ReSearch naturally elicits advanced reasoning capabilities such as reflection and self-correction during the reinforcement learning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19449v1">VecTrans: LLM Transformation Framework for Better Auto-vectorization on High-performance CPU</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated great capabilities in code generation, yet their effective application in compiler optimizations remains an open challenge due to issues such as hallucinations and a lack of domain-specific reasoning. Vectorization, a crucial optimization for enhancing code performance, often fails because of the compiler's inability to recognize complex code patterns, which commonly require extensive empirical expertise. LLMs, with their ability to capture intricate patterns, thus providing a promising solution to this challenge. This paper presents VecTrans, a novel framework that leverages LLMs to enhance compiler-based code vectorization. VecTrans first employs compiler analysis to identify potentially vectorizable code regions. It then utilizes an LLM to refactor these regions into patterns that are more amenable to the compiler's auto-vectorization. To ensure semantic correctness, VecTrans further integrates a hybrid validation mechanism at the intermediate representation (IR) level. With the above efforts, VecTrans combines the adaptability of LLMs with the precision of compiler vectorization, thereby effectively opening up the vectorization opportunities. Experimental results show that among all 50 TSVC functions unvectorizable by Clang, GCC, and BiShengCompiler, VecTrans successfully vectorizes 23 cases (46%) and achieves an average speedup of 2.02x, greatly surpassing state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00599v3">VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 17 pages, 14 figures, technical report
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video LLMs) have recently exhibited remarkable capabilities in general video understanding. However, they mainly focus on holistic comprehension and struggle with capturing fine-grained spatial and temporal details. Besides, the lack of high-quality object-level video instruction data and a comprehensive benchmark further hinders their advancements. To tackle these challenges, we introduce the VideoRefer Suite to empower Video LLM for finer-level spatial-temporal video understanding, i.e., enabling perception and reasoning on any objects throughout the video. Specially, we thoroughly develop VideoRefer Suite across three essential aspects: dataset, model, and benchmark. Firstly, we introduce a multi-agent data engine to meticulously curate a large-scale, high-quality object-level video instruction dataset, termed VideoRefer-700K. Next, we present the VideoRefer model, which equips a versatile spatial-temporal object encoder to capture precise regional and sequential representations. Finally, we meticulously create a VideoRefer-Bench to comprehensively assess the spatial-temporal understanding capability of a Video LLM, evaluating it across various aspects. Extensive experiments and analyses demonstrate that our VideoRefer model not only achieves promising performance on video referring benchmarks but also facilitates general video understanding capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.11960v3">MCRanker: Generating Diverse Criteria On-the-Fly to Improve Point-wise LLM Rankers</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      The most recent pointwise Large Language Model (LLM) rankers have achieved remarkable ranking results. However, these rankers are hindered by two major drawbacks: (1) they fail to follow a standardized comparison guidance during the ranking process, and (2) they struggle with comprehensive considerations when dealing with complicated passages. To address these shortcomings, we propose to build a ranker that generates ranking scores based on a set of criteria from various perspectives. These criteria are intended to direct each perspective in providing a distinct yet synergistic evaluation. Our research, which examines eight datasets from the BEIR benchmark demonstrates that incorporating this multi-perspective criteria ensemble approach markedly enhanced the performance of pointwise LLM rankers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20021v2">Think Carefully and Check Again! Meta-Generation Unlocking LLMs for Low-Resource Cross-Lingual Summarization</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Cross-lingual summarization (CLS) aims to generate a summary for the source text in a different target language. Currently, instruction-tuned large language models (LLMs) excel at various English tasks. However, unlike languages such as English, Chinese or Spanish, for those relatively low-resource languages with limited usage or data, recent studies have shown that LLMs' performance on CLS tasks remains unsatisfactory even with few-shot settings. This raises the question: Are LLMs capable of handling cross-lingual summarization tasks for low-resource languages? To resolve this question, we fully explore the potential of large language models on cross-lingual summarization task for low-resource languages through our four-step zero-shot method: Summarization, Improvement, Translation and Refinement (SITR) with correspondingly designed prompts. We test our proposed method with multiple LLMs on two well-known cross-lingual summarization datasets with various low-resource target languages. The results show that: i) GPT-3.5 and GPT-4 significantly and consistently outperform other baselines when using our zero-shot SITR methods. ii) By employing our proposed method, we unlock the potential of LLMs, enabling them to effectively handle cross-lingual summarization tasks for relatively low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20016v2">Vulnerability of LLMs to Vertically Aligned Text Manipulations</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Text classification involves categorizing a given text, such as determining its sentiment or identifying harmful content. With the advancement of large language models (LLMs), these models have become highly effective at performing text classification tasks. However, they still show vulnerabilities to variations in text formatting. Recent research demonstrates that modifying input formats, such as vertically aligning words for encoder-based models, can substantially lower accuracy in text classification tasks. While easily understood by humans, these inputs can significantly mislead models, posing a potential risk of bypassing detection in real-world scenarios involving harmful or sensitive information. With the expanding application of LLMs, a crucial question arises: Do decoder-based LLMs exhibit similar vulnerabilities to vertically formatted text input? In this paper, we investigate the impact of vertical text input on the performance of various LLMs across multiple text classification datasets and analyze the underlying causes. Our findings are as follows: (i) Vertical text input significantly degrades the accuracy of LLMs in text classification tasks. (ii) Chain of Thought (CoT) reasoning does not help LLMs recognize vertical input or mitigate its vulnerability, but few-shot learning with careful analysis does. (iii) We explore the underlying cause of the vulnerability by analyzing the inherent issues in tokenization and attention matrices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19353v1">QUAD: Quantization and Parameter-Efficient Tuning of LLM with Activation Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-03-25
      | 💬 18 pages, 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in diverse applications but suffer inefficiency due to massive scale. While quantization reduces computational costs, existing methods degrade accuracy in medium-sized LLMs (e.g., Llama-3-8B) due to activation outliers. To address this, we propose QUAD (Quantization with Activation Decomposition), a framework leveraging Singular Value Decomposition (SVD) to suppress activation outliers for effective 4-bit quantization. QUAD estimates activation singular vectors offline using calibration data to construct an orthogonal transformation matrix P, shifting outliers to additional dimensions in full precision while quantizing rest components to 4-bit. Additionally, QUAD enables parameter-efficient fine-tuning via adaptable full-precision outlier weights, narrowing the accuracy gap between quantized and full-precision models. Experiments demonstrate that QUAD achieves 94% ~ 96% accuracy under W4A4 quantization and 98% accuracy with W4A4/A8 and parameter-efficient fine-tuning for Llama-3 and Qwen-2.5 models. Our code is available at \href{https://github.com/hyx1999/Quad}{repository}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19326v1">Process or Result? Manipulated Ending Tokens Can Mislead Reasoning LLMs to Ignore the Correct Reasoning Steps</a></div>
    <div class="paper-meta">
      📅 2025-03-25
    </div>
    <details class="paper-abstract">
      Recent reasoning large language models (LLMs) have demonstrated remarkable improvements in mathematical reasoning capabilities through long Chain-of-Thought. The reasoning tokens of these models enable self-correction within reasoning chains, enhancing robustness. This motivates our exploration: how vulnerable are reasoning LLMs to subtle errors in their input reasoning chains? We introduce "Compromising Thought" (CPT), a vulnerability where models presented with reasoning tokens containing manipulated calculation results tend to ignore correct reasoning steps and adopt incorrect results instead. Through systematic evaluation across multiple reasoning LLMs, we design three increasingly explicit prompting methods to measure CPT resistance, revealing that models struggle significantly to identify and correct these manipulations. Notably, contrary to existing research suggesting structural alterations affect model performance more than content modifications, we find that local ending token manipulations have greater impact on reasoning outcomes than structural changes. Moreover, we discover a security vulnerability in DeepSeek-R1 where tampered reasoning tokens can trigger complete reasoning cessation. Our work enhances understanding of reasoning robustness and highlights security considerations for reasoning-intensive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18929v1">Trajectory Balance with Asynchrony: Decoupling Exploration and Learning for Fast, Scalable LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a critical component of large language model (LLM) post-training. However, existing on-policy algorithms used for post-training are inherently incompatible with the use of experience replay buffers, which can be populated scalably by distributed off-policy actors to enhance exploration as compute increases. We propose efficiently obtaining this benefit of replay buffers via Trajectory Balance with Asynchrony (TBA), a massively scalable LLM RL system. In contrast to existing approaches, TBA uses a larger fraction of compute on search, constantly generating off-policy data for a central replay buffer. A training node simultaneously samples data from this buffer based on reward or recency to update the policy using Trajectory Balance (TB), a diversity-seeking RL objective introduced for GFlowNets. TBA offers three key advantages: (1) decoupled training and search, speeding up training wall-clock time by 4x or more; (2) improved diversity through large-scale off-policy sampling; and (3) scalable search for sparse reward settings. On mathematical reasoning, preference-tuning, and automated red-teaming (diverse and representative post-training tasks), TBA produces speed and performance improvements over strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18891v1">AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) based on large language models (LLMs) have demonstrated significant potential in collaborative problem-solving. However, they still face substantial challenges of low communication efficiency and suboptimal task performance, making the careful design of the agents' communication topologies particularly important. Inspired by the management theory that roles in an efficient team are often dynamically adjusted, we propose AgentDropout, which identifies redundant agents and communication across different communication rounds by optimizing the adjacency matrices of the communication graphs and eliminates them to enhance both token efficiency and task performance. Compared to state-of-the-art methods, AgentDropout achieves an average reduction of 21.6% in prompt token consumption and 18.4% in completion token consumption, along with a performance improvement of 1.14 on the tasks. Furthermore, the extended experiments demonstrate that AgentDropout achieves notable domain transferability and structure robustness, revealing its reliability and effectiveness. We release our code at https://github.com/wangzx1219/AgentDropout.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18869v1">Reimagining Memory Access for LLM Inference: Compression-Aware Memory Controller Design</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 9 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The efficiency of Large Language Model~(LLM) inference is often constrained by substantial memory bandwidth and capacity demands. Existing techniques, such as pruning, quantization, and mixture of experts/depth, reduce memory capacity and/or bandwidth consumption at the cost of slight degradation in inference quality. This paper introduces a design solution that further alleviates memory bottlenecks by enhancing the on-chip memory controller in AI accelerators to achieve two main objectives: (1) significantly reducing memory capacity and bandwidth usage through lossless block compression~(e.g., LZ4 and ZSTD) of model weights and key-value (KV) cache without compromising inference quality, and (2) enabling memory bandwidth and energy consumption to scale proportionally with context-dependent dynamic quantization. These goals are accomplished by equipping the on-chip memory controller with mechanisms to improve fine-grained bit-level accessibility and compressibility of weights and KV cache through LLM-aware configuration of in-memory placement and representation. Experimental results on publicly available LLMs demonstrate the effectiveness of this approach, showing memory footprint reductions of 25.2\% for model weights and 46.9\% for KV cache. In addition, our hardware prototype at 4\,GHz and 32 lanes (7\,nm) achieves 8\,TB/s throughput with a modest area overhead (under 3.8\,mm\(^2\)), which underscores the viability of LLM-aware memory control as a key to efficient large-scale inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18825v1">EconEvals: Benchmarks and Litmus Tests for LLM Agents in Unknown Environments</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      We develop benchmarks for LLM agents that act in, learn from, and strategize in unknown environments, the specifications of which the LLM agent must learn over time from deliberate exploration. Our benchmarks consist of decision-making tasks derived from key problems in economics. To forestall saturation, the benchmark tasks are synthetically generated with scalable difficulty levels. Additionally, we propose litmus tests, a new kind of quantitative measure for LLMs and LLM agents. Unlike benchmarks, litmus tests quantify differences in character, values, and tendencies of LLMs and LLM agents, by considering their behavior when faced with tradeoffs (e.g., efficiency versus equality) where there is no objectively right or wrong behavior. Overall, our benchmarks and litmus tests assess the abilities and tendencies of LLM agents in tackling complex economic problems in diverse settings spanning procurement, scheduling, task allocation, and pricing -- applications that should grow in importance as such agents are further integrated into the economy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18809v1">Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have shown remarkable capabilities in various artificial intelligence problems. However, they fail to plan reliably, even when prompted with a detailed definition of the planning task. Attempts to improve their planning capabilities, such as chain-of-thought prompting, fine-tuning, and explicit "reasoning" still yield incorrect plans and usually fail to generalize to larger tasks. In this paper, we show how to use LLMs to generate correct plans, even for out-of-distribution tasks of increasing size. For a given planning domain, we ask an LLM to generate several domain-dependent heuristic functions in the form of Python code, evaluate them on a set of training tasks within a greedy best-first search, and choose the strongest one. The resulting LLM-generated heuristics solve many more unseen test tasks than state-of-the-art domain-independent heuristics for classical planning. They are even competitive with the strongest learning algorithm for domain-dependent planning. These findings are especially remarkable given that our proof-of-concept implementation is based on an unoptimized Python planner and the baselines all build upon highly optimized C++ code. In some domains, the LLM-generated heuristics expand fewer states than the baselines, revealing that they are not only efficiently computable, but sometimes even more informative than the state-of-the-art heuristics. Overall, our results show that sampling a set of planning heuristic function programs can significantly improve the planning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18792v1">REALM: A Dataset of Real-World LLM Use Cases</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models, such as the GPT series, have driven significant industrial applications, leading to economic and societal transformations. However, a comprehensive understanding of their real-world applications remains limited. To address this, we introduce REALM, a dataset of over 94,000 LLM use cases collected from Reddit and news articles. REALM captures two key dimensions: the diverse applications of LLMs and the demographics of their users. It categorizes LLM applications and explores how users' occupations relate to the types of applications they use. By integrating real-world data, REALM offers insights into LLM adoption across different domains, providing a foundation for future research on their evolving societal roles. A dedicated dashboard https://realm-e7682.web.app/ presents the data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18773v1">BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      The growing adoption of long-context Large Language Models (LLMs) has introduced significant memory and computational challenges in autoregressive decoding due to the expanding Key-Value (KV) cache. KV cache quantization has emerged as a promising solution, with prior work showing that 4-bit or even 2-bit quantization can maintain model accuracy while reducing memory costs. However, despite these benefits, preliminary implementations for the low-bit KV cache struggle to deliver the expected speedup due to quantization and dequantization overheads and the lack of Tensor Cores utilization. In this work, we propose BitDecoding, a GPU-optimized framework that unlocks Tensor Cores for efficient decoding with low-bit KV cache. Efficiently leveraging Tensor Cores for low-bit KV cache is challenging due to the dynamic nature of KV cache generation at each decoding step. BitDecoding addresses these challenges with a Tensor Cores-Centric BitFusion Scheme that ensures data layout compatibility to enable high utilization of Tensor Cores. Additionally, BitDecoding incorporates a warp-efficient parallel decoding kernel and a fine-grained asynchronous pipeline, minimizing dequantization overhead and improving computational efficiency. Experiments show that BitDecoding achieves up to 7.5x speedup on RTX 4090, 4.8x on A100, and 8.9x on H100, compared to FP16 FlashDecoding-v2. It also outperforms the state-of-the-art low-bit KV cache implementation (QServe) by up to 4.3x. On LLaMA-3.1-8B with a 128K sequence length, BitDecoding reduces single-batch decoding latency by 3x, demonstrating its effectiveness in long-context generation scenarios. The code is available at https://github.com/DD-DuDa/BitDecoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16514v2">VeriMind: Agentic LLM for Automated Verilog Generation with a Novel Evaluation Metric</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Designing Verilog modules requires meticulous attention to correctness, efficiency, and adherence to design specifications. However, manually writing Verilog code remains a complex and time-consuming task that demands both expert knowledge and iterative refinement. Leveraging recent advancements in large language models (LLMs) and their structured text generation capabilities, we propose VeriMind, an agentic LLM framework for Verilog code generation that significantly automates and optimizes the synthesis process. Unlike traditional LLM-based code generators, VeriMind employs a structured reasoning approach: given a user-provided prompt describing design requirements, the system first formulates a detailed train of thought before the final Verilog code is generated. This multi-step methodology enhances interpretability, accuracy, and adaptability in hardware design. In addition, we introduce a novel evaluation metric-pass@ARC-which combines the conventional pass@k measure with Average Refinement Cycles (ARC) to capture both success rate and the efficiency of iterative refinement. Experimental results on diverse hardware design tasks demonstrated that our approach achieved up to $8.3\%$ improvement on pass@k metric and $8.1\%$ on pass@ARC metric. These findings underscore the transformative potential of agentic LLMs in automated hardware design, RTL development, and digital system synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13178v2">Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 17 pages, 3 fugures
    </div>
    <details class="paper-abstract">
      Post-training Quantization (PTQ) technique has been extensively adopted for large language models (LLMs) compression owing to its efficiency and low resource requirement. However, current research lacks a in-depth analysis of the superior and applicable scenarios of each PTQ strategy. In addition, existing algorithms focus primarily on performance, overlooking the trade-off among model size, performance, and quantization bitwidth. To mitigate these confusions, we provide a novel benchmark for LLMs PTQ in this paper. Firstly, in order to support our benchmark, we propose a comprehensive taxonomy for existing mainstream methods by scrutinizing their computational strategies (e.g., optimization-based, compensation-based, etc.). Then, we conduct extensive experiments with the baseline within each class, covering models with various sizes (7B-70B), bitwidths, training levels (LLaMA1/2/3/3.1), architectures (Mixtral, DeepSeekMoE and Mamba) and modality (LLaVA1.5 and VILA1.5) on a wide range of evaluation metrics.Through comparative analysis on the results, we summarize the superior of each PTQ strategy and modelsize-bitwidth trade-off considering the performance. For example, our benchmark reveals that compensation-based technique demonstrates outstanding cross-architecture robustness and extremely low-bit PTQ for ultra large models should be reexamined. Finally, we further accordingly claim that a practical combination of compensation and other PTQ strategy can achieve SOTA various robustness. We believe that our benchmark will provide valuable recommendations for the deployment of LLMs and future research on PTQ approaches.We conduct an repository for our benchmark at https://github.com/zjq0455/PTQ_Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18666v1">AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Agents built on LLMs are increasingly deployed across diverse domains, automating complex decision-making and task execution. However, their autonomy introduces safety risks, including security vulnerabilities, legal violations, and unintended harmful actions. Existing mitigation methods, such as model-based safeguards and early enforcement strategies, fall short in robustness, interpretability, and adaptability. To address these challenges, we propose AgentSpec, a lightweight domain-specific language for specifying and enforcing runtime constraints on LLM agents. With AgentSpec, users define structured rules that incorporate triggers, predicates, and enforcement mechanisms, ensuring agents operate within predefined safety boundaries. We implement AgentSpec across multiple domains, including code execution, embodied agents, and autonomous driving, demonstrating its adaptability and effectiveness. Our evaluation shows that AgentSpec successfully prevents unsafe executions in over 90% of code agent cases, eliminates all hazardous actions in embodied agent tasks, and enforces 100% compliance by autonomous vehicles (AVs). Despite its strong safety guarantees, AgentSpec remains computationally lightweight, with overheads in milliseconds. By combining interpretability, modularity, and efficiency, AgentSpec provides a practical and scalable solution for enforcing LLM agent safety across diverse applications. We also automate the generation of rules using LLMs and assess their effectiveness. Our evaluation shows that the rules generated by OpenAI o1 achieve a precision of 95.56% and recall of 70.96% for embodied agents, successfully identifying 87.26% of the risky code, and prevent AVs from breaking laws in 5 out of 8 scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14502v3">How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05843v3">From Objects to Events: Unlocking Complex Visual Understanding in Object Detectors via LLM-guided Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 13 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Our key innovation lies in bridging the semantic gap between object detection and event understanding without requiring expensive task-specific training. The proposed plug-and-play framework interfaces with any open-vocabulary detector while extending their inherent capabilities across architectures. At its core, our approach combines (i) a symbolic regression mechanism exploring relationship patterns among detected entities and (ii) a LLM-guided strategically guiding the search toward meaningful expressions. These discovered symbolic rules transform low-level visual perception into interpretable event understanding, providing a transparent reasoning path from objects to events with strong transferability across domains.We compared our training-free framework against specialized event recognition systems across diverse application domains. Experiments demonstrate that our framework enhances multiple object detector architectures to recognize complex events such as illegal fishing activities (75% AUROC, +8.36% improvement), construction safety violations (+15.77%), and abnormal crowd behaviors (+23.16%). The code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18599v1">Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 15 pages, 14 figures, and 4 tables
    </div>
    <details class="paper-abstract">
      Modern Large Language Model serving system batches multiple requests to achieve high throughput, while batching attention operations is challenging, rendering memory bandwidth a critical bottleneck. The community relies on high-end GPUs with multiple high-bandwidth memory channels. Unfortunately, HBM's high bandwidth often comes at the expense of limited memory capacity, which reduces core utilization and increases costs. Recent advancements enabling longer contexts for LLMs have substantially increased the key-value cache size, further intensifying the pressures on memory capacity. The literature has explored KV cache quantization techniques, which commonly use low bitwidth for most values, selectively using higher bitwidth for outlier values. While this approach helps achieve high accuracy and low bitwidth simultaneously, it comes with the limitation that cost for online outlier detection is excessively high, negating the advantages. We propose Oaken, an acceleration solution that achieves high accuracy and high performance simultaneously through co-designing algorithm and hardware. To effectively find a sweet spot in the accuracy-performance trade-off space of KV cache quantization, Oaken employs an online-offline hybrid approach, setting outlier thresholds offline, which are then used to determine the quantization scale online. To translate the proposed algorithmic technique into tangible performance gains, Oaken also comes with custom quantization engines and memory management units that can be integrated with any LLM accelerators. We built an Oaken accelerator on top of an LLM accelerator, LPU, and conducted a comprehensive evaluation. Our experiments show that for a batch size of 256, Oaken achieves up to 1.58x throughput improvement over NVIDIA A100 GPU, incurring a minimal accuracy loss of only 0.54\% on average, compared to state-of-the-art KV cache quantization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11702v3">Toward a method for LLM-enabled Indoor Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 7 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Indoor navigation presents unique challenges due to complex layouts, lack of GPS signals, and accessibility concerns. Existing solutions often struggle with real-time adaptability and user-specific needs. In this work, we explore the potential of a Large Language Model (LLM), i.e., ChatGPT, to generate natural, context-aware navigation instructions from indoor map images. We design and evaluate test cases across different real-world environments, analyzing the effectiveness of LLMs in interpreting spatial layouts, handling user constraints, and planning efficient routes. Our findings demonstrate the potential of LLMs for supporting personalized indoor navigation, with an average of 50.54% correct indications and a maximum of 77.78%. The results do not appear to depend on the complexity of the layout or the complexity of the expected path, but rather on the number of points of interest and the abundance of visual information, which negatively affect the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21321v2">LLM Post-Training: A Deep Dive into Reasoning Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 32 pages, 7 figures, 3 tables, 377 references. Github Repo: https://github.com/mbzuai-oryx/Awesome-LLM-Post-training
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed the natural language processing landscape and brought to life diverse applications. Pretraining on vast web-scale data has laid the foundation for these models, yet the research community is now increasingly shifting focus toward post-training techniques to achieve further breakthroughs. While pretraining provides a broad linguistic foundation, post-training methods enable LLMs to refine their knowledge, improve reasoning, enhance factual accuracy, and align more effectively with user intents and ethical considerations. Fine-tuning, reinforcement learning, and test-time scaling have emerged as critical strategies for optimizing LLMs performance, ensuring robustness, and improving adaptability across various real-world tasks. This survey provides a systematic exploration of post-training methodologies, analyzing their role in refining LLMs beyond pretraining, addressing key challenges such as catastrophic forgetting, reward hacking, and inference-time trade-offs. We highlight emerging directions in model alignment, scalable adaptation, and inference-time reasoning, and outline future research directions. We also provide a public repository to continually track developments in this fast-evolving field: https://github.com/mbzuai-oryx/Awesome-LLM-Post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18432v1">Teaching LLMs for Step-Level Automatic Math Correction via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Automatic math correction aims to check students' solutions to mathematical problems via artificial intelligence technologies. Most existing studies focus on judging the final answer at the problem level, while they ignore detailed feedback on each step in a math problem-solving process, which requires abilities of semantic understanding and reasoning. In this paper, we propose a reinforcement learning (RL)-based method to boost large language model (LLM) for step-level automatic math correction, named StepAMC. Particularly, we convert the step-level automatic math correction within the text classification task into an RL problem to enhance the reasoning capabilities of LLMs. Then, we design a space-constrained policy network to improve the stability of RL. Then, we introduce a fine-grained reward network to convert the binary human feedback into a continuous value. We conduct extensive experiments over two benchmark datasets and the results show that our model outperforms the eleven strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17174v3">CSCE: Boosting LLM Reasoning by Simultaneous Enhancing of Causal Significance and Consistency</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 6 pages,4 figures. This paper has been accepted for presentation at IEEE International Conference on Multimedia & Expo 2025
    </div>
    <details class="paper-abstract">
      Chain-based reasoning methods like chain of thought (CoT) play a rising role in solving reasoning tasks for large language models (LLMs). However, the causal hallucinations between a step of reasoning and corresponding state transitions are becoming a significant obstacle to advancing LLMs' reasoning capabilities, especially in long-range reasoning tasks. This paper proposes a non-chain-based reasoning framework for simultaneous consideration of causal significance and consistency, i.e., the Causal Significance and Consistency Enhancer (CSCE). We customize LLM's loss function utilizing treatment effect assessments to enhance its reasoning ability from two aspects: causal significance and consistency. This ensures that the model captures essential causal relationships and maintains robust and consistent performance across various scenarios. Additionally, we transform the reasoning process from the cascading multiple one-step reasoning commonly used in Chain-Based methods, like CoT, to a causal-enhanced method that outputs the entire reasoning process in one go, further improving the model's reasoning efficiency. Extensive experiments show that our method improves both the reasoning success rate and speed. These improvements further demonstrate that non-chain-based methods can also aid LLMs in completing reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18377v1">Maximum Redundancy Pruning: A Principle-Driven Layerwise Sparsity Allocation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities, but their enormous size poses significant challenges for deployment in real-world applications. To address this issue, researchers have sought to apply network pruning techniques to LLMs. A critical challenge in pruning is allocation the sparsity for each layer. Recent sparsity allocation methods is often based on heuristics or search that can easily lead to suboptimal performance. In this paper, we conducted an extensive investigation into various LLMs and revealed three significant discoveries: (1) the layerwise pruning sensitivity (LPS) of LLMs is highly non-uniform, (2) the choice of pruning metric affects LPS, and (3) the performance of a sparse model is related to the uniformity of its layerwise redundancy level. Based on these observations, we propose that the layerwise sparsity of LLMs should adhere to three principles: \emph{non-uniformity}, \emph{pruning metric dependency}, and \emph{uniform layerwise redundancy level} in the pruned model. To this end, we proposed Maximum Redundancy Pruning (MRP), an iterative pruning algorithm that prunes in the most redundant layers (\emph{i.e.}, those with the highest non-outlier ratio) at each iteration. The achieved layerwise sparsity aligns with the outlined principles. We conducted extensive experiments on publicly available LLMs, including the LLaMA2 and OPT, across various benchmarks. Experimental results validate the effectiveness of MRP, demonstrating its superiority over previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12486v3">EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive reasoning capabilities in well-defined problems with clear solutions, such as mathematics and coding. However, they still struggle with complex real-world scenarios like business negotiations, which require strategic reasoning-an ability to navigate dynamic environments and align long-term goals amidst uncertainty. Existing methods for strategic reasoning face challenges in adaptability, scalability, and transferring strategies to new contexts. To address these issues, we propose explicit policy optimization (EPO) for strategic reasoning, featuring an LLM that provides strategies in open-ended action space and can be plugged into arbitrary LLM agents to motivate goal-directed behavior. To improve adaptability and policy transferability, we train the strategic reasoning model via multi-turn reinforcement learning (RL) using process rewards and iterative self-play, without supervised fine-tuning (SFT) as a preliminary step. Experiments across social and physical domains demonstrate EPO's ability of long-term goal alignment through enhanced strategic reasoning, achieving state-of-the-art performance on social dialogue and web navigation tasks. Our findings reveal various collaborative reasoning mechanisms emergent in EPO and its effectiveness in generating novel strategies, underscoring its potential for strategic reasoning in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18320v1">Bridging Writing Manner Gap in Visual Instruction Tuning by Creating LLM-aligned Instructions</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In the realm of Large Multi-modal Models (LMMs), the instruction quality during the visual instruction tuning stage significantly influences the performance of modality alignment. In this paper, we assess the instruction quality from a unique perspective termed \textbf{Writing Manner}, which encompasses the selection of vocabulary, grammar and sentence structure to convey specific semantics. We argue that there exists a substantial writing manner gap between the visual instructions and the base Large Language Models (LLMs) within LMMs. This gap forces the pre-trained base LLMs to deviate from their original writing styles, leading to capability degradation of both base LLMs and LMMs. To bridge the writing manner gap while preserving the original semantics, we propose directly leveraging the base LLM to align the writing manner of soft-format visual instructions with that of the base LLM itself, resulting in novel LLM-aligned instructions. The manual writing manner evaluation results demonstrate that our approach successfully minimizes the writing manner gap. By utilizing LLM-aligned instructions, the baseline models LLaVA-7B and QwenVL demonstrate enhanced resistance to hallucinations and non-trivial comprehensive improvements across all $15$ visual and language benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18316v1">Knowledge Transfer from LLMs to Provenance Analysis: A Semantic-Augmented Method for APT Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Advanced Persistent Threats (APTs) have caused significant losses across a wide range of sectors, including the theft of sensitive data and harm to system integrity. As attack techniques grow increasingly sophisticated and stealthy, the arms race between cyber defenders and attackers continues to intensify. The revolutionary impact of Large Language Models (LLMs) has opened up numerous opportunities in various fields, including cybersecurity. An intriguing question arises: can the extensive knowledge embedded in LLMs be harnessed for provenance analysis and play a positive role in identifying previously unknown malicious events? To seek a deeper understanding of this issue, we propose a new strategy for taking advantage of LLMs in provenance-based threat detection. In our design, the state-of-the-art LLM offers additional details in provenance data interpretation, leveraging their knowledge of system calls, software identity, and high-level understanding of application execution context. The advanced contextualized embedding capability is further utilized to capture the rich semantics of event descriptions. We comprehensively examine the quality of the resulting embeddings, and it turns out that they offer promising avenues. Subsequently, machine learning models built upon these embeddings demonstrated outstanding performance on real-world data. In our evaluation, supervised threat detection achieves a precision of 99.0%, and semi-supervised anomaly detection attains a precision of 96.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18313v1">DeepFund: Will LLM be Professional at Fund Investment? A Live Arena Perspective</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across various domains, but their effectiveness in financial decision making, particularly in fund investment, remains inadequately evaluated. Current benchmarks primarily assess LLMs understanding of financial documents rather than their ability to manage assets or analyze trading opportunities in dynamic market conditions. A critical limitation in existing evaluation methodologies is the backtesting approach, which suffers from information leakage when LLMs are evaluated on historical data they may have encountered during pretraining. This paper introduces DeepFund, a comprehensive platform for evaluating LLM based trading strategies in a simulated live environment. Our approach implements a multi agent framework where LLMs serve as both analysts and managers, creating a realistic simulation of investment decision making. The platform employs a forward testing methodology that mitigates information leakage by evaluating models on market data released after their training cutoff dates. We provide a web interface that visualizes model performance across different market conditions and investment parameters, enabling detailed comparative analysis. Through DeepFund, we aim to provide a more accurate and fair assessment of LLMs capabilities in fund investment, offering insights into their potential real world applications in financial markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18305v1">Enhancing LLM-based Code Translation in Repository Context via Triple Knowledge-Augmented</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have behaved well in function-level code translation without repository-level context. However, the performance of LLMs in repository-level context code translation remains suboptimal due to complex dependencies and context, hindering their adoption in industrial settings. In this work, we propose a novel LLM-based code translation technique K-Trans, which leverages triple knowledge augmentation to enhance LLM's translation quality under repository context in real-world software development. First, K-Trans constructs a translation knowledge base by extracting relevant information from target-language codebases, the repository being translated, and prior translation results. Second, for each function to be translated, K-Trans retrieves relevant triple knowledge, including target-language code samples, dependency usage examples, and successful translation function pairs, serving as references to enhance LLM for translation. Third, K-Trans constructs a knowledge-augmented translation prompt using the retrieved triple knowledge and employs LLMs to generate the translated code while preserving repository context. It further leverages LLMs for self-debugging, enhancing translation correctness. The experiments show that K-Trans substantially outperforms the baseline adapted from previous work by 19.4%/40.2% relative improvement in pass@1 and 0.138 in CodeBLEU. It is important to note that the results also demonstrate that each knowledge significantly contributes to K-Trans's effectiveness in handling repository-level context code translation, with dependency usage examples making the most notable contribution. Moreover, as the self-evolution process progresses, the knowledge base continuously enhances the LLM's performance across various aspects of the repository-level code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18293v1">Fact-checking AI-generated news reports: Can LLMs catch their own lies?</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In this paper, we evaluate the ability of Large Language Models (LLMs) to assess the veracity of claims in ''news reports'' generated by themselves or other LLMs. Our goal is to determine whether LLMs can effectively fact-check their own content, using methods similar to those used to verify claims made by humans. Our findings indicate that LLMs are more effective at assessing claims in national or international news stories than in local news stories, better at evaluating static information than dynamic information, and better at verifying true claims compared to false ones. We hypothesize that this disparity arises because the former types of claims are better represented in the training data. Additionally, we find that incorporating retrieved results from a search engine in a Retrieval-Augmented Generation (RAG) setting significantly reduces the number of claims an LLM cannot assess. However, this approach also increases the occurrence of incorrect assessments, partly due to irrelevant or low-quality search results. This diagnostic study highlights the need for future research on fact-checking machine-generated reports to prioritize improving the precision and relevance of retrieved information to better support fact-checking efforts. Furthermore, claims about dynamic events and local news may require human-in-the-loop fact-checking systems to ensure accuracy and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18292v1">Jenga: Effective Memory Management for Serving LLM with Heterogeneity</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 16 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used but expensive to run, especially as inference workloads grow. To lower costs, maximizing the request batch size by managing GPU memory efficiently is crucial. While PagedAttention has recently been proposed to improve the efficiency of memory management, we find that the growing heterogeneity in the embeddings dimensions, attention, and access patterns of modern LLM architectures introduces new challenges for memory allocation. In this paper, we present Jenga, a novel memory allocation framework for heterogeneous embeddings in LLMs. Jenga tackles two key challenges: (1) minimizing memory fragmentation when managing embeddings of different sizes, and (2) enabling flexible caching and eviction policies tailored to the specific token-dependency patterns of various layers. Jenga employs a two-level memory allocator, leveraging the least common multiple (LCM) of embedding sizes to optimize memory usage and providing APIs to express layer-specific caching logic to enhance memory reuse. We implemente Jenga on vLLM, a state-of-the-art LLM inference engine, and evaluate it with diverse LLMs, datasets, and GPU configurations. Evaluations show that Jenga improves GPU memory utilization by up to 79.6%, and increases serving throughput by up to 4.92x (1.80x on average).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18273v1">Analyzing Islamophobic Discourse Using Semi-Coded Terms and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Islamophobia started evolving into a global phenomenon by attracting followers across the globe, particularly in Western societies. Thus, understanding Islamophobia's global spread and online dissemination is crucial. This paper performs a large-scale analysis of specialized, semi-coded Islamophobic terms such as (muzrat, pislam, mudslime, mohammedan, muzzies) floated on extremist social platforms, i.e., 4Chan, Gab, Telegram, etc. First, we use large language models (LLMs) to show their ability to understand these terms. Second, using Google Perspective API, we also find that Islamophobic text is more toxic compared to other kinds of hate speech. Finally, we use BERT topic modeling approach to extract different topics and Islamophobic discourse on these social platforms. Our findings indicate that LLMs understand these Out-Of-Vocabulary (OOV) slurs; however, measures are still required to control such discourse. Our topic modeling also indicates that Islamophobic text is found across various political, conspiratorial, and far-right movements and is particularly directed against Muslim immigrants. Taken altogether, we performed the first study on Islamophobic semi-coded terms and shed a global light on Islamophobia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15299v2">Inside-Out: Hidden Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average relative gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) put a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19217v1">LLM Benchmarking with LLaMA2: Evaluating Code Development Performance Across Multiple Programming Languages</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has opened new possibilities for automating various tasks in software development. This paper evaluates the capabilities of the Llama 2-70B model in automating these tasks for scientific applications written in commonly used programming languages. Using representative test problems, we assess the model's capacity to generate code, documentation, and unit tests, as well as its ability to translate existing code between commonly used programming languages. Our comprehensive analysis evaluates the compilation, runtime behavior, and correctness of the generated and translated code. Additionally, we assess the quality of automatically generated code, documentation and unit tests. Our results indicate that while Llama 2-70B frequently generates syntactically correct and functional code for simpler numerical tasks, it encounters substantial difficulties with more complex, parallelized, or distributed computations, requiring considerable manual corrections. We identify key limitations and suggest areas for future improvements to better leverage AI-driven automation in scientific computing workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19182v1">Evaluating Bias in LLMs for Job-Resume Matching: Gender, Race, and Education</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 NAACL 2025: Industry Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer the potential to automate hiring by matching job descriptions with candidate resumes, streamlining recruitment processes, and reducing operational costs. However, biases inherent in these models may lead to unfair hiring practices, reinforcing societal prejudices and undermining workplace diversity. This study examines the performance and fairness of LLMs in job-resume matching tasks within the English language and U.S. context. It evaluates how factors such as gender, race, and educational background influence model decisions, providing critical insights into the fairness and reliability of LLMs in HR applications. Our findings indicate that while recent models have reduced biases related to explicit attributes like gender and race, implicit biases concerning educational background remain significant. These results highlight the need for ongoing evaluation and the development of advanced bias mitigation strategies to ensure equitable hiring practices when using LLMs in industry settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19114v1">Understanding and Improving Information Preservation in Prompt Compression for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 21 pages, 6 figures, 23 tables
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their successful application to a broad range of tasks. However, in information-intensive tasks, the prompt length can grow fast, leading to increased computational requirements, performance degradation, and induced biases from irrelevant or redundant information. Recently, various prompt compression techniques have been introduced to optimize the trade-off between reducing input length and retaining performance. We propose a holistic evaluation framework that allows for in-depth analysis of prompt compression methods. We focus on three key aspects, besides compression ratio: (i) downstream task performance, (ii) grounding in the input context, and (iii) information preservation. Through this framework, we investigate state-of-the-art soft and hard compression methods, showing that they struggle to preserve key details from the original prompt, limiting their performance on complex tasks. We demonstrate that modifying soft prompting methods to control better the granularity of the compressed information can significantly improve their effectiveness -- up to +23\% in downstream task performance, more than +8 BERTScore points in grounding, and 2.7x more entities preserved in compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19092v1">Rankers, Judges, and Assistants: Towards Understanding the Interplay of LLMs in Information Retrieval Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integral to information retrieval (IR), powering ranking, evaluation, and AI-assisted content creation. This widespread adoption necessitates a critical examination of potential biases arising from the interplay between these LLM-based components. This paper synthesizes existing research and presents novel experiment designs that explore how LLM-based rankers and assistants influence LLM-based judges. We provide the first empirical evidence of LLM judges exhibiting significant bias towards LLM-based rankers. Furthermore, we observe limitations in LLM judges' ability to discern subtle system performance differences. Contrary to some previous findings, our preliminary study does not find evidence of bias against AI-generated content. These results highlight the need for a more holistic view of the LLM-driven information ecosystem. To this end, we offer initial guidelines and a research agenda to ensure the reliable use of LLMs in IR evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19090v1">LLM-Based Insight Extraction for Contact Center Analytics and Cost-Efficient Deployment</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large Language Models have transformed the Contact Center industry, manifesting in enhanced self-service tools, streamlined administrative processes, and augmented agent productivity. This paper delineates our system that automates call driver generation, which serves as the foundation for tasks such as topic modeling, incoming call classification, trend detection, and FAQ generation, delivering actionable insights for contact center agents and administrators to consume. We present a cost-efficient LLM system design, with 1) a comprehensive evaluation of proprietary, open-weight, and fine-tuned models and 2) cost-efficient strategies, and 3) the corresponding cost analysis when deployed in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17167v2">QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM Quality Checks</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      We introduce QualityFlow, a dynamic agentic workflow for program synthesis. Given the English description of a programming problem and a set of unit tests, the model's goal is to synthesize the correct program that solves the problem and passes the tests. QualityFlow includes large language model (LLM) agents resembling a software development team, including code generation, testing, and self-debugging. We propose the LLM Quality Checker, which explicitly "imagines" whether the synthesized programs' execution would conform to the unit tests. The Quality Checks dynamically control the workflow, including actions to submit the final answer, clarify the problem statement, and revert previous workflow steps. Our experiments show that the Quality Checker can precisely accept any correct program, mitigate faulty synthesized tests, and prevent potential workflow deviation. QualityFlow establishes the state-of-the-art results on four program synthesis benchmarks: MBPP, HumanEval, and stricter evaluations from MBPP-EvalPlus and HumanEval-EvalPlus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13773v2">Mitigating KV Cache Competition to Enhance User Experience in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In Large Language Model (LLM) serving, the KV-cache (KVC) bottleneck causes high tail Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT), impairing user experience, particularly in time-sensitive applications. However, satisfying both TTFT and TBT service-level objectives (SLOs) is challenging. To address this, we propose a system, named CacheOPT for mitigating KV Cache competition, based on key insights from our measurements, incorporating novel components. First, it estimates a request's output length, bounding the deviation with a high specified probability, adjusted based on the request arrival rate. Second, it allocates the estimated KVC demand to a request, and reuses other requests' allocated KVC to avoid preemptions while reducing waiting time. Third, it proactively allocates KVC before instead of at the time a request exhausts its allocation and reserves KVC globally to prevent preemptions. Fourth, it chooses a request that has long TBT SLO, long job remaining time and short preemption time to preempt. Fifth, it selects the shortest-latency strategy between swapping and recomputation for preemptions. Experiments show that CacheOPT achieves up to 3.29$\times$ and 2.83$\times$ lower tail TBT and tail TTFT, 47\% and 53\% higher TTFT and TBT SLO attainments, and supports up to 1.58$\times$ higher request arrival rate than the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06364v2">EconoServe: Maximizing Multi-Resource Utilization with SLO Guarantees in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) continue to grow, reducing costs and alleviating GPU demands has become increasingly critical. However, existing schedulers primarily target either GPU compute or Key-Value Cache (KVC) utilization, failing to fully optimize both GPU compute and KVC usage during each iteration or guarantee timely KVC allocations when needed. To address these challenges, we conducted a trace-based experimental analysis and made insightful observations, leading to the design of a system called EconoServe. EconoServe maximizes multi-resource utilization while ensuring service-level objective (SLO) guarantees in LLM serving. To enable adding prompts to a batch to maximize GPU utilization in each iteration, EconoServe maintains separate waiting queues for prompt processing tasks (PTs) and generation tasks (GTs). It batches GTs with the same predicted response lengths (RL) to save scheduling time and allocates KVC space for the predicted RL to avoid KVC allocation failures. It further has a novel KVC pipelining method, allowing sharing allocated but unused KVC space to enhance KVC utilization. In addition, it prioritizes queued requests that occupy more KVC to release KVC earlier and satisfy request service-level-objective (SLO). Experimental results demonstrate that EconoServe increases throughput by up to 4$\times$ with the same level of latency, generates up to 91\% lower job completion time and up to 91\% higher SLO satisfaction ratio compared to vLLM. It also reduces the number of GPUs used in DistServe by up to 78\% while maintaining the same level of goodput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19007v1">Option Discovery Using LLM-guided Semantic Hierarchical Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable promise in reasoning and decision-making, yet their integration with Reinforcement Learning (RL) for complex robotic tasks remains underexplored. In this paper, we propose an LLM-guided hierarchical RL framework, termed LDSC, that leverages LLM-driven subgoal selection and option reuse to enhance sample efficiency, generalization, and multi-task adaptability. Traditional RL methods often suffer from inefficient exploration and high computational cost. Hierarchical RL helps with these challenges, but existing methods often fail to reuse options effectively when faced with new tasks. To address these limitations, we introduce a three-stage framework that uses LLMs for subgoal generation given natural language description of the task, a reusable option learning and selection method, and an action-level policy, enabling more effective decision-making across diverse tasks. By incorporating LLMs for subgoal prediction and policy guidance, our approach improves exploration efficiency and enhances learning performance. On average, LDSC outperforms the baseline by 55.9\% in average reward, demonstrating its effectiveness in complex RL settings. More details and experiment videos could be found in \href{https://raaslab.org/projects/LDSC/}{this link\footnote{https://raaslab.org/projects/LDSC}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18178v1">The Power of Small LLMs in Geometry Generation for Physical Simulations</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 24 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Engineers widely rely on simulation platforms like COMSOL or ANSYS to model and optimise processes. However, setting up such simulations requires expertise in defining geometry, generating meshes, establishing boundary conditions, and configuring solvers. This research aims to simplify this process by enabling engineers to describe their setup in plain language, allowing a Large Language Model (LLM) to generate the necessary input files for their specific application. This novel approach allows establishing a direct link between natural language and complex engineering tasks. Building on previous work that evaluated various LLMs for generating input files across simple and complex geometries, this study demonstrates that small LLMs - specifically, Phi-3 Mini and Qwen-2.5 1.5B - can be fine-tuned to generate precise engineering geometries in GMSH format. Through Low-Rank Adaptation (LoRA), we curated a dataset of 480 instruction-output pairs encompassing simple shapes (squares, rectangles, circles, and half circles) and more complex structures (I-beams, cylindrical pipes, and bent pipes). The fine-tuned models produced high-fidelity outputs, handling routine geometry generation with minimal intervention. While challenges remain with geometries involving combinations of multiple bodies, this study demonstrates that fine-tuned small models can outperform larger models like GPT-4o in specialised tasks, offering a precise and resource-efficient alternative for engineering applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12334v2">When neural implant meets multimodal LLM: A dual-loop system for neuromodulation and naturalistic neuralbehavioral research</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      We propose a novel dual-loop system that synergistically combines responsive neurostimulation (RNS) implants with artificial intelligence-driven wearable devices for treating post-traumatic stress disorder (PTSD) and enabling naturalistic brain research. In PTSD Therapy Mode, an implanted closed-loop neural device monitors amygdala activity and provides on-demand stimulation upon detecting pathological theta oscillations, while an ensemble of wearables (smart glasses, smartwatches, smartphones) uses multimodal large language model (LLM) analysis of sensory data to detect environmental or physiological PTSD triggers and deliver timely audiovisual interventions. Logged events from both the neural and wearable loops are analyzed to personalize trigger detection and progressively transition patients to non-invasive interventions. In Neuroscience Research Mode, the same platform is adapted for real-world brain activity capture. Wearable-LLM systems recognize naturalistic events (social interactions, emotional situations, compulsive behaviors, decision making) and signal implanted RNS devices (via wireless triggers) to record synchronized intracranial data during these moments. This approach builds on recent advances in mobile intracranial EEG recording and closed-loop neuromodulation in humans (BRAIN Initiative, 2023) (Mobbs et al., 2021). We discuss how our interdisciplinary system could revolutionize PTSD therapy and cognitive neuroscience by enabling 24/7 monitoring, context-aware intervention, and rich data collection outside traditional labs. The vision is a future where AI-enhanced devices continuously collaborate with the human brain, offering therapeutic support and deep insights into neural function, with the resulting real-world context rich neural data, in turn, accelerating the development of more biologically-grounded and human-centric AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15793v2">DNA Bench: When Silence is Smarter -- Benchmarking Over-Reasoning in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Test-time scaling has significantly improved large language model performance, enabling deeper reasoning to solve complex problems. However, this increased reasoning capability also leads to excessive token generation and unnecessary problem-solving attempts. We introduce Don\'t Answer Bench (DNA Bench), a new benchmark designed to evaluate LLMs ability to robustly understand the tricky reasoning triggers and avoiding unnecessary generation. DNA Bench consists of 150 adversarially designed prompts that are easy for humans to understand and respond to, but surprisingly not for many of the recent prominent LLMs. DNA Bench tests models abilities across different capabilities, such as instruction adherence, hallucination avoidance, redundancy filtering, and unanswerable question recognition. We evaluate reasoning LLMs (RLMs), including DeepSeek-R1, OpenAI O3-mini, Claude-3.7-sonnet and compare them against a powerful non-reasoning model, e.g., GPT-4o. Our experiments reveal that RLMs generate up to 70x more tokens than necessary, often failing at tasks that simpler non-reasoning models handle efficiently with higher accuracy. Our findings underscore the need for more effective training and inference strategies in RLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12018v2">Atom of Thoughts for Markov LLM Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through training-time scaling, and test-time scaling further enhances their capabilities by conducting effective reasoning during inference. However, as the scale of reasoning increases, existing test-time scaling methods suffer from accumulated historical information, which not only wastes computational resources but also interferes with effective reasoning. To address this issue, we observe that complex reasoning can be achieved by solving a series of independent and self-contained subquestions. These subquestions are essentially \textit{atomic questions}, exhibiting the memoryless property similar to Markov processes. Based on this observation, we propose Atom of Thoughts (\our), where each state transition consists of decomposing the current question into a dependency-based directed acyclic graph and contracting its subquestions, forming a simplified question that maintains answer equivalence with the original problem. This answer preservation enables the iterative \textit{decomposition-contraction} process to naturally form a meaningful Markov reasoning process. Furthermore, these atomic states can be seamlessly integrated into existing test-time scaling methods, enabling \our to serve as a plug-in enhancement for improving reasoning capabilities. Experiments across six benchmarks demonstrate the effectiveness of \our both as a standalone framework and a plug-in enhancement. Notably, on HotpotQA, when applied to gpt-4o-mini, \our achieves an \textbf{80.6\%} F1 score, surpassing o3-mini by \textbf{3.4\%} and DeepSeek-R1 by \textbf{10.6\%}. The code is available at \href{https://github.com/qixucen/atom}{https://github.com/qixucen/atom}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18141v1">AGIR: Assessing 3D Gait Impairment with Reasoning based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Assessing gait impairment plays an important role in early diagnosis, disease monitoring, and treatment evaluation for neurodegenerative diseases. Despite its widespread use in clinical practice, it is limited by subjectivity and a lack of precision. While recent deep learning-based approaches have consistently improved classification accuracies, they often lack interpretability, hindering their utility in clinical decision-making. To overcome these challenges, we introduce AGIR, a novel pipeline consisting of a pre-trained VQ-VAE motion tokenizer and a subsequent Large Language Model (LLM) fine-tuned over pairs of motion tokens and Chain-of-Thought (CoT) reasonings. To fine-tune an LLM for pathological gait analysis, we first introduce a multimodal dataset by adding rationales dedicated to MDS-UPDRS gait score assessment to an existing PD gait dataset. We then introduce a two-stage supervised fine-tuning (SFT) strategy to enhance the LLM's motion comprehension with pathology-specific knowledge. This strategy includes: 1) a generative stage that aligns gait motions with analytic descriptions through bidirectional motion-description generation, 2) a reasoning stage that integrates logical Chain-of-Thought (CoT) reasoning for impairment assessment with UPDRS gait score. Validation on an existing dataset and comparisons with state-of-the-art methods confirm the robustness and accuracy of our pipeline, demonstrating its ability to assign gait impairment scores from motion input with clinically meaningful rationales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18129v1">GeoBenchX: Benchmarking LLMs for Multistep Geospatial Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Github with code and benchmark set: https://github.com/Solirinai/GeoBenchX
    </div>
    <details class="paper-abstract">
      In this paper, we establish a benchmark for evaluating large language models (LLMs) on multi-step geospatial tasks relevant to commercial GIS practitioners. We assess seven leading commercial LLMs (Sonnet 3.5 and 3.7, Haiku 3.5, Gemini 2.0, GPT-4o, GPT-4o mini, and o3-mini) using a simple tool-calling agent equipped with 23 geospatial functions. Our benchmark comprises tasks across four categories of increasing complexity, with both solvable and intentionally unsolvable tasks to test hallucination rejection. We develop an LLM-as-Judge evaluation framework to compare agent solutions against reference implementations. Results show Sonnet 3.5 and GPT-4o achieve the best overall performance, with Claude models excelling on solvable tasks while OpenAI models better identify unsolvable scenarios. We observe significant differences in token usage, with Anthropic models consuming substantially more tokens than competitors. Common errors include misunderstanding geometrical relationships, relying on outdated knowledge, and inefficient data manipulation. The resulting benchmark set, evaluation framework, and data generation pipeline are released as open-source resources, providing one more standardized method for ongoing evaluation of LLMs for GeoAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14504v2">Aligning Multimodal LLM with Human Preference: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Project page: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Alignment
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can handle a wide variety of general tasks with simple prompts, without the need for task-specific training. Multimodal Large Language Models (MLLMs), built upon LLMs, have demonstrated impressive potential in tackling complex tasks involving visual, auditory, and textual data. However, critical issues related to truthfulness, safety, o1-like reasoning, and alignment with human preference remain insufficiently addressed. This gap has spurred the emergence of various alignment algorithms, each targeting different application scenarios and optimization goals. Recent studies have shown that alignment algorithms are a powerful approach to resolving the aforementioned challenges. In this paper, we aim to provide a comprehensive and systematic review of alignment algorithms for MLLMs. Specifically, we explore four key aspects: (1) the application scenarios covered by alignment algorithms, including general image understanding, multi-image, video, and audio, and extended multimodal applications; (2) the core factors in constructing alignment datasets, including data sources, model responses, and preference annotations; (3) the benchmarks used to evaluate alignment algorithms; and (4) a discussion of potential future directions for the development of alignment algorithms. This work seeks to help researchers organize current advancements in the field and inspire better alignment methods. The project page of this paper is available at https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18072v1">On the effectiveness of LLMs for automatic grading of open-ended questions in Spanish</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Grading is a time-consuming and laborious task that educators must face. It is an important task since it provides feedback signals to learners, and it has been demonstrated that timely feedback improves the learning process. In recent years, the irruption of LLMs has shed light on the effectiveness of automatic grading. In this paper, we explore the performance of different LLMs and prompting techniques in automatically grading short-text answers to open-ended questions. Unlike most of the literature, our study focuses on a use case where the questions, answers, and prompts are all in Spanish. Experimental results comparing automatic scores to those of human-expert evaluators show good outcomes in terms of accuracy, precision and consistency for advanced LLMs, both open and proprietary. Results are notably sensitive to prompt styles, suggesting biases toward certain words or content in the prompt. However, the best combinations of models and prompt strategies, consistently surpasses an accuracy of 95% in a three-level grading task, which even rises up to more than 98% when the it is simplified to a binary right or wrong rating problem, which demonstrates the potential that LLMs have to implement this type of automation in education applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18018v1">Lost in Cultural Translation: Do LLMs Struggle with Math Across Cultural Contexts?</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced various fields, particularly coding, mathematical reasoning, and logical problem solving. However, a critical question remains: Do these mathematical reasoning abilities persist when LLMs are presented with culturally adapted math problems? Specifically, how do LLMs perform when faced with math problems embedded in cultural contexts that have no significant representation in main stream web-scale AI training data? To explore this, we generated six synthetic cultural datasets from GSM8K, a widely used benchmark for assessing LLMs' mathematical reasoning skills. While preserving the mathematical logic and numerical values of the original GSM8K test set, we modify cultural elements such as personal names, food items, place names, etc. These culturally adapted datasets provide a more reliable framework for evaluating LLMs' mathematical reasoning under shifting cultural contexts. Our findings reveal that LLMs struggle with math problems when cultural references change, even though the underlying mathematical structure remains constant. Smaller models exhibit greater performance drops compared to larger models. Interestingly, our results also suggest that cultural familiarity can enhance mathematical reasoning. Even models with no explicit mathematical training but exposure to relevant cultural contexts sometimes outperform larger, mathematically proficient models on culturally embedded math problems. This study highlights the impact of cultural context on the mathematical reasoning abilities of LLMs, underscoring the need for more diverse and representative training data to improve robustness in real-world applications. The benchmark data sets and script for reproducing the results are available at https://github.com/akarim23131/Lost_in_Cultural_Translation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17994v1">Instructing the Architecture Search for Spatial-temporal Sequence Forecasting with LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Spatial-temporal sequence forecasting (STSF) is a long-standing research problem with widespread real-world applications. Neural architecture search (NAS), which automates the neural network design, has been shown effective in tackling the STSF problem. However, the existing NAS methods for STSF focus on generating architectures in a time-consuming data-driven fashion, which heavily limits their ability to use background knowledge and explore the complicated search trajectory. Large language models (LLMs) have shown remarkable ability in decision-making with comprehensive internal world knowledge, but how it could benefit NAS for STSF remains unexplored. In this paper, we propose a novel NAS method for STSF based on LLM. Instead of directly generate architectures with LLM, We inspire the LLM's capability with a multi-level enhancement mechanism. Specifically, on the step-level, we decompose the generation task into decision steps with powerful prompt engineering and inspire LLM to serve as instructor for architecture search based on its internal knowledge. On the instance-level, we utilize a one-step tuning framework to quickly evaluate the architecture instance and a memory bank to cumulate knowledge to improve LLM's search ability. On the task-level, we propose a two-stage architecture search, balancing the exploration stage and optimization stage, to reduce the possibility of being trapped in local optima. Extensive experimental results demonstrate that our method can achieve competitive effectiveness with superior efficiency against existing NAS methods for STSF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13558v3">LASER: Tuning-Free LLM-Driven Attention Control for Efficient Text-conditioned Image-to-Animation</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Revolutionary advancements in text-to-image models have unlocked new dimensions for sophisticated content creation, such as text-conditioned image editing, enabling the modification of existing images based on textual guidance. This capability allows for the generation of diverse images that convey highly complex visual concepts. However, existing methods primarily focus on generating new images from text-image pairs and struggle to produce fine-grained animations from existing images and textual guidance without fine-tuning. In this paper, we introduce LASER, a tuning-free LLM-driven attention control framework that follows a progressive process: LLM planning, feature-attention injection, and stable animation generation. LASER leverages a large language model (LLM) to refine general descriptions into fine-grained prompts, guiding pre-trained text-to-image models to generate aligned keyframes with subtle variations. The LLM also generates control signals for feature and attention injections, enabling seamless text-guided image morphing for various transformations without additional fine-tuning. By using the same initial noise inversion from the input image, LASER receives LLM-controlled injections during denoising and leverages interpolated text embeddings to produce a series of coherent animation frames. We propose a Text-conditioned Image-to-Animation Benchmark to validate the effectiveness and efficacy of LASER. Extensive experiments demonstrate that LASER achieves impressive results in consistent and efficient animation generation, establishing it as a powerful tool for producing detailed animations and opening new avenues in digital content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08493v2">Intelligent LiDAR Navigation: Leveraging External Information and Semantic Maps with LLM as Copilot</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Traditional robot navigation systems primarily utilize occupancy grid maps and laser-based sensing technologies, as demonstrated by the popular move_base package in ROS. Unlike robots, humans navigate not only through spatial awareness and physical distances but also by integrating external information, such as elevator maintenance updates from public notification boards and experiential knowledge, like the need for special access through certain doors. With the development of Large Language Models (LLMs), which possesses text understanding and intelligence close to human performance, there is now an opportunity to infuse robot navigation systems with a level of understanding akin to human cognition. In this study, we propose using osmAG (Area Graph in OpensStreetMap textual format), an innovative semantic topometric hierarchical map representation, to bridge the gap between the capabilities of ROS move_base and the contextual understanding offered by LLMs. Our methodology employs LLMs as an actual copilot in robot navigation, enabling the integration of a broader range of informational inputs while maintaining the robustness of traditional robotic navigation systems. Our code, demo, map, experiment results can be accessed at https://github.com/xiexiexiaoxiexie/Intelligent-LiDAR-Navigation-LLM-as-Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17965v1">Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 14 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance on a range of downstream NLP tasks by generating text that closely resembles human writing. However, the ease of achieving this similarity raises concerns from potential malicious uses at scale by bad actors, as LLM-generated text becomes increasingly difficult to discern from human text. Although detection methods have been developed to address this issue, bad actors can further manipulate LLM-generated texts to make them less detectable. In this work, we study how further editing texts with Reinforcement Learning from Human Feedback (RLHF), which aligns model outputs with human preferences, affects (a) the quality of generated texts for two tasks, and (b) the performance of LLM-generated text detectors, looking at both training-based and zero-shot detection methods. Although RLHF improves the quality of LLM-generated texts, we find that it also tends to produce more detectable, lengthy, and repetitive outputs. Additionally, we observe that training-based detectors are vulnerable to short texts and to texts that incorporate code, whereas zero-shot detectors exhibit greater robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11355v3">Nuclear Deployed: Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Please visit https://llm-catastrophic-risks.github.io for a quick tour of our research. Our code is available at https://github.com/pillowsofwind/LLM-CBRN-Risks
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are evolving into autonomous decision-makers, raising concerns about catastrophic risks in high-stakes scenarios, particularly in Chemical, Biological, Radiological and Nuclear (CBRN) domains. Based on the insight that such risks can originate from trade-offs between the agent's Helpful, Harmlessness and Honest (HHH) goals, we build a novel three-stage evaluation framework, which is carefully constructed to effectively and naturally expose such risks. We conduct 14,400 agentic simulations across 12 advanced LLMs, with extensive experiments and analysis. Results reveal that LLM agents can autonomously engage in catastrophic behaviors and deception, without being deliberately induced. Furthermore, stronger reasoning abilities often increase, rather than mitigate, these risks. We also show that these agents can violate instructions and superior commands. On the whole, we empirically prove the existence of catastrophic risks in autonomous LLM agents. We release our code to foster further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17953v1">Smoke and Mirrors: Jailbreaking LLM-based Code Generation via Implicit Malicious Prompts</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has revolutionized natural language processing and significantly impacted code generation tasks, enhancing software development efficiency and productivity. Notably, LLMs like GPT-4 have demonstrated remarkable proficiency in text-to-code generation tasks. However, the growing reliance on LLMs for code generation necessitates a critical examination of the safety implications associated with their outputs. Existing research efforts have primarily focused on verifying the functional correctness of LLMs, overlooking their safety in code generation. This paper introduces a jailbreaking approach, CodeJailbreaker, designed to uncover safety concerns in LLM-based code generation. The basic observation is that existing safety mechanisms for LLMs are built through the instruction-following paradigm, where malicious intent is explicitly articulated within the instruction of the prompt. Consequently, CodeJailbreaker explores to construct a prompt whose instruction is benign and the malicious intent is implicitly encoded in a covert channel, i.e., the commit message, to bypass the safety mechanism. Experiments on the recently-released RMCBench benchmark demonstrate that CodeJailbreaker markedly surpasses the conventional jailbreaking strategy, which explicitly conveys malicious intents in the instructions, in terms of the attack effectiveness across three code generation tasks. This study challenges the traditional safety paradigms in LLM-based code generation, emphasizing the need for enhanced safety measures in safeguarding against implicit malicious cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17922v1">WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      With the advancements in long-context inference capabilities of large language models (LLMs), the KV cache has become one of the foundational components. However, its substantial GPU memory consumption makes KV cache compression a key technique for enabling efficient LLM inference in industrial scenarios. While recent studies have focused on optimizing the memory occupied by the KV cache, they overlook two critical factors: preserving semantic coherence and considering task-specific characteristic during compression. To address these limitations, we propose a novel task-adaptive KV cache window selection method, WindowKV. WindowKV dynamically selects local semantic windows consisting of consecutive tokens, according to task-specific characteristics, ensuring the retained KV cache captures continuous, essential context. Additionally, we introduce an intra-group layer KV cache indices sharing strategy to reduce computational overhead, achieving a balance between performance and efficiency. We rigorously evaluate WindowKV on the LongBench benchmark, and the results demonstrate that it maintains a performance comparable to full KV cache retention while using only 12% of the original KV cache, significantly reducing memory requirements. Furthermore, our method also achieves state-of-the-art results in the Needle-in-a-Haystack evaluation, highlighting its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18995v1">LLMs in the Classroom: Outcomes and Perceptions of Questions Written with the Aid of AI</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Accepted to AAAI 2025 Technical Track on AI Alignment
    </div>
    <details class="paper-abstract">
      We randomly deploy questions constructed with and without use of the LLM tool and gauge the ability of the students to correctly answer, as well as their ability to correctly perceive the difference between human-authored and LLM-authored questions. In determining whether the questions written with the aid of ChatGPT were consistent with the instructor's questions and source text, we computed representative vectors of both the human and ChatGPT questions using SBERT and compared cosine similarity to the course textbook. A non-significant Mann-Whitney U test (z = 1.018, p = .309) suggests that students were unable to perceive whether questions were written with or without the aid of ChatGPT. However, student scores on LLM-authored questions were almost 9% lower (z = 2.702, p < .01). This result may indicate that either the AI questions were more difficult or that the students were more familiar with the instructor's style of questions. Overall, the study suggests that while there is potential for using LLM tools to aid in the construction of assessments, care must be taken to ensure that the questions are fair, well-composed, and relevant to the course material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18991v1">SRMIR: Shadow Reward Models Based on Introspective Reasoning for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human preferences and values is vital for application. However, current alignment methods face three main limitations: (1) reliance on costly human annotation; (2) alignment tax; (3) shallow alignment vulnerable to jailbreak attacks. Additionally, current alignment datasets often suffer from uneven distributions, leading to overrepresentation of some topics and neglect of others. To address these issues, we propose SRMIR (Shadow Reward Models Based on Introspective Reasoning), inspired by shadow models in membership inference attacks. We first construct a balanced safety Chain of Draft (CoD) dataset across $7$ harmful types with structured prompt leveraging the introspective reasoning capabilities of LLMs, then train a set of specialized reward models to guide policy optimization through Group Relative Policy Optimization (GRPO). We apply two strategies, linear combination and categorized approach, to integrate shadow reward models for policy optimization. By comparison, we find that the latter achieves superior alignment despite higher computational costs. Experiments across several LLMs demonstrate SRMIR significantly outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18986v1">SplitFrozen: Split Learning with Device-side Model Frozen for Fine-Tuning LLM on Heterogeneous Resource-Constrained Devices</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on private, on-device data can empower tailored personalized AI agents. However, fine-tuning LLMs on resource-constrained edge devices faces significant challenges, including excessive computation overhead, device heterogeneity, and data imbalance. This paper proposes SplitFrozen, a split learning framework that enables efficient LLM fine-tuning by strategically freezing device-side model layers while centralizing parameter-efficient fine-tuning on the server. Our framework partitions LLMs into device-side frozen layers and server-side fine-tuning layers, where heterogeneous resource-constrained devices execute only forward propagation. To minimize server-side training costs, we integrate Low-Rank Adaptation (LoRA) into the server-side layers. A pipeline parallelism strategy further optimizes training efficiency by decoupling device-server computations and leveraging decomposed backward propagation. Experiments on GPT-2 with the MRPC, MNLI-matched, and SST-2 datasets demonstrate that SplitFrozen outperforms FedLoRA and SplitLoRA by 69.4\% model accuracy under extremely imbalanced data, while reducing up to 86.8\% device-side computations and 50.2\% total training time. Experiments also validate the scalability of SplitFrozen on content generation task using Llama-3.2 model on GSM8K dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17885v1">Reasoning with LLMs for Zero-Shot Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Automating software vulnerability detection (SVD) remains a critical challenge in an era of increasingly complex and interdependent software systems. Despite significant advances in Large Language Models (LLMs) for code analysis, prevailing evaluation methodologies often lack the \textbf{context-aware robustness} necessary to capture real-world intricacies and cross-component interactions. To address these limitations, we present \textbf{VulnSage}, a comprehensive evaluation framework and a dataset curated from diverse, large-scale open-source system software projects developed in C/C++. Unlike prior datasets, it leverages a heuristic noise pre-filtering approach combined with LLM-based reasoning to ensure a representative and minimally noisy spectrum of vulnerabilities. The framework supports multi-granular analysis across function, file, and inter-function levels and employs four diverse zero-shot prompt strategies: Baseline, Chain-of-Thought, Think, and Think & Verify. Through this evaluation, we uncover that structured reasoning prompts substantially improve LLM performance, with Think & Verify reducing ambiguous responses from 20.3% to 9.1% while increasing accuracy. We further demonstrate that code-specialized models consistently outperform general-purpose alternatives, with performance varying significantly across vulnerability types, revealing that no single approach universally excels across all security contexts. Link to dataset and codes: https://github.com/Erroristotle/VulnSage.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17882v1">Think Before Refusal : Triggering Safety Reflection in LLMs to Mitigate False Refusal Behavior</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 18 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated that fine-tuning and human alignment can render LLMs harmless. In practice, such "harmlessness" behavior is mainly achieved by training models to reject harmful requests, such as "Explain how to burn down my neighbor's house", where the model appropriately declines to respond. However, this approach can inadvertently result in false refusal, where models reject benign queries as well, such as "Tell me how to kill a Python process". In this work, we demonstrate that prompting safety reflection before generating a response can mitigate false refusal behavior. Building on this finding, we introduce the Think-Before-Refusal (TBR) schema and conduct safety-aware instruction fine-tuning incorporating safety reflection. In an ablation study across 15 pre-trained models, we show that models fine-tuned with safety reflection significantly reduce false refusal behavior while maintaining safety and overall performance compared to those fine-tuned without safety reflection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17213v6">Plurals: A System for Guiding LLMs Via Simulated Social Ensembles</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 CHI 2025
    </div>
    <details class="paper-abstract">
      Recent debates raised concerns that language models may favor certain viewpoints. But what if the solution is not to aim for a 'view from nowhere' but rather to leverage different viewpoints? We introduce Plurals, a system and Python library for pluralistic AI deliberation. Plurals consists of Agents (LLMs, optionally with personas) which deliberate within customizable Structures, with Moderators overseeing deliberation. Plurals is a generator of simulated social ensembles. Plurals integrates with government datasets to create nationally representative personas, includes deliberation templates inspired by deliberative democracy, and allows users to customize both information-sharing structures and deliberation behavior within Structures. Six case studies demonstrate fidelity to theoretical constructs and efficacy. Three randomized experiments show simulated focus groups produced output resonant with an online sample of the relevant audiences (chosen over zero-shot generation in 75% of trials). Plurals is both a paradigm and a concrete system for pluralistic AI. The Plurals library is available at https://github.com/josh-ashkinaze/plurals and will be continually updated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20576v3">Smart Routing: Cost-Effective Multi-LLM Serving in AIOS</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as service endpoints in systems, the surge in query volume creates significant scheduling challenges. Existing scheduling frameworks mainly target at latency optimization while neglecting the capability of LLMs to serve different level of queries, which could lead to computational resource waste. For example, those simple queries can be safely handled by small, fast and cheap LLMs, while those complex and difficult queries need to be handled by large, slow, and expensive LLMs. This paper addresses this challenge by proposing an efficient capability-cost coordinated scheduling framework, ECCOS, for multi-LLM serving, which explicitly constrains response quality and workload to optimize LLM inference cost. Specifically, it introduces the two-stage scheduling by designing a multi-objective predictor and a constrained optimizer. The predictor estimates both model capabilities and computational costs through training-based and retrieval-based approaches, while the optimizer determines cost-optimal assignments under quality and workload constraints. It also introduces QAServe, a dataset for sample-wise response quality and costs collected by zero-shot prompting different LLMs on knowledge QA and mathematical reasoning. Extensive experiments demonstrate that ECCOS improves success rates by 6.30% while reducing costs by 10.15% compared to existing methods, consuming less than 0.5% of LLM response time. The code is available at: https://github.com/agiresearch/ECCOS, and the proposed smart routing mechanism has been integrated into AIOS, the AI Agent Operating System, at https://github.com/agiresearch/AIOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05400v3">Dynamic Noise Preference Optimization for LLM Self-Improvement via Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Although LLMs have achieved significant success, their reliance on large volumes of human-annotated data has limited their potential for further scaling. In this situation, utilizing self-generated synthetic data has become crucial for fine-tuning LLMs without extensive human annotation. However, current methods often fail to ensure consistent improvements across iterations, with performance stagnating after only minimal updates. To overcome these challenges, we introduce Dynamic Noise Preference Optimization (DNPO). DNPO employs a dynamic sample labeling mechanism to construct preference pairs for training and introduces controlled, trainable noise into the preference optimization process. Our approach effectively prevents stagnation and enables continuous improvement. In experiments with Zephyr-7B, DNPO consistently outperforms existing methods, showing an average performance boost of 2.6% across multiple benchmarks. Additionally, DNPO shows a significant improvement in model-generated data quality, with a 29.4% win-loss rate gap compared to the baseline in GPT-4 evaluations. This highlights its effectiveness in enhancing model performance through iterative refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21526v2">Not All LLM-Generated Data Are Equal: Rethinking Data Weighting in Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 ICLR 2025 camera ready
    </div>
    <details class="paper-abstract">
      Synthetic data augmentation via large language models (LLMs) allows researchers to leverage additional training data, thus enhancing the performance of downstream tasks, especially when real-world data is scarce. However, the generated data can deviate from the real-world data, and this misalignment can bring deficient outcomes while applying the trained model to applications. Therefore, we proposed efficient weighted-loss approaches to align synthetic data with real-world distribution by emphasizing high-quality and diversified data generated by LLMs with using merely a little real-world data. We empirically assessed the effectiveness of our method on multiple text classification tasks, and the results showed leveraging our approaches on a BERT-level model robustly outperformed standard cross-entropy and other data weighting approaches, providing potential solutions to effectively leveraging synthetic data from any suitable data generator for model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17793v1">Every Sample Matters: Leveraging Mixture-of-Experts and High-Quality Data for Efficient and Accurate Code LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in code large language models (LLMs) have demonstrated remarkable capabilities in code generation and understanding. It is still challenging to build a code LLM with comprehensive performance yet ultimate efficiency. Many attempts have been released in the open source community to break the trade-off between performance and efficiency, such as the Qwen Coder series and the DeepSeek Coder series. This paper introduces yet another attempt in this area, namely Ling-Coder-Lite. We leverage the efficient Mixture-of-Experts (MoE) architecture along with a set of high-quality data curation methods (especially those based on program analytics) to build an efficient yet powerful code LLM. Ling-Coder-Lite exhibits on-par performance on 12 representative coding benchmarks compared to state-of-the-art models of similar size, such as Qwen2.5-Coder-7B and DeepSeek-Coder-V2-Lite, while offering competitive latency and throughput. In practice, we achieve a 50\% reduction in deployment resources compared to the similar-sized dense model without performance loss. To facilitate further research and development in this area, we open-source our models as well as a substantial portion of high-quality data for the annealing and post-training stages. The models and data can be accessed at~\url{https://huggingface.co/inclusionAI/Ling-Coder-lite}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17783v1">Energy-Aware LLMs: A step towards sustainable AI for downstream applications</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 This work has been submitted to V. International Conference on Electrical, Computer and Energy Technologies (ICECET 2025) for possible publication
    </div>
    <details class="paper-abstract">
      Advanced Large Language Models (LLMs) have revolutionized various fields, including communication networks, sparking an innovation wave that has led to new applications and services, and significantly enhanced solution schemes. Despite all these impressive developments, most LLMs typically require huge computational resources, resulting in terribly high energy consumption. Thus, this research study proposes an end-to-end pipeline that investigates the trade-off between energy efficiency and model performance for an LLM during fault ticket analysis in communication networks. It further evaluates the pipeline performance using two real-world datasets for the tasks of root cause analysis and response feedback in a communication network. Our results show that an appropriate combination of quantization and pruning techniques is able to reduce energy consumption while significantly improving model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14744v5">Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to face hallucination issues due to the data they trained on often containing human bias; whether this is reflected in the decision-making process of LLM Agents remains under-explored. As LLM Agents are increasingly employed in intricate social environments, a pressing and natural question emerges: Can we utilize LLM Agents' systematic hallucinations to mirror human cognitive biases, thus exhibiting irrational social intelligence? In this paper, we probe the irrational behavior among contemporary LLM Agents by melding practical social science experiments with theoretical insights. Specifically, We propose CogMir, an open-ended Multi-LLM Agents framework that utilizes hallucination properties to assess and enhance LLM Agents' social intelligence through cognitive biases. Experimental results on CogMir subsets show that LLM Agents and humans exhibit high consistency in irrational and prosocial decision-making under uncertain conditions, underscoring the prosociality of LLM Agents as social entities and highlighting the significance of hallucination properties. Additionally, the CogMir framework demonstrates its potential as a valuable platform for encouraging more research into the social intelligence of LLM Agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17755v1">Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 preprint, submitted to ACL ARR 2025, 21 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are often used as automated judges to evaluate text, but their effectiveness can be hindered by various unintentional biases. We propose using linear classifying probes, trained by leveraging differences between contrasting pairs of prompts, to directly access LLMs' latent knowledge and extract more accurate preferences. Through extensive experiments using models of varying size from four different families and six diverse datasets assessing text quality evaluation and common sense reasoning, we demonstrate that both supervised and unsupervised probing approaches consistently outperform traditional generation-based judgement while maintaining similar computational costs. These probes generalise under domain shifts and can even outperform finetuned evaluators with the same training data size. Our results suggest linear probing offers an accurate, robust and computationally efficient approach for LLM-as-judge tasks while providing interpretable insights into how models encode judgement-relevant knowledge. Our data and code will be openly released in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17741v1">RustMap: Towards Project-Scale C-to-Rust Migration via Program Analysis and LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Migrating existing C programs into Rust is increasingly desired, as Rust offers superior memory safety while maintaining C's high performance. However, vastly different features between C and Rust--e.g., distinct definitions and usages of pointers and references--pose significant challenges beyond mere syntactic translation. Existing automated translation tools, such as C2Rust, may rely too much on syntactic, template-based translation and generate unsafe Rust code that is hard for human developers to read, maintain, or even compile. More semantic-aware translation that produces safer, idiomatic, and runnable Rust code is much needed. This paper introduces a novel dependency-guided and large language model (LLM)-based C-to-Rust translation approach, RustMap, based on three key ideas: (1) Utilize LLM capabilities to produce idiomatic Rust code from given small pieces of C code, (2) Mitigate LLM limitations in handling large codebases by breaking project-scale C programs into smaller units for translation according to their usage dependencies and composing them into a runnable Rust program, and (3) Enhance the correctness of the translated Rust program by using test cases to check input/output equivalence, isolate faulty code when execution states deviate, and iteratively refine the translation using feedback from compilation and test errors. We empirically evaluate RustMap on 126 real-world programs, including 125 from Rosetta Code and a 7000+ line bzip2 implementation using GPT-4o as the LLM. RustMap shows promising results, guiding GPT-4o to produce idiomatic, readable, and functional Rust code with significantly less unsafe code than other tools, and revealing non-trivial translation patterns reusable for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07376v2">NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 Accepted by TPAMI 2025
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN), as a crucial research problem of Embodied AI, requires an embodied agent to navigate through complex 3D environments following natural language instructions. Recent research has highlighted the promising capacity of large language models (LLMs) in VLN by improving navigational reasoning accuracy and interpretability. However, their predominant use in an offline manner usually suffers from substantial domain gap between the VLN task and the LLM training corpus. This paper introduces a novel strategy called Navigational Chain-of-Thought (NavCoT), where we fulfill parameter-efficient in-domain training to enable self-guided navigational decision, leading to a significant mitigation of the domain gap in a cost-effective manner. Specifically, at each timestep, the LLM is prompted to forecast the navigational chain-of-thought by: 1) acting as a world model to imagine the next observation according to the instruction, 2) selecting the candidate observation that best aligns with the imagination, and 3) determining the action based on the reasoning from the prior steps. Through constructing formalized labels for training, the LLM can learn to generate desired and reasonable chain-of-thought outputs for improving the action decision. Experimental results across various training settings and popular VLN benchmarks (e.g., Room-to-Room (R2R), Room-across-Room (RxR), Room-for-Room (R4R)) show the significant superiority of NavCoT over the direct action prediction variants. Through simple parameter-efficient finetuning, our NavCoT outperforms a recent GPT4-based approach with ~7% relative improvement on the R2R dataset. We believe that NavCoT will help unlock more task-adaptive and scalable LLM-based embodied agents, which are helpful for developing real-world robotics applications. Code is available at https://github.com/expectorlin/NavCoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07920v3">Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-Ranking</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Cross-encoders distilled from large language models (LLMs) are often more effective re-rankers than cross-encoders fine-tuned on manually labeled data. However, distilled models do not match the effectiveness of their teacher LLMs. We hypothesize that this effectiveness gap is due to the fact that previous work has not applied the best-suited methods for fine-tuning cross-encoders on manually labeled data (e.g., hard-negative sampling, deep sampling, and listwise loss functions). To close this gap, we create a new dataset, Rank-DistiLLM. Cross-encoders trained on Rank-DistiLLM achieve the effectiveness of LLMs while being up to 173 times faster and 24 times more memory efficient. Our code and data is available at https://github.com/webis-de/ECIR-25.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17710v1">Slide2Text: Leveraging LLMs for Personalized Textbook Generation from PowerPoint Presentations</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      The rapid advancements in Large Language Models (LLMs) have revolutionized educational technology, enabling innovative approaches to automated and personalized content creation. This paper introduces Slide2Text, a system that leverages LLMs to transform PowerPoint presentations into customized textbooks. By extracting slide content using OCR, organizing it into a coherent structure, and generating tailored materials such as explanations, exercises, and references, Slide2Text streamlines the textbook creation process. Flexible customization options further enhance its adaptability to diverse educational needs. The system highlights the potential of LLMs in modernizing textbook creation and improving educational accessibility. Future developments will explore multimedia inputs and advanced user customization features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11183v2">Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Recent advancements in tree search algorithms guided by verifiers have significantly enhanced the reasoning capabilities of large language models (LLMs), but at the cost of increased computational resources. In this work, we identify two key challenges contributing to this inefficiency: $\textit{over-exploration}$ due to redundant states with semantically equivalent content, and $\textit{under-exploration}$ caused by high variance in verifier scoring leading to frequent trajectory switching. To address these issues, we propose FETCH, an e$\textbf{f}$fici$\textbf{e}$nt $\textbf{t}$ree sear$\textbf{ch}$ framework, which is a flexible, plug-and-play system compatible with various tree search algorithms. Our framework mitigates over-exploration by merging semantically similar states using agglomerative clustering of text embeddings obtained from a fine-tuned SimCSE model. To tackle under-exploration, we enhance verifiers by incorporating temporal difference learning with adjusted $\lambda$-returns during training to reduce variance, and employing a verifier ensemble to aggregate scores during inference. Experiments on GSM8K, GSM-Plus, and MATH datasets demonstrate that our methods significantly improve reasoning accuracy and computational efficiency across four different tree search algorithms, paving the way for more practical applications of LLM-based reasoning. The code is available at https://github.com/Soistesimmer/Fetch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17707v1">PipeBoost: Resilient Pipelined Architecture for Fast Serverless LLM Scaling</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      This paper presents PipeBoost, a low-latency LLM serving system for multi-GPU (serverless) clusters, which can rapidly launch inference services in response to bursty requests without preemptively over-provisioning GPUs. Many LLM inference tasks rely on the same base model (e.g., LoRA). To leverage this, PipeBoost introduces fault-tolerant pipeline parallelism across both model loading and inference stages. This approach maximizes aggregate PCIe bandwidth and parallel computation across GPUs, enabling faster generation of the first token. PipeBoost also introduces recovery techniques that enable uninterrupted inference services by utilizing the shared advantages of multiple GPUs. Experimental results show that, compared to state-of-the-art low-latency LLM serving systems, PipeBoost reduces inference latency by 31% to 49.8%. For certain models (e.g., OPT-1.3B), PipeBoost achieves cold-start latencies in the range of a few hundred microseconds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17684v1">Can LLMs Automate Fact-Checking Article Writing?</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 10 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Automatic fact-checking aims to support professional fact-checkers by offering tools that can help speed up manual fact-checking. Yet, existing frameworks fail to address the key step of producing output suitable for broader dissemination to the general public: while human fact-checkers communicate their findings through fact-checking articles, automated systems typically produce little or no justification for their assessments. Here, we aim to bridge this gap. We argue for the need to extend the typical automatic fact-checking pipeline with automatic generation of full fact-checking articles. We first identify key desiderata for such articles through a series of interviews with experts from leading fact-checking organizations. We then develop QRAFT, an LLM-based agentic framework that mimics the writing workflow of human fact-checkers. Finally, we assess the practical usefulness of QRAFT through human evaluations with professional fact-checkers. Our evaluation shows that while QRAFT outperforms several previously proposed text-generation approaches, it lags considerably behind expert-written articles. We hope that our work will enable further research in this new and important direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17662v1">Enhancing Persona Consistency for LLMs' Role-Playing using Persona-Aware Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have achieved breakthrough progress in many dialogue generation tasks. However, their lack of emotion and fine-grained role awareness limits the model's ability to provide personalized and diverse interactions further. Current methods face high costs in collecting high-quality annotated data for scenarios such as role-playing, and traditional human alignment methods are difficult to deploy due to the inherent diversity of model behavior in role-playing scenarios. Inspired by the alignment of models for safety behaviors through RLHF (Reinforcement Learning from Human Feedback), in this paper, we revisit model role-playing behavior from the perspective of persona alignment and propose a novel annotation-free framework named \textbf{\underline{P}}ersona-Aware \textbf{\underline{C}}ontrastive \textbf{\underline{L}}earning (PCL) to align LLMs' behavior during role-playing, enhancing the model's role consistency. Specifically, we first design a role chain method to encourage the model to self-question based on the role characteristics and dialogue context to adjust personality consistency. Then, we further enhance the model's role-playing strategy through iterative contrastive learning between the use of role characteristics and not. Experiments on both black-box and white-box LLMs show that LLMs equipped with PCL significantly outperform vanilla LLMs under automatic evaluation methods (CharEval \& GPT-4) and human expert evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17645v1">A Modular Dataset to Demonstrate LLM Abstraction Capability</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 7 pages, 5 figures. Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive capabilities but struggle with reasoning errors due to hallucinations and flawed logic. To investigate their internal representations of reasoning, we introduce ArrangementPuzzle, a novel puzzle dataset with structured solutions and automated stepwise correctness verification. We trained a classifier model on LLM activations on this dataset and found that it achieved over 80% accuracy in predicting reasoning correctness, implying that LLMs internally distinguish between correct and incorrect reasoning steps, with the strongest representations in middle-late Transformer layers. Further analysis reveals that LLMs encode abstract reasoning concepts within the middle activation layers of the transformer architecture, distinguishing logical from semantic equivalence. These findings provide insights into LLM reasoning mechanisms and contribute to improving AI reliability and interpretability, thereby offering the possibility to manipulate and refine LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17620v1">A Case Study of Scalable Content Annotation Using Multi-LLM Consensus and Human Review</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 4 pages, GenAICHI 2025 accepted
    </div>
    <details class="paper-abstract">
      Content annotation at scale remains challenging, requiring substantial human expertise and effort. This paper presents a case study in code documentation analysis, where we explore the balance between automation efficiency and annotation accuracy. We present MCHR (Multi-LLM Consensus with Human Review), a novel semi-automated framework that enhances annotation scalability through the systematic integration of multiple LLMs and targeted human review. Our framework introduces a structured consensus-building mechanism among LLMs and an adaptive review protocol that strategically engages human expertise. Through our case study, we demonstrate that MCHR reduces annotation time by 32% to 100% compared to manual annotation while maintaining high accuracy (85.5% to 98%) across different difficulty levels, from basic binary classification to challenging open-set scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17604v1">OmniScience: A Domain-Specialized LLM for Scientific Reasoning and Discovery</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in advancing scientific knowledge and addressing complex challenges. In this work, we introduce OmniScience, a specialized large reasoning model for general science, developed through three key components: (1) domain adaptive pretraining on a carefully curated corpus of scientific literature, (2) instruction tuning on a specialized dataset to guide the model in following domain-specific tasks, and (3) reasoning-based knowledge distillation through fine-tuning to significantly enhance its ability to generate contextually relevant and logically sound responses. We demonstrate the versatility of OmniScience by developing a battery agent that efficiently ranks molecules as potential electrolyte solvents or additives. Comprehensive evaluations reveal that OmniScience is competitive with state-of-the-art large reasoning models on the GPQA Diamond and domain-specific battery benchmarks, while outperforming all public reasoning and non-reasoning models with similar parameter counts. We further demonstrate via ablation experiments that domain adaptive pretraining and reasoning-based knowledge distillation are critical to attain our performance levels, across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17587v1">ConSol: Sequential Probability Ratio Testing to Find Consistent LLM Reasoning Paths Efficiently</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) integrating explicit reasoning, such as OpenAI's o3-mini, DeepSeek-R1, and QWQ-32B, enable smaller models to solve complex tasks by generating intermediate reasoning steps prior to providing answers. However, this approach significantly increases computational costs, both monetarily and environmentally. The widely-used self-consistency method further exacerbates these costs by aggregating multiple reasoning paths to improve accuracy, often requiring between 40 to 64 samples per task. Although aggregation effectively reduces variance and bias, additional sampling can lead to diminishing returns when early samples yield consistent results. To address inefficiencies, we propose leveraging Sequential Probability Ratio Testing (SPRT) to dynamically terminate sampling once sufficient consistency is achieved. We calibrate SPRT parameters specifically for LLM applications, accounting for sensitivity to detect the mode of the distribution. Our experiments demonstrate that incorporating SPRT significantly enhances token efficiency, achieving comparable accuracy to self-consistency methods but at a substantially reduced computational cost. To promote transparency and facilitate reproducibility, we have made the source code and datasets used in our experiments publicly available at our GitHub repository: https://github.com/LiuzLab/consol, or available as a PyPI package: pip install consol. We hope that this resource will support further research and encourage the development of new methods building upon our work.
    </details>
</div>
