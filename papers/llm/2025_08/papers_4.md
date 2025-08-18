# llm - 2025_08

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23145v2">CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Inductive program synthesis, or programming by example, requires synthesizing functions from input-output examples that generalize to unseen inputs. While large language model agents have shown promise in programming tasks guided by natural language, their ability to perform inductive program synthesis is underexplored. Existing evaluation protocols rely on static sets of examples and held-out tests, offering no feedback when synthesized functions are incorrect and failing to reflect real-world scenarios such as reverse engineering. We propose CodeARC, the Code Abstraction and Reasoning Challenge, a new evaluation framework where agents interact with a hidden target function by querying it with new inputs, synthesizing candidate functions, and iteratively refining their solutions using a differential testing oracle. This interactive setting encourages agents to perform function calls and self-correction based on feedback. We construct the first large-scale benchmark for general-purpose inductive program synthesis, featuring 1114 functions. Among 18 models evaluated, o3-mini performs best with a success rate of 52.7%, highlighting the difficulty of this task. Fine-tuning LLaMA-3.1-8B-Instruct on curated synthesis traces yields up to a 31% relative performance gain. CodeARC provides a more realistic and challenging testbed for evaluating LLM-based program synthesis and inductive reasoning. Our code, data, and models are publicly available at https://github.com/Anjiang-Wei/CodeARC
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18602v2">LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized algorithm development, yet their application in symbolic regression, where algorithms automatically discover symbolic expressions from data, remains constrained and is typically designed manually by human experts. In this paper, we propose a meta learning framework that enables LLMs to automatically design selection operators for evolutionary symbolic regression algorithms. We first identify two key limitations in existing LLM-based algorithm evolution techniques: a lack of semantic guidance and code bloat. The absence of semantic awareness can lead to ineffective exchange of useful code components, and bloat results in unnecessarily complex components, both of which can reduce the interpretability of the designed algorithm or hinder evolutionary learning progress. To address these issues, we enhance the LLM-based evolution framework for meta symbolic regression with two key innovations: a complementary, semantics-aware selection operator and bloat control. Additionally, we embed domain knowledge into the prompt, enabling the LLM to generate more effective and contextually relevant selection operators. Our experimental results on symbolic regression benchmarks show that LLMs can devise selection operators that outperform nine expert-designed baselines, achieving state-of-the-art performance. Moreover, the evolved operator can further improve the state-of-the-art symbolic regression algorithm, achieving the best performance among 26 symbolic regression and machine learning algorithms across 116 regression datasets. This demonstrates that LLMs can exceed expert-level algorithm design for symbolic regression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06060v1">LLMs for Resource Allocation: A Participatory Budgeting Approach to Inferring Preferences</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Published in the Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly expected to handle complex decision-making tasks, yet their ability to perform structured resource allocation remains underexplored. Evaluating their reasoning is also difficult due to data contamination and the static nature of existing benchmarks. We present a dual-purpose framework leveraging Participatory Budgeting (PB) both as (i) a practical setting for LLM-based resource allocation and (ii) an adaptive benchmark for evaluating their reasoning capabilities. We task LLMs with selecting project subsets under feasibility (e.g., budget) constraints via three prompting strategies: greedy selection, direct optimization, and a hill-climbing-inspired refinement. We benchmark LLMs' allocations against a utility-maximizing oracle. Interestingly, we also test whether LLMs can infer structured preferences from natural-language voter input or metadata, without explicit votes. By comparing allocations based on inferred preferences to those from ground-truth votes, we evaluate LLMs' ability to extract preferences from open-ended input. Our results underscore the role of prompt design and show that LLMs hold promise for mechanism design with unstructured inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10970v4">The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The "LLM-as-an-annotator" and "LLM-as-a-judge" paradigms employ Large Language Models (LLMs) as annotators, judges, and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure, the Alternative Annotator Test (alt-test), that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM annotators and judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming the open-source LLMs we examine, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06047v1">ArchXBench: A Complex Digital Systems Benchmark Suite for LLM Driven RTL Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Published in 7th ACM/IEEE International Symposium on Machine Learning for CAD
    </div>
    <details class="paper-abstract">
      Modern SoC datapaths include deeply pipelined, domain-specific accelerators, but their RTL implementation and verification are still mostly done by hand. While large language models (LLMs) exhibit advanced code-generation abilities for programming languages like Python, their application to Verilog-like RTL remains in its nascent stage. This is reflected in the simple arithmetic and control circuits currently used to evaluate generative capabilities in existing benchmarks. In this paper, we introduce ArchXBench, a six-level benchmark suite that encompasses complex arithmetic circuits and other advanced digital subsystems drawn from domains such as cryptography, image processing, machine learning, and signal processing. Architecturally, some of these designs are purely combinational, others are multi-cycle or pipelined, and many require hierarchical composition of modules. For each benchmark, we provide a problem description, design specification, and testbench, enabling rapid research in the area of LLM-driven agentic approaches for complex digital systems design. Using zero-shot prompting with Claude Sonnet 4, GPT 4.1, o4-mini-high, and DeepSeek R1 under a pass@5 criterion, we observed that o4-mini-high successfully solves the largest number of benchmarks, 16 out of 30, spanning Levels 1, 2, and 3. From Level 4 onward, however, all models consistently fail, highlighting a clear gap in the capabilities of current state-of-the-art LLMs and prompting/agentic approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08775v3">Layers at Similar Depths Generate Similar Activations Across LLM Architectures</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      How do the latent spaces used by independently-trained LLMs relate to one another? We study the nearest neighbor relationships induced by activations at different layers of 24 open-weight LLMs, and find that they 1) tend to vary from layer to layer within a model, and 2) are approximately shared between corresponding layers of different models. Claim 2 shows that these nearest neighbor relationships are not arbitrary, as they are shared across models, but Claim 1 shows that they are not "obvious" either, as there is no single set of nearest neighbor relationships that is universally shared. Together, these suggest that LLMs generate a progression of activation geometries from layer to layer, but that this entire progression is largely shared between models, stretched and squeezed to fit into different architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05298v2">GhostShell: Streaming LLM Function Calls for Concurrent Embodied Programming</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 17 pages, 5 figures, conference
    </div>
    <details class="paper-abstract">
      We present GhostShell, a novel approach that leverages Large Language Models (LLMs) to enable streaming and concurrent behavioral programming for embodied systems. In contrast to conventional methods that rely on pre-scheduled action sequences or behavior trees, GhostShell drives embodied systems to act on-the-fly by issuing function calls incrementally as tokens are streamed from the LLM. GhostShell features a streaming XML function token parser, a dynamic function interface mapper, and a multi-channel scheduler that orchestrates intra-channel synchronous and inter-channel asynchronous function calls, thereby coordinating serial-parallel embodied actions across multiple robotic components under LLM guidance. We evaluate GhostShell on our robotic prototype COCO through comprehensive grounded experiments across 34 real-world interaction tasks and multiple LLM backends. The results demonstrate that our approach achieves a state-of-the-art Behavioral Correctness Metric of 0.85 with Claude-4-Sonnet, and up to 66X faster response times compared to native LLM function calling APIs. GhostShell also proves effective in long-horizon multimodal tasks, exhibiting strong robustness and generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06004v1">When a Paper Has 1000 Authors: Rethinking Citation Metrics in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Author-level citation metrics provide a practical, interpretable, and scalable signal of scholarly influence in a complex research ecosystem. It has been widely used as a proxy in hiring decisions. However, the past five years have seen the rapid emergence of large-scale publications in the field of large language models and foundation models, with papers featuring hundreds to thousands of co-authors and receiving tens of thousands of citations within months. For example, Gemini has 1361 authors and has been cited around 4600 times in 19 months. In such cases, traditional metrics, such as total citation count and the $h$-index, fail to meaningfully distinguish individual contributions. Therefore, we propose the following research question: How can one identify standout researchers among thousands of co-authors in large-scale LLM papers? This question is particularly important in scenarios such as academic hiring and funding decisions. In this paper, we introduce a novel citation metric designed to address this challenge by balancing contributions across large-scale and small-scale publications. We propose the SBCI index, analyze its theoretical properties, and evaluate its behavior on synthetic publication datasets. Our results demonstrate that the proposed metric provides a more robust and discriminative assessment of individual scholarly impact in the era of large-scale collaborations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06000v1">Hand by Hand: LLM Driving EMS Assistant for Operational Skill Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Accepted by IJCAI 2025
    </div>
    <details class="paper-abstract">
      Operational skill learning, inherently physical and reliant on hands-on practice and kinesthetic feedback, has yet to be effectively replicated in large language model (LLM)-supported training. Current LLM training assistants primarily generate customized textual feedback, neglecting the crucial kinesthetic modality. This gap derives from the textual and uncertain nature of LLMs, compounded by concerns on user acceptance of LLM driven body control. To bridge this gap and realize the potential of collaborative human-LLM action, this work explores human experience of LLM driven kinesthetic assistance. Specifically, we introduced an "Align-Analyze-Adjust" strategy and developed FlightAxis, a tool that integrates LLM with Electrical Muscle Stimulation (EMS) for flight skill acquisition, a representative operational skill domain. FlightAxis learns flight skills from manuals and guides forearm movements during simulated flight tasks. Our results demonstrate high user acceptance of LLM-mediated body control and significantly reduced task completion times. Crucially, trainees reported that this kinesthetic assistance enhanced their awareness of operation flaws and fostered increased engagement in the training process, rather than relieving perceived load. This work demonstrated the potential of kinesthetic LLM training in operational skill acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05995v1">Optimizing Prompt Sequences using Monte Carlo Tree Search for LLM-Based Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in code generation and structured reasoning; however, their performance often degrades on complex tasks that require consistent multi-step planning. Recent work has explored combining LLMs with Monte Carlo Tree Search (MCTS), yet existing approaches primarily focus on generating heuristic-based code for optimization or target simpler tasks where correctness alone is sufficient. In this work, we propose MCTS-OPS, a novel neural-symbolic framework that formulates prompt selection as a sequential decision process guided by MCTS. Our method explores and refines multi-step prompt sequences for the goal of improving code generation quality and enhancing the problem-solving capabilities of LLMs in general optimization. Experiments on network optimization show significant improvement over the baselines, both in the success rate of executing the generated code and in the optimization results with the specified objective and constraints (2$\sim$4$\times$ higher reward and 3$\times$ lower standard deviation). Moreover, it improves the chance of attaining the optimal solution by about 10\% of cases, compared to baseline methods in hard problems. These results highlight the promise of combining symbolic planning with LLMs for robust, high-quality code generation in complex domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05028v2">Evaluation of LLMs in AMR Parsing</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 27 pages, 32 figures
    </div>
    <details class="paper-abstract">
      AMR (Abstract Meaning Representation) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05954v1">Bifrost-1: Bridging Multimodal LLMs and Diffusion Models with Patch-level CLIP Latents</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Project Page: https://bifrost-1.github.io
    </div>
    <details class="paper-abstract">
      There is growing interest in integrating high-fidelity visual synthesis capabilities into large language models (LLMs) without compromising their strong reasoning capabilities. Existing methods that directly train LLMs or bridge LLMs and diffusion models usually suffer from costly training since the backbone LLMs have not seen image representations during pretraining. We present Bifrost-1, a unified framework that bridges pretrained multimodal LLMs (MLLMs) and diffusion models using patch-level CLIP image embeddings as latent variables, which are natively aligned with the MLLM's CLIP visual encoder. These patch-level image embeddings are integrated into the diffusion model with a lightweight adaptation of its ControlNet. To retain the original multimodal reasoning capabilities of MLLMs, we equip the MLLM with a visual generation branch initialized from the original MLLM parameters when predicting the patch-level image embeddings. By seamlessly integrating pretrained MLLMs and diffusion models with patch-level CLIP latents, our framework enables high-fidelity controllable image generation with significant training efficiency. Our experiments demonstrate that Bifrost-1 achieves comparable or better performance than previous methods in terms of visual fidelity and multimodal understanding, with substantially lower compute during training. We also provide comprehensive ablation studies showing the effectiveness of our design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05953v1">SCALEFeedback: A Large-Scale Dataset of Synthetic Computer Science Assignments for LLM-generated Educational Feedback Research</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Using LLMs to give educational feedback to students for their assignments has attracted much attention in the AI in Education field. Yet, there is currently no large-scale open-source dataset of student assignments that includes detailed assignment descriptions, rubrics, and student submissions across various courses. As a result, research on generalisable methodology for automatic generation of effective and responsible educational feedback remains limited. In the current study, we constructed a large-scale dataset of Synthetic Computer science Assignments for LLM-generated Educational Feedback research (SCALEFeedback). We proposed a Sophisticated Assignment Mimicry (SAM) framework to generate the synthetic dataset by one-to-one LLM-based imitation from real assignment descriptions, student submissions to produce their synthetic versions. Our open-source dataset contains 10,000 synthetic student submissions spanning 155 assignments across 59 university-level computer science courses. Our synthetic submissions achieved BERTScore F1 0.84, PCC of 0.62 for assignment marks and 0.85 for length, compared to the corresponding real-world assignment dataset, while ensuring perfect protection of student private information. All these results of our SAM framework outperformed results of a naive mimicry method baseline. The LLM-generated feedback for our synthetic assignments demonstrated the same level of effectiveness compared to that of real-world assignment dataset. Our research showed that one-to-one LLM imitation is a promising method for generating open-source synthetic educational datasets that preserve the original dataset's semantic meaning and student data distribution, while protecting student privacy and institutional copyright. SCALEFeedback enhances our ability to develop LLM-based generalisable methods for offering high-quality, automated educational feedback in a scalable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05952v1">Dean of LLM Tutors: Exploring Comprehensive and Automated Evaluation of LLM-generated Educational Feedback via LLM Feedback Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The use of LLM tutors to provide automated educational feedback to students on student assignment submissions has received much attention in the AI in Education field. However, the stochastic nature and tendency for hallucinations in LLMs can undermine both quality of learning experience and adherence to ethical standards. To address this concern, we propose a method that uses LLM feedback evaluators (DeanLLMs) to automatically and comprehensively evaluate feedback generated by LLM tutor for submissions on university assignments before it is delivered to students. This allows low-quality feedback to be rejected and enables LLM tutors to improve the feedback they generated based on the evaluation results. We first proposed a comprehensive evaluation framework for LLM-generated educational feedback, comprising six dimensions for feedback content, seven for feedback effectiveness, and three for hallucination types. Next, we generated a virtual assignment submission dataset covering 85 university assignments from 43 computer science courses using eight commonly used commercial LLMs. We labelled and open-sourced the assignment dataset to support the fine-tuning and evaluation of LLM feedback evaluators. Our findings show that o3-pro demonstrated the best performance in zero-shot labelling of feedback while o4-mini demonstrated the best performance in few-shot labelling of feedback. Moreover, GPT-4.1 achieved human expert level performance after fine-tuning (Accuracy 79.8%, F1-score 79.4%; human average Accuracy 78.3%, F1-score 82.6%). Finally, we used our best-performance model to evaluate 2,000 assignment feedback instances generated by 10 common commercial LLMs, 200 each, to compare the quality of feedback generated by different LLMs. Our LLM feedback evaluator method advances our ability to automatically provide high-quality and reliable educational feedback to students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06753v1">Pushing the Envelope of LLM Inference on AI-PC</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The advent of ultra-low-bit LLM models (1/1.58/2-bit), which match the perplexity and end-task performance of their full-precision counterparts using the same model size, is ushering in a new era of LLM inference for resource-constrained environments such as edge devices and AI PCs. While these quantization advances promise models that are more cost-effective in terms of latency, memory, throughput, and energy consumption, the computational efficiency of state-of-the-art (SOTA) inference runtimes (e.g., bitnet.cpp) used to deploy them remains underexplored. In this work, we take a bottom-up approach: we first design and implement 1-bit and 2-bit microkernels optimized for modern CPUs, achieving peak computational efficiency across a variety of CPU platforms. We integrate these microkernels into a state-of-the-art LLM inference framework, namely PyTorch-TPP, and present end-to-end inference results with 2-bit models that outperform the current SOTA runtime bitnet.cpp by up to 2.2x, and deliver up to 7x speedup compared to the 16-bit model inference. Our optimized runtime advances the state of LLM inference on AI PCs and edge devices, paving the way for efficient deployment of ultra-low-bit LLM models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03628v2">LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations at eBay</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Sellers at eBay are recommended keyphrases to bid on to enhance the performance of their advertising campaigns. The relevance of these keyphrases is crucial in avoiding the overcrowding of search systems with irrelevant items and maintaining a positive seller perception. It is essential that keyphrase recommendations align with both seller and Search judgments regarding auctions. Due to the difficulty in procuring negative human judgment at scale, employing LLM-as-a-judge to mimic seller judgment has been established as the norm in several studies. This study introduces a novel two-step LLM distillation process from a LLM-judge used to debias our Embedding Based Retrieval (EBR) model from the various biases that exist in click-data. We distill from an LLM teacher via a cross-encoder assistant into a bi-encoder student using a multi-task training approach, ultimately employing the student bi-encoder to retrieve relevant advertiser keyphrases. We show that integrating a knowledge distillation process from LLMs in a multi-task training setup enhances bi-encoder performance in retrieving relevant advertiser keyphrases at eBay.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06709v1">Play Favorites: A Statistical Method to Measure Self-Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can serve as judges that offer rapid and reliable assessments of other LLM outputs. However, models may systematically assign overly favorable ratings to their own outputs, a phenomenon known as self-bias, which can distort evaluations of true model performance. Previous studies often conflate genuine differences in model quality with bias or incorrectly assume that evaluations from LLMs and humans follow the same rating distributions. In this work, we present a statistical framework that explicitly formalizes assumptions under which self-bias can be identified and estimated. Our method models the difference in the scoring distribution that LLM-as-a-judge assigns to its own completions compared to other models, while accounting for the underlying quality of the completions provided by an independent, third-party judge (e.g., humans). Our method reliably isolates and quantifies self-bias, even when models vary in ability, ensuring that genuine performance differences are not mistaken for self-bias. We conduct an empirical analysis of self-bias on a large dataset (>5000 prompt-completion pairs) consisting of expert human annotations and judgments from nine different LLM judges. We find that some models, such as GPT-4o and Claude 3.5 Sonnet, systematically assign higher scores to their own outputs. These models also display family-bias; systematically assigning higher ratings to outputs produced by other models of the same family. Our findings highlight potential pitfalls of using LLM judges and offer practical guidance to mitigate biases when interpreting automated evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08954v3">Should you use LLMs to simulate opinions? Quality checks for early-stage deliberation</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The emergent capabilities of large language models (LLMs) have prompted interest in using them as surrogates for human subjects in opinion surveys. However, prior evaluations of LLM-based opinion simulation have relied heavily on costly, domain-specific survey data, and mixed empirical results leave their reliability in question. To enable cost-effective, early-stage evaluation, we introduce a quality control assessment designed to test the viability of LLM-simulated opinions on Likert-scale tasks without requiring large-scale human data for validation. This assessment comprises two key tests: \emph{logical consistency} and \emph{alignment with stakeholder expectations}, offering a low-cost, domain-adaptable validation tool. We apply our quality control assessment to an opinion simulation task relevant to AI-assisted content moderation and fact-checking workflows -- a socially impactful use case -- and evaluate seven LLMs using a baseline prompt engineering method (backstory prompting), as well as fine-tuning and in-context learning variants. None of the models or methods pass the full assessment, revealing several failure modes. We conclude with a discussion of the risk management implications and release \texttt{TopicMisinfo}, a benchmark dataset with paired human and LLM annotations simulated by various models and approaches, to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06601v1">Deep Ignorance: Filtering Pretraining Data Builds Tamper-Resistant Safeguards into Open-Weight LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 https://deepignorance.ai/
    </div>
    <details class="paper-abstract">
      Open-weight AI systems offer unique benefits, including enhanced transparency, open research, and decentralized access. However, they are vulnerable to tampering attacks which can efficiently elicit harmful behaviors by modifying weights or activations. Currently, there is not yet a robust science of open-weight model risk management. Existing safety fine-tuning methods and other post-training techniques have struggled to make LLMs resistant to more than a few dozen steps of adversarial fine-tuning. In this paper, we investigate whether filtering text about dual-use topics from training data can prevent unwanted capabilities and serve as a more tamper-resistant safeguard. We introduce a multi-stage pipeline for scalable data filtering and show that it offers a tractable and effective method for minimizing biothreat proxy knowledge in LLMs. We pretrain multiple 6.9B-parameter models from scratch and find that they exhibit substantial resistance to adversarial fine-tuning attacks on up to 10,000 steps and 300M tokens of biothreat-related text -- outperforming existing post-training baselines by over an order of magnitude -- with no observed degradation to unrelated capabilities. However, while filtered models lack internalized dangerous knowledge, we find that they can still leverage such information when it is provided in context (e.g., via search tool augmentation), demonstrating a need for a defense-in-depth approach. Overall, these findings help to establish pretraining data curation as a promising layer of defense for open-weight AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06583v1">Discerning minds or generic tutors? Evaluating instructional guidance capabilities in Socratic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      The conversational capabilities of large language models hold significant promise for enabling scalable and interactive tutoring. While prior research has primarily examined their capacity for Socratic questioning, it often overlooks a critical dimension: adaptively guiding learners based on their cognitive states. This study shifts focus from mere question generation to the broader instructional guidance capability. We ask: Can LLMs emulate expert tutors who dynamically adjust strategies in response to learners' understanding? To investigate this, we propose GuideEval, a benchmark grounded in authentic educational dialogues that evaluates pedagogical guidance through a three-phase behavioral framework: (1) Perception, inferring learner states; (2) Orchestration, adapting instructional strategies; and (3) Elicitation, stimulating proper reflections. Empirical findings reveal that existing LLMs frequently fail to provide effective adaptive scaffolding when learners exhibit confusion or require redirection. Furthermore, we introduce a behavior-guided finetuning strategy that leverages behavior-prompted instructional dialogues, significantly enhancing guidance performance. By shifting the focus from isolated content evaluation to learner-centered interaction, our work advocates a more dialogic paradigm for evaluating Socratic LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06479v1">The Problem of Atypicality in LLM-Powered Psychiatry</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 Preprint of 8/8/2025 -- please cite published version. This article has been published in the Journal of Medical Ethics (2025) following peer review and can also be viewed on the journal's website at 10.1136/jme-2025-110972
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly proposed as scalable solutions to the global mental health crisis. But their deployment in psychiatric contexts raises a distinctive ethical concern: the problem of atypicality. Because LLMs generate outputs based on population-level statistical regularities, their responses -- while typically appropriate for general users -- may be dangerously inappropriate when interpreted by psychiatric patients, who often exhibit atypical cognitive or interpretive patterns. We argue that standard mitigation strategies, such as prompt engineering or fine-tuning, are insufficient to resolve this structural risk. Instead, we propose dynamic contextual certification (DCC): a staged, reversible and context-sensitive framework for deploying LLMs in psychiatry, inspired by clinical translation and dynamic safety models from artificial intelligence governance. DCC reframes chatbot deployment as an ongoing epistemic and ethical process that prioritises interpretive safety over static performance benchmarks. Atypicality, we argue, cannot be eliminated -- but it can, and must, be proactively managed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06467v1">LLM Unlearning using Gradient Ratio-Based Influence Estimation and Noise Injection</a></div>
    <div class="paper-meta">
      📅 2025-08-08
      | 💬 14 Pages, 3 Figures, 11 Tables
    </div>
    <details class="paper-abstract">
      The growing legal and ethical scrutiny of large language models (LLMs) necessitates effective machine unlearning, particularly for sensitive or unauthorized data. Existing empirical methods often yield incomplete forgetting or unintended degradation of unrelated knowledge due to poor localization. In this work, we propose GRIN: a modular and targeted framework for LLM unlearning. GRIN introduces a novel gradient-ratio-based metric to identify parameters most responsible for memorizing forget data. We then perform selective noise injection into these parameters prior to fine-tuning, which improves unlearning performance while maintaining model utility. Finally, we propose new evaluation metrics tailored to the LLM setting and validate our approach on standard benchmarks such as TOFU, WMDP, and SafePKU.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06447v1">SlimInfer: Accelerating Long-Context LLM Inference via Dynamic Token Pruning</a></div>
    <div class="paper-meta">
      📅 2025-08-08
    </div>
    <details class="paper-abstract">
      Long-context inference for Large Language Models (LLMs) is heavily limited by high computational demands. While several existing methods optimize attention computation, they still process the full set of hidden states at each layer, limiting overall efficiency. In this work, we propose SlimInfer, an innovative framework that aims to accelerate inference by directly pruning less critical prompt tokens during the forward pass. Our key insight is an information diffusion phenomenon: As information from critical tokens propagates through layers, it becomes distributed across the entire sequence. This diffusion process suggests that LLMs can maintain their semantic integrity when excessive tokens, even including these critical ones, are pruned in hidden states. Motivated by this, SlimInfer introduces a dynamic fine-grained pruning mechanism that accurately removes redundant tokens of hidden state at intermediate layers. This layer-wise pruning naturally enables an asynchronous KV cache manager that prefetches required token blocks without complex predictors, reducing both memory usage and I/O costs. Extensive experiments show that SlimInfer can achieve up to $\mathbf{2.53\times}$ time-to-first-token (TTFT) speedup and $\mathbf{1.88\times}$ end-to-end latency reduction for LLaMA3.1-8B-Instruct on a single RTX 4090, without sacrificing performance on LongBench. Our code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09032v2">Teaching LLMs How to Learn with Contextual Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Prompting Large Language Models (LLMs), or providing context on the expected model of operation, is an effective way to steer the outputs of such models to satisfy human desiderata after they have been trained. But in rapidly evolving domains, there is often need to fine-tune LLMs to improve either the kind of knowledge in their memory or their abilities to perform open ended reasoning in new domains. When human's learn new concepts, we often do so by linking the new material that we are studying to concepts we have already learned before. To that end, we ask, "can prompting help us teach LLMs how to learn". In this work, we study a novel generalization of instruction tuning, called contextual fine-tuning, to fine-tune LLMs. Our method leverages instructional prompts designed to mimic human cognitive strategies in learning and problem-solving to guide the learning process during training, aiming to improve the model's interpretation and understanding of domain-specific knowledge. We empirically demonstrate that this simple yet effective modification improves the ability of LLMs to be fine-tuned rapidly on new datasets both within the medical and financial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05525v1">The World According to LLMs: How Geographic Origin Influences LLMs' Entity Deduction Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Conference on Language Modeling 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been extensively tuned to mitigate explicit biases, yet they often exhibit subtle implicit biases rooted in their pre-training data. Rather than directly probing LLMs with human-crafted questions that may trigger guardrails, we propose studying how models behave when they proactively ask questions themselves. The 20 Questions game, a multi-turn deduction task, serves as an ideal testbed for this purpose. We systematically evaluate geographic performance disparities in entity deduction using a new dataset, Geo20Q+, consisting of both notable people and culturally significant objects (e.g., foods, landmarks, animals) from diverse regions. We test popular LLMs across two gameplay configurations (canonical 20-question and unlimited turns) and in seven languages (English, Hindi, Mandarin, Japanese, French, Spanish, and Turkish). Our results reveal geographic disparities: LLMs are substantially more successful at deducing entities from the Global North than the Global South, and the Global West than the Global East. While Wikipedia pageviews and pre-training corpus frequency correlate mildly with performance, they fail to fully explain these disparities. Notably, the language in which the game is played has minimal impact on performance gaps. These findings demonstrate the value of creative, free-form evaluation frameworks for uncovering subtle biases in LLMs that remain hidden in standard prompting setups. By analyzing how models initiate and pursue reasoning goals over multiple turns, we find geographic and cultural disparities embedded in their reasoning processes. We release the dataset (Geo20Q+) and code at https://sites.google.com/view/llmbias20q/home.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05512v1">RankArena: A Unified Platform for Evaluating Retrieval, Reranking and RAG with Human and LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accept at CIKM 2025
    </div>
    <details class="paper-abstract">
      Evaluating the quality of retrieval-augmented generation (RAG) and document reranking systems remains challenging due to the lack of scalable, user-centric, and multi-perspective evaluation tools. We introduce RankArena, a unified platform for comparing and analysing the performance of retrieval pipelines, rerankers, and RAG systems using structured human and LLM-based feedback as well as for collecting such feedback. RankArena supports multiple evaluation modes: direct reranking visualisation, blind pairwise comparisons with human or LLM voting, supervised manual document annotation, and end-to-end RAG answer quality assessment. It captures fine-grained relevance feedback through both pairwise preferences and full-list annotations, along with auxiliary metadata such as movement metrics, annotation time, and quality ratings. The platform also integrates LLM-as-a-judge evaluation, enabling comparison between model-generated rankings and human ground truth annotations. All interactions are stored as structured evaluation datasets that can be used to train rerankers, reward models, judgment agents, or retrieval strategy selectors. Our platform is publicly available at https://rankarena.ngrok.io/, and the Demo video is provided https://youtu.be/jIYAP4PaSSI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05496v1">InfiAlign: A Scalable and Sample-Efficient Framework for Aligning LLMs to Enhance Reasoning Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited impressive reasoning abilities on a wide range of complex tasks. However, enhancing these capabilities through post-training remains resource intensive, particularly in terms of data and computational cost. Although recent efforts have sought to improve sample efficiency through selective data curation, existing methods often rely on heuristic or task-specific strategies that hinder scalability. In this work, we introduce InfiAlign, a scalable and sample-efficient post-training framework that integrates supervised fine-tuning (SFT) with Direct Preference Optimization (DPO) to align LLMs for enhanced reasoning. At the core of InfiAlign is a robust data selection pipeline that automatically curates high-quality alignment data from open-source reasoning datasets using multidimensional quality metrics. This pipeline enables significant performance gains while drastically reducing data requirements and remains extensible to new data sources. When applied to the Qwen2.5-Math-7B-Base model, our SFT model achieves performance on par with DeepSeek-R1-Distill-Qwen-7B, while using only approximately 12% of the training data, and demonstrates strong generalization across diverse reasoning tasks. Additional improvements are obtained through the application of DPO, with particularly notable gains in mathematical reasoning tasks. The model achieves an average improvement of 3.89% on AIME 24/25 benchmarks. Our results highlight the effectiveness of combining principled data selection with full-stage post-training, offering a practical solution for aligning large reasoning models in a scalable and data-efficient manner. The model checkpoints are available at https://huggingface.co/InfiX-ai/InfiAlign-Qwen-7B-SFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05469v1">Let's Measure Information Step-by-Step: LLM-Based Evaluation Beyond Vibes</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 13 pages
    </div>
    <details class="paper-abstract">
      We develop mechanisms for evaluating AI systems without ground truth by exploiting a connection between gaming resistance and output quality. The data processing inequality ensures post-hoc attempts to game a metric degrades both information content and task performance. We prove that f-mutual information measures are the unique gaming resistant mechanisms under natural conditions, with the overseer acting as an agent. While Shannon mutual information faces exponential sample complexity, bounded measures like total variation distance remain tractable. Empirically, across ten domains from translation to peer review, all information-theoretic mechanisms achieve perfect discrimination (d > 0.5) between faithful and strategic agents. In contrast, LLM judges exhibit systematic evaluation inversion, preferring fabricated content over accurate summaries. Our mechanisms show 10-100x better robustness to adversarial manipulation than current practices. We also find performance follows an inverted-U curve with compression ratio, peaking at 10:1 where agent responses exhibit optimal information diversity (3 effective dimensions), giving a bias-variance perspective on when our approach is expected to be most effective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01545v2">Getting out of the Big-Muddy: Escalation of Commitment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in autonomous decision-making roles across high-stakes domains. However, since models are trained on human-generated data, they may inherit cognitive biases that systematically distort human judgment, including escalation of commitment, where decision-makers continue investing in failing courses of action due to prior investment. Understanding when LLMs exhibit such biases presents a unique challenge. While these biases are well-documented in humans, it remains unclear whether they manifest consistently in LLMs or require specific triggering conditions. This paper investigates this question using a two-stage investment task across four experimental conditions: model as investor, model as advisor, multi-agent deliberation, and compound pressure scenario. Across N = 6,500 trials, we find that bias manifestation in LLMs is highly context-dependent. In individual decision-making contexts (Studies 1-2, N = 4,000), LLMs demonstrate strong rational cost-benefit logic with minimal escalation of commitment. However, multi-agent deliberation reveals a striking hierarchy effect (Study 3, N = 500): while asymmetrical hierarchies show moderate escalation rates (46.2%), symmetrical peer-based decision-making produces near-universal escalation (99.2%). Similarly, when subjected to compound organizational and personal pressures (Study 4, N = 2,000), models exhibit high degrees of escalation of commitment (68.95% average allocation to failing divisions). These findings reveal that LLM bias manifestation depends critically on social and organizational context rather than being inherent, with significant implications for the deployment of multi-agent systems and unsupervised operations where such conditions may emerge naturally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09994v2">TIME: Temporal-Sensitive Multi-Dimensional Instruction Tuning and Robust Benchmarking for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Video large language models have achieved remarkable performance in tasks such as video question answering, however, their temporal understanding remains suboptimal. To address this limitation, we curate a dedicated instruction fine-tuning dataset that focuses on enhancing temporal comprehension across five key dimensions. In order to reduce reliance on costly temporal annotations, we introduce a multi-task prompt fine-tuning approach that seamlessly integrates temporal-sensitive tasks into existing instruction datasets without requiring additional annotations. Furthermore, we develop a novel benchmark for temporal-sensitive video understanding that not only fills the gaps in dimension coverage left by existing benchmarks but also rigorously filters out potential shortcuts, ensuring a more accurate evaluation. Extensive experimental results demonstrate that our approach significantly enhances the temporal understanding of video-LLMs while avoiding reliance on shortcuts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05421v1">LLM-based Multi-Agent Copilot for Quantum Sensor</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 13 pages,4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLM) exhibit broad utility but face limitations in quantum sensor development, stemming from interdisciplinary knowledge barriers and involving complex optimization processes. Here we present QCopilot, an LLM-based multi-agent framework integrating external knowledge access, active learning, and uncertainty quantification for quantum sensor design and diagnosis. Comprising commercial LLMs with few-shot prompt engineering and vector knowledge base, QCopilot employs specialized agents to adaptively select optimization methods, automate modeling analysis, and independently perform problem diagnosis. Applying QCopilot to atom cooling experiments, we generated 10${}^{\rm{8}}$ sub-$\rm{\mu}$K atoms without any human intervention within a few hours, representing $\sim$100$\times$ speedup over manual experimentation. Notably, by continuously accumulating prior knowledge and enabling dynamic modeling, QCopilot can autonomously identify anomalous parameters in multi-parameter experimental settings. Our work reduces barriers to large-scale quantum sensor deployment and readily extends to other quantum information systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05370v1">Simulating LLM training workloads for heterogeneous compute and network infrastructure</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The growing demand for large-scale GPU clusters in distributed model training presents a significant barrier to innovation, particularly in model optimization, performance tuning, and system-level enhancements. To address this challenge, LLM training simulators are employed to estimate training time and guide design decisions. However, the state-of-the-art LLM training simulators assume homogeneous compute and network infrastructure. In practice, device heterogeneity is inevitable due to resource sharing in cloud environments, frequent shifts in device generations, and inherent intra-chip interconnect heterogeneity. To address the gap between state-of-the-art and practical requirements, we propose the design of a heterogeneity-aware distributed LLM simulator capable of predicting training time while enabling abstractions to specify custom configurations for device groups and device-to-parallelism mapping. We present the design requirements and challenges in building a heterogeneity-aware distributed ML training simulator, and design components such as non-uniform workload partitioning. Our initial simulation results demonstrate the impact of heterogeneity on the model computation and communication time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05344v1">NomicLaw: Emergent Trust and Strategic Argumentation in LLMs During Collaborative Law-Making</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have extended their capabilities from basic text processing to complex reasoning tasks, including legal interpretation, argumentation, and strategic interaction. However, empirical understanding of LLM behavior in open-ended, multi-agent settings especially those involving deliberation over legal and ethical dilemmas remains limited. We introduce NomicLaw, a structured multi-agent simulation where LLMs engage in collaborative law-making, responding to complex legal vignettes by proposing rules, justifying them, and voting on peer proposals. We quantitatively measure trust and reciprocity via voting patterns and qualitatively assess how agents use strategic language to justify proposals and influence outcomes. Experiments involving homogeneous and heterogeneous LLM groups demonstrate how agents spontaneously form alliances, betray trust, and adapt their rhetoric to shape collective decisions. Our results highlight the latent social reasoning and persuasive capabilities of ten open-source LLMs and provide insights into the design of future AI systems capable of autonomous negotiation, coordination and drafting legislation in legal settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02253v3">Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 26 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Effective agent performance relies on the ability to compose tools and agents into effective workflows. However, progress in Large Language Model (LLM) planning and reasoning is limited by the scarcity of scalable, reliable evaluation data. This study addresses this limitation by identifying a suitable workflow domain for LLM application. I introduce NL2Flow, a fully automated system for parametrically generating planning problems, which are expressed in natural language, a structured intermediate representation, and formal PDDL, and rigorously evaluating the quality of generated plans. NL2Flow generates a dataset of 2296 low-difficulty problems in automated workflow generation and evaluates multiple open-sourced, instruct-tuned LLMs without task-specific optimization or architectural modifications. Results reveal that the highest performing model achieved 86% success in generating valid plans and 69% in generating optimal plans, specifically for problems with feasible plans. Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. To investigate the potential of LLMs as natural language-to-JSON translators for workflow definition, and to facilitate integration with downstream symbolic computation tools and a symbolic planner, I evaluated the LLM's translation performance on natural language workflow descriptions. I observed that translating natural language into a JSON representation of a workflow problem yielded a lower success rate than generating a plan directly, suggesting that unnecessary decomposition of the reasoning task may degrade performance and highlighting the benefit of models capable of reasoning directly from natural language to action. As LLM reasoning scales to increasingly complex problems, understanding the shifting bottlenecks and sources of error within these systems will be crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06608v5">Nexus:Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Monolithic serving with chunked prefill improves GPU utilization by batching prefill and decode together, but suffers from fine-grained phase interference. Engine-level prefill-decode (PD) disaggregation avoids interference but incurs higher hardware and coordination overhead. Prior intra-GPU disaggregation approaches multiplex prefill and decode within a single GPU, using SLO-based tuning guided by heuristics from offline profiling or reactive feedback loops. However, these methods respond reactively to performance issues rather than anticipating them, limiting adaptability under dynamic workloads. We ask: can we achieve proactive intra-GPU disaggregation that adapts effectively to dynamic workloads? The key challenge lies in managing the conflicting resource demands of prefill and decode under varying conditions. We first show that GPU resources exhibit diminishing returns -- beyond a saturation point, more allocation yields minimal latency benefit. Second, we observe that memory bandwidth contention becomes a critical bottleneck. These insights motivate a design that dynamically partitions GPU resources across prefill and decode phases, while jointly considering compute capacity, memory footprint, and bandwidth contention. Evaluated on diverse LLMs and workloads, our system Nexus achieves up to 2.2x higher throughput, 20x lower TTFT, and 2.5x lower TBT than vLLM; outperforms SGLang by up to 2x; and matches or exceeds disaggregated vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05311v1">A Novel Architecture for Symbolic Reasoning with Decision Trees and LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      We propose a hybrid architecture that integrates decision tree-based symbolic reasoning with the generative capabilities of large language models (LLMs) within a coordinated multi-agent framework. Unlike prior approaches that loosely couple symbolic and neural modules, our design embeds decision trees and random forests as callable oracles within a unified reasoning system. Tree-based modules enable interpretable rule inference and causal logic, while LLM agents handle abductive reasoning, generalization, and interactive planning. A central orchestrator maintains belief state consistency and mediates communication across agents and external tools, enabling reasoning over both structured and unstructured inputs. The system achieves strong performance on reasoning benchmarks. On \textit{ProofWriter}, it improves entailment consistency by +7.2\% through logic-grounded tree validation. On GSM8k, it achieves +5.3\% accuracy gains in multistep mathematical problems via symbolic augmentation. On \textit{ARC}, it boosts abstraction accuracy by +6.0\% through integration of symbolic oracles. Applications in clinical decision support and scientific discovery show how the system encodes domain rules symbolically while leveraging LLMs for contextual inference and hypothesis generation. This architecture offers a robust, interpretable, and extensible solution for general-purpose neuro-symbolic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05299v1">VS-LLM: Visual-Semantic Depression Assessment based on LLM for Drawing Projection Test</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The Drawing Projection Test (DPT) is an essential tool in art therapy, allowing psychologists to assess participants' mental states through their sketches. Specifically, through sketches with the theme of "a person picking an apple from a tree (PPAT)", it can be revealed whether the participants are in mental states such as depression. Compared with scales, the DPT can enrich psychologists' understanding of an individual's mental state. However, the interpretation of the PPAT is laborious and depends on the experience of the psychologists. To address this issue, we propose an effective identification method to support psychologists in conducting a large-scale automatic DPT. Unlike traditional sketch recognition, DPT more focus on the overall evaluation of the sketches, such as color usage and space utilization. Moreover, PPAT imposes a time limit and prohibits verbal reminders, resulting in low drawing accuracy and a lack of detailed depiction. To address these challenges, we propose the following efforts: (1) Providing an experimental environment for automated analysis of PPAT sketches for depression assessment; (2) Offering a Visual-Semantic depression assessment based on LLM (VS-LLM) method; (3) Experimental results demonstrate that our method improves by 17.6% compared to the psychologist assessment method. We anticipate that this work will contribute to the research in mental state assessment based on PPAT sketches' elements recognition. Our datasets and codes are available at https://github.com/wmeiqi/VS-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05298v1">GhostShell: Streaming LLM Function Calls for Concurrent Embodied Programming</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 17 pages, 5 figures, conference
    </div>
    <details class="paper-abstract">
      We present GhostShell, a novel approach that leverages Large Language Models (LLMs) to enable streaming and concurrent behavioral programming for embodied systems. In contrast to conventional methods that rely on pre-scheduled action sequences or behavior trees, GhostShell drives embodied systems to act on-the-fly by issuing function calls incrementally as tokens are streamed from the LLM. GhostShell features a streaming XML function token parser, a dynamic function interface mapper, and a multi-channel scheduler that orchestrates intra-channel synchronous and inter-channel asynchronous function calls, thereby coordinating serial-parallel embodied actions across multiple robotic components as directed by the LLM. We evaluate GhostShell on our robot prototype COCO through comprehensive grounded experiments across 34 real-world interaction tasks and multiple LLMs. The results demonstrate that our approach achieves state-of-the-art Behavioral Correctness Metric of 0.85 with Claude-4 Sonnet and up to 66X faster response times compared to LLM native function calling APIs. GhostShell also proves effective in long-horizon multimodal tasks, demonstrating strong robustness and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05289v1">RLHF Fine-Tuning of LLMs for Alignment with Implicit User Feedback in Conversational Recommenders</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Conversational recommender systems (CRS) based on Large Language Models (LLMs) need to constantly be aligned to the user preferences to provide satisfying and context-relevant item recommendations. The traditional supervised fine-tuning cannot capture the implicit feedback signal, e.g., dwell time, sentiment polarity, or engagement patterns. In this paper, we share a fine-tuning solution using human feedback reinforcement learning (RLHF) to maximize implied user feedback (IUF) in a multi-turn recommendation context. We specify a reward model $R_{\phi}$ learnt on weakly-labelled engagement information and maximize user-centric utility by optimizing the foundational LLM M_{\theta} through a proximal policy optimization (PPO) approach. The architecture models conversational state transitions $s_t \to a_t \to s_{t +1}$, where the action $a_t$ is associated with LLM-generated item suggestions only on condition of conversation history in the past. The evaluation across synthetic and real-world datasets (e.g.REDIAL, OpenDialKG) demonstrates that our RLHF-fine-tuned models can perform better in terms of top-$k$ recommendation accuracy, coherence, and user satisfaction compared to (arrow-zero-cmwrquca-teja-falset ensuite 2Round group-deca States penalty give up This paper shows that implicit signal alignment can be efficient in achieving scalable and user-adaptive design of CRS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05282v1">ASCoT: An Adaptive Self-Correction Chain-of-Thought Method for Late-Stage Fragility in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting has significantly advanced the reasoning capabilities of Large Language Models (LLMs), yet the reliability of these reasoning chains remains a critical challenge. A widely held "cascading failure" hypothesis suggests that errors are most detrimental when they occur early in the reasoning process. This paper challenges that assumption through systematic error-injection experiments, revealing a counter-intuitive phenomenon we term "Late-Stage Fragility": errors introduced in the later stages of a CoT chain are significantly more likely to corrupt the final answer than identical errors made at the beginning. To address this specific vulnerability, we introduce the Adaptive Self-Correction Chain-of-Thought (ASCoT) method. ASCoT employs a modular pipeline in which an Adaptive Verification Manager (AVM) operates first, followed by the Multi-Perspective Self-Correction Engine (MSCE). The AVM leverages a Positional Impact Score function I(k) that assigns different weights based on the position within the reasoning chains, addressing the Late-Stage Fragility issue by identifying and prioritizing high-risk, late-stage steps. Once these critical steps are identified, the MSCE applies robust, dual-path correction specifically to the failure parts. Extensive experiments on benchmarks such as GSM8K and MATH demonstrate that ASCoT achieves outstanding accuracy, outperforming strong baselines, including standard CoT. Our work underscores the importance of diagnosing specific failure modes in LLM reasoning and advocates for a shift from uniform verification strategies to adaptive, vulnerability-aware correction mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05266v1">Understanding and Mitigating Errors of LLM-Generated RTL Code</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 14 pages, 26 figures
    </div>
    <details class="paper-abstract">
      Despite the promising potential of large language model (LLM) based register-transfer-level (RTL) code generation, the overall success rate remains unsatisfactory. Errors arise from various factors, with limited understanding of specific failure causes hindering improvement. To address this, we conduct a comprehensive error analysis and manual categorization. Our findings reveal that most errors stem not from LLM reasoning limitations, but from insufficient RTL programming knowledge, poor understanding of circuit concepts, ambiguous design descriptions, or misinterpretation of complex multimodal inputs. Leveraging in-context learning, we propose targeted error correction techniques. Specifically, we construct a domain-specific knowledge base and employ retrieval-augmented generation (RAG) to supply necessary RTL knowledge. To mitigate ambiguity errors, we introduce design description rules and implement a rule-checking mechanism. For multimodal misinterpretation, we integrate external tools to convert inputs into LLM-compatible meta-formats. For remaining errors, we adopt an iterative debugging loop (simulation-error localization-correction). Integrating these techniques into an LLM-based framework significantly improves performance. We incorporate these error correction techniques into a foundational LLM-based RTL code generation framework, resulting in significantly improved performance. Experimental results show that our enhanced framework achieves 91.0\% accuracy on the VerilogEval benchmark, surpassing the baseline code generation approach by 32.7\%, demonstrating the effectiveness of our methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05257v1">MoBE: Mixture-of-Basis-Experts for Compressing MoE-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The Mixture-of-Experts (MoE) architecture has become a predominant paradigm for scaling large language models (LLMs). Despite offering strong performance and computational efficiency, large MoE-based LLMs like DeepSeek-V3-0324 and Kimi-K2-Instruct present serious challenges due to substantial memory requirements in deployment. While recent works have explored MoE compression to address this issue, existing methods often suffer from considerable accuracy drops (e.g., 7-14% relatively) even at modest compression rates. This paper introduces a novel Mixture-of-Basis-Experts (MoBE) method that achieves model compression while incurring minimal accuracy drops. Specifically, each up/gate matrix in an expert is decomposed via a rank decomposition as W = AB, where matrix A is unique to each expert. The relatively larger matrix B is further re-parameterized as a linear combination of basis matrices {Bi} shared across all experts within a given MoE layer. The factorization is learned by minimizing the reconstruction error relative to the original weight matrices. Experiments demonstrate that MoBE achieves notably lower accuracy drops compared to prior works. For instance, MoBE can reduce the parameter counts of Qwen3-235B-A22B-2507, DeepSeek-V3-0324 (671B) and Kimi-K2-Instruct (1T) by 24%-30% with only 1%-2% accuracy drop (about 2% drops when measured relatively).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05242v1">CodeBoost: Boosting Code LLMs by Squeezing Knowledge from Code Snippets with RL</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Technical report. Project page: https://github.com/sijieaaa/CodeBoost
    </div>
    <details class="paper-abstract">
      Code large language models (LLMs) have become indispensable tools for building efficient and automated coding pipelines. Existing models are typically post-trained using reinforcement learning (RL) from general-purpose LLMs using "human instruction-final answer" pairs, where the instructions are usually from manual annotations. However, collecting high-quality coding instructions is both labor-intensive and difficult to scale. On the other hand, code snippets are abundantly available from various sources. This imbalance presents a major bottleneck in instruction-based post-training. We propose CodeBoost, a post-training framework that enhances code LLMs purely from code snippets, without relying on human-annotated instructions. CodeBoost introduces the following key components: (1) maximum-clique curation, which selects a representative and diverse training corpus from code; (2) bi-directional prediction, which enables the model to learn from both forward and backward prediction objectives; (3) error-aware prediction, which incorporates learning signals from both correct and incorrect outputs; (4) heterogeneous augmentation, which diversifies the training distribution to enrich code semantics; and (5) heterogeneous rewarding, which guides model learning through multiple reward types including format correctness and execution feedback from both successes and failures. Extensive experiments across several code LLMs and benchmarks verify that CodeBoost consistently improves performance, demonstrating its effectiveness as a scalable and effective training pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05232v1">Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Traditional parameter-efficient fine-tuning (PEFT) methods such as LoRA are tightly coupled with the base model architecture, which constrains their applicability across heterogeneous pretrained large language models (LLMs). To address this limitation, we introduce Cross-LoRA, a data-free framework for transferring LoRA modules between diverse base models without requiring additional training data. Cross-LoRA consists of two key components: (a) LoRA-Align, which performs subspace alignment between source and target base models through rank-truncated singular value decomposition (SVD) and Frobenius-optimal linear transformation, ensuring compatibility under dimension mismatch; and (b) LoRA-Shift, which applies the aligned subspaces to project source LoRA weight updates into the target model parameter space. Both components are data-free, training-free, and enable lightweight adaptation on a commodity GPU in 20 minutes. Experiments on ARCs, OBOA and HellaSwag show that Cross-LoRA achieves relative gains of up to 5.26% over base models. Across other commonsense reasoning benchmarks, Cross-LoRA maintains performance comparable to that of directly trained LoRA adapters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14964v2">Efficient Knowledge Injection in LLMs via Self-Distillation</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      In many practical applications, large language models (LLMs) need to acquire new knowledge not present in their pre-training data. Efficiently leveraging this knowledge usually relies on supervised fine-tuning or retrieval-augmented generation (RAG). Although RAG has emerged as the industry standard for knowledge injection, fine-tuning has not yet achieved comparable success. This paper proposes utilizing prompt distillation, a self-distillation-based method previously explored primarily for style alignment and instruction tuning, to internalize new factual knowledge from free-form documents. Unlike prior methods, our approach requires neither larger teacher models nor structured knowledge formats. Across multiple LLM sizes and model families, we show that prompt distillation outperforms standard supervised fine-tuning and can even surpass RAG. We analyze the key factors contributing to prompt distillation's effectiveness and examine how it scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05165v1">Aligning LLMs on a Budget: Inference-Time Alignment with Heuristic Reward Models</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Aligning LLMs with user preferences is crucial for real-world use but often requires costly fine-tuning or expensive inference, forcing trade-offs between alignment quality and computational cost. Existing inference-time methods typically ignore this balance, focusing solely on the optimized policy's performance. We propose HIA (Heuristic-Guided Inference-time Alignment), a tuning-free, black-box-compatible approach that uses a lightweight prompt optimizer, heuristic reward models, and two-stage filtering to reduce inference calls while preserving alignment quality. On real-world prompt datasets, HelpSteer and ComPRed, HIA outperforms best-of-N sampling, beam search, and greedy search baselines in multi-objective, goal-conditioned tasks under the same inference budget. We also find that HIA is effective under low-inference budgets with as little as one or two response queries, offering a practical solution for scalable, personalized LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20367v5">Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      With the rapid evolution of large language models (LLM), reinforcement learning (RL) has emerged as a pivotal technique for code generation and optimization in various domains. This paper presents a systematic survey of the application of RL in code optimization and generation, highlighting its role in enhancing compiler optimization, resource allocation, and the development of frameworks and tools. Subsequent sections first delve into the intricate processes of compiler optimization, where RL algorithms are leveraged to improve efficiency and resource utilization. The discussion then progresses to the function of RL in resource allocation, emphasizing register allocation and system optimization. We also explore the burgeoning role of frameworks and tools in code generation, examining how RL can be integrated to bolster their capabilities. This survey aims to serve as a comprehensive resource for researchers and practitioners interested in harnessing the power of RL to advance code generation and optimization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05149v1">Speech LLMs in Low-Resource Scenarios: Data Volume Requirements and the Impact of Pretraining on High-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted at Interspeech 2025. 5 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in handling spoken inputs for high-resource languages, reaching state-of-the-art performance in various tasks. However, their applicability is still less explored in low-resource settings. This work investigates the use of Speech LLMs for low-resource Automatic Speech Recognition using the SLAM-ASR framework, where a trainable lightweight projector connects a speech encoder and a LLM. Firstly, we assess training data volume requirements to match Whisper-only performance, re-emphasizing the challenges of limited data. Secondly, we show that leveraging mono- or multilingual projectors pretrained on high-resource languages reduces the impact of data scarcity, especially with small training sets. Using multilingual LLMs (EuroLLM, Salamandra) with whisper-large-v3-turbo, we evaluate performance on several public benchmarks, providing insights for future research on optimizing Speech LLMs for low-resource languages and multilinguality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05129v1">Navigating Through Paper Flood: Advancing LLM-based Paper Evaluation through Domain-Aware Retrieval and Latent Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      With the rapid and continuous increase in academic publications, identifying high-quality research has become an increasingly pressing challenge. While recent methods leveraging Large Language Models (LLMs) for automated paper evaluation have shown great promise, they are often constrained by outdated domain knowledge and limited reasoning capabilities. In this work, we present PaperEval, a novel LLM-based framework for automated paper evaluation that addresses these limitations through two key components: 1) a domain-aware paper retrieval module that retrieves relevant concurrent work to support contextualized assessments of novelty and contributions, and 2) a latent reasoning mechanism that enables deep understanding of complex motivations and methodologies, along with comprehensive comparison against concurrently related work, to support more accurate and reliable evaluation. To guide the reasoning process, we introduce a progressive ranking optimization strategy that encourages the LLM to iteratively refine its predictions with an emphasis on relative comparison. Experiments on two datasets demonstrate that PaperEval consistently outperforms existing methods in both academic impact and paper quality evaluation. In addition, we deploy PaperEval in a real-world paper recommendation system for filtering high-quality papers, which has gained strong engagement on social media -- amassing over 8,000 subscribers and attracting over 10,000 views for many filtered high-quality papers -- demonstrating the practical effectiveness of PaperEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05113v1">EasySize: Elastic Analog Circuit Sizing via LLM-Guided Heuristic Search</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Analog circuit design is a time-consuming, experience-driven task in chip development. Despite advances in AI, developing universal, fast, and stable gate sizing methods for analog circuits remains a significant challenge. Recent approaches combine Large Language Models (LLMs) with heuristic search techniques to enhance generalizability, but they often depend on large model sizes and lack portability across different technology nodes. To overcome these limitations, we propose EasySize, the first lightweight gate sizing framework based on a finetuned Qwen3-8B model, designed for universal applicability across process nodes, design specifications, and circuit topologies. EasySize exploits the varying Ease of Attainability (EOA) of performance metrics to dynamically construct task-specific loss functions, enabling efficient heuristic search through global Differential Evolution (DE) and local Particle Swarm Optimization (PSO) within a feedback-enhanced flow. Although finetuned solely on 350nm node data, EasySize achieves strong performance on 5 operational amplifier (Op-Amp) netlists across 180nm, 45nm, and 22nm technology nodes without additional targeted training, and outperforms AutoCkt, a widely-used Reinforcement Learning based sizing framework, on 86.67\% of tasks with more than 96.67\% of simulation resources reduction. We argue that EasySize can significantly reduce the reliance on human expertise and computational resources in gate sizing, thereby accelerating and simplifying the analog circuit design process. EasySize will be open-sourced at a later date.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03440v3">LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 11 pages, 7 figures, working in progress
    </div>
    <details class="paper-abstract">
      Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. This paper explores the `Soft Thinking' capabilities of various LLMs by examining the models' internal behavior using a suite of probing techniques. Contrary to the common belief that Soft Thinking enables the simultaneous exploration of diverse reasoning paths, our findings reveal that LLMs predominantly rely on the most influential component of the soft inputs during subsequent decoding steps. This reliance hinders the exploration of different reasoning paths and reduces vanilla Soft Thinking to a form of greedy decoding, obscuring the advantage of transmitting more information through Soft Tokens. To tackle this issue, we explore sampling strategies to introduce \emph{randomness}, employing methods such as Dirichlet resampling and the Gumbel-Softmax trick. Our experiments demonstrate that incorporating randomness can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking. Notably, the Gumbel-Softmax trick provides adequate randomness with controlled smoothness, resulting in superior performance across eight reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03864v2">DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Enhancing the capability of large language models (LLMs) in reasoning has gained significant attention in recent years. Previous studies have demonstrated the effectiveness of various prompting strategies in aiding LLMs in reasoning (called "reasoning actions"), such as step-by-step thinking, reflecting before answering, solving with programs, and their combinations. However, these approaches often applied static, predefined reasoning actions uniformly to all questions, without considering the specific characteristics of each question or the capability of the task-solving LLM. In this paper, we propose DOTS, an approach enabling LLMs to reason dynamically via optimal reasoning trajectory search, tailored to the specific characteristics of each question and the inherent capability of the task-solving LLM. Our approach involves three key steps: i) defining atomic reasoning action modules that can be composed into various reasoning action trajectories; ii) searching for the optimal action trajectory for each training question through iterative exploration and evaluation for the specific task-solving LLM; and iii) using the collected optimal trajectories to train an LLM to plan for the reasoning trajectories of unseen questions. In particular, we propose two learning paradigms, i.e., fine-tuning an external LLM as a planner to guide the task-solving LLM, or directly fine-tuning the task-solving LLM with an internalized capability for reasoning actions planning. Our experiments across eight reasoning tasks show that our method consistently outperforms static reasoning techniques and the vanilla instruction tuning approach. Further analysis reveals that our method enables LLMs to adjust their computation based on problem complexity, allocating deeper thinking and reasoning to harder problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01674v2">CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted to COLM 2025. Project Website: https://cupid.kixlab.org/
    </div>
    <details class="paper-abstract">
      Personalization of Large Language Models (LLMs) often assumes users hold static preferences that reflect globally in all tasks. In reality, humans hold dynamic preferences that change depending on the context. As users interact with an LLM in various contexts, they naturally reveal their contextual preferences, which a model must infer and apply in future contexts to ensure alignment. To assess this, we introduce CUPID, a benchmark of 756 human-curated interaction session histories between users and LLM-based chat assistants. In each interaction session, the user provides a request in a specific context and expresses their preference through multi-turn feedback. Given a new user request and prior interaction sessions, our benchmark assesses whether LLMs can infer the preference relevant to this request and generate a response that satisfies this preference. With CUPID, we evaluated 10 open and proprietary LLMs, revealing that state-of-the-art LLMs struggle to infer preferences from multi-turn interactions and fail to discern what previous context is relevant to a new request -- under 50% precision and 65% recall. Our work highlights the need to advance LLM capabilities for more contextually personalized interactions and proposes CUPID as a resource to drive these improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06518v2">Medal Matters: Probing LLMs' Failure Cases Through Olympic Rankings</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 COLM 2025 ORIGen Workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in natural language processing tasks, yet their internal knowledge structures remain poorly understood. This study examines these structures through the lens of historical Olympic medal tallies, evaluating LLMs on two tasks: (1) retrieving medal counts for specific teams and (2) identifying rankings of each team. While state-of-the-art LLMs excel in recalling medal counts, they struggle with providing rankings, highlighting a key difference between their knowledge organization and human reasoning. These findings shed light on the limitations of LLMs' internal knowledge integration and suggest directions for improvement. To facilitate further research, we release our code, dataset, and model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05028v1">Evaluation of LLMs in AMR Parsing</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 27 pages, 32 figures
    </div>
    <details class="paper-abstract">
      Meaning Representation (AMR) is a semantic formalism that encodes sentence meaning as rooted, directed, acyclic graphs, where nodes represent concepts and edges denote semantic relations. Finetuning decoder only Large Language Models (LLMs) represent a promising novel straightfoward direction for AMR parsing. This paper presents a comprehensive evaluation of finetuning four distinct LLM architectures, Phi 3.5, Gemma 2, LLaMA 3.2, and DeepSeek R1 LLaMA Distilled using the LDC2020T02 Gold AMR3.0 test set. Our results have shown that straightfoward finetuning of decoder only LLMs can achieve comparable performance to complex State of the Art (SOTA) AMR parsers. Notably, LLaMA 3.2 demonstrates competitive performance against SOTA AMR parsers given a straightforward finetuning approach. We achieved SMATCH F1: 0.804 on the full LDC2020T02 test split, on par with APT + Silver (IBM) at 0.804 and approaching Graphene Smatch (MBSE) at 0.854. Across our analysis, we also observed a consistent pattern where LLaMA 3.2 leads in semantic performance while Phi 3.5 excels in structural validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.17963v3">M$^{2}$Chat: Empowering VLM for Multimodal LLM Interleaved Text-Image Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      While current LLM chatbots like GPT-4V bridge the gap between human instructions and visual representations to enable text-image generations, they still lack efficient alignment methods for high-fidelity performance on multiple downstream tasks. In this paper, we propose \textbf{$M^{2}Chat$}, a novel unified multimodal LLM framework for generating interleaved text-image conversation across various scenarios. Specifically, we propose an $M^{3}Adapter$ that efficiently integrates granular low-level visual information and high-level semantic features from multi-modality prompts. Upon the well-aligned fused feature, $M^{3}Adapter$ tailors a learnable gating strategy to balance the model creativity and consistency across various tasks adaptively. Moreover, to further enhance the effectiveness of $M^{3}Adapter$ while preserving the coherence of semantic context comprehension, we introduce a two-stage $M^{3}FT$ fine-tuning strategy. This strategy optimizes disjoint groups of parameters for image-text alignment and visual-instruction respectively. Extensive experiments demonstrate our $M^{2}Chat$ surpasses state-of-the-art counterparts across diverse benchmarks, showcasing its prowess in interleaving generation, storytelling, and multimodal dialogue systems. The demo and code are available at \red{https://mattie-e.github.io/M2Chat.github.io}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05012v1">Making Prompts First-Class Citizens for Adaptive LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Modern LLM pipelines increasingly resemble data-centric systems: they retrieve external context, compose intermediate outputs, validate results, and adapt based on runtime feedback. Yet, the central element guiding this process -- the prompt -- remains a brittle, opaque string, disconnected from the surrounding dataflow. This disconnect limits reuse, optimization, and runtime control. In this paper, we describe our vision and an initial design for SPEAR, a language and runtime that fills this prompt management gap by making prompts structured, adaptive, and first-class components of the execution model. SPEAR enables (1) runtime prompt refinement -- modifying prompts dynamically in response to execution-time signals such as confidence, latency, or missing context; and (2) structured prompt management -- organizing prompt fragments into versioned views with support for introspection and logging. SPEAR defines a prompt algebra that governs how prompts are constructed and adapted within a pipeline. It supports multiple refinement modes (manual, assisted, and automatic), giving developers a balance between control and automation. By treating prompt logic as structured data, SPEAR enables optimizations such as operator fusion, prefix caching, and view reuse. Preliminary experiments quantify the behavior of different refinement modes compared to static prompts and agentic retries, as well as the impact of prompt-level optimizations such as operator fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05004v1">R-Zero: Self-Evolving Reasoning LLM from Zero Data</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05853v2">"Mango Mango, How to Let The Lettuce Dry Without A Spinner?": Exploring User Perceptions of Using An LLM-Based Conversational Assistant Toward Cooking Partner</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 To appear at CSCW 2025
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has created numerous potentials for integration with conversational assistants (CAs) assisting people in their daily tasks, particularly due to their extensive flexibility. However, users' real-world experiences interacting with these assistants remain unexplored. In this research, we chose cooking, a complex daily task, as a scenario to explore people's successful and unsatisfactory experiences while receiving assistance from an LLM-based CA, Mango Mango. We discovered that participants value the system's ability to offer customized instructions based on context, provide extensive information beyond the recipe, and assist them in dynamic task planning. However, users expect the system to be more adaptive to oral conversation and provide more suggestive responses to keep them actively involved. Recognizing that users began treating our LLM-CA as a personal assistant or even a partner rather than just a recipe-reading tool, we propose five design considerations for future development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04975v1">Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Traditionally, traders and quantitative analysts address alpha decay by manually crafting formulaic alphas, mathematical expressions that identify patterns or signals in financial data, through domain expertise and trial-and-error. This process is often time-consuming and difficult to scale. With recent advances in large language models (LLMs), it is now possible to automate the generation of such alphas by leveraging the reasoning capabilities of LLMs. This paper introduces a novel framework that integrates a prompt-based LLM with a Transformer model for stock price prediction. The LLM first generates diverse and adaptive alphas using structured inputs such as historical stock features (Close, Open, High, Low, Volume), technical indicators, sentiment scores of both target and related companies. These alphas, instead of being used directly for trading, are treated as high-level features that capture complex dependencies within the financial data. To evaluate the effectiveness of these LLM-generated formulaic alphas, the alpha features are then fed into prediction models such as Transformer, LSTM, TCN, SVR, and Random Forest to forecast future stock prices. Experimental results demonstrate that the LLM-generated alphas significantly improve predictive accuracy. Moreover, the accompanying natural language reasoning provided by the LLM enhances the interpretability and transparency of the predictions, supporting more informed financial decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13213v4">Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Published in Transactions on Machine Learning Research (TMLR)
    </div>
    <details class="paper-abstract">
      We study 15 large language models (LLMs) fine-tuned for chat and find that their maximum softmax probabilities (MSPs) are consistently miscalibrated on multiple-choice Q&A. However, those MSPs might still encode useful uncertainty information. Specifically, we hypothesized that wrong answers would be associated with smaller MSPs compared to correct answers. Via rigorous statistical testing, we show that this hypothesis holds for models which perform well on the underlying Q&A task. We also find a strong direction correlation between Q&A accuracy and MSP correctness prediction, while finding no correlation between Q&A accuracy and calibration error. This suggests that within the current fine-tuning paradigm, we can expect correctness prediction but not calibration to improve as LLM capabilities progress. To demonstrate the utility of correctness prediction, we show that when models have the option to abstain, performance can be improved by selectively abstaining based on the MSP of the initial model response, using only a small amount of labeled data to choose the MSP threshold.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07132v3">Interactive Data Harmonization with LLM Agents: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Data harmonization is an essential task that entails integrating datasets from diverse sources. Despite years of research in this area, it remains a time-consuming and challenging task due to schema mismatches, varying terminologies, and differences in data collection methodologies. This paper presents the case for agentic data harmonization as a means to both empower experts to harmonize their data and to streamline the process. We introduce Harmonia, a system that combines LLM-based reasoning, an interactive user interface, and a library of data harmonization primitives to automate the synthesis of data harmonization pipelines. We demonstrate Harmonia in a clinical data harmonization scenario, where it helps to interactively create reusable pipelines that map datasets to a standard format. Finally, we discuss challenges and open problems, and suggest research directions for advancing our vision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19028v3">Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 29 pages, 9 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04030v2">OpenCodeInstruct: A Large-scale Instruction Tuning Dataset for Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software development by enabling code generation, automated debugging, and complex reasoning. However, their continued advancement is constrained by the scarcity of high-quality, publicly available supervised fine-tuning (SFT) datasets tailored for coding tasks. To bridge this gap, we introduce OpenCodeInstruct, the largest open-access instruction tuning dataset, comprising 5 million diverse samples. Each sample includes a programming question, solution, test cases, execution feedback, and LLM-generated quality assessments. We fine-tune various base models, including LLaMA and Qwen, across multiple scales (1B+, 3B+, and 7B+) using our dataset. Comprehensive evaluations on popular benchmarks (HumanEval, MBPP, LiveCodeBench, and BigCodeBench) demonstrate substantial performance improvements achieved by SFT with OpenCodeInstruct. We also present a detailed methodology encompassing seed data curation, synthetic instruction and solution generation, and filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05835v1">NanoCodec: Towards High-Quality Ultra Fast Speech LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Accepted to Interspeech 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced audio processing by leveraging audio codecs to discretize audio into tokens, enabling the application of language modeling techniques to speech data. However, existing audio codecs often operate at high frame rates, leading to slow training and inference, particularly for autoregressive models. To address this, there is growing interest in low frame-rate audio codecs, which reduce the number of autoregressive steps required to generate one second of audio. In this paper, we conduct ablation studies to examine the impact of frame rate, bitrate, and causality on codec reconstruction quality. Based on our findings, we introduce NanoCodec, a state-of-the-art audio codec that achieves high-quality compression at just 12.5 frames per second (FPS). NanoCodec outperforms related works across various bitrate ranges, establishing a new benchmark for low-latency and efficient Speech LLM training and inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.17008v2">Benchmarking LLMs on the Semantic Overlap Summarization Task</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Semantic Overlap Summarization (SOS) is a constrained multi-document summarization task, where the constraint is to capture the common/overlapping information between two alternative narratives. In this work, we perform a benchmarking study of popular Large Language Models (LLMs) exclusively on the SOS task. Additionally, we introduce the PrivacyPolicyPairs (3P) dataset to expand the space of SOS benchmarks in terms of quantity and variety. This dataset provides 135 high-quality SOS data samples sourced from privacy policy documents. We then use a standard prompting taxonomy called TELeR to create and evaluate 905,216 distinct LLM-generated summaries over two SOS datasets from different domains, and we further conduct human evaluation on a subset of 540 samples. We conclude the paper by analyzing models' performances and the reliability of automatic evaluation. The code and datasets used to conduct this study are available at https://anonymous.4open.science/r/llm_eval-E16D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05728v1">CLAPP: The CLASS LLM Agent for Pair Programming</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 Code: https://github.com/santiagocasas/clapp, Streamlit app: https://classclapp.streamlit.app
    </div>
    <details class="paper-abstract">
      We introduce CLAPP (CLASS LLM Agent for Pair Programming), an interactive AI assistant designed to support researchers working with the Einstein-Boltzmann solver CLASS. CLAPP leverages large language models (LLMs) and domain-specific retrieval to provide conversational coding support for CLASS-answering questions, generating code, debugging errors, and producing plots. Its architecture combines multi-agent LLM orchestration, semantic search across CLASS documentation, and a live Python execution environment. Deployed as a user-friendly web application, CLAPP lowers the entry barrier for scientists unfamiliar with AI tools and enables more productive human-AI collaboration in computational and numerical cosmology. The app is available at https://classclapp.streamlit.app
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05702v1">Semantic Reasoning Meets Numerical Precision: An LLM-Powered Multi-Agent System for Power Grid Control</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      The increasing penetration of Distributed Energy Resources (DERs), widespread adoption of Electric Vehicles (EVs), and the growing frequency of extreme weather events have significantly increased the complexity of power grid planning, operation, and management. Traditional rule-based systems and numerical optimization approaches often struggle with the scale, dynamics, and adaptability required by modern power networks. This paper introduces Grid-Agent, an autonomous, AI-driven framework that combines Large Language Models (LLMs) with multi-agent reinforcement learning to detect and remediate grid violations in real time. Grid-Agent integrates semantic reasoning with numerical precision through a modular agent architecture: a planning agent generates coordinated action sequences using numerical power flow solvers, while a validation agent evaluates system stability and action effectiveness via sandboxed execution with safety rollbacks. To ensure scalability, Grid-Agent incorporates an adaptive multiscale network representation that dynamically selects optimal encoding schemes based on network size and complexity. The framework enables coordinated violation resolution through optimizing switch configurations, battery deployment, and load curtailment strategies. Experimental results in standard IEEE and CIGRE test systems (IEEE 69-bus, CIGRE MV, and IEEE 30-bus) demonstrate superior violation mitigation performance. Additionally, the framework's built-in data collection and learning capabilities enable continuous learning and adaptation to diverse network topologies. The autonomous nature of the framework makes it particularly suitable for modern smart grid applications requiring rapid response to dynamic operating conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06577v1">Leveraging LLMs for Privacy-Aware Predictions in Participatory Budgeting</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Participatory Budgeting (PB) empowers citizens to propose and vote on public investment projects. Yet, despite its democratic potential, PB initiatives often suffer from low participation rates, limiting their visibility and perceived legitimacy. In this work, we aim to strengthen PB elections in two key ways: by supporting project proposers in crafting better proposals, and by helping PB organizers manage large volumes of submissions in a transparent manner. We propose a privacy-preserving approach to predict which PB proposals are likely to be funded, using only their textual descriptions and anonymous historical voting records -- without relying on voter demographics or personally identifiable information. We evaluate the performance of GPT 4 Turbo in forecasting proposal outcomes across varying contextual scenarios, observing that the LLM's prior knowledge needs to be complemented by past voting data to obtain predictions reflecting real-world PB voting behavior. Our findings highlight the potential of AI-driven tools to support PB processes by improving transparency, planning efficiency, and civic engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05625v1">How Do LLMs Persuade? Linear Probes Can Uncover Persuasion Dynamics in Multi-Turn Conversations</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have started to demonstrate the ability to persuade humans, yet our understanding of how this dynamic transpires is limited. Recent work has used linear probes, lightweight tools for analyzing model representations, to study various LLM skills such as the ability to model user sentiment and political perspective. Motivated by this, we apply probes to study persuasion dynamics in natural, multi-turn conversations. We leverage insights from cognitive science to train probes on distinct aspects of persuasion: persuasion success, persuadee personality, and persuasion strategy. Despite their simplicity, we show that they capture various aspects of persuasion at both the sample and dataset levels. For instance, probes can identify the point in a conversation where the persuadee was persuaded or where persuasive success generally occurs across the entire dataset. We also show that in addition to being faster than expensive prompting-based approaches, probes can do just as well and even outperform prompting in some settings, such as when uncovering persuasion strategy. This suggests probes as a plausible avenue for studying other complex behaviours such as deception and manipulation, especially in multi-turn settings and large-scale dataset analysis where prompting-based methods would be computationally inefficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05622v1">Simulating Human-Like Learning Dynamics with LLM-Empowered Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Capturing human learning behavior based on deep learning methods has become a major research focus in both psychology and intelligent systems. Recent approaches rely on controlled experiments or rule-based models to explore cognitive processes. However, they struggle to capture learning dynamics, track progress over time, or provide explainability. To address these challenges, we introduce LearnerAgent, a novel multi-agent framework based on Large Language Models (LLMs) to simulate a realistic teaching environment. To explore human-like learning dynamics, we construct learners with psychologically grounded profiles-such as Deep, Surface, and Lazy-as well as a persona-free General Learner to inspect the base LLM's default behavior. Through weekly knowledge acquisition, monthly strategic choices, periodic tests, and peer interaction, we can track the dynamic learning progress of individual learners over a full-year journey. Our findings are fourfold: 1) Longitudinal analysis reveals that only Deep Learner achieves sustained cognitive growth. Our specially designed "trap questions" effectively diagnose Surface Learner's shallow knowledge. 2) The behavioral and cognitive patterns of distinct learners align closely with their psychological profiles. 3) Learners' self-concept scores evolve realistically, with the General Learner developing surprisingly high self-efficacy despite its cognitive limitations. 4) Critically, the default profile of base LLM is a "diligent but brittle Surface Learner"-an agent that mimics the behaviors of a good student but lacks true, generalizable understanding. Extensive simulation experiments demonstrate that LearnerAgent aligns well with real scenarios, yielding more insightful findings about LLMs' behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05616v1">TrajEvo: Trajectory Prediction Heuristics Design via LLM-driven Evolution</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 arXiv admin note: substantial text overlap with arXiv:2505.04480
    </div>
    <details class="paper-abstract">
      Trajectory prediction is a critical task in modeling human behavior, especially in safety-critical domains such as social robotics and autonomous vehicle navigation. Traditional heuristics based on handcrafted rules often lack accuracy and generalizability. Although deep learning approaches offer improved performance, they typically suffer from high computational cost, limited explainability, and, importantly, poor generalization to out-of-distribution (OOD) scenarios. In this paper, we introduce TrajEvo, a framework that leverages Large Language Models (LLMs) to automatically design trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to generate and refine prediction heuristics from past trajectory data. We propose two key innovations: Cross-Generation Elite Sampling to encourage population diversity, and a Statistics Feedback Loop that enables the LLM to analyze and improve alternative predictions. Our evaluations demonstrate that TrajEvo outperforms existing heuristic methods across multiple real-world datasets, and notably surpasses both heuristic and deep learning methods in generalizing to an unseen OOD real-world dataset. TrajEvo marks a promising step toward the automated design of fast, explainable, and generalizable trajectory prediction heuristics. We release our source code to facilitate future research at https://github.com/ai4co/trajevo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05571v1">Fairy$\pm i$: the First 2-bit Complex LLM with All Parameters in $\{\pm1, \pm i\}$</a></div>
    <div class="paper-meta">
      📅 2025-08-07
      | 💬 13 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Quantization-Aware Training (QAT) integrates quantization into the training loop, enabling LLMs to learn robust low-bit representations, and is widely recognized as one of the most promising research directions. All current QAT research focuses on minimizing quantization error on full-precision models, where the full-precision accuracy acts as an upper bound (accuracy ceiling). No existing method has even attempted to surpass this ceiling. To break this ceiling, we propose a new paradigm: raising the ceiling (full-precision model), and then still quantizing it efficiently into 2 bits. We propose Fairy$\pm i$, the first 2-bit quantization framework for complex-valued LLMs. Specifically, our method leverages the representational advantages of the complex domain to boost full-precision accuracy. We map weights to the fourth roots of unity $\{\pm1, \pm i\}$, forming a perfectly symmetric and information-theoretically optimal 2-bit representation. Importantly, each quantized weight has either a zero real or imaginary part, enabling multiplication-free inference using only additions and element swaps. Experimental results show that Fairy$\pm i$ outperforms the ceiling of existing 2-bit quantization approaches in terms of both PPL and downstream tasks, while maintaining strict storage and compute efficiency. This work opens a new direction for building highly accurate and practical LLMs under extremely low-bit constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v3">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/JARVIS-Xs/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00255v2">SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers</a></div>
    <div class="paper-meta">
      📅 2025-08-07
    </div>
    <details class="paper-abstract">
      This study evaluates large language models (LLMs) in generating code from algorithm descriptions in recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a dual-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implements solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful non-reasoning and reasoning LLMs as foundational models. The best-performing LLM using \ModelName~achieves only 39% execution accuracy, highlighting the benchmark's difficulty. Our analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We make available our benchmark and code at https://github.com/xyzCS/SciReplicate-Bench and project homepage at https://xyzcs.github.io/scireplicate.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00222v3">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16781v2">Evaluating Robustness of LLMs in Question Answering on Multilingual Noisy OCR Data</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted at CIKM 2025
    </div>
    <details class="paper-abstract">
      Optical Character Recognition (OCR) plays a crucial role in digitizing historical and multilingual documents, yet OCR errors - imperfect extraction of text, including character insertion, deletion, and substitution can significantly impact downstream tasks like question-answering (QA). In this work, we conduct a comprehensive analysis of how OCR-induced noise affects the performance of Multilingual QA Systems. To support this analysis, we introduce a multilingual QA dataset MultiOCR-QA, comprising 50K question-answer pairs across three languages, English, French, and German. The dataset is curated from OCR-ed historical documents, which include different levels and types of OCR noise. We then evaluate how different state-of-the-art Large Language models (LLMs) perform under different error conditions, focusing on three major OCR error types. Our findings show that QA systems are highly prone to OCR-induced errors and perform poorly on noisy OCR text. By comparing model performance on clean versus noisy texts, we provide insights into the limitations of current approaches and emphasize the need for more noise-resilient QA systems in historical digitization contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06850v4">The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables remarkable capabilities in natural language processing and generation. However, these systems introduce unprecedented security vulnerabilities that extend beyond traditional content generation attacks to system-level compromise. This paper presents a comprehensive evaluation of the security of LLMs used as reasoning engines within autonomous agents, highlighting how they can be exploited as attack vectors capable of achieving complete computer takeover. We focus on how different attack surfaces and trust boundaries - Direct Prompt Injection, RAG Backdoor, and Inter Agent Trust - can be leveraged to orchestrate such takeovers. We demonstrate that adversaries can effectively coerce popular LLMs (including GPT-4, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 18 state-of-the-art LLMs reveals an alarming scenario: 94.4% of models succumb to Direct Prompt Injection and 83.3% are vulnerable to the more stealth and evasive RAG Backdoor Attack. Notably, we tested trust boundaries within multi-agent systems, where LLM agents interact and influence each other, and we revealed a critical security flaw: LLMs which successfully resist direct injection or RAG backdoor will execute identical payloads when requested by peer agents. Our findings show that 100.0% of tested LLMs can be compromised through Inter-Agent Trust Exploitation attacks and that every model exhibits context-dependent security behaviors that create exploitable blind spots. Our results also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11773v3">AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      A major challenge in developing robust and generalizable Human Activity Recognition (HAR) systems for smart homes is the lack of large and diverse labeled datasets. Variations in home layouts, sensor configurations, and individual behaviors further exacerbate this issue. To address this, we leverage the idea of embodied AI agents-virtual agents that perceive and act within simulated environments guided by internal world models. We introduce AgentSense, a virtual data generation pipeline in which agents live out daily routines in simulated smart homes, with behavior guided by Large Language Models (LLMs). The LLM generates diverse synthetic personas and realistic routines grounded in the environment, which are then decomposed into fine-grained actions. These actions are executed in an extended version of the VirtualHome simulator, which we augment with virtual ambient sensors that record the agents' activities. Our approach produces rich, privacy-preserving sensor data that reflects real-world diversity. We evaluate AgentSense on five real HAR datasets. Models pretrained on the generated data consistently outperform baselines, especially in low-resource settings. Furthermore, combining the generated virtual sensor data with a small amount of real data achieves performance comparable to training on full real-world datasets. These results highlight the potential of using LLM-guided embodied agents for scalable and cost-effective sensor data generation in HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03440v2">LLMs Have a Heart of Stone: Demystifying the Soft Thinking Ability of Large Reasoning Models</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 10 pages, 7 figures, working in progress
    </div>
    <details class="paper-abstract">
      Human cognition naturally engages with abstract and fluid concepts, whereas existing reasoning models often rely on generating discrete tokens, potentially constraining their expressive capabilities. Recent advancements aim to address this limitation by enabling large language models (LLMs) to generate soft, abstract tokens, thus facilitating reasoning within a continuous concept space. This paper explores the `Soft Thinking' capabilities of various LLMs by examining the models' internal behavior using a suite of probing techniques. Contrary to the common belief that Soft Thinking enables the simultaneous exploration of diverse reasoning paths, our findings reveal that LLMs predominantly rely on the most influential component of the soft inputs during subsequent decoding steps. This reliance hinders the exploration of different reasoning paths and reduces vanilla Soft Thinking to a form of greedy decoding, obscuring the advantage of transmitting more information through Soft Tokens. To tackle this issue, we explore sampling strategies to introduce \emph{randomness}, employing methods such as Dirichlet resampling and the Gumbel-Softmax trick. Our experiments demonstrate that incorporating randomness can alleviate the limitations of vanilla approaches and unleash the potential of Soft Thinking. Notably, the Gumbel-Softmax trick provides adequate randomness with controlled smoothness, resulting in superior performance across eight reasoning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04530v1">StyliTruth : Unlocking Stylized yet Truthful LLM Generation via Disentangled Steering</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Generating stylized large language model (LLM) responses via representation editing is a promising way for fine-grained output control. However, there exists an inherent trade-off: imposing a distinctive style often degrades truthfulness. Existing representation editing methods, by naively injecting style signals, overlook this collateral impact and frequently contaminate the model's core truthfulness representations, resulting in reduced answer correctness. We term this phenomenon stylization-induced truthfulness collapse. We attribute this issue to latent coupling between style and truth directions in certain key attention heads, and propose StyliTruth, a mechanism that preserves stylization while keeping truthfulness intact. StyliTruth separates the style-relevant and truth-relevant subspaces in the model's representation space via an orthogonal deflation process. This decomposition enables independent control of style and truth in their own subspaces, minimizing interference. By designing adaptive, token-level steering vectors within each subspace, we dynamically and precisely control the generation process to maintain both stylistic fidelity and truthfulness. We validate our method on multiple styles and languages. Extensive experiments and analyses show that StyliTruth significantly reduces stylization-induced truthfulness collapse and outperforms existing inference-time intervention methods in balancing style adherence with truthfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22716v2">From Sufficiency to Reflection: Reinforcement-Guided Thinking Quality in Retrieval-Augmented Reasoning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Reinforcement learning-based retrieval-augmented generation (RAG) methods enhance the reasoning abilities of large language models (LLMs). However, most rely only on final-answer rewards, overlooking intermediate reasoning quality. This paper analyzes existing RAG reasoning models and identifies three main failure patterns: (1) information insufficiency, meaning the model fails to retrieve adequate support; (2) faulty reasoning, where logical or content-level flaws appear despite sufficient information; and (3) answer-reasoning inconsistency, where a valid reasoning chain leads to a mismatched final answer. We propose TIRESRAG-R1, a novel framework using a think-retrieve-reflect process and a multi-dimensional reward system to improve reasoning and stability. TIRESRAG-R1 introduces: (1) a sufficiency reward to encourage thorough retrieval; (2) a reasoning quality reward to assess the rationality and accuracy of the reasoning chain; and (3) a reflection reward to detect and revise errors. It also employs a difficulty-aware reweighting strategy and training sample filtering to boost performance on complex tasks. Experiments on four multi-hop QA datasets show that TIRESRAG-R1 outperforms prior RAG methods and generalizes well to single-hop tasks. The code and data are available at: https://github.com/probe2/TIRESRAG-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04451v1">Automatic LLM Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Red teaming is critical for identifying vulnerabilities and building trust in current LLMs. However, current automated methods for Large Language Models (LLMs) rely on brittle prompt templates or single-turn attacks, failing to capture the complex, interactive nature of real-world adversarial dialogues. We propose a novel paradigm: training an AI to strategically `break' another AI. By formalizing red teaming as a Markov Decision Process (MDP) and employing a hierarchical Reinforcement Learning (RL) framework, we effectively address the inherent sparse reward and long-horizon challenges. Our generative agent learns coherent, multi-turn attack strategies through a fine-grained, token-level harm reward, enabling it to uncover subtle vulnerabilities missed by existing baselines. This approach sets a new state-of-the-art, fundamentally reframing LLM red teaming as a dynamic, trajectory-based process (rather than a one-step test) essential for robust AI deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15299v4">Inside-Out: Hidden Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average relative gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) put a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04440v1">StepFun-Formalizer: Unlocking the Autoformalization Potential of LLMs through Knowledge-Reasoning Fusion</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 24 pages, 17 figures, under review
    </div>
    <details class="paper-abstract">
      Autoformalization aims to translate natural-language mathematical statements into a formal language. While LLMs have accelerated progress in this area, existing methods still suffer from low accuracy. We identify two key abilities for effective autoformalization: comprehensive mastery of formal-language domain knowledge, and reasoning capability of natural language problem understanding and informal-formal alignment. Without the former, a model cannot identify the correct formal objects; without the latter, it struggles to interpret real-world contexts and map them precisely into formal expressions. To address these gaps, we introduce ThinkingF, a data synthesis and training pipeline that improves both abilities. First, we construct two datasets: one by distilling and selecting large-scale examples rich in formal knowledge, and another by generating informal-to-formal reasoning trajectories guided by expert-designed templates. We then apply SFT and RLVR with these datasets to further fuse and refine the two abilities. The resulting 7B and 32B models exhibit both comprehensive formal knowledge and strong informal-to-formal reasoning. Notably, StepFun-Formalizer-32B achieves SOTA BEq@1 scores of 40.5% on FormalMATH-Lite and 26.7% on ProverBench, surpassing all prior general-purpose and specialized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04428v1">\textsc{SimInstruct}: A Responsible Tool for Collecting Scaffolding Dialogues Between Experts and LLM-Simulated Novices</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      High-quality, multi-turn instructional dialogues between novices and experts are essential for developing AI systems that support teaching, learning, and decision-making. These dialogues often involve scaffolding -- the process by which an expert supports a novice's thinking through questions, feedback, and step-by-step guidance. However, such data are scarce due to privacy concerns in recording and the vulnerability inherent in help-seeking. We present SimInstruct, a scalable, expert-in-the-loop tool for collecting scaffolding dialogues. Using teaching development coaching as an example domain, SimInstruct simulates novice instructors via LLMs, varying their teaching challenges and LLM's persona traits, while human experts provide multi-turn feedback, reasoning, and instructional support. This design enables the creation of realistic, pedagogically rich dialogues without requiring real novice participants. Our results reveal that persona traits, such as extroversion and introversion, meaningfully influence how experts engage. Compared to real mentoring recordings, SimInstruct dialogues demonstrate comparable pedagogical relevance and cognitive depth. Experts also reported the process as engaging and reflective, improving both data quality and their own professional insight. We further fine-tuned a LLaMA model to be an expert model using the augmented dataset, which outperformed GPT-4o in instructional quality. Our analysis highlights GPT-4o's limitations in weak reflective questioning, overuse of generic praise, a condescending tone, and a tendency to overwhelm novices with excessive suggestions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04412v1">Beyond Pixels: Exploring DOM Downsampling for LLM-Based Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Frontier LLMs only recently enabled serviceable, autonomous web agents. At that, a model poses as an instantaneous domain model backend. Ought to suggest interaction, it is consulted with a web-based task and respective application state. The key problem lies in application state serialisation $\unicode{x2013}$ referred to as snapshot. State-of-the-art web agents are premised on grounded GUI snapshots, i.e., screenshots enhanced with visual cues. Not least to resemble human perception, but for images representing relatively cheap means of model input. LLM vision still lag behind code interpretation capabilities. DOM snapshots, which structurally resemble HTML, impose a desired alternative. Vast model input token size, however, disables reliable implementation with web agents to date. We propose D2Snap, a first-of-its-kind DOM downsampling algorithm. Based on a GPT-4o backend, we evaluate D2Snap on tasks sampled from the Online-Mind2Web dataset. The success rate of D2Snap-downsampled DOM snapshots (67%) matches a grounded GUI snapshot baseline (65%) $\unicode{x2013}$ within the same input token order of magnitude (1e3). Our best evaluated configurations $\unicode{x2013}$ one token order above, but within the model's context window $\unicode{x2013}$ outperform this baseline by 8%. Our evaluation, moreover, yields that DOM-inherent hierarchy embodies a strong UI feature for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04405v1">FlexQ: Efficient Post-training INT6 Quantization for LLM Serving via Algorithm-System Co-Design</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate exceptional performance but entail significant memory and computational costs, restricting their practical deployment. While existing INT4/INT8 quantization reduces these costs, they often degrade accuracy or lack optimal efficiency. INT6 quantization offers a superior trade-off between model accuracy and inference efficiency, but lacks hardware support in modern GPUs, forcing emulation via higher-precision arithmetic units that limit acceleration. In this paper, we propose FlexQ, a novel post-training INT6 quantization framework combining algorithmic innovation with system-level optimizations. FlexQ employs uniform 6-bit weight quantization across all layers, with adaptive retention of 8-bit activations in layers identified through layer-wise sensitivity analysis. To maximize hardware efficiency, we develop a specialized high-performance GPU kernel supporting matrix multiplication for W6A6 and W6A8 representations via Binary Tensor Core (BTC) equivalents, effectively bypassing the lack of native INT6 tensor cores. Evaluations on LLaMA models show FlexQ maintains near-FP16 accuracy, with perplexity increases of no more than 0.05. The proposed kernel achieves an average 1.39$\times$ speedup over ABQ-LLM on LLaMA-2-70B linear layers. End-to-end, FlexQ delivers 1.33$\times$ inference acceleration and 1.21$\times$ memory savings over SmoothQuant. Code is released at https://github.com/FlyFoxPlayer/FlexQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04401v1">Why are LLMs' abilities emergent?</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      The remarkable success of Large Language Models (LLMs) in generative tasks has raised fundamental questions about the nature of their acquired capabilities, which often appear to emerge unexpectedly without explicit training. This paper examines the emergent properties of Deep Neural Networks (DNNs) through both theoretical analysis and empirical observation, addressing the epistemological challenge of "creation without understanding" that characterises contemporary AI development. We explore how the neural approach's reliance on nonlinear, stochastic processes fundamentally differs from symbolic computational paradigms, creating systems whose macro-level behaviours cannot be analytically derived from micro-level neuron activities. Through analysis of scaling laws, grokking phenomena, and phase transitions in model capabilities, I demonstrate that emergent abilities arise from the complex dynamics of highly sensitive nonlinear systems rather than simply from parameter scaling alone. My investigation reveals that current debates over metrics, pre-training loss thresholds, and in-context learning miss the fundamental ontological nature of emergence in DNNs. I argue that these systems exhibit genuine emergent properties analogous to those found in other complex natural phenomena, where systemic capabilities emerge from cooperative interactions among simple components without being reducible to their individual behaviours. The paper concludes that understanding LLM capabilities requires recognising DNNs as a new domain of complex dynamical systems governed by universal principles of emergence, similar to those operating in physics, chemistry, and biology. This perspective shifts the focus from purely phenomenological definitions of emergence to understanding the internal dynamic transformations that enable these systems to acquire capabilities that transcend their individual components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03329v2">Industrial LLM-based Code Optimization under Regulation: A Mixture-of-Agents Approach</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Submitted to ASE'25 Industry Showcase
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) for code optimization have enabled industrial platforms to automate software performance engineering at unprecedented scale and speed. Yet, organizations in regulated industries face strict constraints on which LLMs they can use - many cannot utilize commercial models due to data privacy regulations and compliance requirements, creating a significant challenge for achieving high-quality code optimization while maintaining cost-effectiveness. We address this by implementing a Mixture-of-Agents (MoA) approach that directly synthesizes code from multiple specialized LLMs, comparing it against TurinTech AI's vanilla Genetic Algorithm (GA)-based ensemble system and individual LLM optimizers using real-world industrial codebases. Our key contributions include: (1) First MoA application to industrial code optimization using real-world codebases; (2) Empirical evidence that MoA excels with open-source models, achieving 14.3% to 22.2% cost savings and 28.6% to 32.2% faster optimization times for regulated environments; (3) Deployment guidelines demonstrating GA's advantage with commercial models while both ensembles outperform individual LLMs; and (4) Real-world validation across 50 code snippets and seven LLM combinations, generating over 8,700 variants, addresses gaps in industrial LLM ensemble evaluation. This provides actionable guidance for organizations balancing regulatory compliance with optimization performance in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04353v1">LUST: A Multi-Modal Framework with Hierarchical LLM-based Scoring for Learned Thematic Significance Tracking in Multimedia Content</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 5 pages and 4 figures
    </div>
    <details class="paper-abstract">
      This paper introduces the Learned User Significance Tracker (LUST), a framework designed to analyze video content and quantify the thematic relevance of its segments in relation to a user-provided textual description of significance. LUST leverages a multi-modal analytical pipeline, integrating visual cues from video frames with textual information extracted via Automatic Speech Recognition (ASR) from the audio track. The core innovation lies in a hierarchical, two-stage relevance scoring mechanism employing Large Language Models (LLMs). An initial "direct relevance" score, $S_{d,i}$, assesses individual segments based on immediate visual and auditory content against the theme. This is followed by a "contextual relevance" score, $S_{c,i}$, that refines the assessment by incorporating the temporal progression of preceding thematic scores, allowing the model to understand evolving narratives. The LUST framework aims to provide a nuanced, temporally-aware measure of user-defined significance, outputting an annotated video with visualized relevance scores and comprehensive analytical logs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12286v3">The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable and widely adopted, benchmarks play a central role in assessing their practical utility. For example, SWE-Bench Verified has emerged as a critical benchmark for evaluating LLMs' software engineering abilities, particularly their aptitude for resolving real-world GitHub issues. Recent LLMs show impressive performance on SWE-Bench, leading to optimism about their capacity for complex coding tasks. However, current evaluation protocols may overstate these models' true capabilities. It is crucial to distinguish LLMs' generalizable problem-solving ability and other learned artifacts. In this work, we introduce two diagnostic tasks: file path identification from issue descriptions alone and ground truth function reproduction with only the current file context and issue description to probe models' underlying knowledge. We present empirical evidence that performance gains on SWE-Bench-Verified may be partially driven by memorization rather than genuine problem-solving. We show that state-of-the-art models achieve up to 76% accuracy in identifying buggy file paths using only issue descriptions, without access to repository structure. This performance is merely up to 53% on tasks from repositories not included in SWE-Bench, pointing to possible data contamination or memorization. Similar patterns are also observed for the function reproduction task, where the verbatim similarity is much higher on SWE-Bench Verified than on other similar coding benchmarks (up to 35% consecutive 5-gram accuracy on SWE-Bench Verified and Full, but only up to 18% for tasks in other benchmarks). These findings raise concerns about the validity of existing results and underscore the need for more robust, contamination-resistant benchmarks to reliably evaluate LLMs' coding abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13287v5">PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      Selecting a sample generation scheme from multiple prompt-based generative models, including large language models (LLMs) and prompt-guided image and video generation models, is typically addressed by choosing the model that maximizes an averaged evaluation score. However, this score-based selection overlooks the possibility that different models achieve the best generation performance for different types of text prompts. An online identification of the best generation model for various input prompts can reduce the costs associated with querying sub-optimal models. In this work, we explore the possibility of varying rankings of text-based generative models for different text prompts and propose an online learning framework to predict the best data generation model for a given input prompt. The proposed PAK-UCB algorithm addresses a contextual bandit (CB) setting with shared context variables across the arms, utilizing the generated data to update kernel-based functions that predict the score of each model available for unseen text prompts. Additionally, we leverage random Fourier features (RFF) to accelerate the online learning process of PAK-UCB. Our numerical experiments on real and simulated text-to-image and image-to-text generative models show that RFF-UCB performs successfully in identifying the best generation model across different sample types. The code is available at: github.com/yannxiaoyanhu/dgm-online-select.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04279v1">Mockingbird: How does LLM perform in general machine learning tasks?</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now being used with increasing frequency as chat bots, tasked with the summarizing information or generating text and code in accordance with user instructions. The rapid increase in reasoning capabilities and inference speed of LLMs has revealed their remarkable potential for applications extending beyond the domain of chat bots to general machine learning tasks. This work is conducted out of the curiosity about such potential. In this work, we propose a framework Mockingbird to adapt LLMs to general machine learning tasks and evaluate its performance and scalability on several general machine learning tasks. The core concept of this framework is instructing LLMs to role-play functions and reflect on its mistakes to improve itself. Our evaluation and analysis result shows that LLM-driven machine learning methods, such as Mockingbird, can achieve acceptable results on common machine learning tasks; however, solely reflecting on its own currently cannot outperform the effect of domain-specific documents and feedback from human experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14448v2">How Far Can LLMs Improve from Experience? Measuring Test-Time Learning Ability in LLMs with Human Comparison</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      As evaluation designs of large language models may shape our trajectory toward artificial general intelligence, comprehensive and forward-looking assessment is essential. Existing benchmarks primarily assess static knowledge, while intelligence also entails the ability to rapidly learn from experience. To this end, we advocate for the evaluation of Test-time Learning, the capacity to improve performance in experience-based, reasoning-intensive tasks during test time. In this work, we propose semantic games as effective testbeds for evaluating test-time learning, due to their resistance to saturation and inherent demand for strategic reasoning. We introduce an objective evaluation framework that compares model performance under both limited and cumulative experience settings, and contains four forms of experience representation. To provide a comparative baseline, we recruit eight human participants to complete the same task. Results show that LLMs exhibit measurable test-time learning capabilities; however, their improvements are less stable under cumulative experience and progress more slowly than those observed in humans. These findings underscore the potential of LLMs as general-purpose learning machines, while also revealing a substantial intellectual gap between models and humans, irrespective of how well LLMs perform on static benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04257v1">KVSink: Understanding and Enhancing the Preservation of Attention Sinks in KV Cache Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 Published as a conference paper at COLM 2025
    </div>
    <details class="paper-abstract">
      Key-Value (KV) cache quantization has become a widely adopted optimization technique for efficient large language models (LLMs) inference by reducing KV cache memory usage and mitigating memory-bound constraints. Recent studies have emphasized the importance of preserving the original precision of KVs for the first few tokens to ensure the protection of attention sinks. While this approach has proven effective in mitigating performance degradation, its underlying principles remain insufficiently understood. Moreover, it fails to address the recent discovery that attention sinks can emerge beyond the initial token positions. In this work, we elucidate the underlying mechanisms of attention sinks during inference by examining their role in the cross-layer evolution of extreme activation outliers. Additionally, we provide a comprehensive analysis of the interplay between attention sinks and KV cache quantization. Based on our enhanced understanding, we introduce \textit{\textbf{KVSink}}, a plug-and-play method that effectively predicts sink tokens with negligible overhead, enabling more thorough preservation. Extensive experiments demonstrate that KVSink outperforms the existing Preserve-First-N (PFN) strategy, offering more effective preservation of attention sinks during KV cache quantization. Moreover, when applied to the well-established KVQuant method, KVSink further improves perplexity (PPL) and reduces reliance on 16-bit numerical outliers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04231v1">Empowering Time Series Forecasting with LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) powered agents have emerged as effective planners for Automated Machine Learning (AutoML) systems. While most existing AutoML approaches focus on automating feature engineering and model architecture search, recent studies in time series forecasting suggest that lightweight models can often achieve state-of-the-art performance. This observation led us to explore improving data quality, rather than model architecture, as a potentially fruitful direction for AutoML on time series data. We propose DCATS, a Data-Centric Agent for Time Series. DCATS leverages metadata accompanying time series to clean data while optimizing forecasting performance. We evaluated DCATS using four time series forecasting models on a large-scale traffic volume forecasting dataset. Results demonstrate that DCATS achieves an average 6% error reduction across all tested models and time horizons, highlighting the potential of data-centric approaches in AutoML for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04206v1">ViLLA-MMBench: A Unified Benchmark Suite for LLM-Augmented Multimodal Movie Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-06
      | 💬 17 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Recommending long-form video content demands joint modeling of visual, audio, and textual modalities, yet most benchmarks address only raw features or narrow fusion. We present ViLLA-MMBench, a reproducible, extensible benchmark for LLM-augmented multimodal movie recommendation. Built on MovieLens and MMTF-14K, it aligns dense item embeddings from three modalities: audio (block-level, i-vector), visual (CNN, AVF), and text. Missing or sparse metadata is automatically enriched using state-of-the-art LLMs (e.g., OpenAI Ada), generating high-quality synopses for thousands of movies. All text (raw or augmented) is embedded with configurable encoders (Ada, LLaMA-2, Sentence-T5), producing multiple ready-to-use sets. The pipeline supports interchangeable early-, mid-, and late-fusion (concatenation, PCA, CCA, rank-aggregation) and multiple backbones (MF, VAECF, VBPR, AMR, VMF) for ablation. Experiments are fully declarative via a single YAML file. Evaluation spans accuracy (Recall, nDCG) and beyond-accuracy metrics: cold-start rate, coverage, novelty, diversity, fairness. Results show LLM-based augmentation and strong text embeddings boost cold-start and coverage, especially when fused with audio-visual features. Systematic benchmarking reveals universal versus backbone- or metric-specific combinations. Open-source code, embeddings, and configs enable reproducible, fair multimodal RS research and advance principled generative AI integration in large-scale recommendation. Code: https://recsys-lab.github.io/ViLLA-MMBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12342v2">CRAB: A Benchmark for Evaluating Curation of Retrieval-Augmented LLMs in Biomedicine</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Recent development in Retrieval-Augmented Large Language Models (LLMs) have shown great promise in biomedical applications. How ever, a critical gap persists in reliably evaluating their curation ability the process by which models select and integrate relevant references while filtering out noise. To address this, we introduce the benchmark for Curation of Retrieval-Augmented LLMs in Biomedicine (CRAB), the first multilingual benchmark tailored for evaluating the biomedical curation of retrieval-augmented LLMs, available in English, French, German and Chinese. By incorporating a novel citation-based evaluation metric, CRAB quantifies the curation performance of retrieval-augmented LLMs in biomedicine. Experimental results reveal significant discrepancies in the curation performance of mainstream LLMs, underscoring the urgent need to improve it in the domain of biomedicine. Our dataset is available at https://huggingface.co/datasets/zhm0/CRAB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.04199v1">Reasoning Beyond Labels: Measuring LLM Sentiment in Low-Resource, Culturally Nuanced Contexts</a></div>
    <div class="paper-meta">
      📅 2025-08-06
    </div>
    <details class="paper-abstract">
      Sentiment analysis in low-resource, culturally nuanced contexts challenges conventional NLP approaches that assume fixed labels and universal affective expressions. We present a diagnostic framework that treats sentiment as a context-dependent, culturally embedded construct, and evaluate how large language models (LLMs) reason about sentiment in informal, code-mixed WhatsApp messages from Nairobi youth health groups. Using a combination of human-annotated data, sentiment-flipped counterfactuals, and rubric-based explanation evaluation, we probe LLM interpretability, robustness, and alignment with human reasoning. Framing our evaluation through a social-science measurement lens, we operationalize and interrogate LLMs outputs as an instrument for measuring the abstract concept of sentiment. Our findings reveal significant variation in model reasoning quality, with top-tier LLMs demonstrating interpretive stability, while open models often falter under ambiguity or sentiment shifts. This work highlights the need for culturally sensitive, reasoning-aware AI evaluation in complex, real-world communication.
    </details>
</div>
