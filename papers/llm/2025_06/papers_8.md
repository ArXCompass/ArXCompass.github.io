# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12097v1">UCD: Unlearning in LLMs via Contrastive Decoding</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to remove specific information, e.g. sensitive or undesirable content, from large language models (LLMs) while preserving overall performance. We propose an inference-time unlearning algorithm that uses contrastive decoding, leveraging two auxiliary smaller models, one trained without the forget set and one trained with it, to guide the outputs of the original model using their difference during inference. Our strategy substantially improves the tradeoff between unlearning effectiveness and model utility. We evaluate our approach on two unlearning benchmarks, TOFU and MUSE. Results show notable gains in both forget quality and retained performance in comparison to prior approaches, suggesting that incorporating contrastive decoding can offer an efficient, practical avenue for unlearning concepts in large-scale models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10966v1">GENMANIP: LLM-driven Simulation for Generalizable Instruction-Following Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Robotic manipulation in real-world settings remains challenging, especially regarding robust generalization. Existing simulation platforms lack sufficient support for exploring how policies adapt to varied instructions and scenarios. Thus, they lag behind the growing interest in instruction-following foundation models like LLMs, whose adaptability is crucial yet remains underexplored in fair comparisons. To bridge this gap, we introduce GenManip, a realistic tabletop simulation platform tailored for policy generalization studies. It features an automatic pipeline via LLM-driven task-oriented scene graph to synthesize large-scale, diverse tasks using 10K annotated 3D object assets. To systematically assess generalization, we present GenManip-Bench, a benchmark of 200 scenarios refined via human-in-the-loop corrections. We evaluate two policy types: (1) modular manipulation systems integrating foundation models for perception, reasoning, and planning, and (2) end-to-end policies trained through scalable data collection. Results show that while data scaling benefits end-to-end methods, modular systems enhanced with foundation models generalize more effectively across diverse scenarios. We anticipate this platform to facilitate critical insights for advancing policy generalization in realistic conditions. Project Page: https://genmanip.axi404.top/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10949v1">Monitoring Decomposition Attacks in LLMs with Lightweight Sequential Monitors</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Current LLM safety defenses fail under decomposition attacks, where a malicious goal is decomposed into benign subtasks that circumvent refusals. The challenge lies in the existing shallow safety alignment techniques: they only detect harm in the immediate prompt and do not reason about long-range intent, leaving them blind to malicious intent that emerges over a sequence of seemingly benign instructions. We therefore propose adding an external monitor that observes the conversation at a higher granularity. To facilitate our study of monitoring decomposition attacks, we curate the largest and most diverse dataset to date, including question-answering, text-to-image, and agentic tasks. We verify our datasets by testing them on frontier LLMs and show an 87% attack success rate on average on GPT-4o. This confirms that decomposition attack is broadly effective. Additionally, we find that random tasks can be injected into the decomposed subtasks to further obfuscate malicious intents. To defend in real time, we propose a lightweight sequential monitoring framework that cumulatively evaluates each subtask. We show that a carefully prompt engineered lightweight monitor achieves a 93% defense success rate, beating reasoning models like o3 mini as a monitor. Moreover, it remains robust against random task injection and cuts cost by 90% and latency by 50%. Our findings suggest that lightweight sequential monitors are highly effective in mitigating decomposition attacks and are viable in deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15724v2">LLM-Cure: LLM-based Competitor User Review Analysis for Feature Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      The exponential growth of the mobile app market underscores the importance of constant innovation and rapid response to user demands. As user satisfaction is paramount to the success of a mobile application (app), developers typically rely on user reviews, which represent user feedback that includes ratings and comments to identify areas for improvement. However, the sheer volume of user reviews poses challenges in manual analysis, necessitating automated approaches. Existing automated approaches either analyze only the target apps reviews, neglecting the comparison of similar features to competitors or fail to provide suggestions for feature enhancement. To address these gaps, we propose a Large Language Model (LLM)-based Competitive User Review Analysis for Feature Enhancement) (LLM-Cure), an approach powered by LLMs to automatically generate suggestion s for mobile app feature improvements. More specifically, LLM-Cure identifies and categorizes features within reviews by applying LLMs. When provided with a complaint in a user review, LLM-Cure curates highly rated (4 and 5 stars) reviews in competing apps related to the complaint and proposes potential improvements tailored to the target application. We evaluate LLM-Cure on 1,056,739 reviews of 70 popular Android apps. Our evaluation demonstrates that LLM-Cure significantly outperforms the state-of-the-art approaches in assigning features to reviews by up to 13% in F1-score, up to 16% in recall and up to 11% in precision. Additionally, LLM-Cure demonstrates its capability to provide suggestions for resolving user complaints. We verify the suggestions using the release notes that reflect the changes of features in the target mobile app. LLM-Cure achieves a promising average of 73% of the implementation of the provided suggestions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03710v2">Improving LLM Safety Alignment with Dual-Objective Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10922v1">Robustly Improving LLM Fairness in Realistic Settings via Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in high-stakes hiring applications, making decisions that directly impact people's careers and livelihoods. While prior studies suggest simple anti-bias prompts can eliminate demographic biases in controlled evaluations, we find these mitigations fail when realistic contextual details are introduced. We address these failures through internal bias mitigation: by identifying and neutralizing sensitive attribute directions within model activations, we achieve robust bias reduction across all tested scenarios. Across leading commercial (GPT-4o, Claude 4 Sonnet, Gemini 2.5 Flash) and open-source models (Gemma-2 27B, Gemma-3, Mistral-24B), we find that adding realistic context such as company names, culture descriptions from public careers pages, and selective hiring constraints (e.g.,``only accept candidates in the top 10\%") induces significant racial and gender biases (up to 12\% differences in interview rates). When these biases emerge, they consistently favor Black over White candidates and female over male candidates across all tested models and scenarios. Moreover, models can infer demographics and become biased from subtle cues like college affiliations, with these biases remaining invisible even when inspecting the model's chain-of-thought reasoning. To address these limitations, our internal bias mitigation identifies race and gender-correlated directions and applies affine concept editing at inference time. Despite using directions from a simple synthetic dataset, the intervention generalizes robustly, consistently reducing bias to very low levels (typically under 1\%, always below 2.5\%) while largely maintaining model performance. Our findings suggest that practitioners deploying LLMs for hiring should adopt more realistic evaluation methodologies and consider internal mitigation strategies for equitable outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10903v1">Beyond Gold Standards: Epistemic Ensemble of LLM Judges for Formal Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Autoformalization plays a crucial role in formal mathematical reasoning by enabling the automatic translation of natural language statements into formal languages. While recent advances using large language models (LLMs) have shown promising results, methods for automatically evaluating autoformalization remain underexplored. As one moves to more complex domains (e.g., advanced mathematics), human evaluation requires significant time and domain expertise, especially as the complexity of the underlying statements and background knowledge increases. LLM-as-a-judge presents a promising approach for automating such evaluation. However, existing methods typically employ coarse-grained and generic evaluation criteria, which limit their effectiveness for advanced formal mathematical reasoning, where quality hinges on nuanced, multi-granular dimensions. In this work, we take a step toward addressing this gap by introducing a systematic, automatic method to evaluate autoformalization tasks. The proposed method is based on an epistemically and formally grounded ensemble (EFG) of LLM judges, defined on criteria encompassing logical preservation (LP), mathematical consistency (MC), formal validity (FV), and formal quality (FQ), resulting in a transparent assessment that accounts for different contributing factors. We validate the proposed framework to serve as a proxy for autoformalization assessment within the domain of formal mathematics. Overall, our experiments demonstrate that the EFG ensemble of LLM judges is a suitable emerging proxy for evaluation, more strongly correlating with human assessments than a coarse-grained model, especially when assessing formal qualities. These findings suggest that LLM-as-judges, especially when guided by a well-defined set of atomic properties, could offer a scalable, interpretable, and reliable support for evaluating formal mathematical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14432v2">PLAY2PROMPT: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 ACL 2025 Long Paper (Findings)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated with specialized external tools, yet many tasks demand zero-shot tool usage with minimal or noisy documentation. Existing solutions rely on manual rewriting or labeled data for validation, making them inapplicable in true zero-shot settings. To address these challenges, we propose PLAY2PROMPT, an automated framework that systematically "plays" with each tool to explore its input-output behaviors. Through this iterative trial-and-error process, PLAY2PROMPT refines tool documentation and generates usage examples without any labeled data. These examples not only guide LLM inference but also serve as validation to further enhance tool utilization. Extensive experiments on real-world tasks demonstrate that PLAY2PROMPT significantly improves zero-shot tool performance across both open and closed models, offering a scalable and effective solution for domain-specific tool integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10885v1">Slimming Down LLMs Without Losing Their Minds</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      This paper investigates and validates the impact of fine-tuning on large language model performance, focusing on parameter-efficient methods (LoRA and QLoRA). We evaluate model capabilities across three key domains: (1) commonsense reasoning (HellaSwag), (2) mathematical reasoning (GSM8K), and (3) multi-domain knowledge (MMLU-CS). Our findings demonstrate that: (1) LoRA-based methods effectively improve task-specific performance while maintaining computational efficiency, and (2) performance strongly depends on alignment between fine-tuning dataset and benchmark tasks. The study provides both theoretical insights into parameter-efficient mechanisms and practical guidance for developers implementing efficient LLM adaptation with limited resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10859v1">Precise Zero-Shot Pointwise Ranking with LLMs through Post-Aggregated Global Context Information</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted by SIGIR 2025
    </div>
    <details class="paper-abstract">
      Recent advancements have successfully harnessed the power of Large Language Models (LLMs) for zero-shot document ranking, exploring a variety of prompting strategies. Comparative approaches like pairwise and listwise achieve high effectiveness but are computationally intensive and thus less practical for larger-scale applications. Scoring-based pointwise approaches exhibit superior efficiency by independently and simultaneously generating the relevance scores for each candidate document. However, this independence ignores critical comparative insights between documents, resulting in inconsistent scoring and suboptimal performance. In this paper, we aim to improve the effectiveness of pointwise methods while preserving their efficiency through two key innovations: (1) We propose a novel Global-Consistent Comparative Pointwise Ranking (GCCP) strategy that incorporates global reference comparisons between each candidate and an anchor document to generate contrastive relevance scores. We strategically design the anchor document as a query-focused summary of pseudo-relevant candidates, which serves as an effective reference point by capturing the global context for document comparison. (2) These contrastive relevance scores can be efficiently Post-Aggregated with existing pointwise methods, seamlessly integrating essential Global Context information in a training-free manner (PAGC). Extensive experiments on the TREC DL and BEIR benchmark demonstrate that our approach significantly outperforms previous pointwise methods while maintaining comparable efficiency. Our method also achieves competitive performance against comparative methods that require substantially more computational resources. More analyses further validate the efficacy of our anchor construction strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10829v1">LLM-Driven Personalized Answer Generation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 This is the preprint version of a paper accepted at AIED 2025. The final version will be published by Springer
    </div>
    <details class="paper-abstract">
      Online learning has experienced rapid growth due to its flexibility and accessibility. Personalization, adapted to the needs of individual learners, is crucial for enhancing the learning experience, particularly in online settings. A key aspect of personalization is providing learners with answers customized to their specific questions. This paper therefore explores the potential of Large Language Models (LLMs) to generate personalized answers to learners' questions, thereby enhancing engagement and reducing the workload on educators. To evaluate the effectiveness of LLMs in this context, we conducted a comprehensive study using the StackExchange platform in two distinct areas: language learning and programming. We developed a framework and a dataset for validating automatically generated personalized answers. Subsequently, we generated personalized answers using different strategies, including 0-shot, 1-shot, and few-shot scenarios. The generated answers were evaluated using three methods: 1. BERTScore, 2. LLM evaluation, and 3. human evaluation. Our findings indicated that providing LLMs with examples of desired answers (from the learner or similar learners) can significantly enhance the LLMs' ability to tailor responses to individual learners' needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10822v1">ReCUT: Balancing Reasoning Length and Accuracy in LLMs via Stepwise Trails and Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Recent advances in Chain-of-Thought (CoT) prompting have substantially improved the reasoning capabilities of Large Language Models (LLMs). However, these methods often suffer from overthinking, leading to unnecessarily lengthy or redundant reasoning traces. Existing approaches attempt to mitigate this issue through curating multiple reasoning chains for training LLMs, but their effectiveness is often constrained by the quality of the generated data and prone to overfitting. To address the challenge, we propose Reasoning Compression ThroUgh Stepwise Trials (ReCUT), a novel method aimed at balancing the accuracy and length of reasoning trajectory. Specifically, ReCUT employs a stepwise exploration mechanism and a long-short switched sampling strategy, enabling LLMs to incrementally generate diverse reasoning paths. These paths are evaluated and used to construct preference pairs to train two specialized models (Gemini LLMs)-one optimized for reasoning accuracy, the other for shorter reasoning. A final integrated model is obtained by interpolating the parameters of these two models. Experimental results across multiple math reasoning datasets and backbone models demonstrate that ReCUT significantly reduces reasoning lengths by approximately 30-50%, while maintaining or improving reasoning accuracy compared to various baselines. All codes and data will be released via https://github.com/NEUIR/ReCUT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18765v2">A Vision for Auto Research with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      This paper introduces Agent-Based Auto Research, a structured multi-agent framework designed to automate, coordinate, and optimize the full lifecycle of scientific research. Leveraging the capabilities of large language models (LLMs) and modular agent collaboration, the system spans all major research phases, including literature review, ideation, methodology planning, experimentation, paper writing, peer review response, and dissemination. By addressing issues such as fragmented workflows, uneven methodological expertise, and cognitive overload, the framework offers a systematic and scalable approach to scientific inquiry. Preliminary explorations demonstrate the feasibility and potential of Auto Research as a promising paradigm for self-improving, AI-driven research processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10779v1">Improving Named Entity Transcription with Contextual LLM-based Revision</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      With recent advances in modeling and the increasing amount of supervised training data, automatic speech recognition (ASR) systems have achieved remarkable performance on general speech. However, the word error rate (WER) of state-of-the-art ASR remains high for named entities. Since named entities are often the most critical keywords, misrecognizing them can affect all downstream applications, especially when the ASR system functions as the front end of a complex system. In this paper, we introduce a large language model (LLM) revision mechanism to revise incorrect named entities in ASR predictions by leveraging the LLM's reasoning ability as well as local context (e.g., lecture notes) containing a set of correct named entities. Finally, we introduce the NER-MIT-OpenCourseWare dataset, containing 45 hours of data from MIT courses for development and testing. On this dataset, our proposed technique achieves up to 30\% relative WER reduction for named entities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10769v1">Different Questions, Different Models: Fine-Grained Evaluation of Uncertainty and Calibration in Clinical QA with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Accurate and well-calibrated uncertainty estimates are essential for deploying large language models (LLMs) in high-stakes domains such as clinical decision support. We present a fine-grained evaluation of uncertainty estimation methods for clinical multiple-choice question answering, covering ten open-source LLMs (general-purpose, biomedical, and reasoning models) across two datasets, eleven medical specialties, and six question types. We compare standard single-generation and sampling-based methods, and present a case study exploring simple, single-pass estimators based on behavioral signals in reasoning traces. These lightweight methods approach the performance of Semantic Entropy while requiring only one generation. Our results reveal substantial variation across specialties and question types, underscoring the importance of selecting models based on both the nature of the question and model-specific strengths.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06971v2">Chain-of-Code Collapse: Reasoning Failures in LLMs via Adversarial Prompting in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in tasks requiring complex reasoning, such as code generation, mathematical problem solving, and algorithmic synthesis -- especially when aided by reasoning tokens and Chain-of-Thought prompting. Yet, a core question remains: do these models truly reason, or do they merely exploit shallow statistical patterns? In this paper, we introduce Chain-of-Code Collapse, where we systematically investigate the robustness of reasoning LLMs by introducing a suite of semantically faithful yet adversarially structured prompt perturbations. Our evaluation -- spanning 700 perturbed code generations derived from LeetCode-style problems -- applies transformations such as storytelling reframing, irrelevant constraint injection, example reordering, and numeric perturbation. We observe that while certain modifications severely degrade performance (with accuracy drops up to -42.1%), others surprisingly improve model accuracy by up to 35.3%, suggesting sensitivity not only to semantics but also to surface-level prompt dynamics. These findings expose the fragility and unpredictability of current reasoning systems, underscoring the need for more principles approaches to reasoning alignments and prompting robustness. We release our perturbation datasets and evaluation framework to promote further research in trustworthy and resilient LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10764v1">OPT-BENCH: Evaluating LLM Agent on Large-Scale Search Spaces Optimization Problems</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in solving diverse tasks. However, their proficiency in iteratively optimizing complex solutions through learning from previous feedback remains insufficiently explored. To bridge this gap, we present OPT-BENCH, a comprehensive benchmark designed to evaluate LLM agents on large-scale search space optimization problems. OPT-BENCH includes 20 real-world machine learning tasks sourced from Kaggle and 10 classical NP problems, offering a diverse and challenging environment for assessing LLM agents on iterative reasoning and solution refinement. To enable rigorous evaluation, we introduce OPT-Agent, an end-to-end optimization framework that emulates human reasoning when tackling complex problems by generating, validating, and iteratively improving solutions through leveraging historical feedback. Through extensive experiments on 9 state-of-the-art LLMs from 6 model families, we analyze the effects of optimization iterations, temperature settings, and model architectures on solution quality and convergence. Our results demonstrate that incorporating historical context significantly enhances optimization performance across both ML and NP tasks. All datasets, code, and evaluation tools are open-sourced to promote further research in advancing LLM-driven optimization and iterative reasoning. Project page: \href{https://github.com/OliverLeeXZ/OPT-BENCH}{https://github.com/OliverLeeXZ/OPT-BENCH}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10737v1">TaxoAdapt: Aligning LLM-Based Multidimensional Taxonomy Construction to Evolving Research Corpora</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 Accepted to ACL 2025 Main Conference. Code available at: https://github.com/pkargupta/taxoadapt
    </div>
    <details class="paper-abstract">
      The rapid evolution of scientific fields introduces challenges in organizing and retrieving scientific literature. While expert-curated taxonomies have traditionally addressed this need, the process is time-consuming and expensive. Furthermore, recent automatic taxonomy construction methods either (1) over-rely on a specific corpus, sacrificing generalizability, or (2) depend heavily on the general knowledge of large language models (LLMs) contained within their pre-training datasets, often overlooking the dynamic nature of evolving scientific domains. Additionally, these approaches fail to account for the multi-faceted nature of scientific literature, where a single research paper may contribute to multiple dimensions (e.g., methodology, new tasks, evaluation metrics, benchmarks). To address these gaps, we propose TaxoAdapt, a framework that dynamically adapts an LLM-generated taxonomy to a given corpus across multiple dimensions. TaxoAdapt performs iterative hierarchical classification, expanding both the taxonomy width and depth based on corpus' topical distribution. We demonstrate its state-of-the-art performance across a diverse set of computer science conferences over the years to showcase its ability to structure and capture the evolution of scientific fields. As a multidimensional method, TaxoAdapt generates taxonomies that are 26.51% more granularity-preserving and 50.41% more coherent than the most competitive baselines judged by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13572v3">VeriContaminated: Assessing LLM-Driven Verilog Coding for Data Contamination</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized code generation, achieving exceptional results on various established benchmarking frameworks. However, concerns about data contamination - where benchmark data inadvertently leaks into pre-training or fine-tuning datasets - raise questions about the validity of these evaluations. While this issue is known, limiting the industrial adoption of LLM-driven software engineering, hardware coding has received little to no attention regarding these risks. For the first time, we analyze state-of-the-art (SOTA) evaluation frameworks for Verilog code generation (VerilogEval and RTLLM), using established methods for contamination detection (CCD and Min-K% Prob). We cover SOTA commercial and open-source LLMs (CodeGen2.5, Minitron 4b, Mistral 7b, phi-4 mini, LLaMA-{1,2,3.1}, GPT-{2,3.5,4o}, Deepseek-Coder, and CodeQwen 1.5), in baseline and fine-tuned models (RTLCoder and Verigen). Our study confirms that data contamination is a critical concern. We explore mitigations and the resulting trade-offs for code quality vs fairness (i.e., reducing contamination toward unbiased benchmarking).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10641v1">Spelling-out is not Straightforward: LLMs' Capability of Tokenization from Token to Characters</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can spell out tokens character by character with high accuracy, yet they struggle with more complex character-level tasks, such as identifying compositional subcomponents within tokens. In this work, we investigate how LLMs internally represent and utilize character-level information during the spelling-out process. Our analysis reveals that, although spelling out is a simple task for humans, it is not handled in a straightforward manner by LLMs. Specifically, we show that the embedding layer does not fully encode character-level information, particularly beyond the first character. As a result, LLMs rely on intermediate and higher Transformer layers to reconstruct character-level knowledge, where we observe a distinct "breakthrough" in their spelling behavior. We validate this mechanism through three complementary analyses: probing classifiers, identification of knowledge neurons, and inspection of attention weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.09092v1">CUDA-LLM: LLMs Can Write Efficient CUDA Kernels</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities in general-purpose code generation. However, generating the code which is deeply hardware-specific, architecture-aware, and performance-critical, especially for massively parallel GPUs, remains a complex challenge. In this work, we explore the use of LLMs for the automated generation and optimization of CUDA programs, with the goal of producing high-performance GPU kernels that fully exploit the underlying hardware. To address this challenge, we propose a novel framework called \textbf{Feature Search and Reinforcement (FSR)}. FSR jointly optimizes compilation and functional correctness, as well as the runtime performance, which are validated through extensive and diverse test cases, and measured by actual kernel execution latency on the target GPU, respectively. This approach enables LLMs not only to generate syntactically and semantically correct CUDA code but also to iteratively refine it for efficiency, tailored to the characteristics of the GPU architecture. We evaluate FSR on representative CUDA kernels, covering AI workloads and computational intensive algorithms. Our results show that LLMs augmented with FSR consistently guarantee correctness rates. Meanwhile, the automatically generated kernels can outperform general human-written code by a factor of up to 179$\times$ in execution speeds. These findings highlight the potential of combining LLMs with performance reinforcement to automate GPU programming for hardware-specific, architecture-sensitive, and performance-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2506.09443v1">LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable intelligence across various tasks, which has inspired the development and widespread adoption of LLM-as-a-Judge systems for automated model testing, such as red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising concerns about their robustness and, consequently, their trustworthiness. Existing evaluation methods adopted by LLM-based judges are often piecemeal and lack a unified framework for comprehensive assessment. Furthermore, prompt template and model selections for improving judge robustness have been rarely explored, and their performance in real-world settings remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. RobustJudge investigates the impact of attack methods and defense strategies (RQ1), explores the influence of prompt template and model selection (RQ2), and assesses the robustness of real-world LLM-as-a-Judge applications (RQ3).Our main findings are: (1) LLM-as-a-Judge systems are still vulnerable to a range of adversarial attacks, including Combined Attack and PAIR, while defense mechanisms such as Re-tokenization and LLM-based Detectors offer improved protection; (2) Robustness is highly sensitive to the choice of prompt template and judge models. Our proposed prompt template optimization method can improve robustness, and JudgeLM-13B demonstrates strong performance as a robust open-source judge; (3) Applying RobustJudge to Alibaba's PAI platform reveals previously unreported vulnerabilities. The source code of RobustJudge is provided at https://github.com/S3IC-Lab/RobustJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.15724v2">LLM-Cure: LLM-based Competitor User Review Analysis for Feature Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-06-12
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      The exponential growth of the mobile app market underscores the importance of constant innovation and rapid response to user demands. As user satisfaction is paramount to the success of a mobile application (app), developers typically rely on user reviews, which represent user feedback that includes ratings and comments to identify areas for improvement. However, the sheer volume of user reviews poses challenges in manual analysis, necessitating automated approaches. Existing automated approaches either analyze only the target apps reviews, neglecting the comparison of similar features to competitors or fail to provide suggestions for feature enhancement. To address these gaps, we propose a Large Language Model (LLM)-based Competitive User Review Analysis for Feature Enhancement) (LLM-Cure), an approach powered by LLMs to automatically generate suggestion s for mobile app feature improvements. More specifically, LLM-Cure identifies and categorizes features within reviews by applying LLMs. When provided with a complaint in a user review, LLM-Cure curates highly rated (4 and 5 stars) reviews in competing apps related to the complaint and proposes potential improvements tailored to the target application. We evaluate LLM-Cure on 1,056,739 reviews of 70 popular Android apps. Our evaluation demonstrates that LLM-Cure significantly outperforms the state-of-the-art approaches in assigning features to reviews by up to 13% in F1-score, up to 16% in recall and up to 11% in precision. Additionally, LLM-Cure demonstrates its capability to provide suggestions for resolving user complaints. We verify the suggestions using the release notes that reflect the changes of features in the target mobile app. LLM-Cure achieves a promising average of 73% of the implementation of the provided suggestions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10161v1">Can LLMs Generate Good Stories? Insights and Challenges from a Narrative Planning Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 In 2025 IEEE Conference on Games (CoG)
    </div>
    <details class="paper-abstract">
      Story generation has been a prominent application of Large Language Models (LLMs). However, understanding LLMs' ability to produce high-quality stories remains limited due to challenges in automatic evaluation methods and the high cost and subjectivity of manual evaluation. Computational narratology offers valuable insights into what constitutes a good story, which has been applied in the symbolic narrative planning approach to story generation. This work aims to deepen the understanding of LLMs' story generation capabilities by using them to solve narrative planning problems. We present a benchmark for evaluating LLMs on narrative planning based on literature examples, focusing on causal soundness, character intentionality, and dramatic conflict. Our experiments show that GPT-4 tier LLMs can generate causally sound stories at small scales, but planning with character intentionality and dramatic conflict remains challenging, requiring LLMs trained with reinforcement learning for complex reasoning. The results offer insights on the scale of stories that LLMs can generate while maintaining quality from different aspects. Our findings also highlight interesting problem solving behaviors and shed lights on challenges and considerations for applying LLM narrative planning in game environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13305v2">Computation Mechanism Behind LLM Position Generalization</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 ACL 2025 Main Long Paper
    </div>
    <details class="paper-abstract">
      Most written natural languages are composed of sequences of words and sentences. Similar to humans, large language models (LLMs) exhibit flexibility in handling textual positions - a phenomenon we term position generalization. They can understand texts with position perturbations and generalize to longer texts than those encountered during training with the latest techniques. These phenomena suggest that LLMs handle positions tolerantly, but how LLMs computationally process positional relevance remains largely unexplored. This work connects the linguistic phenomenon with LLMs' computational mechanisms. We show how LLMs enforce certain computational mechanisms for the aforementioned tolerance in position perturbations. Despite the complex design of the self-attention mechanism, this work reveals that LLMs learn a counterintuitive disentanglement of attention logits. Their values show a 0.959 linear correlation with an approximation of the arithmetic sum of positional relevance and semantic importance. Furthermore, we identify a prevalent pattern in intermediate features, which we prove theoretically enables this effect. The pattern, which is different from how randomly initialized parameters would behave, suggests that it is a learned behavior rather than a natural result of the model architecture. Based on these findings, we provide computational explanations and criteria for LLMs' position flexibilities. This work takes a pioneering step in linking position generalization with modern LLMs' internal mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10125v1">D-LiFT: Improving LLM-based Decompiler Backend via Code Quality-driven Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Decompilers, which reconstruct human-readable source code from binary executables, are vital to many security tasks. Yet, despite recent advances, their output often suffers from syntactic and semantic errors and remains difficult to read. Recently, with the advent of large language models (LLMs), researchers began to explore the potential of LLMs to refine decompiler output. Nevertheless, our study of these approaches reveals significant limitations, such as introducing new errors and relying on unreliable accuracy validation. In this paper, we present D-LiFT, an automated decompiler backend that harnesses and further trains LLMs to improve the quality of decompiled code via reinforcement learning (RL). Unlike prior work that overlooks preserving accuracy, D-LiFT adheres to a key principle for enhancing the quality of decompiled code: \textit{preserving accuracy while improving readability}. Central to D-LiFT, we propose D-SCORE, an integrated quality assessment system to score the decompiled code from multiple aspects. In line with our principle, D-SCORE assigns low scores to any inaccurate output and only awards higher scores for readability to code that passes the accuracy check. Specifically, D-SCORE first verifies the syntactic and semantic correctness via the compiler and symbolic execution; only if a candidate is deemed accurate, it then evaluates readability using established metrics to compare the LLM output with the original decompiled code. The score will then be fed back to the LLM for fine-tuning. Our implementation, based on Ghidra and a range of LLMs, demonstrates significant improvements for the accurate decompiled code from the coreutils and util-linux projects. Compared to baseline LLMs without D-SCORE-driven fine-tuning, D-LiFT produces 55.3% more improved decompiled functions, as measured by D-SCORE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10106v1">One For All: LLM-based Heterogeneous Mission Planning in Precision Agriculture</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted to International Federation of Automatic Control (IFAC) Sensing, Control and Automation Technologies for Agriculture - 8th AGRICONTROL 2025
    </div>
    <details class="paper-abstract">
      Artificial intelligence is transforming precision agriculture, offering farmers new tools to streamline their daily operations. While these technological advances promise increased efficiency, they often introduce additional complexity and steep learning curves that are particularly challenging for non-technical users who must balance tech adoption with existing workloads. In this paper, we present a natural language (NL) robotic mission planner that enables non-specialists to control heterogeneous robots through a common interface. By leveraging large language models (LLMs) and predefined primitives, our architecture seamlessly translates human language into intermediate descriptions that can be executed by different robotic platforms. With this system, users can formulate complex agricultural missions without writing any code. In the work presented in this paper, we extend our previous system tailored for wheeled robot mission planning through a new class of experiments involving robotic manipulation and computer vision tasks. Our results demonstrate that the architecture is both general enough to support a diverse set of robots and powerful enough to execute complex mission requests. This work represents a significant step toward making robotic automation in precision agriculture more accessible to non-technical users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20834v4">Token-Efficient RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Title updated to "Token-Efficient RL for LLM Reasoning" to better reflect algorithmic focus. Revised abstract, intro, and conclusion. Paper shortened and typos fixed
    </div>
    <details class="paper-abstract">
      We propose reinforcement learning (RL) strategies tailored for reasoning in large language models (LLMs) under strict memory and compute limits, with a particular focus on compatibility with LoRA fine-tuning. Building on early policy gradient methods with baseline subtraction, we design critic-free methods that operate on a small, informative subset of output tokens to reduce memory usage and stabilize training. We introduce S-GRPO, a stochastic variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching approach for fine-grained credit assignment. Applied to Qwen2-1.5B, our methods raise accuracy on the SVAMP benchmark from 46% to over 70% and show strong performance on multi-digit multiplication. Surprisingly, full-token GRPO under LoRA fails to improve over the base model, suggesting that selective token-level optimization may act as an implicit regularizer in low-parameter training regimes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10095v1">When Meaning Stays the Same, but Models Drift: Evaluating Quality of Service under Token-Level Behavioral Instability in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 This paper was developed for presentation at ICML 2025 Tokshop Workshop, but is now submitted as a standalone contribution
    </div>
    <details class="paper-abstract">
      We investigate how large language models respond to prompts that differ only in their token-level realization but preserve the same semantic intent, a phenomenon we call prompt variance. We propose Prompt-Based Semantic Shift (PBSS), a diagnostic framework for measuring behavioral drift in LLMs under semantically equivalent prompt rewordings. Applied to ten constrained tasks, PBSS reveals consistent, model-specific response shifts, suggesting statistical regularities linked to tokenization and decoding. These results highlight an overlooked dimension of model evaluation stability under rephrasing and suggest that tokenization strategies and decoding dynamics may contribute to post-training quality of service instability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10093v1">Leveraging LLMs for Mission Planning in Precision Agriculture</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Published in Proceedings of 2025 International Conference on Robotics and Automation (ICRA)
    </div>
    <details class="paper-abstract">
      Robotics and artificial intelligence hold significant potential for advancing precision agriculture. While robotic systems have been successfully deployed for various tasks, adapting them to perform diverse missions remains challenging, particularly because end users often lack technical expertise. In this paper, we present an end-to-end system that leverages large language models (LLMs), specifically ChatGPT, to enable users to assign complex data collection tasks to autonomous robots using natural language instructions. To enhance reusability, mission plans are encoded using an existing IEEE task specification standard, and are executed on robots via ROS2 nodes that bridge high-level mission descriptions with existing ROS libraries. Through extensive experiments, we highlight the strengths and limitations of LLMs in this context, particularly regarding spatial reasoning and solving complex routing challenges, and show how our proposed implementation overcomes them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10060v1">Textual Bayes: Quantifying Uncertainty in LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) are becoming increasingly capable of solving challenging real-world tasks, accurately quantifying their uncertainty remains a critical open problem, which limits their applicability in high-stakes domains. This challenge is further compounded by the closed-source, black-box nature of many state-of-the-art LLMs. Moreover, LLM-based systems can be highly sensitive to the prompts that bind them together, which often require significant manual tuning (i.e., prompt engineering). In this work, we address these challenges by viewing LLM-based systems through a Bayesian lens. We interpret prompts as textual parameters in a statistical model, allowing us to use a small training dataset to perform Bayesian inference over these prompts. This novel perspective enables principled uncertainty quantification over both the model's textual parameters and its downstream predictions, while also incorporating prior beliefs about these parameters expressed in free-form text. To perform Bayesian inference, a difficult problem even for well-studied data modalities, we introduce Metropolis-Hastings through LLM Proposals (MHLP), a novel Markov chain Monte Carlo (MCMC) algorithm that combines prompt optimization techniques with standard MCMC methods. MHLP is a turnkey modification to existing LLM pipelines, including those that rely exclusively on closed-source models. Empirically, we demonstrate that our method yields improvements in both predictive accuracy and uncertainty quantification (UQ) on a range of LLM benchmarks and UQ tasks. More broadly, our work demonstrates a viable path for incorporating methods from the rich Bayesian literature into the era of LLMs, paving the way for more reliable and calibrated LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10054v1">Omni-DPO: A Dual-Perspective Paradigm for Dynamic Preference Learning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) has become a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based approaches typically treat all preference pairs uniformly, ignoring critical variations in their inherent quality and learning utility, leading to suboptimal data utilization and performance. To address this challenge, we propose Omni-DPO, a dual-perspective optimization framework that jointly accounts for (1) the inherent quality of each preference pair and (2) the model's evolving performance on those pairs. By adaptively weighting samples according to both data quality and the model's learning dynamics during training, Omni-DPO enables more effective training data utilization and achieves better performance. Experimental results on various models and benchmarks demonstrate the superiority and generalization capabilities of Omni-DPO. On textual understanding tasks, Gemma-2-9b-it finetuned with Omni-DPO beats the leading LLM, Claude 3 Opus, by a significant margin of 6.7 points on the Arena-Hard benchmark. On mathematical reasoning tasks, Omni-DPO consistently outperforms the baseline methods across all benchmarks, providing strong empirical evidence for the effectiveness and robustness of our approach. Code and models will be available at https://github.com/pspdada/Omni-DPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11150v1">ADAgent: LLM Agent for Alzheimer's Disease Analysis with Collaborative Coordinator</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Alzheimer's disease (AD) is a progressive and irreversible neurodegenerative disease. Early and precise diagnosis of AD is crucial for timely intervention and treatment planning to alleviate the progressive neurodegeneration. However, most existing methods rely on single-modality data, which contrasts with the multifaceted approach used by medical experts. While some deep learning approaches process multi-modal data, they are limited to specific tasks with a small set of input modalities and cannot handle arbitrary combinations. This highlights the need for a system that can address diverse AD-related tasks, process multi-modal or missing input, and integrate multiple advanced methods for improved performance. In this paper, we propose ADAgent, the first specialized AI agent for AD analysis, built on a large language model (LLM) to address user queries and support decision-making. ADAgent integrates a reasoning engine, specialized medical tools, and a collaborative outcome coordinator to facilitate multi-modal diagnosis and prognosis tasks in AD. Extensive experiments demonstrate that ADAgent outperforms SOTA methods, achieving significant improvements in accuracy, including a 2.7% increase in multi-modal diagnosis, a 0.7% improvement in multi-modal prognosis, and enhancements in MRI and PET diagnosis tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11148v1">LLM-to-Phy3D: Physically Conform Online 3D Object Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      The emergence of generative artificial intelligence (GenAI) and large language models (LLMs) has revolutionized the landscape of digital content creation in different modalities. However, its potential use in Physical AI for engineering design, where the production of physically viable artifacts is paramount, remains vastly underexplored. The absence of physical knowledge in existing LLM-to-3D models often results in outputs detached from real-world physical constraints. To address this gap, we introduce LLM-to-Phy3D, a physically conform online 3D object generation that enables existing LLM-to-3D models to produce physically conforming 3D objects on the fly. LLM-to-Phy3D introduces a novel online black-box refinement loop that empowers large language models (LLMs) through synergistic visual and physics-based evaluations. By delivering directional feedback in an iterative refinement process, LLM-to-Phy3D actively drives the discovery of prompts that yield 3D artifacts with enhanced physical performance and greater geometric novelty relative to reference objects, marking a substantial contribution to AI-driven generative design. Systematic evaluations of LLM-to-Phy3D, supported by ablation studies in vehicle design optimization, reveal various LLM improvements gained by 4.5% to 106.7% in producing physically conform target domain 3D designs over conventional LLM-to-3D models. The encouraging results suggest the potential general use of LLM-to-Phy3D in Physical AI for scientific and engineering applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09996v1">From Judgment to Interference: Early Stopping LLM Harmful Outputs via Streaming Content Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 22 pages, 7 figures, and 9 tables
    </div>
    <details class="paper-abstract">
      Though safety alignment has been applied to most large language models (LLMs), LLM service providers generally deploy a subsequent moderation as the external safety guardrail in real-world products. Existing moderators mainly practice a conventional full detection, which determines the harmfulness based on the complete LLM output, causing high service latency. Recent works pay more attention to partial detection where moderators oversee the generation midway and early stop the output if harmfulness is detected, but they directly apply moderators trained with the full detection paradigm to incomplete outputs, introducing a training-inference gap that lowers the performance. In this paper, we explore how to form a data-and-model solution that natively supports partial detection. For the data, we construct FineHarm, a dataset consisting of 29K prompt-response pairs with fine-grained annotations to provide reasonable supervision for token-level training. Then, we propose the streaming content monitor, which is trained with dual supervision of response- and token-level labels and can follow the output stream of LLM to make a timely judgment of harmfulness. Experiments show that SCM gains 0.95+ in macro F1 score that is comparable to full detection, by only seeing the first 18% of tokens in responses on average. Moreover, the SCM can serve as a pseudo-harmfulness annotator for improving safety alignment and lead to a higher harmlessness score than DPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09998v1">Flipping Against All Odds: Reducing LLM Coin Flip Bias via Verbalized Rejection Sampling</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Technical Report v1 (21 pages, 14 figures)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can often accurately describe probability distributions using natural language, yet they still struggle to generate faithful samples from them. This mismatch limits their use in tasks requiring reliable stochasticity, such as Monte Carlo methods, agent-based simulations, and randomized decision-making. We investigate this gap between knowledge and sampling in the context of Bernoulli distributions. We introduce Verbalized Rejection Sampling (VRS), a natural-language adaptation of classical rejection sampling that prompts the LLM to reason about and accept or reject proposed samples. Despite relying on the same Bernoulli mechanism internally, VRS substantially reduces sampling bias across models. We provide theoretical analysis showing that, under mild assumptions, VRS improves over direct sampling, with gains attributable to both the algorithm and prompt design. More broadly, our results show how classical probabilistic tools can be verbalized and embedded into LLM workflows to improve reliability, without requiring access to model internals or heavy prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09983v1">Step-by-step Instructions and a Simple Tabular Output Format Improve the Dependency Parsing Accuracy of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 9 pages, 2 figures, accepted for SyntaxFest 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled impressive performance in various tasks. However, standard prompting often struggles to produce structurally valid and accurate outputs, especially in dependency parsing. We propose a novel step-by-step instruction strategy, where universal part-of-speech tagging precedes the prediction of syntactic heads and dependency labels, and a simplified CoNLL-U like output format, our method achieves state-of-the-art accuracy on Universal Dependencies datasets across 17 languages without hallucination or contamination. We further show that multilingual fine-tuning simultaneously improves cross-language generalization performance. Our results highlight the effectiveness of explicit reasoning steps in LLM-based parsing and offer a scalable, format-consistent alternative to bracket-based approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v6">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 13 pages, 6 figures, VLDB 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve strong performances on various benchmark datasets without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09968v1">SRLAgent: Enhancing Self-Regulated Learning Skills through Gamification and LLM Assistance</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Self-regulated learning (SRL) is crucial for college students navigating increased academic demands and independence. Insufficient SRL skills can lead to disorganized study habits, low motivation, and poor time management, undermining learners ability to thrive in challenging environments. Through a formative study involving 59 college students, we identified key challenges students face in developing SRL skills, including difficulties with goal-setting, time management, and reflective learning. To address these challenges, we introduce SRLAgent, an LLM-assisted system that fosters SRL skills through gamification and adaptive support from large language models (LLMs). Grounded in Zimmermans three-phase SRL framework, SRLAgent enables students to engage in goal-setting, strategy execution, and self-reflection within an interactive game-based environment. The system offers real-time feedback and scaffolding powered by LLMs to support students independent study efforts. We evaluated SRLAgent using a between-subjects design, comparing it to a baseline system (SRL without Agent features) and a traditional multimedia learning condition. Results showed significant improvements in SRL skills within the SRLAgent group (p < .001, Cohens d = 0.234) and higher engagement compared to the baselines. This work highlights the value of embedding SRL scaffolding and real-time AI support within gamified environments, offering design implications for educational technologies that aim to promote deeper learning and metacognitive skill development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06845v5">7B Fully Open Source Moxin-LLM/VLM -- From Pretraining to GRPO-based Reinforcement Learning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed, adhering to principles of open science, open source, open data, and open access. We release the pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. After pre-training the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO) following DeepSeek R1 to finetune our model, leading to the Moxin Reasoning model. Moreover, we develop our vision language model based on our Moxin model. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14259v4">Let's Fuse Step by Step: A Generative Fusion Decoding Algorithm with LLMs for Robust and Instruction-Aware ASR and OCR</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      We propose "Generative Fusion Decoding" (GFD), a novel shallow fusion framework designed to integrate large language models (LLMs) into cross-modal text recognition systems for automatic speech recognition (ASR) and optical character recognition (OCR). We derive the necessary formulations to enable GFD to operate across mismatched token spaces of different models by calculating likelihood at the byte level, thereby enabling seamless fusion and synchronous progression during the decoding process. GFD is plug-and-play by design, making it readily compatible with various auto-regressive models without the need for any re-training. GFD proves effective for general ASR and OCR tasks through intermediate and frequent interactions with LLMs, surpassing cascaded methods in English and Mandarin benchmarks. In addition, GFD transfers in-context learning abilities of LLMs and allows for adaptive ASR in instruction-aware and long-context settings, yielding significant WER reductions of up to 17.7\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08174v2">LLM-BT-Terms: Back-Translation as a Framework for Terminology Standardization and Dynamic Semantic Embedding</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      The rapid expansion of English technical terminology presents a significant challenge to traditional expert-based standardization, particularly in rapidly developing areas such as artificial intelligence and quantum computing. Manual approaches face difficulties in maintaining consistent multilingual terminology. To address this, we introduce LLM-BT, a back-translation framework powered by large language models (LLMs) designed to automate terminology verification and standardization through cross-lingual semantic alignment. Our key contributions include: (1) term-level consistency validation: by performing English -> intermediate language -> English back-translation, LLM-BT achieves high term consistency across different models (such as GPT-4, DeepSeek, and Grok). Case studies demonstrate over 90 percent of terms are preserved either exactly or semantically; (2) multi-path verification workflow: we develop a novel pipeline described as Retrieve -> Generate -> Verify -> Optimize, which supports both serial paths (e.g., English -> Simplified Chinese -> Traditional Chinese -> English) and parallel paths (e.g., English -> Chinese / Portuguese -> English). BLEU scores and term-level accuracy indicate strong cross-lingual robustness, with BLEU scores exceeding 0.45 and Portuguese term accuracy reaching 100 percent; (3) back-translation as semantic embedding: we reinterpret back-translation as a form of dynamic semantic embedding that uncovers latent trajectories of meaning. In contrast to static embeddings, LLM-BT offers transparent, path-based embeddings shaped by the evolution of the models. This reframing positions back-translation as an active mechanism for multilingual terminology standardization, fostering collaboration between machines and humans - machines preserve semantic integrity, while humans provide cultural interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19384v2">The Remarkable Robustness of LLMs: Stages of Inference?</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 For Github code see https://github.com/vdlad/Remarkable-Robustness-of-LLMs. Send all correspondence to the first author
    </div>
    <details class="paper-abstract">
      We investigate the robustness of Large Language Models (LLMs) to structural interventions by deleting and swapping adjacent layers during inference. Surprisingly, models retain 72-95% of their original top-1 prediction accuracy without any fine-tuning. We find that performance degradation is not uniform across layers: interventions to the early and final layers cause the most degradation, while the model is remarkably robust to dropping middle layers. This pattern of localized sensitivity motivates our hypothesis of four stages of inference, observed across diverse model families and sizes: (1) detokenization, where local context is integrated to lift raw token embeddings into higher-level representations; (2) feature engineering, where task- and entity-specific features are iteratively refined; (3) prediction ensembling, where hidden states are aggregated into plausible next-token predictions; and (4) residual sharpening, where irrelevant features are suppressed to finalize the output distribution. Synthesizing behavioral and mechanistic evidence, we provide a framework for interpreting depth-dependent computations in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05387v2">Advancing Decoding Strategies: Enhancements in Locally Typical Sampling for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 This is the accepted but pre-reviewed version of the chapter that has been accepted for publication in the Springer volume 'Decision-Making in Computational Intelligence-Based Systems,' edited by Witold Pedrycz, Gilberto Rivera, Rose Ma Rodriguez, and Salvador Ibarra Martinez. The chapter is 39 pages long, and it contains 2 figures and 6 tables. This is NOT the final camera-ready version
    </div>
    <details class="paper-abstract">
      This chapter explores advancements in decoding strategies for large language models (LLMs), focusing on enhancing the Locally Typical Sampling (LTS) algorithm. Traditional decoding methods, such as top-k and nucleus sampling, often struggle to balance fluency, diversity, and coherence in text generation. To address these challenges, Adaptive Semantic-Aware Typicality Sampling (ASTS) is proposed as an improved version of LTS, incorporating dynamic entropy thresholding, multi-objective scoring, and reward-penalty adjustments. ASTS ensures contextually coherent and diverse text generation while maintaining computational efficiency. Its performance is evaluated across multiple benchmarks, including story generation and abstractive summarization, using metrics such as perplexity, MAUVE, and diversity scores. Experimental results demonstrate that ASTS outperforms existing sampling techniques by reducing repetition, enhancing semantic alignment, and improving fluency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09886v1">Attention Head Embeddings with Trainable Deep Kernels for Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      We present a novel approach for detecting hallucinations in large language models (LLMs) by analyzing the probabilistic divergence between prompt and response hidden-state distributions. Counterintuitively, we find that hallucinated responses exhibit smaller deviations from their prompts compared to grounded responses, suggesting that hallucinations often arise from superficial rephrasing rather than substantive reasoning. Leveraging this insight, we propose a model-intrinsic detection method that uses distributional distances as principled hallucination scores, eliminating the need for external knowledge or auxiliary models. To enhance sensitivity, we employ deep learnable kernels that automatically adapt to capture nuanced geometric differences between distributions. Our approach outperforms existing baselines, demonstrating state-of-the-art performance on several benchmarks. The method remains competitive even without kernel training, offering a robust, scalable solution for hallucination detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07859v2">Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 ICML 2025 camera-ready; 15 pages, 6 figures, 5 tables
    </div>
    <details class="paper-abstract">
      The Abstraction and Reasoning Corpus (ARC-AGI) poses a significant challenge for large language models (LLMs), exposing limitations in their abstract reasoning abilities. In this work, we leverage task-specific data augmentations throughout the training, generation, and scoring phases, and employ a depth-first search algorithm to generate diverse, high-probability candidate solutions. Furthermore, we utilize the LLM not only as a generator but also as a scorer, using its output probabilities to select the most promising solutions. Our method achieves a score of 71.6% (286.5/400 solved tasks) on the public ARC-AGI evaluation set, demonstrating state-of-the-art performance among publicly available approaches. While concurrent closed-source work has reported higher scores, our method distinguishes itself through its transparency, reproducibility, and remarkably low inference cost, averaging only around 2ct per task on readily available hardware (we assume a price of 36ct/hour for a Nvidia 4090 GPU).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17962v6">Crafting Customisable Characters with LLMs: Introducing SimsChat, a Persona-Driven Role-Playing Agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable ability to comprehend instructions and generate human-like text, enabling sophisticated agent simulation beyond basic behavior replication. However, the potential for creating freely customisable characters remains underexplored. We introduce the Customisable Conversation Agent Framework, which employs LLMs to simulate real-world characters through personalised characteristic feature injection, enabling diverse character creation according to user preferences. We propose the SimsConv dataset, comprising 68 customised characters and 13,971 multi-turn role-playing dialogues across 1,360 real-world scenes. Characters are initially customised using pre-defined elements (career, aspiration, traits, skills), then expanded through personal and social profiles. Building on this, we present SimsChat, a freely customisable role-playing agent incorporating various realistic settings and topic-specified character interactions. Experimental results on both SimsConv and WikiRoleEval datasets demonstrate SimsChat's superior performance in maintaining character consistency, knowledge accuracy, and appropriate question rejection compared to existing models. Our framework provides valuable insights for developing more accurate and customisable human simulacra. Our data and code are publicly available at https://github.com/Bernard-Yang/SimsChat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.01129v4">Emphasising Structured Information: Integrating Abstract Meaning Representation into LLMs for Enhanced Open-Domain Dialogue Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Automatic open-domain dialogue evaluation has attracted increasing attention, yet remains challenging due to the complexity of assessing response appropriateness. Traditional evaluation metrics, typically trained with true positive and randomly selected negative responses, tend to assign higher scores to responses that share greater content similarity with contexts. However, adversarial negative responses, despite possessing high lexical overlap with contexts, can be semantically incongruous. Consequently, existing metrics struggle to effectively evaluate such responses, resulting in low correlations with human judgments. While recent studies have demonstrated the effectiveness of Large Language Models (LLMs) for open-domain dialogue evaluation, they still face challenges in handling adversarial negative examples. We propose a novel evaluation framework that integrates Abstract Meaning Representation (AMR) enhanced domain-specific language models (SLMs) with LLMs. Our SLMs explicitly incorporate AMR graph information through a gating mechanism for enhanced semantic representation learning, while both SLM predictions and AMR knowledge are integrated into LLM prompts for robust evaluation. Extensive experiments on open-domain dialogue evaluation tasks demonstrate the superiority of our method compared to state-of-the-art baselines. Our comprehensive ablation studies reveal that AMR graph information contributes substantially more to performance improvements. Our framework achieves strong correlations with human judgments across multiple datasets, establishing a new benchmark for dialogue evaluation. Our code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14352v2">LogProber: Disentangling confidence from contamination in LLM responses</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      In machine learning, contamination refers to situations where testing data leak into the training set. The issue is particularly relevant for the evaluation of the performance of Large Language Models (LLMs), which are generally trained on gargantuan, and generally opaque, corpora of text scraped from the world wide web. Developing tools to detect contamination is therefore crucial to be able to fairly and properly track the evolution of the performance of LLMs. To date, only a few recent studies have attempted to address the issue of quantifying and detecting contamination in short text sequences, such as those commonly found in benchmarks. However, these methods have limitations that can sometimes render them impractical.In the present paper, we introduce LogProber, a novel, efficient algorithm that we show to be able to detect contamination in a black box setting that tries to tackle some of these drawbacks by focusing on the familiarity with the question rather than the answer. Here, we explore the properties of the proposed method in comparison with concurrent approaches, identify its advantages and limitations, and illustrate how different forms of contamination can go undetected depending on the design of the detection algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09796v1">Do LLMs Give Psychometrically Plausible Responses in Educational Assessments?</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted for publication at the 20th Workshop on Innovative Use of NLP for Building Educational Applications (BEA) at ACL 2025
    </div>
    <details class="paper-abstract">
      Knowing how test takers answer items in educational assessments is essential for test development, to evaluate item quality, and to improve test validity. However, this process usually requires extensive pilot studies with human participants. If large language models (LLMs) exhibit human-like response behavior to test items, this could open up the possibility of using them as pilot participants to accelerate test development. In this paper, we evaluate the human-likeness or psychometric plausibility of responses from 18 instruction-tuned LLMs with two publicly available datasets of multiple-choice test items across three subjects: reading, U.S. history, and economics. Our methodology builds on two theoretical frameworks from psychometrics which are commonly used in educational assessment, classical test theory and item response theory. The results show that while larger models are excessively confident, their response distributions can be more human-like when calibrated with temperature scaling. In addition, we find that LLMs tend to correlate better with humans in reading comprehension items compared to other subjects. However, the correlations are not very strong overall, indicating that LLMs should not be used for piloting educational assessments in a zero-shot setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09713v1">A First Look at Bugs in LLM Inference Engines</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language model-specific inference engines (in short as \emph{LLM inference engines}) have become a fundamental component of modern AI infrastructure, enabling the deployment of LLM-powered applications (LLM apps) across cloud and local devices. Despite their critical role, LLM inference engines are prone to bugs due to the immense resource demands of LLMs and the complexities of cross-platform compatibility. However, a systematic understanding of these bugs remains lacking. To bridge this gap, we present the first empirical study on bugs in LLM inference engines. We mine official repositories of 5 widely adopted LLM inference engines, constructing a comprehensive dataset of 929 real-world bugs. Through a rigorous open coding process, we analyze these bugs to uncover their symptoms, root causes, and commonality. Our findings reveal six major bug symptoms and a taxonomy of 28 root causes, shedding light on the key challenges in bug detection and location within LLM inference engines. Based on these insights, we propose a series of actionable implications for researchers, inference engine vendors, and LLM app developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09657v1">Bridging the Gap Between Open-Source and Proprietary LLMs in Table QA</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted for publication at the 19th International Workshop on Semantic Evaluation (SemEval-2025), to be held in conjunction with ACL 2025. 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      This paper presents a system developed for SemEval 2025 Task 8: Question Answering (QA) over tabular data. Our approach integrates several key components: text-to-SQL and text-to-code generation modules, a self-correction mechanism, and a retrieval-augmented generation (RAG). Additionally, it includes an end-to-end (E2E) module, all orchestrated by a large language model (LLM). Through ablation studies, we analyzed the effects of different parts of our pipeline and identified the challenges that are still present in this field. During the evaluation phase of the competition, our solution achieved an accuracy of 80%, resulting in a top-13 ranking among the 38 participating teams. Our pipeline demonstrates a significant improvement in accuracy for open-source models and achieves a performance comparable to proprietary LLMs in QA tasks over tables. The code is available at GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09655v1">DipLLM: Fine-Tuning LLM for Strategic Decision-making in Diplomacy</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted to the 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Diplomacy is a complex multiplayer game that requires both cooperation and competition, posing significant challenges for AI systems. Traditional methods rely on equilibrium search to generate extensive game data for training, which demands substantial computational resources. Large Language Models (LLMs) offer a promising alternative, leveraging pre-trained knowledge to achieve strong performance with relatively small-scale fine-tuning. However, applying LLMs to Diplomacy remains challenging due to the exponential growth of possible action combinations and the intricate strategic interactions among players. To address this challenge, we propose DipLLM, a fine-tuned LLM-based agent that learns equilibrium policies for Diplomacy. DipLLM employs an autoregressive factorization framework to simplify the complex task of multi-unit action assignment into a sequence of unit-level decisions. By defining an equilibrium policy within this framework as the learning objective, we fine-tune the model using only 1.5% of the data required by the state-of-the-art Cicero model, surpassing its performance. Our results demonstrate the potential of fine-tuned LLMs for tackling complex strategic decision-making in multiplayer games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01940v2">AskToAct: Enhancing LLMs Tool Use via Self-Correcting Clarification</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in tool learning. In real-world scenarios, user queries are often ambiguous and incomplete, requiring effective clarification. However, existing interactive clarification approaches face two critical limitations: reliance on manually constructed datasets, which inherently constrains training data scale and diversity, and lack of error correction mechanisms during multi-turn clarification, leading to error accumulation that compromises both accuracy and efficiency. We present AskToAct, which addresses these challenges by exploiting the structural mapping between queries and their tool invocation solutions. Our key insight is that tool parameters naturally represent explicit user intents. By systematically removing key parameters from queries while retaining them as ground truth, we enable automated construction of high-quality training data. We further enhance model robustness through error-correction pairs and selective masking, enabling dynamic error detection during clarification interactions. Comprehensive experiments demonstrate that AskToAct significantly outperforms existing approaches, achieving above 57% accuracy in recovering critical unspecified intents and enhancing clarification efficiency by an average of 10.46% while maintaining high accuracy in tool invocation. Our framework exhibits robust performance across different model architectures and successfully generalizes to entirely unseen APIs without additional training, achieving performance comparable to GPT-4o with substantially fewer computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09632v1">Ties of Trust: a bowtie model to uncover trustor-trustee relationships in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted for publication at The 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT '25). This version corresponds to the camera-ready manuscript submitted to the conference proceedings
    </div>
    <details class="paper-abstract">
      The rapid and unprecedented dominance of Artificial Intelligence (AI), particularly through Large Language Models (LLMs), has raised critical trust challenges in high-stakes domains like politics. Biased LLMs' decisions and misinformation undermine democratic processes, and existing trust models fail to address the intricacies of trust in LLMs. Currently, oversimplified, one-directional approaches have largely overlooked the many relationships between trustor (user) contextual factors (e.g. ideology, perceptions) and trustee (LLMs) systemic elements (e.g. scientists, tool's features). In this work, we introduce a bowtie model for holistically conceptualizing and formulating trust in LLMs, with a core component comprehensively exploring trust by tying its two sides, namely the trustor and the trustee, as well as their intricate relationships. We uncover these relationships within the proposed bowtie model and beyond to its sociotechnical ecosystem, through a mixed-methods explanatory study, that exploits a political discourse analysis tool (integrating ChatGPT), by exploring and responding to the next critical questions: 1) How do trustor's contextual factors influence trust-related actions? 2) How do these factors influence and interact with trustee systemic elements? 3) How does trust itself vary across trustee systemic elements? Our bowtie-based explanatory analysis reveals that past experiences and familiarity significantly shape trustor's trust-related actions; not all trustor contextual factors equally influence trustee systemic elements; and trustee's human-in-the-loop features enhance trust, while lack of transparency decreases it. Finally, this solid evidence is exploited to deliver recommendations, insights and pathways towards building robust trusting ecosystems in LLM-based solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09630v1">In-Context Bias Propagation in LLM-Based Tabular Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Paper accepted at ICML 2025 workshop DIG-BUG
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for synthetic tabular data generation through in-context learning (ICL), offering a practical solution for data augmentation in data scarce scenarios. While prior work has shown the potential of LLMs to improve downstream task performance through augmenting underrepresented groups, these benefits often assume access to a subset of unbiased in-context examples, representative of the real dataset. In real-world settings, however, data is frequently noisy and demographically skewed. In this paper, we systematically study how statistical biases within in-context examples propagate to the distribution of synthetic tabular data, showing that even mild in-context biases lead to global statistical distortions. We further introduce an adversarial scenario where a malicious contributor can inject bias into the synthetic dataset via a subset of in-context examples, ultimately compromising the fairness of downstream classifiers for a targeted and protected subgroup. Our findings demonstrate a new vulnerability associated with LLM-based data generation pipelines that rely on in-context prompts with in sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09627v1">Benchmarking Debiasing Methods for LLM-based Parameter Estimates</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer an inexpensive yet powerful way to annotate text, but are often inconsistent when compared with experts. These errors can bias downstream estimates of population parameters such as regression coefficients and causal effects. To mitigate this bias, researchers have developed debiasing methods such as Design-based Supervised Learning (DSL) and Prediction-Powered Inference (PPI), which promise valid estimation by combining LLM annotations with a limited number of expensive expert annotations. Although these methods produce consistent estimates under theoretical assumptions, it is unknown how they compare in finite samples of sizes encountered in applied research. We make two contributions: First, we study how each method's performance scales with the number of expert annotations, highlighting regimes where LLM bias or limited expert labels significantly affect results. Second, we compare DSL and PPI across a range of tasks, finding that although both achieve low bias with large datasets, DSL often outperforms PPI on bias reduction and empirical efficiency, but its performance is less consistent across datasets. Our findings indicate that there is a bias-variance tradeoff at the level of debiasing methods, calling for more research on developing metrics for quantifying their efficiency in finite samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09601v1">ASTAGEN: Empirical Evaluation of Automated SATD Taxonomy Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Technical debt refers to suboptimal code that degrades software quality. When developers intentionally introduce such debt, it is called self-admitted technical debt (SATD). Since SATD hinders maintenance, identifying its categories is key to uncovering quality issues. Traditionally, constructing such taxonomies requires manually inspecting SATD comments and surrounding code, which is time-consuming, labor-intensive, and often inconsistent due to annotator subjectivity. This study presents ASTAGEN, an initial step toward automating SATD taxonomy generation using large language models (LLMs). Given a comment and its surrounding code, ASTAGEN first generates a concise explanation for each SATD comment, then incrementally generates and updates categories to construct a taxonomy. We evaluate ASTAGEN on SATD datasets from three domains: quantum software, smart contracts, and machine learning. It successfully recovers domain-specific categories reported in prior work, such as Layer Configuration in machine learning. Compared to a naive use of an LLM, ASTAGEN produces more consistent category assignments due to its explanation-driven, iterative design. It also completes taxonomy generation in under two hours and for less than one USD, even on the largest dataset. These results suggest that while full automation remains challenging, ASTAGEN is able to support semi-automated taxonomy construction. Furthermore, our work opens up avenues for future work, such as automatic taxonomy generation in other areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09581v1">Integrating Quantized LLMs into Robotics Systems as Edge AI to Leverage their Natural Language Processing Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 10 pages, 4 figures, Submitted to 3rd edition of the Workshop on Ontologies and Standards for Robotics and Automation (WOSRA) at ICRA 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have experienced great advancements in the last year resulting in an increase of these models in several fields to face natural language tasks. The integration of these models in robotics can also help to improve several aspects such as human-robot interaction, navigation, planning and decision-making. Therefore, this paper introduces llama\_ros, a tool designed to integrate quantized Large Language Models (LLMs) into robotic systems using ROS 2. Leveraging llama.cpp, a highly optimized runtime engine, llama\_ros enables the efficient execution of quantized LLMs as edge artificial intelligence (AI) in robotics systems with resource-constrained environments, addressing the challenges of computational efficiency and memory limitations. By deploying quantized LLMs, llama\_ros empowers robots to leverage the natural language understanding and generation for enhanced decision-making and interaction which can be paired with prompt engineering, knowledge graphs, ontologies or other tools to improve the capabilities of autonomous robots. Additionally, this paper provides insights into some use cases of using llama\_ros for planning and explainability in robotics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17304v2">Meaningless is better: hashing bias-inducing words in LLM prompts improves performance in logical reasoning and statistical learning</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      This paper introduces a novel method, referred to as "hashing", which involves masking potentially bias-inducing words in large language models (LLMs) with hash-like meaningless identifiers to reduce cognitive biases and reliance on external knowledge. The method was tested across three sets of experiments involving a total of 490 prompts. Statistical analysis using chi-square tests showed significant improvements in all tested scenarios, which covered LLama, ChatGPT, Copilot, Gemini and Mixtral models. In the first experiment, hashing decreased the fallacy rate in a modified version of the "Linda" problem aimed at evaluating susceptibility to cognitive biases. In the second experiment, it improved LLM results on the frequent itemset extraction task. In the third experiment, we found hashing is also effective when the Linda problem is presented in a tabular format rather than text, indicating that the technique works across various input representations. Overall, the method was shown to improve bias reduction and incorporation of external knowledge. Despite bias reduction, hallucination rates were inconsistently reduced across types of LLM models. These findings suggest that masking bias-inducing terms can improve LLM performance, although its effectiveness is model- and task-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08184v2">Unable to Forget: Proactive lnterference Reveals Working Memory Limits in LLMs Beyond Context Length</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Information retrieval in Large Language Models (LLMs) is increasingly recognized as intertwined with generation capabilities rather than mere lookup. While longer contexts are often assumed to improve retrieval, the effects of intra-context interference remain understudied. To address this, we adapt the proactive interference (PI) paradigm from cognitive science, where earlier information disrupts recall of newer updates. In humans, susceptibility to such interference is inversely linked to working memory capacity. We introduce PI-LLM, an evaluation that sequentially streams semantically related key-value updates and queries only the final values. Although these final values are clearly positioned just before the query, LLM retrieval accuracy declines log-linearly toward zero as interference accumulates; errors arise from retrieving previously overwritten values. Attempts to mitigate interference via prompt engineering (e.g., instructing models to ignore earlier input) yield limited success. These findings reveal a fundamental constraint on LLMs' ability to disentangle interference and flexibly manipulate information, suggesting a working memory bottleneck beyond mere context access. This calls for approaches that strengthen models' ability to suppress irrelevant content during retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07507v2">Traceable LLM-based validation of statements in knowledge graphs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      This article presents a method for verifying RDF triples using LLMs, with an emphasis on providing traceable arguments. Because the LLMs cannot currently reliably identify the origin of the information used to construct the response to the user prompt, our approach is to avoid using internal LLM factual knowledge altogether. Instead, verified RDF statements are compared to chunks of external documents retrieved through a web search or Wikipedia. To assess the possible application of this retrieval augmented generation (RAG) workflow on biosciences content, we evaluated 1,719 positive statements from the BioRED dataset and the same number of newly generated negative statements. The resulting precision is 88 %, and recall is 44 %. This indicates that the method requires human oversight. We also evaluated the method on the SNLI dataset, which allowed us to compare our approach with models specifically tuned for the natural language inference task. We demonstrate the method on Wikidata, where a SPARQL query is used to automatically retrieve statements needing verification. Overall, the results suggest that LLMs could be used for large-scale verification of statements in KGs, a task previously unfeasible due to human annotation costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09554v1">Understanding the Performance and Power of LLM Inferencing on Edge Accelerators</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional benefits to a wide range of domains, for tasks as diverse as code generation and robot navigation. While LLMs are usually served from cloud data centers, mission-critical and privacy-sensitive applications may require local hosting of open LLM models. Given the large GPU memory footprint needed for LLMs, edge accelerators such as Nvidia Jetson Orin AGX with 64GB of shared GPU-CPU RAM are a compelling choice. However, the feasibility and performance of LLM inference on edge accelerators is under-explored. This study presents a detailed evaluation of LLM inference on the NVIDIA Jetson Orin AGX, on four SOTA models ranging from 2.7B to 32.8B parameters, such as Meta Llama3.1, Microsoft-Phi2, Deepseek-R1-Qwen.We investigate the impact of varying batch sizes, sequence lengths, and quantization levels on latency, throughput, and perplexity, and also explore various custom power modes on the Orin AGX to perform power and energy consumption analysis. Our findings offer interesting insights on the trade-offs between efficiency, inference speed and resource use, e.g., increasing the sequence length causes a decrease in token throughput and quantization causes smaller LLMs to be slower. These results can help optimize LLM serving on edge accelerators for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07736v2">RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08837v2">Design Patterns for Securing LLM Agents against Prompt Injections</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      As AI agents powered by Large Language Models (LLMs) become increasingly versatile and capable of addressing a broad spectrum of tasks, ensuring their security has become a critical challenge. Among the most pressing threats are prompt injection attacks, which exploit the agent's resilience on natural language inputs -- an especially dangerous threat when agents are granted tool access or handle sensitive information. In this work, we propose a set of principled design patterns for building AI agents with provable resistance to prompt injection. We systematically analyze these patterns, discuss their trade-offs in terms of utility and security, and illustrate their real-world applicability through a series of case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08768v2">AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable progress in reasoning abilities and general natural language processing (NLP) tasks, yet their performance on Arabic data, characterized by rich morphology, diverse dialects, and complex script, remains underexplored. This paper presents a comprehensive benchmarking study of multiple reasoning-focused LLMs, with a special emphasis on the newly introduced DeepSeek models, across a suite of fifteen Arabic NLP tasks. We experiment with various strategies, including zero-shot, few-shot, and fine-tuning. This allows us to systematically evaluate performance on datasets covering a range of applications to examine their capacity for linguistic reasoning under different levels of complexity. Our experiments reveal several key findings. First, carefully selecting just three in-context examples delivers an average uplift of over 13 F1 points on classification tasks-boosting sentiment analysis from 35.3% to 87.5% and paraphrase detection from 56.1% to 87.0%. Second, reasoning-focused DeepSeek architectures outperform a strong GPT o4-mini baseline by an average of 12 F1 points on complex inference tasks in the zero-shot setting. Third, LoRA-based fine-tuning yields up to an additional 8 points in F1 and BLEU compared to equivalent increases in model scale. The code is available at https://anonymous.4open.science/r/AraReasoner41299
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09516v1">LLM-Powered CPI Prediction Inference with Online Text Time Series</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 73 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Forecasting the Consumer Price Index (CPI) is an important yet challenging task in economics, where most existing approaches rely on low-frequency, survey-based data. With the recent advances of large language models (LLMs), there is growing potential to leverage high-frequency online text data for improved CPI prediction, an area still largely unexplored. This paper proposes LLM-CPI, an LLM-based approach for CPI prediction inference incorporating online text time series. We collect a large set of high-frequency online texts from a popularly used Chinese social network site and employ LLMs such as ChatGPT and the trained BERT models to construct continuous inflation labels for posts that are related to inflation. Online text embeddings are extracted via LDA and BERT. We develop a joint time series framework that combines monthly CPI data with LLM-generated daily CPI surrogates. The monthly model employs an ARX structure combining observed CPI data with text embeddings and macroeconomic variables, while the daily model uses a VARX structure built on LLM-generated CPI surrogates and text embeddings. We establish the asymptotic properties of the method and provide two forms of constructed prediction intervals. The finite-sample performance and practical advantages of LLM-CPI are demonstrated through both simulation and real data examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09495v1">Bridging Online Behavior and Clinical Insight: A Longitudinal LLM-based Study of Suicidality on YouTube Reveals Novel Digital Markers</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Suicide remains a leading cause of death in Western countries, underscoring the need for new research approaches. As social media becomes central to daily life, digital footprints offer valuable insight into suicidal behavior. Focusing on individuals who attempted suicide while uploading videos to their channels, we investigate: How do suicidal behaviors manifest on YouTube, and how do they differ from expert knowledge? We applied complementary approaches: computational bottom-up, hybrid, and expert-driven top-down, on a novel longitudinal dataset of 181 YouTube channels from individuals with life-threatening attempts, alongside 134 control channels. In the bottom-up approach, we applied LLM-based topic modeling to identify behavioral indicators. Of 166 topics, five were associated with suicide-attempt, with two also showing temporal attempt-related changes ($p<.01$) - Mental Health Struggles ($+0.08$)* and YouTube Engagement ($+0.1$)*. In the hybrid approach, a clinical expert reviewed LLM-derived topics and flagged 19 as suicide-related. However, none showed significant attempt-related temporal effects beyond those identified bottom-up. Notably, YouTube Engagement, a platform-specific indicator, was not flagged by the expert, underscoring the value of bottom-up discovery. In the top-down approach, psychological assessment of suicide attempt narratives revealed that the only significant difference between individuals who attempted before and those attempted during their upload period was the motivation to share this experience: the former aimed to Help Others ($\beta=-1.69$, $p<.01$), while the latter framed it as part of their Personal Recovery ($\beta=1.08$, $p<.01$). By integrating these approaches, we offer a nuanced understanding of suicidality, bridging digital behavior and clinical insights. * Within-group changes in relation to the suicide attempt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20354v3">Rethinking Text-based Protein Understanding: Retrieval or LLM?</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, we identify significant data leakage issues present in current benchmarks. Moreover, conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain. To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a retrieval-enhanced method, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios. Our code and data can be seen at https://github.com/IDEA-XL/RAPM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08952v2">Can LLMs Ground when they (Don't) Know: A Study on Direct and Loaded Political Questions</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Preprint accepted at ACL Main Conference 2025
    </div>
    <details class="paper-abstract">
      Communication among humans relies on conversational grounding, allowing interlocutors to reach mutual understanding even when they do not have perfect knowledge and must resolve discrepancies in each other's beliefs. This paper investigates how large language models (LLMs) manage common ground in cases where they (don't) possess knowledge, focusing on facts in the political domain where the risk of misinformation and grounding failure is high. We examine the ability of LLMs to answer direct knowledge questions and loaded questions that presuppose misinformation. We evaluate whether loaded questions lead LLMs to engage in active grounding and correct false user beliefs, in connection to their level of knowledge and their political bias. Our findings highlight significant challenges in LLMs' ability to engage in grounding and reject false user beliefs, raising concerns about their role in mitigating misinformation in political discourse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09452v1">Learning Obfuscations Of LLM Embedding Sequences: Stained Glass Transform</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Submitted to IEEE S&P 2026
    </div>
    <details class="paper-abstract">
      The high cost of ownership of AI compute infrastructure and challenges of robust serving of large language models (LLMs) has led to a surge in managed Model-as-a-service deployments. Even when enterprises choose on-premises deployments, the compute infrastructure is typically shared across many teams in order to maximize the return on investment. In both scenarios the deployed models operate only on plaintext data, and so enterprise data owners must allow their data to appear in plaintext on a shared or multi-tenant compute infrastructure. This results in data owners with private or sensitive data being hesitant or restricted in what data they use with these types of deployments. In this work we introduce the Stained Glass Transform, a learned, stochastic, and sequence dependent transformation of the word embeddings of an LLM which information theoretically provides privacy to the input of the LLM while preserving the utility of model. We theoretically connect a particular class of Stained Glass Transforms to the theory of mutual information of Gaussian Mixture Models. We then calculate a-postiori privacy estimates, based on mutual information, and verify the privacy and utility of instances of transformed embeddings through token level metrics of privacy and standard LLM performance benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09450v1">UniToMBench: Integrating Perspective-Taking to Improve Theory of Mind in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted at Conference of the North American Chapter of the Association for Computational Linguistics, Student Research Workshop 2025 (NAACL SRW 2025)
    </div>
    <details class="paper-abstract">
      Theory of Mind (ToM), the ability to understand the mental states of oneself and others, remains a challenging area for large language models (LLMs), which often fail to predict human mental states accurately. In this paper, we introduce UniToMBench, a unified benchmark that integrates the strengths of SimToM and TOMBENCH to systematically improve and assess ToM capabilities in LLMs by integrating multi-interaction task designs and evolving story scenarios. Supported by a custom dataset of over 1,000 hand-written scenarios, UniToMBench combines perspective-taking techniques with diverse evaluation metrics to better stimulate social cognition in LLMs. Through evaluation, we observe that while models like GPT-4o and GPT-4o Mini show consistently high accuracy in tasks involving emotional and belief-related scenarios, with results usually above 80%, there is significant variability in their performance across knowledge-based tasks. These results highlight both the strengths and limitations of current LLMs in ToM-related tasks, underscoring the value of UniToMBench as a comprehensive tool for future development. Our code is publicly available here: https://github.com/Shamant/unifiedtombenchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09443v1">LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable intelligence across various tasks, which has inspired the development and widespread adoption of LLM-as-a-Judge systems for automated model testing, such as red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising concerns about their robustness and, consequently, their trustworthiness. Existing evaluation methods adopted by LLM-based judges are often piecemeal and lack a unified framework for comprehensive assessment. Furthermore, prompt template and model selections for improving judge robustness have been rarely explored, and their performance in real-world settings remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. RobustJudge investigates the impact of attack methods and defense strategies (RQ1), explores the influence of prompt template and model selection (RQ2), and assesses the robustness of real-world LLM-as-a-Judge applications (RQ3).Our main findings are: (1) LLM-as-a-Judge systems are still vulnerable to a range of adversarial attacks, including Combined Attack and PAIR, while defense mechanisms such as Re-tokenization and LLM-based Detectors offer improved protection; (2) Robustness is highly sensitive to the choice of prompt template and judge models. Our proposed prompt template optimization method can improve robustness, and JudgeLM-13B demonstrates strong performance as a robust open-source judge; (3) Applying RobustJudge to Alibaba's PAI platform reveals previously unreported vulnerabilities. The source code of RobustJudge is provided at https://github.com/S3IC-Lab/RobustJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09433v1">Mitigating Spurious Correlations in LLMs via Causality-Aware Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated remarkable capabilities in language modeling, recent studies reveal that they often fail on out-of-distribution (OOD) samples due to spurious correlations acquired during pre-training. Here, we aim to mitigate such spurious correlations through causality-aware post-training (CAPT). By decomposing a biased prediction into two unbiased steps, known as \textit{event estimation} and \textit{event intervention}, we reduce LLMs' pre-training biases without incurring additional fine-tuning biases, thus enhancing the model's generalization ability. Experiments on the formal causal inference benchmark CLadder and the logical reasoning dataset PrOntoQA show that 3B-scale language models fine-tuned with CAPT can outperform both traditional SFT and larger LLMs on in-distribution (ID) and OOD tasks using only 100 ID fine-tuning samples, demonstrating the effectiveness and sample efficiency of CAPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09424v1">Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted to ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Detecting deception in an increasingly digital world is both a critical and challenging task. In this study, we present a comprehensive evaluation of the automated deception detection capabilities of Large Language Models (LLMs) and Large Multimodal Models (LMMs) across diverse domains. We assess the performance of both open-source and commercial LLMs on three distinct datasets: real life trial interviews (RLTD), instructed deception in interpersonal scenarios (MU3D), and deceptive reviews (OpSpam). We systematically analyze the effectiveness of different experimental setups for deception detection, including zero-shot and few-shot approaches with random or similarity-based in-context example selection. Our results show that fine-tuned LLMs achieve state-of-the-art performance on textual deception detection tasks, while LMMs struggle to fully leverage cross-modal cues. Additionally, we analyze the impact of auxiliary features, such as non-verbal gestures and video summaries, and examine the effectiveness of different prompting strategies, including direct label generation and chain-of-thought reasoning. Our findings provide key insights into how LLMs process and interpret deceptive cues across modalities, highlighting their potential and limitations in real-world deception detection applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09720v2">NestQuant: Nested Lattice Quantization for Matrix Products and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Post-training quantization (PTQ) has emerged as a critical technique for efficient deployment of large language models (LLMs). This work proposes NestQuant, a novel PTQ scheme for weights and activations that is based on self-similar nested lattices. Recent works have mathematically shown such quantizers to be information-theoretically optimal for low-precision matrix multiplication. We implement a practical low-complexity version of NestQuant based on Gosset lattice, making it a drop-in quantizer for any matrix multiplication step (e.g., in self-attention, MLP etc). For example, NestQuant quantizes weights, KV-cache, and activations of Llama-3-8B to 4 bits, achieving perplexity of 6.6 on wikitext2. This represents more than 55% reduction in perplexity gap with respect to unquantized model (perplexity of 6.14) compared to state-of-the-art Metas SpinQuant (perplexity 7.3), OstQuant (7.3) and QuaRot (8.2). Comparisons on bigger models (up to 70B) and on various LLM evaluation benchmarks confirm uniform superiority of NestQuant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09021v2">Comparing human and LLM proofreading in L2 writing: Impact on lexical and syntactic features</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      This study examines the lexical and syntactic interventions of human and LLM proofreading aimed at improving overall intelligibility in identical second language writings, and evaluates the consistency of outcomes across three LLMs (ChatGPT-4o, Llama3.1-8b, Deepseek-r1-8b). Findings show that both human and LLM proofreading enhance bigram lexical features, which may contribute to better coherence and contextual connectedness between adjacent words. However, LLM proofreading exhibits a more generative approach, extensively reworking vocabulary and sentence structures, such as employing more diverse and sophisticated vocabulary and incorporating a greater number of adjective modifiers in noun phrases. The proofreading outcomes are highly consistent in major lexical and syntactic features across the three models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02460v2">Code-Switching Curriculum Learning for Multilingual Transfer in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 To appear in Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now exhibit near human-level performance in various tasks, but their performance drops drastically after a handful of high-resource languages due to the imbalance in pre-training data. Inspired by the human process of second language acquisition, particularly code-switching$\unicode{x2014}$the practice of language alternation in a conversation$\unicode{x2014}$we propose code-switching curriculum learning (CSCL) to enhance cross-lingual transfer for LLMs. CSCL mimics the stages of human language learning by progressively training models with a curriculum consisting of 1) token-level code-switching, 2) sentence-level code-switching, and 3) monolingual corpora. Using Qwen 2 as our underlying model, we demonstrate the efficacy of the CSCL in improving language transfer to Korean, achieving significant performance gains compared to monolingual continual pre-training methods. Ablation studies reveal that both token- and sentence-level code-switching significantly enhance cross-lingual transfer and that curriculum learning amplifies these effects. We also extend our findings into various languages, including Japanese (high-resource) and Indonesian (low-resource), and using two additional models (Gemma 2 and Phi 3.5). We further show that CSCL mitigates spurious correlations between language resources and safety alignment, presenting a robust, efficient framework for more equitable language transfer in LLMs. We observe that CSCL is effective for low-resource settings where high-quality, monolingual corpora for language transfer are hardly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.15481v3">Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 To appear in ACL 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) have advanced rapidly, concerns regarding their safety have become prominent. In this paper, we discover that code-switching in red-teaming queries can effectively elicit undesirable behaviors of LLMs, which are common practices in natural language. We introduce a simple yet effective framework, CSRT, to synthesize codeswitching red-teaming queries and investigate the safety and multilingual understanding of LLMs comprehensively. Through extensive experiments with ten state-of-the-art LLMs and code-switching queries combining up to 10 languages, we demonstrate that the CSRT significantly outperforms existing multilingual red-teaming techniques, achieving 46.7% more attacks than standard attacks in English and being effective in conventional safety domains. We also examine the multilingual ability of those LLMs to generate and understand codeswitching texts. Additionally, we validate the extensibility of the CSRT by generating codeswitching attack prompts with monolingual data. We finally conduct detailed ablation studies exploring code-switching and propound unintended correlation between resource availability of languages and safety alignment in existing multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08885v2">AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI)</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at https://anonymous.4open.science/r/alkali-B416/README.md.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14479v3">Towards Reliable Proof Generation with LLMs: A Neuro-Symbolic Approach</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 long paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with formal domains that require rigorous logical deduction and symbolic reasoning, such as mathematical proof generation. We propose a neuro-symbolic approach that combines LLMs' generative strengths with structured components to overcome this challenge. As a proof-of-concept, we focus on geometry problems. Our approach is two-fold: (1) we retrieve analogous problems and use their proofs to guide the LLM, and (2) a formal verifier evaluates the generated proofs and provides feedback, helping the model fix incorrect proofs. We demonstrate that our method significantly improves proof accuracy for OpenAI's o1 model (58%-70% improvement); both analogous problems and the verifier's feedback contribute to these gains. More broadly, shifting to LLMs that generate provably correct conclusions could dramatically improve their reliability, accuracy and consistency, unlocking complex tasks and critical real-world applications that require trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08524v2">Teaching Physical Awareness to LLMs through Sounds</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable capabilities in text and multimodal processing, yet they fundamentally lack physical awareness--understanding of real-world physical phenomena. In this work, we present ACORN, a framework that teaches LLMs physical awareness through sound, focusing on fundamental physical phenomena like the Doppler effect, multipath effect, and spatial relationships. To overcome data scarcity, ACORN introduce a physics-based simulator combining real-world sound sources with controlled physical channels to generate diverse training data. Using this simulator, we build AQA-PHY, a comprehensive Audio Question-Answer dataset, and propose an audio encoder that processes both magnitude and phase information. By connecting our audio encoder to state-of-the-art LLMs, we demonstrate reasonable results in both simulated and real-world tasks, such as line-of-sight detection, Doppler effect estimation, and Direction-of-Arrival estimation, paving the way for enabling LLMs to understand physical world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08265v3">LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) as feature enhancers to optimize node representations, which are then used as inputs for graph neural networks (GNNs), has shown significant potential in graph representation learning. However, the fundamental properties of this approach remain underexplored. To address this issue, we propose conducting a more in-depth analysis of this issue based on the interchange intervention method. First, we construct a synthetic graph dataset with controllable causal relationships, enabling precise manipulation of semantic relationships and causal modeling to provide data for analysis. Using this dataset, we conduct interchange interventions to examine the deeper properties of LLM enhancers and GNNs, uncovering their underlying logic and internal mechanisms. Building on the analytical results, we design a plug-and-play optimization module to improve the information transfer between LLM enhancers and GNNs. Experiments across multiple datasets and models validate the proposed module.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.08792v2">ProxyGPT: Enabling User Anonymity in LLM Chatbots via (Un)Trustworthy Volunteer Proxies</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Popular large language model (LLM) chatbots such as ChatGPT and Claude require users to create an account with an email or a phone number before allowing full access to their services. This practice ties users' personally identifiable information (PII) to their sensitive conversational data, thus posing significant privacy risks. Unfortunately, existing private LLM solutions based on cryptography or trusted execution environments (TEEs) remain unpopular due to their prohibitive computational expense and platform restrictions. To enable practical user anonymity in LLM chatbots, we propose ProxyGPT, a privacy-enhancing system that leverages browser interaction proxies to submit user queries on their behalf. Unlike traditional proxy systems, ProxyGPT operates at the "user" layer by proxying user interactions with the browser in identity-required environments, thus easily supporting a wide range of chatbot services. We prevent malicious proxies by performing regular integrity audits using modern web proof protocols for TLS data provenance. We further utilize state-of-the-art LLM prompt guards on the proxy's side to mitigate unwanted user requests. Additionally, we incorporate a give-and-take economy based on Chaum's blind-signature e-cash to incentivize ProxyGPT users to proxy for others. Our system evaluation and user study demonstrate the practicality of our approach, as each chat request only takes a few additional seconds on average to fully complete. To the best of our knowledge, ProxyGPT is the first comprehensive proxy-based solution for privacy-preserving AI chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09397v1">SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 6 pages, 9 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Regardless the advancements in device capabilities, efficient inferencing advanced large language models (LLMs) at the edge remains challenging due to limited device memory and power constraints. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new approach that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose SLED, a method that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server efficiently batches and verifies the tokens utilizing a more precise target model. This approach supports device heterogeneity and reduces server-side memory footprint by avoiding the need to deploy multiple target models. Our initial experiments with Jetson Orin Nano, Raspberry Pi 5, and an RTX 6000 edge server indicate substantial benefits: significantly reduced latency, improved energy efficiency, and increased concurrent inference sessions, all without sacrificing model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09391v1">Comparing human and LLM politeness strategies in free production</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 25 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Polite speech poses a fundamental alignment challenge for large language models (LLMs). Humans deploy a rich repertoire of linguistic strategies to balance informational and social goals -- from positive approaches that build rapport (compliments, expressions of interest) to negative strategies that minimize imposition (hedging, indirectness). We investigate whether LLMs employ a similarly context-sensitive repertoire by comparing human and LLM responses in both constrained and open-ended production tasks. We find that larger models ($\ge$70B parameters) successfully replicate key preferences from the computational pragmatics literature, and human evaluators surprisingly prefer LLM-generated responses in open-ended contexts. However, further linguistic analyses reveal that models disproportionately rely on negative politeness strategies even in positive contexts, potentially leading to misinterpretations. While modern LLMs demonstrate an impressive handle on politeness strategies, these subtle differences raise important questions about pragmatic alignment in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00829v2">An LLM-Empowered Adaptive Evolutionary Algorithm For Multi-Component Deep Learning Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 9
    </div>
    <details class="paper-abstract">
      Multi-objective evolutionary algorithms (MOEAs) are widely used for searching optimal solutions in complex multi-component applications. Traditional MOEAs for multi-component deep learning (MCDL) systems face challenges in enhancing the search efficiency while maintaining the diversity. To combat these, this paper proposes $\mu$MOEA, the first LLM-empowered adaptive evolutionary search algorithm to detect safety violations in MCDL systems. Inspired by the context-understanding ability of Large Language Models (LLMs), $\mu$MOEA promotes the LLM to comprehend the optimization problem and generate an initial population tailed to evolutionary objectives. Subsequently, it employs adaptive selection and variation to iteratively produce offspring, balancing the evolutionary efficiency and diversity. During the evolutionary process, to navigate away from the local optima, $\mu$MOEA integrates the evolutionary experience back into the LLM. This utilization harnesses the LLM's quantitative reasoning prowess to generate differential seeds, breaking away from current optimal solutions. We evaluate $\mu$MOEA in finding safety violations of MCDL systems, and compare its performance with state-of-the-art MOEA methods. Experimental results show that $\mu$MOEA can significantly improve the efficiency and diversity of the evolutionary search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09390v1">Beyond Nash Equilibrium: Bounded Rationality of LLMs and humans in Strategic Decision-making</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used in strategic decision-making settings, yet evidence shows that, like humans, they often deviate from full rationality. In this study, we compare LLMs and humans using experimental paradigms directly adapted from behavioral game-theory research. We focus on two well-studied strategic games, Rock-Paper-Scissors and the Prisoner's Dilemma, which are well known for revealing systematic departures from rational play in human subjects. By placing LLMs in identical experimental conditions, we evaluate whether their behaviors exhibit the bounded rationality characteristic of humans. Our findings show that LLMs reproduce familiar human heuristics, such as outcome-based strategy switching and increased cooperation when future interaction is possible, but they apply these rules more rigidly and demonstrate weaker sensitivity to the dynamic changes in the game environment. Model-level analyses reveal distinctive architectural signatures in strategic behavior, and even reasoning models sometimes struggle to find effective strategies in adaptive situations. These results indicate that current LLMs capture only a partial form of human-like bounded rationality and highlight the need for training methods that encourage flexible opponent modeling and stronger context awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04211v2">MMREC: LLM Based Multi-Modal Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      The importance of recommender systems is growing rapidly due to the exponential increase in the volume of content generated daily. This surge in content presents unique challenges for designing effective recommender systems. Key among these challenges is the need to effectively leverage the vast amounts of natural language data and images that represent user preferences. This paper presents a novel approach to enhancing recommender systems by leveraging Large Language Models (LLMs) and deep learning techniques. The proposed framework aims to improve the accuracy and relevance of recommendations by incorporating multi-modal information processing and by the use of unified latent space representation. The study explores the potential of LLMs to better understand and utilize natural language data in recommendation contexts, addressing the limitations of previous methods. The framework efficiently extracts and integrates text and image information through LLMs, unifying diverse modalities in a latent space to simplify the learning process for the ranking model. Experimental results demonstrate the enhanced discriminative power of the model when utilizing multi-modal information. This research contributes to the evolving field of recommender systems by showcasing the potential of LLMs and multi-modal data integration to create more personalized and contextually relevant recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09101v2">Bridging the Gap Between LLMs and Human Intentions: Progresses and Challenges in Instruction Understanding, Intention Reasoning, and Reliable Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 19 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities in understanding and generation. However, when interacting with human instructions in real-world scenarios, LLMs still face significant challenges, particularly in accurately capturing and comprehending human instructions and intentions. This paper focuses on three challenges in LLM-based text generation tasks: instruction understanding, intention reasoning, and Reliable Dialog Generation. Regarding human complex instruction, LLMs have deficiencies in understanding long contexts and instructions in multi-round conversations. For intention reasoning, LLMs may have inconsistent command reasoning, difficulty reasoning about commands containing incorrect information, difficulty understanding user ambiguous language commands, and a weak understanding of user intention in commands. Besides, In terms of Reliable Dialog Generation, LLMs may have unstable generated content and unethical generation. To this end, we classify and analyze the performance of LLMs in challenging scenarios and conduct a comprehensive evaluation of existing solutions. Furthermore, we introduce benchmarks and categorize them based on the aforementioned three core challenges. Finally, we explore potential directions for future research to enhance the reliability and adaptability of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20367v4">Enhancing Code LLMs with Reinforcement Learning in Code Generation: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a powerful paradigm for enhancing large language models (LLMs) in code generation and optimization. This survey systematically reviews RL-driven techniques across the code development lifecycle, from compiler-level optimizations and resource allocation strategies to end-to-end code synthesis frameworks. We first examine classical and modern RL algorithms -- spanning policy gradients, actor-critic methods, human-feedback alignment, and preference-based optimization -- and their adaptations to the unique challenges of code generation, such as sparse and delayed rewards. Next, we analyze key benchmarks, datasets, and evaluation metrics that drive progress in RL-augmented Code LLMs. Finally, we identify open problems, including the need for richer feedback sources, support for low-level and domain-specific languages, and methods to reduce computational overhead. By consolidating current insights and outlining future directions, this work aims to guide researchers and practitioners in leveraging RL to produce more robust, efficient, and human-aligned code generation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02450v3">Measuring What Makes You Unique: Difference-Aware User Modeling for Enhancing LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 2025 ACL Findings
    </div>
    <details class="paper-abstract">
      Personalizing Large Language Models (LLMs) has become a critical step in facilitating their widespread application to enhance individual life experiences. In pursuit of personalization, distilling key preference information from an individual's historical data as instructional preference context to customize LLM generation has emerged as a promising direction. However, these methods face a fundamental limitation by overlooking the inter-user comparative analysis, which is essential for identifying the inter-user differences that truly shape preferences. To address this limitation, we propose Difference-aware Personalization Learning (DPL), a novel approach that emphasizes extracting inter-user differences to enhance LLM personalization. DPL strategically selects representative users for comparison and establishes a structured standard to extract meaningful, task-relevant differences for customizing LLM generation. Extensive experiments on real-world datasets demonstrate that DPL significantly enhances LLM personalization. We release our code at https://github.com/SnowCharmQ/DPL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08626v2">Leveraging LLMs to Evaluate Usefulness of Document</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      The conventional Cranfield paradigm struggles to effectively capture user satisfaction due to its weak correlation between relevance and satisfaction, alongside the high costs of relevance annotation in building test collections. To tackle these issues, our research explores the potential of leveraging large language models (LLMs) to generate multilevel usefulness labels for evaluation. We introduce a new user-centric evaluation framework that integrates users' search context and behavioral data into LLMs. This framework uses a cascading judgment structure designed for multilevel usefulness assessments, drawing inspiration from ordinal regression techniques. Our study demonstrates that when well-guided with context and behavioral information, LLMs can accurately evaluate usefulness, allowing our approach to surpass third-party labeling methods. Furthermore, we conduct ablation studies to investigate the influence of key components within the framework. We also apply the labels produced by our method to predict user satisfaction, with real-world experiments indicating that these labels substantially improve the performance of satisfaction prediction models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00115v2">SACA: A Scenario-Aware Collision Avoidance Framework for Autonomous Vehicles Integrating LLMs-Driven Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 11 pages,10 figures. This work has been submitted to the IEEE TVT for possible publication
    </div>
    <details class="paper-abstract">
      Reliable collision avoidance under extreme situations remains a critical challenge for autonomous vehicles. While large language models (LLMs) offer promising reasoning capabilities, their application in safety-critical evasive maneuvers is limited by latency and robustness issues. Even so, LLMs stand out for their ability to weigh emotional, legal, and ethical factors, enabling socially responsible and context-aware collision avoidance. This paper proposes a scenario-aware collision avoidance (SACA) framework for extreme situations by integrating predictive scenario evaluation, data-driven reasoning, and scenario-preview-based deployment to improve collision avoidance decision-making. SACA consists of three key components. First, a predictive scenario analysis module utilizes obstacle reachability analysis and motion intention prediction to construct a comprehensive situational prompt. Second, an online reasoning module refines decision-making by leveraging prior collision avoidance knowledge and fine-tuning with scenario data. Third, an offline evaluation module assesses performance and stores scenarios in a memory bank. Additionally, A precomputed policy method improves deployability by previewing scenarios and retrieving or reasoning policies based on similarity and confidence levels. Real-vehicle tests show that, compared with baseline methods, SACA effectively reduces collision losses in extreme high-risk scenarios and lowers false triggering under complex conditions. Project page: https://sean-shiyuez.github.io/SACA/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13287v4">An Online Learning Approach to Prompt-based Selection of Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      Selecting a sample generation scheme from multiple prompt-based generative models, including large language models (LLMs) and prompt-guided image and video generation models, is typically addressed by choosing the model that maximizes an averaged evaluation score. However, this score-based selection overlooks the possibility that different models achieve the best generation performance for different types of text prompts. An online identification of the best generation model for various input prompts can reduce the costs associated with querying sub-optimal models. In this work, we explore the possibility of varying rankings of text-based generative models for different text prompts and propose an online learning framework to predict the best data generation model for a given input prompt. The proposed PAK-UCB algorithm addresses a contextual bandit (CB) setting with shared context variables across the arms, utilizing the generated data to update kernel-based functions that predict the score of each model available for unseen text prompts. Additionally, we leverage random Fourier features (RFF) to accelerate the online learning process of PAK-UCB. Our numerical experiments on real and simulated text-to-image and image-to-text generative models show that RFF-UCB performs successfully in identifying the best generation model across different sample types. The code is available at: github.com/yannxiaoyanhu/dgm-online-select.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09359v1">Taming SQL Complexity: LLM-Based Equivalence Evaluation for Text-to-SQL</a></div>
    <div class="paper-meta">
      📅 2025-06-11
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has significantly advanced Text-to-SQL (NL2SQL) systems, yet evaluating the semantic equivalence of generated SQL remains a challenge, especially given ambiguous user queries and multiple valid SQL interpretations. This paper explores using LLMs to assess both semantic and a more practical "weak" semantic equivalence. We analyze common patterns of SQL equivalence and inequivalence, discuss challenges in LLM-based evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08371v2">Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) support long contexts, they struggle with performance degradation within the context window. Current solutions incur prohibitive training costs, leaving statistical behaviors and cost-effective approaches underexplored. From the decoding perspective, we identify the Posterior Salience Attenuation (PSA) phenomenon, where the salience ratio correlates with long-text performance degradation. Notably, despite the attenuation, gold tokens still occupy high-ranking positions in the decoding space. Motivated by it, we propose the training-free Positional Contrastive Decoding (PCD) that contrasts the logits derived from long-aware attention with those from designed local-aware attention, enabling the model to focus on the gains introduced by large-scale short-to-long training. Through the analysis of long-term decay simulation, we demonstrate that PCD effectively alleviates attention score degradation. Experimental results show that PCD achieves state-of-the-art performance on long-context benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00963v2">PDE-Controller: LLMs for Autoformalization and Reasoning of PDEs</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      While recent AI-for-math has made strides in pure mathematics, areas of applied mathematics, particularly PDEs, remain underexplored despite their significant real-world applications. We present PDE-Controller, a framework that enables large language models (LLMs) to control systems governed by partial differential equations (PDEs). Our approach enables LLMs to transform informal natural language instructions into formal specifications, and then execute reasoning and planning steps to improve the utility of PDE control. We build a holistic solution comprising datasets (both human-written cases and 2 million synthetic samples), math-reasoning models, and novel evaluation metrics, all of which require significant effort. Our PDE-Controller significantly outperforms prompting the latest open source and GPT models in reasoning, autoformalization, and program synthesis, achieving up to a 62% improvement in utility gain for PDE control. By bridging the gap between language generation and PDE systems, we demonstrate the potential of LLMs in addressing complex scientific and engineering challenges. We release all data, model checkpoints, and code at https://pde-controller.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09354v1">"Is This Really a Human Peer Supporter?": Misalignments Between Peer Supporters and Experts in LLM-Supported Interactions</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      Mental health is a growing global concern, prompting interest in AI-driven solutions to expand access to psychosocial support. Peer support, grounded in lived experience, offers a valuable complement to professional care. However, variability in training, effectiveness, and definitions raises concerns about quality, consistency, and safety. Large Language Models (LLMs) present new opportunities to enhance peer support interactions, particularly in real-time, text-based interactions. We present and evaluate an AI-supported system with an LLM-simulated distressed client, context-sensitive LLM-generated suggestions, and real-time emotion visualisations. 2 mixed-methods studies with 12 peer supporters and 5 mental health professionals (i.e., experts) examined the system's effectiveness and implications for practice. Both groups recognised its potential to enhance training and improve interaction quality. However, we found a key tension emerged: while peer supporters engaged meaningfully, experts consistently flagged critical issues in peer supporter responses, such as missed distress cues and premature advice-giving. This misalignment highlights potential limitations in current peer support training, especially in emotionally charged contexts where safety and fidelity to best practices are essential. Our findings underscore the need for standardised, psychologically grounded training, especially as peer support scales globally. They also demonstrate how LLM-supported systems can scaffold this development--if designed with care and guided by expert oversight. This work contributes to emerging conversations on responsible AI integration in mental health and the evolving role of LLMs in augmenting peer-delivered care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06975v3">Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test</a></div>
    <div class="paper-meta">
      📅 2025-06-11
    </div>
    <details class="paper-abstract">
      As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, and full model substitution, showing that it consistently achieves superior statistical power over prior methods under constrained query budgets.
    </details>
</div>
