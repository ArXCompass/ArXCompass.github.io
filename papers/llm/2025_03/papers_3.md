# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17181v1">LLMs Love Python: A Study of LLMs' Bias for Programming Languages and Libraries</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Programming language and library choices are crucial to software reliability and security. Poor or inconsistent choices can lead to increased technical debt, security vulnerabilities, and even catastrophic failures in safety-critical systems. As Large Language Models (LLMs) play an increasing role in code generation, it is essential to understand how they make these decisions. However, little is known about their preferences when selecting programming languages and libraries for different coding tasks. To fill this gap, this study provides the first in-depth investigation into LLM preferences for programming languages and libraries used when generating code. We assess the preferences of eight diverse LLMs by prompting them to complete various coding tasks, including widely-studied benchmarks and the more practical task of generating the initial structural code for new projects (a crucial step that often determines a project's language or library choices). Our findings reveal that LLMs heavily favour Python when solving language-agnostic problems, using it in 90%-97% of cases for benchmark tasks. Even when generating initial project code where Python is not a suitable language, it remains the most-used language in 58% of instances. Moreover, LLMs contradict their own language recommendations in 83% of project initialisation tasks, raising concerns about their reliability in guiding language selection. Similar biases toward well-established libraries further create serious discoverability challenges for newer open-source projects. These results highlight the need to improve LLMs' adaptability to diverse programming contexts and to develop mechanisms for mitigating programming language and library bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15289v2">SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00363v2">SPDZCoder: Combining Expert Knowledge with LLMs for Generating Privacy-Computing Code</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Privacy computing receives increasing attention but writing privacy computing code remains challenging for developers due to limited library functions, necessitating function implementation from scratch, and data-oblivious requirement, contradicting intuitive thinking and usual practices of programmers. Automating the generation of privacy computing code with Large Language Models can streamline development effort and lower the barrier to using privacy computing frameworks. However, existing LLMs still encounter challenges in code translation for privacy-preserving computation, such as translating Python to MP-SPDZ, due to the scarcity of MP-SPDZ data required for effective pre-training or fine-tuning. Moreover, the lack of a benchmark further complicates the evaluation of translation quality. To address the limitations, this work proposes SPDZCoder, a rule-based framework that combines LLMs with expert knowledge for generating privacy-computing code without requiring additional training data. Specifically, SPDZCoder employ a rigorous procedure for collecting high-quality expert knowledge to represent the semantic-expressing differences between Python and MP-SPDZ, and to derive transformation rules for translating Python to MP-SPDZ based on these knowledge. Then, SPDZCoder progressively converts Python code into MP-SPDZ code using transformation rules in a three stage pipeline. To evaluate SPDZCoder, we manually constructed a benchmark dataset, SPDZEval, which comprises six data splits, each representing a distinct class of challenging tasks in MP-SPDZ implementation. Extensive experiments show that SPDZCoder achieves superior performance, significantly surpassing baselines in pass@1 and pass@2. Specifically, SPDZCoder attains an overall correctness of 85.94% and 92.01% in pass@1 and pass@2, respectively, whereas the best-performing baseline achieves 63.58% and 76.36%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17073v1">A Study into Investigating Temporal Robustness of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encapsulate a surprising amount of factual world knowledge. However, their performance on temporal questions and historical knowledge is limited because they often cannot understand temporal scope and orientation or neglect the temporal aspect altogether. In this study, we aim to measure precisely how robust LLMs are for question answering based on their ability to process temporal information and perform tasks requiring temporal reasoning and temporal factual knowledge. Specifically, we design eight time-sensitive robustness tests for factual information to check the sensitivity of six popular LLMs in the zero-shot setting. Overall, we find LLMs lacking temporal robustness, especially to temporal reformulations and the use of different granularities of temporal references. We show how a selection of these eight tests can be used automatically to judge a model's temporal robustness for user questions on the fly. Finally, we apply the findings of this study to improve the temporal QA performance by up to 55 percent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12547v2">LLMSeR: Enhancing Sequential Recommendation via LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Sequential Recommender Systems (SRS) have become a cornerstone of online platforms, leveraging users' historical interaction data to forecast their next potential engagement. Despite their widespread adoption, SRS often grapple with the long-tail user dilemma, resulting in less effective recommendations for individuals with limited interaction records. The advent of Large Language Models (LLMs), with their profound capability to discern semantic relationships among items, has opened new avenues for enhancing SRS through data augmentation. Nonetheless, current methodologies encounter obstacles, including the absence of collaborative signals and the prevalence of hallucination phenomena. In this work, we present LLMSeR, an innovative framework that utilizes Large Language Models (LLMs) to generate pseudo-prior items, thereby improving the efficacy of Sequential Recommender Systems (SRS). To alleviate the challenge of insufficient collaborative signals, we introduce the Semantic Interaction Augmentor (SIA), a method that integrates both semantic and collaborative information to comprehensively augment user interaction data. Moreover, to weaken the adverse effects of hallucination in SRS, we develop the Adaptive Reliability Validation (ARV), a validation technique designed to assess the reliability of the generated pseudo items. Complementing these advancements, we also devise a Dual-Channel Training strategy, ensuring seamless integration of data augmentation into the SRS training process.Extensive experiments conducted with three widely-used SRS models demonstrate the generalizability and efficacy of LLMSeR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17039v1">Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans?</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Studies on evaluation metrics and LLM-as-a-Judge models for automatic text summarization have largely been focused on English, limiting our understanding of their effectiveness in other languages. Through our new dataset BASSE (BAsque and Spanish Summarization Evaluation), we address this situation by collecting human judgments on 2,040 abstractive summaries in Basque and Spanish, generated either manually or by five LLMs with four different prompts. For each summary, annotators evaluated five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We use these data to reevaluate traditional automatic metrics used for evaluating summaries, as well as several LLM-as-a-Judge models that show strong performance on this task in English. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open-sourced judge LLMs perform poorly. We release BASSE and our code publicly, along with the first large-scale Basque summarization dataset containing 22,525 news articles with their subheads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17004v1">Text2Model: Generating dynamic chemical reactor models using large language models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      As large language models have shown remarkable capabilities in conversing via natural language, the question arises as to how LLMs could potentially assist chemical engineers in research and industry with domain-specific tasks. We generate dynamic chemical reactor models in Modelica code format from textual descriptions as user input. We fine-tune Llama 3.1 8B Instruct on synthetically generated Modelica code for different reactor scenarios. We compare the performance of our fine-tuned model to the baseline Llama 3.1 8B Instruct model and GPT4o. We manually assess the models' predictions regarding the syntactic and semantic accuracy of the generated dynamic models. We find that considerable improvements are achieved by the fine-tuned model with respect to both the semantic and the syntactic accuracy of the Modelica models. However, the fine-tuned model lacks a satisfactory ability to generalize to unseen scenarios compared to GPT4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17397v2">Building Multilingual Datasets for Predicting Mental Health Severity through LLMs: Prospects and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into various medical fields, including mental health support systems. However, there is a gap in research regarding the effectiveness of LLMs in non-English mental health support applications. To address this problem, we present a novel multilingual adaptation of widely-used mental health datasets, translated from English into six languages (e.g., Greek, Turkish, French, Portuguese, German, and Finnish). This dataset enables a comprehensive evaluation of LLM performance in detecting mental health conditions and assessing their severity across multiple languages. By experimenting with GPT and Llama, we observe considerable variability in performance across languages, despite being evaluated on the same translated dataset. This inconsistency underscores the complexities inherent in multilingual mental health support, where language-specific nuances and mental health data coverage can affect the accuracy of the models. Through comprehensive error analysis, we emphasize the risks of relying exclusively on LLMs in medical settings (e.g., their potential to contribute to misdiagnoses). Moreover, our proposed approach offers significant cost savings for multilingual tasks, presenting a major advantage for broad-scale implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07871v2">Objection Overruled! Lay People can Distinguish Large Language Models from Lawyers, but still Favour Advice from an LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 ACMConference on Human Factors in Computing Systems (CHI'25)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are seemingly infiltrating every domain, and the legal context is no exception. In this paper, we present the results of three experiments (total N = 288) that investigated lay people's willingness to act upon, and their ability to discriminate between, LLM- and lawyer-generated legal advice. In Experiment 1, participants judged their willingness to act on legal advice when the source of the advice was either known or unknown. When the advice source was unknown, participants indicated that they were significantly more willing to act on the LLM-generated advice. The result of the source unknown condition was replicated in Experiment 2. Intriguingly, despite participants indicating higher willingness to act on LLM-generated advice in Experiments 1 and 2, participants discriminated between the LLM- and lawyer-generated texts significantly above chance-level in Experiment 3. Lastly, we discuss potential explanations and risks of our findings, limitations and future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12923v2">On-Device LLMs for Home Assistant: Dual Role in Intent Detection and Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 WNUT 2025
    </div>
    <details class="paper-abstract">
      This paper investigates whether Large Language Models (LLMs), fine-tuned on synthetic but domain-representative data, can perform the twofold task of (i) slot and intent detection and (ii) natural language response generation for a smart home assistant, while running solely on resource-limited, CPU-only edge hardware. We fine-tune LLMs to produce both JSON action calls and text responses. Our experiments show that 16-bit and 8-bit quantized variants preserve high accuracy on slot and intent detection and maintain strong semantic coherence in generated text, while the 4-bit model, while retaining generative fluency, suffers a noticeable drop in device-service classification accuracy. Further evaluations on noisy human (non-synthetic) prompts and out-of-domain intents confirm the models' generalization ability, obtaining around 80--86\% accuracy. While the average inference time is 5--6 seconds per query -- acceptable for one-shot commands but suboptimal for multi-turn dialogue -- our results affirm that an on-device LLM can effectively unify command interpretation and flexible response generation for home automation without relying on specialized hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16929v1">TEMPO: Temporal Preference Optimization of Video LLMs via Difficulty Scheduling and Pre-SFT Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video LLMs) have achieved significant success by leveraging a two-stage paradigm: pretraining on large-scale video-text data for vision-language alignment, followed by supervised fine-tuning (SFT) for task-specific capabilities. However, existing approaches struggle with temporal reasoning due to weak temporal correspondence in the data and reliance on the next-token prediction paradigm during training. To address these limitations, we propose TEMPO (TEMporal Preference Optimization), a systematic framework that enhances Video LLMs' temporal reasoning capabilities through Direct Preference Optimization (DPO). To facilitate this, we introduce an automated preference data generation pipeline that systematically constructs preference pairs by selecting videos that are rich in temporal information, designing video-specific perturbation strategies, and finally evaluating model responses on clean and perturbed video inputs. Our temporal alignment features two key innovations: curriculum learning which that progressively increases perturbation difficulty to improve model robustness and adaptability; and ``Pre-SFT Alignment'', applying preference optimization before instruction tuning to prioritize fine-grained temporal comprehension. Extensive experiments demonstrate that our approach consistently improves Video LLM performance across multiple benchmarks with a relatively small set of self-generated DPO data. We further analyze the transferability of DPO data across architectures and the role of difficulty scheduling in optimization. Our findings highlight our TEMPO as a scalable and efficient complement to SFT-based methods, paving the way for developing reliable Video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12785v2">Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large Vision-Language Models (LVLMs) typically learn visual capacity through visual instruction tuning, involving updates to both a projector and their LLM backbones. Inspired by the concept of a visual region in the human brain, we investigate the existence of an analogous \textit{visual region} within LLMs that functions as a cognitive core, and explore the potential of efficient training of LVLMs via selective layers tuning. Using Bunny-Llama-3-8B-V for detailed analysis and other three LVLMs for validation across diverse visual and textual tasks, we find that selectively updating 25\% of LLMs layers, when sparsely and uniformly distributed, can preserve nearly 99\% of visual performance and maintain or improve textual task results, while effectively reducing training time. Based on this targeted training approach, we further propose a novel visual region-based pruning paradigm, removing non-critical layers outside the visual region, which can achieve minimal performance loss. This study offers an effective and efficient strategy for LVLM training and inference by activating a layer-wise visual region within LLMs, which proves consistently effective across different models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13879v2">Bridging Social Psychology and LLM Reasoning: Conflict-Aware Meta-Review Generation via Cognitive Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      The rapid growth of scholarly submissions has overwhelmed traditional peer review systems, driving the need for intelligent automation to preserve scientific rigor. While large language models (LLMs) show promise in automating manuscript critiques, their ability to synthesize high-stakes meta-reviews, which require conflict-aware reasoning and consensus derivation, remains underdeveloped. Existing methods fail to effectively handle conflicting viewpoints within differing opinions, and often introduce additional cognitive biases, such as anchoring effects and conformity bias.To overcome these limitations, we propose the Cognitive Alignment Framework (CAF), a dual-process architecture that transforms LLMs into adaptive scientific arbitrators. By operationalizing Kahneman's dual-process theory, CAF introduces a three-step cognitive pipeline: review initialization, incremental integration, and cognitive alignment.Empirical validation shows that CAF outperforms existing LLM-based methods, with sentiment consistency gains reaching up to 19.47\% and content consistency improving by as much as 12.95\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16922v1">RustEvo^2: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become pivotal tools for automating code generation in software development. However, these models face significant challenges in producing version-aware code for rapidly evolving languages like Rust, where frequent Application Programming Interfaces (API) changes across versions lead to compatibility issues and correctness errors. Existing benchmarks lack systematic evaluation of how models navigate API transitions, relying on labor-intensive manual curation and offering limited version-specific insights. To address this gap, we present RustEvo, a novel framework for constructing dynamic benchmarks that evaluate the ability of LLMs to adapt to evolving Rust APIs. RustEvo automates dataset creation by synthesizing 588 API changes (380 from Rust standard libraries, 208 from 15 third-party crates) into programming tasks mirroring real-world challenges. These tasks cover four API evolution categories: Stabilizations, Signature Changes, Behavioral Changes, and Deprecations, reflecting their actual distribution in the Rust ecosystem. Experiments on state-of-the-art (SOTA) LLMs reveal significant performance variations: models achieve a 65.8% average success rate on stabilized APIs but only 38.0% on behavioral changes, highlighting difficulties in detecting semantic shifts without signature alterations. Knowledge cutoff dates strongly influence performance, with models scoring 56.1% on before-cutoff APIs versus 32.5% on after-cutoff tasks. Retrieval-Augmented Generation (RAG) mitigates this gap, improving success rates by 13.5% on average for APIs released after model training. Our findings underscore the necessity of our evolution-aware benchmarks to advance the adaptability of LLMs in fast-paced software ecosystems. The framework and the benchmarks are publicly released at https://github.com/SYSUSELab/RustEvo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16893v1">Improving the End-to-End Efficiency of Offline Inference for Multi-LLM Applications Based on Sampling and Simulation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) have shown great success in many tasks, they are used in various applications. While a lot of works have focused on the efficiency of single-LLM application (e.g., offloading, request scheduling, parallelism strategy selection), multi-LLM applications receive less attention, particularly in offline inference scenarios. In this work, we aim to improve the offline end-to-end inference efficiency of multi-LLM applications in the single-node multi-GPU environment. The problem involves two key decisions: (1) determining which LLMs to run concurrently each time (we may not run all the models at the same time), and (2) selecting a parallelism strategy to use for each LLM. This problem is NP-hard. Naive solutions may not work well because the running time for a model to complete a set of requests depends on the request workload and the selected parallelism strategy, and they lack an accurate model of the running time. As the LLM output lengths are unknown before running, to estimate the model running time, we propose a sampling-then-simulation method which first estimates the output lengths by sampling from an empirical cumulative function we obtained from a large dataset in advance, and then simulates the LLM inference process accordingly. Based on the simulation, we estimate the per-iteration latencys to get the total latency. A greedy method is proposed to optimize the scheduling of the LLMs in the application across the GPUs. We then propose a framework SamuLLM which contains two phases: planning, which calls the greedy method for an application and running, which runs the application and dynamically adjust the model scheduling based on the runtime information. Experiments on 3 applications and a mixed application show that SamuLLM can achieve 1.0-2.4$\times$ end-to-end speedups compared to the competitors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16454v3">Catastrophic Failure of LLM Unlearning via Quantization</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable proficiency in generating text, benefiting from extensive training on vast textual corpora. However, LLMs may also acquire unwanted behaviors from the diverse and sensitive nature of their training data, which can include copyrighted and private content. Machine unlearning has been introduced as a viable solution to remove the influence of such problematic content without the need for costly and time-consuming retraining. This process aims to erase specific knowledge from LLMs while preserving as much model utility as possible. Despite the effectiveness of current unlearning methods, little attention has been given to whether existing unlearning methods for LLMs truly achieve forgetting or merely hide the knowledge, which current unlearning benchmarks fail to detect. This paper reveals that applying quantization to models that have undergone unlearning can restore the "forgotten" information. To thoroughly evaluate this phenomenon, we conduct comprehensive experiments using various quantization techniques across multiple precision levels. We find that for unlearning methods with utility constraints, the unlearned model retains an average of 21\% of the intended forgotten knowledge in full precision, which significantly increases to 83\% after 4-bit quantization. ... Our code is available at: \href{https://github.com/zzwjames/FailureLLMUnlearning}{https://github.com/zzwjames/FailureLLMUnlearning}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14345v4">Bias Testing and Mitigation in LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 Accepted by TOSEM
    </div>
    <details class="paper-abstract">
      As the adoption of LLMs becomes more widespread in software coding ecosystems, a pressing issue has emerged: does the generated code contain social bias and unfairness, such as those related to age, gender, and race? This issue concerns the integrity, fairness, and ethical foundation of software applications that depend on the code generated by these models but are underexplored in the literature. This paper presents a novel bias testing framework that is specifically designed for code generation tasks. Based on this framework, we conduct an extensive empirical study on the biases in code generated by five widely studied LLMs (i.e., PALM-2-CodeChat-bison, Claude-instant-1, GPT-3.5-turbo, GPT-4-turbo, and GPT-4). Our findings reveal that biases are prevalent. For example, 13.47% to 49.10% of the codes generated by these LLMs have biased behaviors towards gender. Moreover, we study five bias mitigation prompt strategies that are commonly used in current code generation scenarios, i.e., zero-shot, one-shot, few-shot, and two Chain-of-Thought (CoT) prompts, with and without provided feedback-driven refinement. Our evaluation results illustrate that using direct prompt engineering strategies has limited effectiveness in mitigating bias, but our test execution feedback can help to reduce the ratio of code biases to a large extent (e.g., from 59.88% to 4.79% for GPT-4).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16851v1">Towards LLM Guardrails via Sparse Representation Steering</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance in natural language generation tasks, yet their uncontrolled outputs pose significant ethical and safety risks. Recently, representation engineering methods have shown promising results in steering model behavior by modifying the rich semantic information encoded in activation vectors. However, due to the difficulty of precisely disentangling semantic directions within high-dimensional representation space, existing approaches suffer from three major limitations: lack of fine-grained control, quality degradation of generated content, and poor interpretability. To address these challenges, we propose a sparse encoding-based representation engineering method, named SRE, which decomposes polysemantic activations into a structured, monosemantic feature space. By leveraging sparse autoencoding, our approach isolates and adjusts only task-specific sparse feature dimensions, enabling precise and interpretable steering of model behavior while preserving content quality. We validate our method on three critical domains, i.e., safety, fairness, and truthfulness using the open-source LLM Gemma-2-2B-it. Experimental results show that SRE achieves superior controllability while maintaining the overall quality of generated content (i.e., controllability and quality), demonstrating its effectiveness as a fine-grained and interpretable activation steering framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00212v4">STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 25 pages, 5 figures
    </div>
    <details class="paper-abstract">
      A fundamental challenge in formal theorem proving by LLMs is the lack of high-quality training data. Although reinforcement learning or expert iteration partially mitigates this issue by alternating between LLM generating proofs and finetuning them on correctly generated ones, performance quickly plateaus due to the scarcity of correct proofs (sparse rewards). To keep improving the models with limited data, we draw inspiration from mathematicians, who continuously develop new results, partly by proposing novel conjectures or exercises (which are often variants of known results) and attempting to solve them. We design the Self-play Theorem Prover (STP) that simultaneously takes on two roles, conjecturer and prover, each providing training signals to the other. The conjecturer is trained iteratively on previously generated conjectures that are barely provable by the current prover, which incentivizes it to generate increasingly challenging conjectures over time. The prover attempts to prove the conjectures with standard expert iteration. We evaluate STP with both Lean and Isabelle formal versifiers. With 51.3 billion tokens generated during the training in Lean, STP proves 28.5% of the statements in the LeanWorkbook dataset, doubling the previous best result of 13.2% achieved through expert iteration. The final model achieves state-of-the-art performance among whole-proof generation methods on miniF2F-test (65.0%, pass@3200), Proofnet-test (23.9%, pass@3200) and PutnamBench (8/644, pass@3200). We release our code, model, and dataset in this URL: https://github.com/kfdong/STP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16789v1">Conversational User-AI Intervention: A Study on Prompt Rewriting for Improved LLM Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 8 pages, ACL style
    </div>
    <details class="paper-abstract">
      Human-LLM conversations are increasingly becoming more pervasive in peoples' professional and personal lives, yet many users still struggle to elicit helpful responses from LLM Chatbots. One of the reasons for this issue is users' lack of understanding in crafting effective prompts that accurately convey their information needs. Meanwhile, the existence of real-world conversational datasets on the one hand, and the text understanding faculties of LLMs on the other, present a unique opportunity to study this problem, and its potential solutions at scale. Thus, in this paper we present the first LLM-centric study of real human-AI chatbot conversations, focused on investigating aspects in which user queries fall short of expressing information needs, and the potential of using LLMs to rewrite suboptimal user prompts. Our findings demonstrate that rephrasing ineffective prompts can elicit better responses from a conversational system, while preserving the user's original intent. Notably, the performance of rewrites improves in longer conversations, where contextual inferences about user needs can be made more accurately. Additionally, we observe that LLMs often need to -- and inherently do -- make \emph{plausible} assumptions about a user's intentions and goals when interpreting prompts. Our findings largely hold true across conversational domains, user intents, and LLMs of varying sizes and families, indicating the promise of using prompt rewriting as a solution for better human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08823v2">ResBench: Benchmarking LLM-Generated FPGA Designs with Resource Awareness</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 to be published in International Symposium on Highly Efficient Accelerators and Reconfigurable Technologies 2025
    </div>
    <details class="paper-abstract">
      Field-Programmable Gate Arrays (FPGAs) are widely used in modern hardware design, yet writing Hardware Description Language (HDL) code for FPGA implementation remains a complex and time-consuming task. Large Language Models (LLMs) have emerged as a promising tool for HDL generation, but existing benchmarks for LLM-based code generation primarily focus on functional correctness while overlooking hardware resource usage. Furthermore, current benchmarks offer limited diversity and do not fully represent the wide range of real-world FPGA applications. To address these shortcomings, we introduce ResBench, the first resource-focused benchmark explicitly designed to distinguish between resource-optimized and inefficient LLM-generated HDL code. ResBench consists of 56 problems across 12 categories, covering applications from finite state machines to financial computing. Our open-source evaluation framework automatically tests LLMs by generating Verilog code, verifying correctness, and measuring resource usage. The experiments, which primarily analyze Lookup Table (LUT) usage, reveal significant differences among LLMs, demonstrating ResBench's capability to identify models that generate more resource-optimized FPGA designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17579v1">Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Whether large language models (LLMs) process language similarly to humans has been the subject of much theoretical and practical debate. We examine this question through the lens of the production-interpretation distinction found in human sentence processing and evaluate the extent to which instruction-tuned LLMs replicate this distinction. Using an empirically documented asymmetry between production and interpretation in humans for implicit causality verbs as a testbed, we find that some LLMs do quantitatively and qualitatively reflect human-like asymmetries between production and interpretation. We demonstrate that whether this behavior holds depends upon both model size - with larger models more likely to reflect human-like patterns and the choice of meta-linguistic prompts used to elicit the behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17569v1">Fairness-Driven LLM-based Causal Discovery with Active Learning and Dynamic Scoring</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Causal discovery (CD) plays a pivotal role in numerous scientific fields by clarifying the causal relationships that underlie phenomena observed in diverse disciplines. Despite significant advancements in CD algorithms that enhance bias and fairness analyses in machine learning, their application faces challenges due to the high computational demands and complexities of large-scale data. This paper introduces a framework that leverages Large Language Models (LLMs) for CD, utilizing a metadata-based approach akin to the reasoning processes of human experts. By shifting from pairwise queries to a more scalable breadth-first search (BFS) strategy, the number of required queries is reduced from quadratic to linear in terms of variable count, thereby addressing scalability concerns inherent in previous approaches. This method utilizes an Active Learning (AL) and a Dynamic Scoring Mechanism that prioritizes queries based on their potential information gain, combining mutual information, partial correlation, and LLM confidence scores to refine the causal graph more efficiently and accurately. This BFS query strategy reduces the required number of queries significantly, thereby addressing scalability concerns inherent in previous approaches. This study provides a more scalable and efficient solution for leveraging LLMs in fairness-driven CD, highlighting the effects of the different parameters on performance. We perform fairness analyses on the inferred causal graphs, identifying direct and indirect effects of sensitive attributes on outcomes. A comparison of these analyses against those from graphs produced by baseline methods highlights the importance of accurate causal graph construction in understanding bias and ensuring fairness in machine learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09778v2">Prompt and circumstance: A word-by-word LLM prompting approach to interlinear glossing for low-resource languages</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Partly automated creation of interlinear glossed text (IGT) has the potential to assist in linguistic documentation. We argue that LLMs can make this process more accessible to linguists because of their capacity to follow natural-language instructions. We investigate the effectiveness of a retrieval-based LLM prompting approach to glossing, applied to the seven languages from the SIGMORPHON 2023 shared task. Our system beats the BERT-based shared task baseline for every language in the morpheme-level score category, and we show that a simple 3-best oracle has higher word-level scores than the challenge winner (a tuned sequence model) in five languages. In a case study on Tsez, we ask the LLM to automatically create and follow linguistic instructions, reducing errors on a confusing grammatical feature. Our results thus demonstrate the potential contributions which LLMs can make in interactive systems for glossing, both in making suggestions to human annotators and following directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17553v1">Autonomous Radiotherapy Treatment Planning Using DOLA: A Privacy-Preserving, LLM-Based Optimization Agent</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 19 pages, 5 figures, preprint
    </div>
    <details class="paper-abstract">
      Radiotherapy treatment planning is a complex and time-intensive process, often impacted by inter-planner variability and subjective decision-making. To address these challenges, we introduce Dose Optimization Language Agent (DOLA), an autonomous large language model (LLM)-based agent designed for optimizing radiotherapy treatment plans while rigorously protecting patient privacy. DOLA integrates the LLaMa3.1 LLM directly with a commercial treatment planning system, utilizing chain-of-thought prompting, retrieval-augmented generation (RAG), and reinforcement learning (RL). Operating entirely within secure local infrastructure, this agent eliminates external data sharing. We evaluated DOLA using a retrospective cohort of 18 prostate cancer patients prescribed 60 Gy in 20 fractions, comparing model sizes (8 billion vs. 70 billion parameters) and optimization strategies (No-RAG, RAG, and RAG+RL) over 10 planning iterations. The 70B model demonstrated significantly improved performance, achieving approximately 16.4% higher final scores than the 8B model. The RAG approach outperformed the No-RAG baseline by 19.8%, and incorporating RL accelerated convergence, highlighting the synergy of retrieval-based memory and reinforcement learning. Optimal temperature hyperparameter analysis identified 0.4 as providing the best balance between exploration and exploitation. This proof of concept study represents the first successful deployment of locally hosted LLM agents for autonomous optimization of treatment plans within a commercial radiotherapy planning system. By extending human-machine interaction through interpretable natural language reasoning, DOLA offers a scalable and privacy-conscious framework, with significant potential for clinical implementation and workflow improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17823v2">A General Framework to Enhance Fine-tuning-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Unlearning has been proposed to remove copyrighted and privacy-sensitive data from Large Language Models (LLMs). Existing approaches primarily rely on fine-tuning-based methods, which can be categorized into gradient ascent-based (GA-based) and suppression-based methods. However, they often degrade model utility (the ability to respond to normal prompts). In this work, we aim to develop a general framework that enhances the utility of fine-tuning-based unlearning methods. To achieve this goal, we first investigate the common property between GA-based and suppression-based methods. We unveil that GA-based methods unlearn by distinguishing the target data (i.e., the data to be removed) and suppressing related generations, which is essentially the same strategy employed by suppression-based methods. Inspired by this finding, we introduce Gated Representation UNlearning (GRUN) which has two components: a soft gate function for distinguishing target data and a suppression module using Representation Fine-tuning (ReFT) to adjust representations rather than model parameters. Experiments show that GRUN significantly improves the unlearning and utility. Meanwhile, it is general for fine-tuning-based methods, efficient and promising for sequential unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17502v1">Large Language Models (LLMs) for Source Code Analysis: applications, models and datasets</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and transformer-based architectures are increasingly utilized for source code analysis. As software systems grow in complexity, integrating LLMs into code analysis workflows becomes essential for enhancing efficiency, accuracy, and automation. This paper explores the role of LLMs for different code analysis tasks, focusing on three key aspects: 1) what they can analyze and their applications, 2) what models are used and 3) what datasets are used, and the challenges they face. Regarding the goal of this research, we investigate scholarly articles that explore the use of LLMs for source code analysis to uncover research developments, current trends, and the intellectual structure of this emerging field. Additionally, we summarize limitations and highlight essential tools, datasets, and key challenges, which could be valuable for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17500v1">Variance Control via Weight Rescaling in LLM Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      The outcome of Large Language Model (LLM) pre-training strongly depends on weight initialization and variance control strategies. Although the importance of initial variance control has been well documented in neural networks in general, the literature on initialization and management of its growth during LLM pre-training, specifically, is somewhat sparse. In this paper, we introduce the Layer Index Rescaling (LIR) weight initialization scheme, and the Target Variance Rescaling (TVR) variance control strategy. Experiments on a 1B parameter LLaMA model demonstrate that better variance management using these techniques yields substantial improvements in downstream task performance (up to 4.6% on common pre-training benchmarks) and reduces extreme activation values, thus mitigating challenges associated with quantization and low-precision training. Our code is available at: https://github.com/bluorion-com/weight_rescaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17479v1">Your voice is your voice: Supporting Self-expression through Speech Generation and LLMs in Augmented and Alternative Communication</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      In this paper, we present Speak Ease: an augmentative and alternative communication (AAC) system to support users' expressivity by integrating multimodal input, including text, voice, and contextual cues (conversational partner and emotional tone), with large language models (LLMs). Speak Ease combines automatic speech recognition (ASR), context-aware LLM-based outputs, and personalized text-to-speech technologies to enable more personalized, natural-sounding, and expressive communication. Through an exploratory feasibility study and focus group evaluation with speech and language pathologists (SLPs), we assessed Speak Ease's potential to enable expressivity in AAC. The findings highlight the priorities and needs of AAC users and the system's ability to enhance user expressivity by supporting more personalized and contextually relevant communication. This work provides insights into the use of multimodal inputs and LLM-driven features to improve AAC systems and support expressivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17439v1">LEMMA: Learning from Errors for MatheMatical Advancement in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 9 pages, 6 figures, 4 tables, under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capability in solving mathematical problems. However, existing approaches primarily focus on improving the quality of correct training data, e.g., distilling high-quality correct solutions from advanced models, neglecting the value contained in error data, potentially hindering the model's reflective ability. Though some studies attempt to leverage error data, they often involve complex mechanisms, such as Monte Carlo Tree Search (MCTS) to explore error nodes. In this work, we propose to enhance LLMs' reasoning ability by Learning from Errors for Mathematical Advancement (LEMMA). LEMMA constructs data consisting of an incorrect solution with an erroneous step and a reflection connection to a correct solution for fine-tuning. Specifically, we systematically analyze the model-generated error types and introduce an error-type grounded mistake augmentation method to collect diverse and representative errors. Correct solutions are either from fixing the errors or generating a fresh start. Through a model-aware smooth reflection connection, the erroneous solution is transferred to the correct one. By fine-tuning on the constructed dataset, the model is able to self-correct errors autonomously within the generation process without relying on external critique models. Experimental results demonstrate that LEMMA achieves significant performance improvements over other strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17422v1">V-Seek: Accelerating LLM Reasoning on Open-hardware Server-class RISC-V Platforms</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      The recent exponential growth of Large Language Models (LLMs) has relied on GPU-based systems. However, CPUs are emerging as a flexible and lower-cost alternative, especially when targeting inference and reasoning workloads. RISC-V is rapidly gaining traction in this area, given its open and vendor-neutral ISA. However, the RISC-V hardware for LLM workloads and the corresponding software ecosystem are not fully mature and streamlined, given the requirement of domain-specific tuning. This paper aims at filling this gap, focusing on optimizing LLM inference on the Sophon SG2042, the first commercially available many-core RISC-V CPU with vector processing capabilities. On two recent state-of-the-art LLMs optimized for reasoning, DeepSeek R1 Distill Llama 8B and DeepSeek R1 Distill QWEN 14B, we achieve 4.32/2.29 token/s for token generation and 6.54/3.68 token/s for prompt processing, with a speed up of up 2.9x/3.0x compared to our baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17421v1">Understanding Social Support Needs in Questions: A Hybrid Approach Integrating Semi-Supervised Learning and LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 55 pages
    </div>
    <details class="paper-abstract">
      Patients are increasingly turning to online health Q&A communities for social support to improve their well-being. However, when this support received does not align with their specific needs, it may prove ineffective or even detrimental. This necessitates a model capable of identifying the social support needs in questions. However, training such a model is challenging due to the scarcity and class imbalance issues of labeled data. To overcome these challenges, we follow the computational design science paradigm to develop a novel framework, Hybrid Approach for SOcial Support need classification (HA-SOS). HA-SOS integrates an answer-enhanced semi-supervised learning approach, a text data augmentation technique leveraging large language models (LLMs) with reliability- and diversity-aware sample selection mechanism, and a unified training process to automatically label social support needs in questions. Extensive empirical evaluations demonstrate that HA-SOS significantly outperforms existing question classification models and alternative semi-supervised learning approaches. This research contributes to the literature on social support, question classification, semi-supervised learning, and text data augmentation. In practice, our HA-SOS framework facilitates online Q&A platform managers and answerers to better understand users' social support needs, enabling them to provide timely, personalized answers and interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15242v2">BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity?</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes of set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their strengths and weaknesses in handling complexity requirements. In particular, token-space reasoning models are unrivaled in code generation but not in complexity understanding, hinting that they may not generalize well to tasks for which no reward was given at training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16402v1">The Emperor's New Clothes in Benchmarking? A Rigorous Examination of Mitigation Strategies for LLM Benchmark Data Contamination</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      Benchmark Data Contamination (BDC)-the inclusion of benchmark testing samples in the training set-has raised increasing concerns in Large Language Model (LLM) evaluation, leading to falsely inflated performance estimates and undermining evaluation reliability. To address this, researchers have proposed various mitigation strategies to update existing benchmarks, including modifying original questions or generating new ones based on them. However, a rigorous examination of the effectiveness of these mitigation strategies remains lacking. In this paper, we design a systematic and controlled pipeline along with two novel metrics-fidelity and contamination resistance-to provide a fine-grained and comprehensive assessment of existing BDC mitigation strategies. Previous assessment methods, such as accuracy drop and accuracy matching, focus solely on aggregate accuracy, often leading to incomplete or misleading conclusions. Our metrics address this limitation by emphasizing question-level evaluation result matching. Extensive experiments with 10 LLMs, 5 benchmarks, 20 BDC mitigation strategies, and 2 contamination scenarios reveal that no existing strategy significantly improves resistance over the vanilla case (i.e., no benchmark update) across all benchmarks, and none effectively balances fidelity and contamination resistance. These findings underscore the urgent need for designing more effective BDC mitigation strategies. Our code repository is available at https://github.com/ASTRAL-Group/BDC_mitigation_assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.12372v5">Is Long Context All You Need? Leveraging LLM's Extended Context for NL2SQL</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 13 pages, 6 figures, VLDB 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across a range of natural language processing tasks. In particular, improvements in reasoning abilities and the expansion of context windows have opened new avenues for leveraging these powerful models. NL2SQL is challenging in that the natural language question is inherently ambiguous, while the SQL generation requires a precise understanding of complex data schema and semantics. One approach to this semantic ambiguous problem is to provide more and sufficient contextual information. In this work, we explore the performance and the latency trade-offs of the extended context window (a.k.a., long context) offered by Google's state-of-the-art LLM (\textit{gemini-1.5-pro}). We study the impact of various contextual information, including column example values, question and SQL query pairs, user-provided hints, SQL documentation, and schema. To the best of our knowledge, this is the first work to study how the extended context window and extra contextual information can help NL2SQL generation with respect to both accuracy and latency cost. We show that long context LLMs are robust and do not get lost in the extended contextual information. Additionally, our long-context NL2SQL pipeline based on Google's \textit{gemini-pro-1.5} achieve strong performances on various benchmark datasets without finetuning and expensive self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16334v1">LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 16 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent findings reveal that much of the knowledge in a Transformer-based Large Language Model (LLM) is encoded in its feed-forward (FFN) layers, where each FNN layer can be interpreted as the summation of sub-updates, each corresponding to a weighted column vector from the FFN's value parameter matrix that often encodes human-interpretable concepts. In light of this, we hypothesize that model performance and behaviors can be further enhanced and controlled by modulating the contributions of these sub-updates based on their relevance to the input or target output style, and propose LLMBRACES, a novel and efficient method that computes relevance scores associated with value vectors in FFN layers and leverages these scores to dynamically adjust the contribution of sub-updates. By optimizing sub-update contributions, LLMBRACES refines the prediction process, leading to more accurate and reliable outputs, much like a 'brace' providing support and stability. Moreover, LLMBRACES can be extended to support conditional control over generation characteristics, such as sentiment, thereby offering fine-grained steering of LLM outputs. Extensive experiments on various LLMs-including Qwen2.5-1.5B, Llama2-7B, and Llama3-8B-demonstrate that LLMBRACES outperforms baseline approaches in both fine-tuning and zero-shot settings while requiring significantly fewer tunable parameters, up to 75% fewer compared to LoRA. Furthermore, LLMBRACES excels in sentiment-controlled generation and toxicity reduction, highlighting its potential for flexible, controlled text generation across applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00906v3">Generative AI and Perceptual Harms: Who's Suspected of using LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into a variety of writing tasks. While these tools can help people by generating ideas or producing higher quality work, like many other AI tools they may risk causing a variety of harms, disproportionately burdening historically marginalized groups. In this work, we introduce and evaluate perceptual harm, a term for the harm caused to users when others perceive or suspect them of using AI. We examined perceptual harms in three online experiments, each of which entailed human participants evaluating the profiles for fictional freelance writers. We asked participants whether they suspected the freelancers of using AI, the quality of their writing, and whether they should be hired. We found some support for perceptual harms against for certain demographic groups, but that perceptions of AI use negatively impacted writing evaluations and hiring outcomes across the board.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06061v2">GPTCoach: Towards LLM-Based Physical Activity Coaching</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Please note that the title has been updated from a previous pre-print (previously: "Supporting Physical Activity Behavior Change with LLM-Based Conversational Agents")
    </div>
    <details class="paper-abstract">
      Mobile health applications show promise for scalable physical activity promotion but are often insufficiently personalized. In contrast, health coaching offers highly personalized support but can be prohibitively expensive and inaccessible. This study draws inspiration from health coaching to explore how large language models (LLMs) might address personalization challenges in mobile health. We conduct formative interviews with 12 health professionals and 10 potential coaching recipients to develop design principles for an LLM-based health coach. We then built GPTCoach, a chatbot that implements the onboarding conversation from an evidence-based coaching program, uses conversational strategies from motivational interviewing, and incorporates wearable data to create personalized physical activity plans. In a lab study with 16 participants using three months of historical data, we find promising evidence that GPTCoach gathers rich qualitative information to offer personalized support, with users feeling comfortable sharing concerns. We conclude with implications for future research on LLM-based physical activity support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.20089v2">Robust LLM safeguarding via refusal feature adversarial training</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16219v1">Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of large language models (LLMs) typically relies on massive computational resources and extensive datasets, limiting accessibility for resource-constrained settings. Our study investigates the potential of reinforcement learning (RL) to improve reasoning in small LLMs, focusing on a 1.5-billion-parameter model, DeepSeek-R1-Distill-Qwen-1.5B, under strict constraints: training on 4 NVIDIA A40 GPUs (48 GB VRAM each) within 24 hours. Adapting the Group Relative Policy Optimization (GRPO) algorithm and curating a compact, high-quality mathematical reasoning dataset, we conducted three experiments to explore model behavior and performance. Our results demonstrate rapid reasoning gains - e.g., AMC23 accuracy rising from 63% to 80% and AIME24 reaching 46.7%, surpassing o1-preview - using only 7,000 samples and a $42 training cost, compared to thousands of dollars for baseline models. However, challenges such as optimization instability and length constraints emerged with prolonged training. These findings highlight the efficacy of RL-based fine-tuning for small LLMs, offering a cost-effective alternative to large-scale approaches. We release our code and datasets as open-source resources, providing insights into trade-offs and laying a foundation for scalable, reasoning-capable LLMs in resource-limited environments. All are available at https://github.com/knoveleng/open-rs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07058v3">Using Contextually Aligned Online Reviews to Measure LLMs' Performance Disparities Across Language Varieties</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted by 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL), theme track
    </div>
    <details class="paper-abstract">
      A language can have different varieties. These varieties can affect the performance of natural language processing (NLP) models, including large language models (LLMs), which are often trained on data from widely spoken varieties. This paper introduces a novel and cost-effective approach to benchmark model performance across language varieties. We argue that international online review platforms, such as Booking.com, can serve as effective data sources for constructing datasets that capture comments in different language varieties from similar real-world scenarios, like reviews for the same hotel with the same rating using the same language (e.g., Mandarin Chinese) but different language varieties (e.g., Taiwan Mandarin, Mainland Mandarin). To prove this concept, we constructed a contextually aligned dataset comprising reviews in Taiwan Mandarin and Mainland Mandarin and tested six LLMs in a sentiment analysis task. Our results show that LLMs consistently underperform in Taiwan Mandarin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16212v1">MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive progress in mathematical reasoning. While data augmentation is promising to enhance mathematical problem-solving ability, current approaches are predominantly limited to instance-level modifications-such as rephrasing or generating syntactic variations-which fail to capture and leverage the intrinsic relational structures inherent in mathematical knowledge. Inspired by human learning processes, where mathematical proficiency develops through systematic exposure to interconnected concepts, we introduce MathFusion, a novel framework that enhances mathematical reasoning through cross-problem instruction synthesis. MathFusion implements this through three fusion strategies: (1) sequential fusion, which chains related problems to model solution dependencies; (2) parallel fusion, which combines analogous problems to reinforce conceptual understanding; and (3) conditional fusion, which creates context-aware selective problems to enhance reasoning flexibility. By applying these strategies, we generate a new dataset, \textbf{MathFusionQA}, followed by fine-tuning models (DeepSeekMath-7B, Mistral-7B, Llama3-8B) on it. Experimental results demonstrate that MathFusion achieves substantial improvements in mathematical reasoning while maintaining high data efficiency, boosting performance by 18.0 points in accuracy across diverse benchmarks while requiring only 45K additional synthetic instructions, representing a substantial improvement over traditional single-instruction approaches. Our datasets, models, and code are publicly available at https://github.com/QizhiPei/mathfusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19146v4">Puzzle: Distillation-Based NAS for Inference-Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer remarkable capabilities, yet their high inference costs restrict wider adoption. While increasing parameter counts improves accuracy, it also broadens the gap between state-of-the-art capabilities and practical deployability. We present Puzzle, a hardware-aware framework that accelerates the inference of LLMs while preserving their capabilities. Using neural architecture search (NAS) at a large-scale, Puzzle optimizes models with tens of billions of parameters. Our approach utilizes blockwise local knowledge distillation (BLD) for parallel architecture exploration and employs mixed-integer programming for precise constraint optimization. We showcase our framework's impact via Llama-3.1-Nemotron-51B-Instruct (Nemotron-51B), a publicly available model derived from Llama-3.1-70B-Instruct. Nemotron-51B achieves a 2.17x inference throughput speedup, fitting on a single NVIDIA H100 GPU while retaining 98.4% of the original model's benchmark accuracies. Notably, it is the most accurate model supporting single H100 GPU inference with large batch sizes, despite training on only 45B tokens, far fewer than the 15T used to train Llama-70B. Lastly, we derive Llama-3.3-Nemotron-49B-Super-Base to demonstrate Puzzle can retain long-context and that lightweight alignment on these derived models allows them to surpass the parent model in specific capabilities. Our work establishes that powerful LLM models can be optimized for efficient deployment with only negligible loss in quality, underscoring that inference performance, not parameter count alone, should guide model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16163v1">SpeCache: Speculative Key-Value Caching for Efficient Generation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) have already achieved remarkable results on long-text tasks, but the limited GPU memory (VRAM) resources struggle to accommodate the linearly growing demand for key-value (KV) cache as the sequence length increases, which has become a bottleneck for the application of LLMs on long sequences. Existing KV cache compression methods include eviction, merging, or quantization of the KV cache to reduce its size. However, compression results in irreversible information forgetting, potentially affecting the accuracy of subsequent decoding. In this paper, we propose SpeCache, which takes full advantage of the large and easily expandable CPU memory to offload the complete KV cache, and dynamically fetches KV pairs back in each decoding step based on their importance measured by low-bit KV cache copy in VRAM. To avoid inference latency caused by CPU-GPU communication, SpeCache speculatively predicts the KV pairs that the next token might attend to, allowing us to prefetch them before the next decoding step which enables parallelization of prefetching and computation. Experiments on LongBench and Needle-in-a-Haystack benchmarks verify that SpeCache effectively reduces VRAM usage while avoiding information forgetting for long sequences without re-training, even with a 10x high KV cache compression ratio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16144v1">Unify and Triumph: Polyglot, Diverse, and Self-Consistent Generation of Unit Tests with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based test generation has gained attention in software engineering, yet most studies evaluate LLMs' ability to generate unit tests in a single attempt for a given language, missing the opportunity to leverage LLM diversity for more robust testing. This paper introduces PolyTest, a novel approach that enhances test generation by exploiting polyglot and temperature-controlled diversity. PolyTest systematically leverages these properties in two complementary ways: (1) Cross-lingual test generation, where tests are generated in multiple languages at zero temperature and then unified; (2) Diverse test sampling, where multiple test sets are generated within the same language at a higher temperature before unification. A key insight is that LLMs can generate diverse yet contradicting tests -- same input, different expected outputs -- across languages and generations. PolyTest mitigates inconsistencies by unifying test sets, fostering self-consistency and improving overall test quality. Unlike single-language or single-attempt approaches, PolyTest enhances testing without requiring on-the-fly execution, making it particularly beneficial for weaker-performing languages. We evaluate PolyTest on Llama3-70B, GPT-4o, and GPT-3.5 using EvalPlus, generating tests in five languages (Java, C, Python, JavaScript, and a CSV-based format) at temperature 0 and sampling multiple sets at temperature 1. We observe that LLMs frequently generate contradicting tests across settings, and that PolyTest significantly improves test quality across all considered metrics -- number of tests, passing rate, statement/branch coverage (up to +9.01%), and mutation score (up to +11.23%). Finally, PolyTest outperforms Pynguin in test generation, passing rate, and mutation score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16071v1">Tuning LLMs by RAG Principles: Towards LLM-native Memory</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Memory, additional information beyond the training of large language models (LLMs), is crucial to various real-world applications, such as personal assistant. The two mainstream solutions to incorporate memory into the generation process are long-context LLMs and retrieval-augmented generation (RAG). In this paper, we first systematically compare these two types of solutions on three renovated/new datasets and show that (1) long-context solutions, although more expensive, shall be easier to capture the big picture and better answer queries which require considering the memory as a whole; and (2) when the queries concern specific information, RAG solutions shall be more competitive especially when the keywords can be explicitly matched. Therefore, we propose a novel method RAG-Tuned-LLM which fine-tunes a relative small (e.g., 7B) LLM using the data generated following the RAG principles, so it can combine the advantages of both solutions. Extensive experiments on three datasets demonstrate that RAG-Tuned-LLM can beat long-context LLMs and RAG methods across a wide range of query types.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16040v1">Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Recently, Test-Time Scaling Large Language Models (LLMs), such as DeepSeek-R1 and OpenAI o1, have demonstrated exceptional capabilities across various domains and tasks, particularly in reasoning. While these models have shown impressive performance on general language tasks, their effectiveness in specialized fields like legal remains unclear. To address this, we present a preliminary evaluation of LLMs in various legal scenarios, covering both Chinese and English legal tasks. Our analysis includes 9 LLMs and 17 legal tasks, with a focus on newly published and more complex challenges such as multi-defendant legal judgments and legal argument reasoning. Our findings indicate that, despite DeepSeek-R1 and OpenAI o1 being among the most powerful models, their legal reasoning capabilities are still lacking. Specifically, these models score below 80\% on seven Chinese legal reasoning tasks and below 80\% on two English legal reasoning tasks. This suggests that, even among the most advanced reasoning models, legal reasoning abilities remain underdeveloped.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16024v1">The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently transformed from text-based assistants to autonomous agents capable of planning, reasoning, and iteratively improving their actions. While numerical reward signals and verifiers can effectively rank candidate actions, they often provide limited contextual guidance. In contrast, natural language feedback better aligns with the generative capabilities of LLMs, providing richer and more actionable suggestions. However, parsing and implementing this feedback effectively can be challenging for LLM-based agents. In this work, we introduce Critique-Guided Improvement (CGI), a novel two-player framework, comprising an actor model that explores an environment and a critic model that generates detailed nature language feedback. By training the critic to produce fine-grained assessments and actionable revisions, and the actor to utilize these critiques, our approach promotes more robust exploration of alternative strategies while avoiding local optima. Experiments in three interactive environments show that CGI outperforms existing baselines by a substantial margin. Notably, even a small critic model surpasses GPT-4 in feedback quality. The resulting actor achieves state-of-the-art performance, demonstrating the power of explicit iterative guidance to enhance decision-making in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.01503v2">LumosCore: Highly Scalable LLM Clusters with Optical Interconnect</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      We propose \emph{LumosCore} to build high-bandwidth and large-scale data center networks for LLM jobs. By replacing the core-layer electrical packet switches by optical circuit switches, \emph{LumosCore} could achieves $2\times$ increase in bandwidth or $8\times$ increase in network size. We offer the detailed design of \emph{LumosCore} at both deployment stage and running stage. At deployment stage, we propose Interleaved Wiring, which is compatible with all possible logical topologies. At running stage, we design polynomial-time algorithms for GPU placement, logical topology generating and OCS reconfiguration to minimize network contention and reduce impact to scheduled jobs. We evaluate \emph{LumosCore} using both testbed experiments and large-scale simulation. Compared to traditional hybrid optical/electrical architectures, \emph{LumosCore} increases the end-to-end training throughput by up to 39.5\% on a 128-node testbed. Compared to the state-of-art Clos architectures, \emph{LumosCore} reduces the average job completion time by up to 34.1\% in a 16k simulation platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01082v4">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Added acknowledgements and minor rewordings to make the intro/abstract more readable. No major change in length or content
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its significant impact on improving text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15904v1">From Structured Prompts to Open Narratives: Measuring Gender Bias in LLMs Through Open-Ended Storytelling</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing, yet concerns persist regarding their tendency to reflect or amplify social biases present in their training data. This study introduces a novel evaluation framework to uncover gender biases in LLMs, focusing on their occupational narratives. Unlike previous methods relying on structured scenarios or carefully crafted prompts, our approach leverages free-form storytelling to reveal biases embedded in the models. Systematic analyses show an overrepresentation of female characters across occupations in six widely used LLMs. Additionally, our findings reveal that LLM-generated occupational gender rankings align more closely with human stereotypes than actual labor statistics. These insights underscore the need for balanced mitigation strategies to ensure fairness while avoiding the reinforcement of new stereotypes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.02901v2">A Comprehensive Survey on Process-Oriented Automatic Text Summarization with Exploration of LLM-Based Methods</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Automatic Text Summarization (ATS), utilizing Natural Language Processing (NLP) algorithms, aims to create concise and accurate summaries, thereby significantly reducing the human effort required in processing large volumes of text. ATS has drawn considerable interest in both academic and industrial circles. Many studies have been conducted in the past to survey ATS methods; however, they generally lack practicality for real-world implementations, as they often categorize previous methods from a theoretical standpoint. Moreover, the advent of Large Language Models (LLMs) has altered conventional ATS methods. In this survey, we aim to 1) provide a comprehensive overview of ATS from a ``Process-Oriented Schema'' perspective, which is best aligned with real-world implementations; 2) comprehensively review the latest LLM-based ATS works; and 3) deliver an up-to-date survey of ATS, bridging the two-year gap in the literature. To the best of our knowledge, this is the first survey to specifically investigate LLM-based ATS methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15885v1">Human or LLM? A Comparative Study on Accessible Code Generation Capability</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Web accessibility is essential for inclusive digital experiences, yet the accessibility of LLM-generated code remains underexplored. This paper presents an empirical study comparing the accessibility of web code generated by GPT-4o and Qwen2.5-Coder-32B-Instruct-AWQ against human-written code. Results show that LLMs often produce more accessible code, especially for basic features like color contrast and alternative text, but struggle with complex issues such as ARIA attributes. We also assess advanced prompting strategies (Zero-Shot, Few-Shot, Self-Criticism), finding they offer some gains but are limited. To address these gaps, we introduce FeedA11y, a feedback-driven ReAct-based approach that significantly outperforms other methods in improving accessibility. Our work highlights the promise of LLMs for accessible code generation and emphasizes the need for feedback-based techniques to address persistent challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15871v1">MASH-VLM: Mitigating Action-Scene Hallucination in Video-LLMs through Disentangled Spatial-Temporal Representations</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted for CVPR 2025
    </div>
    <details class="paper-abstract">
      In this work, we tackle action-scene hallucination in Video Large Language Models (Video-LLMs), where models incorrectly predict actions based on the scene context or scenes based on observed actions. We observe that existing Video-LLMs often suffer from action-scene hallucination due to two main factors. First, existing Video-LLMs intermingle spatial and temporal features by applying an attention operation across all tokens. Second, they use the standard Rotary Position Embedding (RoPE), which causes the text tokens to overemphasize certain types of tokens depending on their sequential orders. To address these issues, we introduce MASH-VLM, Mitigating Action-Scene Hallucination in Video-LLMs through disentangled spatial-temporal representations. Our approach includes two key innovations: (1) DST-attention, a novel attention mechanism that disentangles the spatial and temporal tokens within the LLM by using masked attention to restrict direct interactions between the spatial and temporal tokens; (2) Harmonic-RoPE, which extends the dimensionality of the positional IDs, allowing the spatial and temporal tokens to maintain balanced positions relative to the text tokens. To evaluate the action-scene hallucination in Video-LLMs, we introduce the UNSCENE benchmark with 1,320 videos and 4,078 QA pairs. Extensive experiments demonstrate that MASH-VLM achieves state-of-the-art results on the UNSCENE benchmark, as well as on existing video understanding benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07627v2">Automatic Curriculum Expert Iteration for Reliable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Hallucinations (i.e., generating plausible but inaccurate content) and laziness (i.e. excessive refusals or defaulting to "I don't know") persist as major challenges in LLM reasoning. Current efforts to reduce hallucinations primarily focus on factual errors in knowledge-grounded tasks, often neglecting hallucinations related to faulty reasoning. Meanwhile, some approaches render LLMs overly conservative, limiting their problem-solving capabilities. To mitigate hallucination and laziness in reasoning tasks, we propose Automatic Curriculum Expert Iteration (Auto-CEI) to enhance LLM reasoning and align responses to the model's capabilities--assertively answering within its limits and declining when tasks exceed them. In our method, Expert Iteration explores the reasoning trajectories near the LLM policy, guiding incorrect paths back on track to reduce compounding errors and improve robustness; it also promotes appropriate "I don't know" responses after sufficient reasoning attempts. The curriculum automatically adjusts rewards, incentivizing extended reasoning before acknowledging incapability, thereby pushing the limits of LLM reasoning and aligning its behaviour with these limits. We compare Auto-CEI with various SOTA baselines across logical reasoning, mathematics, and planning tasks, where Auto-CEI achieves superior alignment by effectively balancing assertiveness and conservativeness. The code is available at https://github.com/SalesforceAIResearch/Auto-CEI .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15838v1">Enhancing LLM Code Generation with Ensembles: A Similarity-Based Selection Approach</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Ensemble learning has been widely used in machine learning to improve model robustness, accuracy, and generalization, but has not yet been applied to code generation tasks with large language models (LLMs). We propose an ensemble approach for LLMs in code generation. Instead of relying on the output of a single model, we generate multiple candidate programs from different LLMs and apply a structured voting mechanism to select the most reliable solution. For voting, we compute syntactic and semantic similarity using CodeBLEU and behavioral equivalence using CrossHair's differential behavior analysis. By aggregating these similarity scores, we select the program that best aligns with the consensus among the candidates. We show through experiments that our ensemble approach consistently outperforms standalone LLMs on the well-known HumanEval and the more challenging LiveCodeBench datasets, achieving an accuracy of 90.2% and 50.2%, respectively, on the two datasets. In comparison, the best-performing LLM (GPT-4o) has an accuracy of 83.5% and 43.4%, respectively. Furthermore, even when restricted to free open-source models, our method achieves an accuracy of 80.5% and 41.6%, respectively, demonstrating the viability of our approach in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.13218v2">ULTRA: Unleash LLMs' Potential for Event Argument Extraction through Hierarchical Modeling and Pair-wise Self-Refinement</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 ACL'24 Findings
    </div>
    <details class="paper-abstract">
      Structural extraction of events within discourse is critical since it avails a deeper understanding of communication patterns and behavior trends. Event argument extraction (EAE), at the core of event-centric understanding, is the task of identifying role-specific text spans (i.e., arguments) for a given event. Document-level EAE (DocEAE) focuses on arguments that are scattered across an entire document. In this work, we explore open-source Large Language Models (LLMs) for DocEAE, and propose ULTRA, a hierarchical framework that extracts event arguments more cost-effectively. Further, it alleviates the positional bias issue intrinsic to LLMs. ULTRA sequentially reads text chunks of a document to generate a candidate argument set, upon which non-pertinent candidates are dropped through self-refinement. We introduce LEAFER to address the challenge LLMs face in locating the exact boundary of an argument. ULTRA outperforms strong baselines, including strong supervised models and ChatGPT, by 9.8% when evaluated by Exact Match (EM).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15793v1">DNA Bench: When Silence is Smarter -- Benchmarking Over-Reasoning in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Test-time scaling has significantly improved large language model performance, enabling deeper reasoning to solve complex problems. However, this increased reasoning capability also leads to excessive token generation and unnecessary problem-solving attempts. We introduce Don\'t Answer Bench (DNA Bench), a new benchmark designed to evaluate LLMs ability to robustly understand the tricky reasoning triggers and avoiding unnecessary generation. DNA Bench consists of 150 adversarially designed prompts that are easy for humans to understand and respond to, but surprisingly not for many of the recent prominent LLMs. DNA Bench tests models abilities across different capabilities, such as instruction adherence, hallucination avoidance, redundancy filtering, and unanswerable question recognition. We evaluate reasoning LLMs (RLMs), including DeepSeek-R1, OpenAI O3-mini, Claude-3.7-sonnet and compare them against a powerful non-reasoning model, e.g., GPT-4o. Our experiments reveal that RLMs generate up to 70x more tokens than necessary, often failing at tasks that simpler non-reasoning models handle efficiently with higher accuracy. Our findings underscore the need for more effective training and inference strategies in RLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15783v1">Grammar and Gameplay-aligned RL for Game Description Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Game Description Generation (GDG) is the task of generating a game description written in a Game Description Language (GDL) from natural language text. Previous studies have explored generation methods leveraging the contextual understanding capabilities of Large Language Models (LLMs); however, accurately reproducing the game features of the game descriptions remains a challenge. In this paper, we propose reinforcement learning-based fine-tuning of LLMs for GDG (RLGDG). Our training method simultaneously improves grammatical correctness and fidelity to game concepts by introducing both grammar rewards and concept rewards. Furthermore, we adopt a two-stage training strategy where Reinforcement Learning (RL) is applied following Supervised Fine-Tuning (SFT). Experimental results demonstrate that our proposed method significantly outperforms baseline methods using SFT alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15772v1">Detecting LLM-Written Peer Reviews</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 26 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Editors of academic journals and program chairs of conferences require peer reviewers to write their own reviews. However, there is growing concern about the rise of lazy reviewing practices, where reviewers use large language models (LLMs) to generate reviews instead of writing them independently. Existing tools for detecting LLM-generated content are not designed to differentiate between fully LLM-generated reviews and those merely polished by an LLM. In this work, we employ a straightforward approach to identify LLM-generated reviews - doing an indirect prompt injection via the paper PDF to ask the LLM to embed a watermark. Our focus is on presenting watermarking schemes and statistical tests that maintain a bounded family-wise error rate, when a venue evaluates multiple reviews, with a higher power as compared to standard methods like Bonferroni correction. These guarantees hold without relying on any assumptions about human-written reviews. We also consider various methods for prompt injection including font embedding and jailbreaking. We evaluate the effectiveness and various tradeoffs of these methods, including different reviewer defenses. We find a high success rate in the embedding of our watermarks in LLM-generated reviews across models. We also find that our approach is resilient to common reviewer defenses, and that the bounds on error rates in our statistical tests hold in practice while having the power to flag LLM-generated reviews, while Bonferroni correction is infeasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21474v2">Estimating Causal Effects of Text Interventions Leveraging LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Quantifying the effects of textual interventions in social systems, such as reducing anger in social media posts to see its impact on engagement, is challenging. Real-world interventions are often infeasible, necessitating reliance on observational data. Traditional causal inference methods, typically designed for binary or discrete treatments, are inadequate for handling the complex, high-dimensional textual data. This paper addresses these challenges by proposing CausalDANN, a novel approach to estimate causal effects using text transformations facilitated by large language models (LLMs). Unlike existing methods, our approach accommodates arbitrary textual interventions and leverages text-level classifiers with domain adaptation ability to produce robust effect estimates against domain shifts, even when only the control group is observed. This flexibility in handling various text interventions is a key advancement in causal estimation for textual data, offering opportunities to better understand human behaviors and develop effective interventions within social systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05132v3">3D-GRAND: A Million-Scale Dataset for 3D-LLMs with Better Grounding and Less Hallucination</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 CVPR 2025. Project website: https://3d-grand.github.io
    </div>
    <details class="paper-abstract">
      The integration of language and 3D perception is crucial for embodied agents and robots that comprehend and interact with the physical world. While large language models (LLMs) have demonstrated impressive language understanding and generation capabilities, their adaptation to 3D environments (3D-LLMs) remains in its early stages. A primary challenge is a lack of large-scale datasets with dense grounding between language and 3D scenes. We introduce 3D-GRAND, a pioneering large-scale dataset comprising 40,087 household scenes paired with 6.2 million densely-grounded scene-language instructions. Our results show that instruction tuning with 3D-GRAND significantly enhances grounding capabilities and reduces hallucinations in 3D-LLMs. As part of our contributions, we propose a comprehensive benchmark 3D-POPE to systematically evaluate hallucination in 3D-LLMs, enabling fair comparisons of models. Our experiments highlight a scaling effect between dataset size and 3D-LLM performance, emphasizing the importance of large-scale 3D-text datasets for embodied AI research. Our results demonstrate early signals for effective sim-to-real transfer, indicating that models trained on large synthetic data can perform well on real-world 3D scans. Through 3D-GRAND and 3D-POPE, we aim to equip the embodied AI community with resources and insights to lead to more reliable and better-grounded 3D-LLMs. Project website: https://3d-grand.github.io
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05868v3">EmojiPrompt: Generative Prompt Obfuscation for Privacy-Preserving Communication with Cloud-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted to the 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL 2025)
    </div>
    <details class="paper-abstract">
      Cloud-based Large Language Models (LLMs) such as ChatGPT have become increasingly integral to daily operations. Nevertheless, they also introduce privacy concerns: firstly, numerous studies underscore the risks to user privacy posed by jailbreaking cloud-based LLMs; secondly, the LLM service providers have access to all user data, which deters individuals from confidently utilizing such services. To address such concerns, we propose a simple yet effective paradigm, EmojiPrompt, to protect user privacy. At its core, EmojiPrompt performs generative transformation, obfuscating private data within prompts with linguistic and non-linguistic elements before submitting them to cloud-based LLMs. We evaluate EmojiPrompt's performance across 8 datasets from various domains. We also propose simulated inference attacks to assess EmojiPrompt's ability to preserve user privacy. The results demonstrate that EmojiPrompt effectively obfuscates user private data, while largely maintaining, or even enhancing, performances compared to the unobfuscated version. Furthermore, EmojiPrompt's atomic-level obfuscation allows it to function exclusively with cloud-based LLMs. For source code, please refer to: https://github.com/agiresearch/EmojiCrypt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16674v1">Through the LLM Looking Glass: A Socratic Self-Assessment of Donkeys, Elephants, and Markets</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      While detecting and avoiding bias in LLM-generated text is becoming increasingly important, media bias often remains subtle and subjective, making it particularly difficult to identify and mitigate. In this study, we assess media bias in LLM-generated content and LLMs' ability to detect subtle ideological bias. We conduct this evaluation using two datasets, PoliGen and EconoLex, covering political and economic discourse, respectively. We evaluate eight widely used LLMs by prompting them to generate articles and analyze their ideological preferences via self-assessment. By using self-assessment, the study aims to directly measure the models' biases rather than relying on external interpretations, thereby minimizing subjective judgments about media bias. Our results reveal a consistent preference of Democratic over Republican positions across all models. Conversely, in economic topics, biases vary among Western LLMs, while those developed in China lean more strongly toward socialism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10838v2">Who Relies More on World Knowledge and Bias for Syntactic Ambiguity Resolution: Humans or LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 Accepted at NAACL 2025 main
    </div>
    <details class="paper-abstract">
      This study explores how recent large language models (LLMs) navigate relative clause attachment {ambiguity} and use world knowledge biases for disambiguation in six typologically diverse languages: English, Chinese, Japanese, Korean, Russian, and Spanish. We describe the process of creating a novel dataset -- MultiWho -- for fine-grained evaluation of relative clause attachment preferences in ambiguous and unambiguous contexts. Our experiments with three LLMs indicate that, contrary to humans, LLMs consistently exhibit a preference for local attachment, displaying limited responsiveness to syntactic variations or language-specific attachment patterns. Although LLMs performed well in unambiguous cases, they rigidly prioritized world knowledge biases, lacking the flexibility of human language processing. These findings highlight the need for more diverse, pragmatically nuanced multilingual training to improve LLMs' handling of complex structures and human-like comprehension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.06586v2">ContextGPT: Infusing LLMs Knowledge into Neuro-Symbolic Activity Recognition Models</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Context-aware Human Activity Recognition (HAR) is a hot research area in mobile computing, and the most effective solutions in the literature are based on supervised deep learning models. However, the actual deployment of these systems is limited by the scarcity of labeled data that is required for training. Neuro-Symbolic AI (NeSy) provides an interesting research direction to mitigate this issue, by infusing common-sense knowledge about human activities and the contexts in which they can be performed into HAR deep learning classifiers. Existing NeSy methods for context-aware HAR rely on knowledge encoded in logic-based models (e.g., ontologies) whose design, implementation, and maintenance to capture new activities and contexts require significant human engineering efforts, technical knowledge, and domain expertise. Recent works show that pre-trained Large Language Models (LLMs) effectively encode common-sense knowledge about human activities. In this work, we propose ContextGPT: a novel prompt engineering approach to retrieve from LLMs common-sense knowledge about the relationship between human activities and the context in which they are performed. Unlike ontologies, ContextGPT requires limited human effort and expertise. An extensive evaluation carried out on two public datasets shows how a NeSy model obtained by infusing common-sense knowledge from ContextGPT is effective in data scarcity scenarios, leading to similar (and sometimes better) recognition rates than logic-based approaches with a fraction of the effort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16585v1">Distributed LLMs and Multimodal Large Language Models: A Survey on Advances, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      Language models (LMs) are machine learning models designed to predict linguistic patterns by estimating the probability of word sequences based on large-scale datasets, such as text. LMs have a wide range of applications in natural language processing (NLP) tasks, including autocomplete and machine translation. Although larger datasets typically enhance LM performance, scalability remains a challenge due to constraints in computational power and resources. Distributed computing strategies offer essential solutions for improving scalability and managing the growing computational demand. Further, the use of sensitive datasets in training and deployment raises significant privacy concerns. Recent research has focused on developing decentralized techniques to enable distributed training and inference while utilizing diverse computational resources and enabling edge AI. This paper presents a survey on distributed solutions for various LMs, including large language models (LLMs), vision language models (VLMs), multimodal LLMs (MLLMs), and small language models (SLMs). While LLMs focus on processing and generating text, MLLMs are designed to handle multiple modalities of data (e.g., text, images, and audio) and to integrate them for broader applications. To this end, this paper reviews key advancements across the MLLM pipeline, including distributed training, inference, fine-tuning, and deployment, while also identifying the contributions, limitations, and future areas of improvement. Further, it categorizes the literature based on six primary focus areas of decentralization. Our analysis describes gaps in current methodologies for enabling distributed solutions for LMs and outline future research directions, emphasizing the need for novel solutions to enhance the robustness and applicability of distributed LMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16561v1">FutureGen: LLM-RAG Approach to Generate the Future Work of Scientific Article</a></div>
    <div class="paper-meta">
      📅 2025-03-20
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The future work section of a scientific article outlines potential research directions by identifying gaps and limitations of a current study. This section serves as a valuable resource for early-career researchers seeking unexplored areas and experienced researchers looking for new projects or collaborations. In this study, we generate future work suggestions from key sections of a scientific article alongside related papers and analyze how the trends have evolved. We experimented with various Large Language Models (LLMs) and integrated Retrieval-Augmented Generation (RAG) to enhance the generation process. We incorporate a LLM feedback mechanism to improve the quality of the generated content and propose an LLM-as-a-judge approach for evaluation. Our results demonstrated that the RAG-based approach with LLM feedback outperforms other methods evaluated through qualitative and quantitative metrics. Moreover, we conduct a human evaluation to assess the LLM as an extractor and judge. The code and dataset for this project are here, code: HuggingFace
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16416v1">Survey on Evaluation of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      The emergence of LLM-based agents represents a paradigm shift in AI, enabling autonomous systems to plan, reason, use tools, and maintain memory while interacting with dynamic environments. This paper provides the first comprehensive survey of evaluation methodologies for these increasingly capable agents. We systematically analyze evaluation benchmarks and frameworks across four critical dimensions: (1) fundamental agent capabilities, including planning, tool use, self-reflection, and memory; (2) application-specific benchmarks for web, software engineering, scientific, and conversational agents; (3) benchmarks for generalist agents; and (4) frameworks for evaluating agents. Our analysis reveals emerging trends, including a shift toward more realistic, challenging evaluations with continuously updated benchmarks. We also identify critical gaps that future research must address-particularly in assessing cost-efficiency, safety, and robustness, and in developing fine-grained, and scalable evaluation methods. This survey maps the rapidly evolving landscape of agent evaluation, reveals the emerging trends in the field, identifies current limitations, and proposes directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14924v1">UTFix: Change Aware Unit Test Repairing using LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 26 pages, International Conference on Object-oriented Programming, Systems, Languages, and Applications (OOPSLA) 2025
    </div>
    <details class="paper-abstract">
      Software updates, including bug repair and feature additions, are frequent in modern applications but they often leave test suites outdated, resulting in undetected bugs and increased chances of system failures. A recent study by Meta revealed that 14%-22% of software failures stem from outdated tests that fail to reflect changes in the codebase. This highlights the need to keep tests in sync with code changes to ensure software reliability. In this paper, we present UTFix, a novel approach for repairing unit tests when their corresponding focal methods undergo changes. UTFix addresses two critical issues: assertion failure and reduced code coverage caused by changes in the focal method. Our approach leverages language models to repair unit tests by providing contextual information such as static code slices, dynamic code slices, and failure messages. We evaluate UTFix on our generated synthetic benchmarks (Tool-Bench), and real-world benchmarks. Tool- Bench includes diverse changes from popular open-source Python GitHub projects, where UTFix successfully repaired 89.2% of assertion failures and achieved 100% code coverage for 96 tests out of 369 tests. On the real-world benchmarks, UTFix repairs 60% of assertion failures while achieving 100% code coverage for 19 out of 30 unit tests. To the best of our knowledge, this is the first comprehensive study focused on unit test in evolving Python projects. Our contributions include the development of UTFix, the creation of Tool-Bench and real-world benchmarks, and the demonstration of the effectiveness of LLM-based methods in addressing unit test failures due to software evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14887v1">Pseudo-Relevance Feedback Can Improve Zero-Shot LLM-Based Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Pseudo-relevance feedback (PRF) refines queries by leveraging initially retrieved documents to improve retrieval effectiveness. In this paper, we investigate how large language models (LLMs) can facilitate PRF for zero-shot LLM-based dense retrieval, extending the recently proposed PromptReps method. Specifically, our approach uses LLMs to extract salient passage features-such as keywords and summaries-from top-ranked documents, which are then integrated into PromptReps to produce enhanced query representations. Experiments on passage retrieval benchmarks demonstrate that incorporating PRF significantly boosts retrieval performance. Notably, smaller rankers with PRF can match the effectiveness of larger rankers without PRF, highlighting PRF's potential to improve LLM-driven search while maintaining an efficient balance between effectiveness and resource usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14882v1">Communication-Efficient Distributed On-Device LLM Inference Over Wireless Networks</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 arXiv admin note: text overlap with arXiv:2502.12559
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable success across various application domains, but their enormous sizes and computational demands pose significant challenges for deployment on resource-constrained edge devices. To address this issue, we propose a novel distributed on-device LLM inference framework that leverages tensor parallelism to partition the neural network tensors (e.g., weight matrices) of one LLM across multiple edge devices for collaborative inference. A key challenge in tensor parallelism is the frequent all-reduce operations for aggregating intermediate layer outputs across participating devices, which incurs significant communication overhead. To alleviate this bottleneck, we propose an over-the-air computation (AirComp) approach that harnesses the analog superposition property of wireless multiple-access channels to perform fast all-reduce steps. To utilize the heterogeneous computational capabilities of edge devices and mitigate communication distortions, we investigate a joint model assignment and transceiver optimization problem to minimize the average transmission error. The resulting mixed-timescale stochastic non-convex optimization problem is intractable, and we propose an efficient two-stage algorithm to solve it. Moreover, we prove that the proposed algorithm converges almost surely to a stationary point of the original problem. Comprehensive simulation results will show that the proposed framework outperforms existing benchmark schemes, achieving up to 5x inference speed acceleration and improving inference accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11843v5">From Commands to Prompts: LLM-based Semantic File System for AIOS</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted by International Conference on Learning Representations 2025(ICLR2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in the development of intelligent applications and systems such as LLM-based agents and agent operating systems (AIOS). However, when these applications and systems interact with the underlying file system, the file system still remains the traditional paradigm: reliant on manual navigation through precise commands. This paradigm poses a bottleneck to the usability of these systems as users are required to navigate complex folder hierarchies and remember cryptic file names. To address this limitation, we propose an LLM-based semantic file system ( LSFS ) for prompt-driven file management. Unlike conventional approaches, LSFS incorporates LLMs to enable users or agents to interact with files through natural language prompts, facilitating semantic file management. At the macro-level, we develop a comprehensive API set to achieve semantic file management functionalities, such as semantic file retrieval, file update monitoring and summarization, and semantic file rollback). At the micro-level, we store files by constructing semantic indexes for them, design and implement syscalls of different semantic operations (e.g., CRUD, group by, join) powered by vector database. Our experiments show that LSFS offers significant improvements over traditional file systems in terms of user convenience, the diversity of supported functions, and the accuracy and efficiency of file operations. Additionally, with the integration of LLM, our system enables more intelligent file management tasks, such as content summarization and version comparison, further enhancing its capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09956v2">DeepSeek-Inspired Exploration of RL-based LLMs and Synergy with Wireless Networks: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 25 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL)-based large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, have gained significant attention for their exceptional capabilities in natural language processing and multimodal data understanding. Meanwhile, the rapid expansion of information services has driven the growing need for intelligence, efficient, and adaptable wireless networks. Wireless networks require the empowerment of RL-based LLMs while these models also benefit from wireless networks to broaden their application scenarios. Specifically, RL-based LLMs can enhance wireless communication systems through intelligent resource allocation, adaptive network optimization, and real-time decision-making. Conversely, wireless networks provide a vital infrastructure for the efficient training, deployment, and distributed inference of RL-based LLMs, especially in decentralized and edge computing environments. This mutual empowerment highlights the need for a deeper exploration of the interplay between these two domains. We first review recent advancements in wireless communications, highlighting the associated challenges and potential solutions. We then discuss the progress of RL-based LLMs, focusing on key technologies for LLM training, challenges, and potential solutions. Subsequently, we explore the mutual empowerment between these two fields, highlighting key motivations, open challenges, and potential solutions. Finally, we provide insights into future directions, applications, and their societal impact to further explore this intersection, paving the way for next-generation intelligent communication systems. Overall, this survey provides a comprehensive overview of the relationship between RL-based LLMs and wireless networks, offering a vision where these domains empower each other to drive innovations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15726v1">Reinforcement Learning Environment with LLM-Controlled Adversary in D&D 5th Edition Combat</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Preprint. Submitted to the 31st International Conference on Neural Information Processing (ICONIP 2024)
    </div>
    <details class="paper-abstract">
      The objective of this study is to design and implement a reinforcement learning (RL) environment using D\&D 5E combat scenarios to challenge smaller RL agents through interaction with a robust adversarial agent controlled by advanced Large Language Models (LLMs) like GPT-4o and LLaMA 3 8B. This research employs Deep Q-Networks (DQN) for the smaller agents, creating a testbed for strategic AI development that also serves as an educational tool by simulating dynamic and unpredictable combat scenarios. We successfully integrated sophisticated language models into the RL framework, enhancing strategic decision-making processes. Our results indicate that while RL agents generally outperform LLM-controlled adversaries in standard metrics, the strategic depth provided by LLMs significantly enhances the overall AI capabilities in this complex, rule-based setting. The novelty of our approach and its implications for mastering intricate environments and developing adaptive strategies are discussed, alongside potential innovations in AI-driven interactive simulations. This paper aims to demonstrate how integrating LLMs can create more robust and adaptable AI systems, providing valuable insights for further research and educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09516v2">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities during inference to use search engines is not optimal, since the LLM does not learn how to optimally interact with the search engine. This paper introduces Search-R1, an extension of the DeepSeek-R1 model where the LLM learns -- solely through reinforcement learning (RL) -- to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM rollouts with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 26% (Qwen2.5-7B), 21% (Qwen2.5-3B), and 10% (LLaMA3.2-3B) over strong baselines. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at https://github.com/PeterGriffinJin/Search-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15669v1">ECO: An LLM-Driven Efficient Code Optimizer for Warehouse Scale Computers</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      With the end of Moore's Law, optimizing code for performance has become paramount for meeting ever-increasing compute demands, particularly in hyperscale data centers where even small efficiency gains translate to significant resource and energy savings. Traditionally, this process requires significant programmer effort to identify optimization opportunities, modify the code to implement the optimization, and carefully deploy and measure the optimization's impact. Despite a significant amount of work on automating program edits and promising results in small-scale settings, such performance optimizations have remained elusive in large real-world production environments, due to the scale, high degree of complexity, and reliability required. This paper introduces ECO (Efficient Code Optimizer), a system that automatically refactors source code to improve performance at scale. To achieve these performance gains, ECO searches through historical commits at scale to create a dictionary of performance anti-patterns that these commits addressed. These anti-patterns are used to search for similar patterns in a code base of billions of lines of code, pinpointing other code segments with similar potential optimization opportunities. Using a fine-tuned LLM, ECO then automatically refactors the code to generate and apply similar edits. Next, ECO verifies the transformed code, submits it for code review, and measures the impact of the optimization in production. Currently deployed on Google's hyperscale production fleet, this system has driven >25k changed lines of production code, across over 6.4k submitted commits, with a >99.5% production success rate. Over the past year, ECO has consistently resulted in significant performance savings every quarter. On average, the savings produced per quarter are equivalent to over 500k normalized CPU cores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18011v2">StructTest: Benchmarking LLMs' Reasoning through Compositional Structured Outputs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) demands robust, unbiased, and scalable evaluation methods. However, human annotations are costly to scale, model-based evaluations are susceptible to stylistic biases, and target-answer-based benchmarks are vulnerable to data contamination and cheating. To address these limitations, we propose StructTest, a novel benchmark that evaluates LLMs on their ability to follow compositional instructions and generate structured outputs, providing an unbiased, cost-effective, and difficult-to-cheat evaluation framework. Assessments are conducted deterministically using a rule-based evaluator, which can be easily extended to new tasks and datasets. By testing structured outputs across diverse domains including Summarization, Code, HTML, and Math, and evaluating 17 popular LLMs, we demonstrate that StructTest remains challenging even for top-performing models like Deepseek-V3/R1 and GPT-4o, establishing it as a robust proxy for measuring reasoning capabilities. We believe StructTest offers a critical and complementary approach to achieving objective and comprehensive model evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15655v1">R$^2$: A LLM Based Novel-to-Screenplay Generation Framework with Causal Plot Graphs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Automatically adapting novels into screenplays is important for the TV, film, or opera industries to promote products with low costs. The strong performances of large language models (LLMs) in long-text generation call us to propose a LLM based framework Reader-Rewriter (R$^2$) for this task. However, there are two fundamental challenges here. First, the LLM hallucinations may cause inconsistent plot extraction and screenplay generation. Second, the causality-embedded plot lines should be effectively extracted for coherent rewriting. Therefore, two corresponding tactics are proposed: 1) A hallucination-aware refinement method (HAR) to iteratively discover and eliminate the affections of hallucinations; and 2) a causal plot-graph construction method (CPC) based on a greedy cycle-breaking algorithm to efficiently construct plot lines with event causalities. Recruiting those efficient techniques, R$^2$ utilizes two modules to mimic the human screenplay rewriting process: The Reader module adopts a sliding window and CPC to build the causal plot graphs, while the Rewriter module generates first the scene outlines based on the graphs and then the screenplays. HAR is integrated into both modules for accurate inferences of LLMs. Experimental results demonstrate the superiority of R$^2$, which substantially outperforms three existing approaches (51.3%, 22.6%, and 57.1% absolute increases) in pairwise comparison at the overall win rate for GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12065v2">Formalizing Complex Mathematical Statements with LLMs: A Study on Mathematical Definitions</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Thanks to their linguistic capabilities, LLMs offer an opportunity to bridge the gap between informal mathematics and formal languages through autoformalization. However, it is still unclear how well LLMs generalize to sophisticated and naturally occurring mathematical statements. To address this gap, we investigate the task of autoformalizing real-world mathematical definitions -- a critical component of mathematical discourse. Specifically, we introduce two novel resources for autoformalisation, collecting definitions from Wikipedia (Def_Wiki) and arXiv papers (Def_ArXiv). We then systematically evaluate a range of LLMs, analyzing their ability to formalize definitions into Isabelle/HOL. Furthermore, we investigate strategies to enhance LLMs' performance including refinement through external feedback from Proof Assistants, and formal definition grounding, where we guide LLMs through relevant contextual elements from formal mathematical libraries. Our findings reveal that definitions present a greater challenge compared to existing benchmarks, such as miniF2F. In particular, we found that LLMs still struggle with self-correction, and aligning with relevant mathematical libraries. At the same time, structured refinement methods and definition grounding strategies yield notable improvements of up to 16% on self-correction capabilities and 43% on the reduction of undefined errors, highlighting promising directions for enhancing LLM-based autoformalization in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15621v1">LLaVA-MORE: A Comparative Study of LLMs and Visual Backbones for Enhanced Visual Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Recent progress in Multimodal Large Language Models (MLLMs) has highlighted the critical roles of both the visual backbone and the underlying language model. While prior work has primarily focused on scaling these components to billions of parameters, the trade-offs between model size, architecture, and performance remain underexplored. Additionally, inconsistencies in training data and evaluation protocols have hindered direct comparisons, making it difficult to derive optimal design choices. In this paper, we introduce LLaVA-MORE, a new family of MLLMs that integrates recent language models with diverse visual backbones. To ensure fair comparisons, we employ a unified training protocol applied consistently across all architectures. Our analysis systematically explores both small- and medium-scale LLMs -- including Phi-4, LLaMA-3.1, and Gemma-2 -- to evaluate multimodal reasoning, generation, and instruction following, while examining the relationship between model size and performance. Beyond evaluating the LLM impact on final results, we conduct a comprehensive study of various visual encoders, ranging from CLIP-based architectures to alternatives such as DINOv2, SigLIP, and SigLIP2. Additional experiments investigate the effects of increased image resolution and variations in pre-training datasets. Overall, our results provide insights into the design of more effective MLLMs, offering a reproducible evaluation framework that facilitates direct comparisons and can guide future model development. Our source code and trained models are publicly available at: https://github.com/aimagelab/LLaVA-MORE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15620v1">Does Context Matter? ContextualJudgeBench for Evaluating LLM-based Judges in Contextual Settings</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 23 pages, 13 figures, 6 tables
    </div>
    <details class="paper-abstract">
      The large language model (LLM)-as-judge paradigm has been used to meet the demand for a cheap, reliable, and fast evaluation of model outputs during AI system development and post-deployment monitoring. While judge models -- LLMs finetuned to specialize in assessing and critiquing model outputs -- have been touted as general purpose evaluators, they are typically evaluated only on non-contextual scenarios, such as instruction following. The omission of contextual settings -- those where external information is used as context to generate an output -- is surprising given the increasing prevalence of retrieval-augmented generation (RAG) and summarization use cases. Contextual assessment is uniquely challenging, as evaluation often depends on practitioner priorities, leading to conditional evaluation criteria (e.g., comparing responses based on factuality and then considering completeness if they are equally factual). To address the gap, we propose ContextualJudgeBench, a judge benchmark with 2,000 challenging response pairs across eight splits inspired by real-world contextual evaluation scenarios. We build our benchmark with a multi-pronged data construction pipeline that leverages both existing human annotations and model-based perturbations. Our comprehensive study across 11 judge models and 9 general purpose models, reveals that the contextual information and its assessment criteria present a significant challenge to even state-of-the-art models. For example, OpenAI's o1, the best-performing model, barely reaches 55% consistent accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00057v2">Improved YOLOv12 with LLM-Generated Synthetic Data for Enhanced Apple Detection and Benchmarking Against YOLOv11 and YOLOv10</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 8 pages, 5 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      This study evaluated the performance of the YOLOv12 object detection model, and compared against the performances YOLOv11 and YOLOv10 for apple detection in commercial orchards based on the model training completed entirely on synthetic images generated by Large Language Models (LLMs). The YOLOv12n configuration achieved the highest precision at 0.916, the highest recall at 0.969, and the highest mean Average Precision (mAP@50) at 0.978. In comparison, the YOLOv11 series was led by YOLO11x, which achieved the highest precision at 0.857, recall at 0.85, and mAP@50 at 0.91. For the YOLOv10 series, YOLOv10b and YOLOv10l both achieved the highest precision at 0.85, with YOLOv10n achieving the highest recall at 0.8 and mAP@50 at 0.89. These findings demonstrated that YOLOv12, when trained on realistic LLM-generated datasets surpassed its predecessors in key performance metrics. The technique also offered a cost-effective solution by reducing the need for extensive manual data collection in the agricultural field. In addition, this study compared the computational efficiency of all versions of YOLOv12, v11 and v10, where YOLOv11n reported the lowest inference time at 4.7 ms, compared to YOLOv12n's 5.6 ms and YOLOv10n's 5.9 ms. Although YOLOv12 is new and more accurate than YOLOv11, and YOLOv10, YOLO11n still stays the fastest YOLO model among YOLOv10, YOLOv11 and YOLOv12 series of models. (Index: YOLOv12, YOLOv11, YOLOv10, YOLOv13, YOLOv14, YOLOv15, YOLOE, YOLO Object detection)
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15571v1">LLM-Aided Customizable Profiling of Code Data Based On Programming Language Concepts</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Data profiling is critical in machine learning for generating descriptive statistics, supporting both deeper understanding and downstream tasks like data valuation and curation. This work addresses profiling specifically in the context of code datasets for Large Language Models (code-LLMs), where data quality directly influences tasks such as code generation and summarization. Characterizing code datasets in terms of programming language concepts enables better insights and targeted data curation. Our proposed methodology decomposes code data profiling into two phases: (1) an offline phase where LLMs are leveraged to derive and learn rules for extracting syntactic and semantic concepts across various programming languages, including previously unseen or low-resource languages, and (2) an online deterministic phase applying these derived rules for efficient real-time analysis. This hybrid approach is customizable, extensible to new syntactic and semantic constructs, and scalable to multiple languages. Experimentally, our LLM-aided method achieves a mean accuracy of 90.33% for syntactic extraction rules and semantic classification accuracies averaging 80% and 77% across languages and semantic concepts, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16548v1">SemanticScanpath: Combining Gaze and Speech for Situated Human-Robot Interaction Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have substantially improved the conversational capabilities of social robots. Nevertheless, for an intuitive and fluent human-robot interaction, robots should be able to ground the conversation by relating ambiguous or underspecified spoken utterances to the current physical situation and to the intents expressed non verbally by the user, for example by using referential gaze. Here we propose a representation integrating speech and gaze to enable LLMs to obtain higher situated awareness and correctly resolve ambiguous requests. Our approach relies on a text-based semantic translation of the scanpath produced by the user along with the verbal requests and demonstrates LLM's capabilities to reason about gaze behavior, robustly ignoring spurious glances or irrelevant objects. We validate the system across multiple tasks and two scenarios, showing its generality and accuracy, and demonstrate its implementation on a robotic platform, closing the loop from request interpretation to execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15478v1">SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 29 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents need to perform multi-turn interactions in real-world tasks. However, existing multi-turn RL algorithms for optimizing LLM agents fail to perform effective credit assignment over multiple turns while leveraging the generalization capabilities of LLMs and it remains unclear how to develop such algorithms. To study this, we first introduce a new benchmark, ColBench, where an LLM agent interacts with a human collaborator over multiple turns to solve realistic tasks in backend programming and frontend design. Building on this benchmark, we propose a novel RL algorithm, SWEET-RL (RL with Step-WisE Evaluation from Training-time information), that uses a carefully designed optimization objective to train a critic model with access to additional training-time information. The critic provides step-level rewards for improving the policy model. Our experiments demonstrate that SWEET-RL achieves a 6% absolute improvement in success and win rates on ColBench compared to other state-of-the-art multi-turn RL algorithms, enabling Llama-3.1-8B to match or exceed the performance of GPT4-o in realistic collaborative content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13213v3">Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      We study 15 large language models (LLMs) fine-tuned for chat and find that their maximum softmax probabilities (MSPs) are consistently miscalibrated on multiple-choice Q&A. However, those MSPs might still encode useful uncertainty information. Specifically, we hypothesized that wrong answers would be associated with smaller MSPs compared to correct answers. Via rigorous statistical testing, we show that this hypothesis holds for models which perform well on the underlying Q&A task. We also find a strong direction correlation between Q&A accuracy and MSP correctness prediction, while finding no correlation between Q&A accuracy and calibration error. This suggests that within the current fine-tuning paradigm, we can expect correctness prediction but not calibration to improve as LLM capabilities progress. To demonstrate the utility of correctness prediction, we show that when models have the option to abstain, performance can be improved by selectively abstaining based on the MSP of the initial model response, using only a small amount of labeled data to choose the MSP threshold.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15374v1">Real-world validation of a multimodal LLM-powered pipeline for High-Accuracy Clinical Trial Patient Matching leveraging EHR data</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Background: Patient recruitment in clinical trials is hindered by complex eligibility criteria and labor-intensive chart reviews. Prior research using text-only models have struggled to address this problem in a reliable and scalable way due to (1) limited reasoning capabilities, (2) information loss from converting visual records to text, and (3) lack of a generic EHR integration to extract patient data. Methods: We introduce a broadly applicable, integration-free, LLM-powered pipeline that automates patient-trial matching using unprocessed documents extracted from EHRs. Our approach leverages (1) the new reasoning-LLM paradigm, enabling the assessment of even the most complex criteria, (2) visual capabilities of latest LLMs to interpret medical records without lossy image-to-text conversions, and (3) multimodal embeddings for efficient medical record search. The pipeline was validated on the n2c2 2018 cohort selection dataset (288 diabetic patients) and a real-world dataset composed of 485 patients from 30 different sites matched against 36 diverse trials. Results: On the n2c2 dataset, our method achieved a new state-of-the-art criterion-level accuracy of 93\%. In real-world trials, the pipeline yielded an accuracy of 87\%, undermined by the difficulty to replicate human decision-making when medical records lack sufficient information. Nevertheless, users were able to review overall eligibility in under 9 minutes per patient on average, representing an 80\% improvement over traditional manual chart reviews. Conclusion: This pipeline demonstrates robust performance in clinical trial patient matching without requiring custom integration with site systems or trial-specific tailoring, thereby enabling scalable deployment across sites seeking to leverage AI for patient matching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20227v2">LLM Reasoning Engine: Specialized Training for Enhanced Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted to NAACL 2025 KnowledgeNLP
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance in various natural language processing tasks but face challenges in mathematical reasoning, where complex problem-solving requires both linguistic understanding and mathematical reasoning skills. Existing approaches to address this challenge often rely on ensemble methods and suffer from the problem of data scarcity in target domains. In this work, we present a novel method to enhance LLMs' capabilities in mathematical reasoning tasks. Motivated by the need to bridge this gap, our approach incorporates a question paraphrase strategy, which aims at diversifying the linguistic forms of mathematical questions to improve generalization. Additionally, specialized training objectives are employed to guide the model's learning process, focusing on enhancing its understanding of mathematical concepts and reasoning processes. We conduct experiments on four datasets using different LLMs, and demonstrate the effectiveness of our approach in improving LLMs' performance on mathematical reasoning tasks. Our findings underscore the significance of our methodology in the advancement of large language models and its potential implications for real-world applications that require mathematical reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15341v1">Uncertainty-Guided Chain-of-Thought for Code Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) reasoning has been demonstrated as an effective technique for improving the problem-solving capabilities of large language models (LLMs) in the context of code generation. However, existing CoT methods often exhibit a tendency toward "overthinking", where the LLM consistently applies reasoning strategies without adequately considering the task's underlying complexity. This results in the LLMs allocating excessive computational resources, in terms of tokens, to relatively simple tasks or problems where the correct answer is already evident. Additionally, this overthinking may lead LLMs down incorrect reasoning paths, resulting in incorrect code generation. In this paper, we introduce UnCertainty-Aware Chain-of-Thought (UnCert-CoT), an LLM-based approach designed to enhance code generation by incorporating an uncertainty-aware CoT reasoning mechanism, which focuses computational resources on targeting points where LLMs are more prone to error. We propose two confidence-based uncertainty measures: Entropy-based and Probability Differential-based methods. When uncertainty is high, UnCert-CoT activates CoT-decoding to generate multiple reasoning paths and selects the final code that exhibits the highest likelihood of correctness. In contrast, LLM directly generates the code when uncertainty is low. This uncertainty judgment mechanism allows LLMs to prioritize complex tasks and avoid unnecessary steps in simpler cases, thereby improving overall efficiency and accuracy in code generation. Our experimental results demonstrate that UnCert-CoT significantly enhances code generation accuracy on challenging benchmark MHPP(Mostly Hard Python Problems), it achieves improvements up to 6.1% on PassRate accuracy, particularly in situations where traditional LLMs are prone to errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14703v3">Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset designed for LLMs with Psychometrics</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted to NAACL2025 Findings
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to their adaptation in various domains as conversational agents. We wonder: can personality tests be applied to these agents to analyze their behavior, similar to humans? We introduce TRAIT, a new benchmark consisting of 8K multi-choice questions designed to assess the personality of LLMs. TRAIT is built on two psychometrically validated small human questionnaires, Big Five Inventory (BFI) and Short Dark Triad (SD-3), enhanced with the ATOMIC-10X knowledge graph to a variety of real-world scenarios. TRAIT also outperforms existing personality tests for LLMs in terms of reliability and validity, achieving the highest scores across four key metrics: Content Validity, Internal Validity, Refusal Rate, and Reliability. Using TRAIT, we reveal two notable insights into personalities of LLMs: 1) LLMs exhibit distinct and consistent personality, which is highly influenced by their training data (e.g., data used for alignment tuning), and 2) current prompting techniques have limited effectiveness in eliciting certain traits, such as high psychopathy or low conscientiousness, suggesting the need for further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15338v1">Solla: Towards a Speech-Oriented LLM That Hears Acoustic Context</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown remarkable ability to process not only text but also multimodal inputs such as speech and audio. However, most existing models primarily focus on analyzing input signals using text instructions, overlooking scenarios in which speech instructions and audio are mixed and serve as inputs to the model. To address these challenges, we introduce Solla, a novel framework designed to understand speech-based questions and hear the acoustic context concurrently. Solla incorporates an audio tagging module to effectively identify and represent audio events, as well as an ASR-assisted prediction method to improve comprehension of spoken content. To rigorously evaluate Solla and other publicly available models, we propose a new benchmark dataset called SA-Eval, which includes three tasks: audio event classification, audio captioning, and audio question answering. SA-Eval has diverse speech instruction with various speaking styles, encompassing two difficulty levels, easy and hard, to capture the range of real-world acoustic conditions. Experimental results show that Solla performs on par with or outperforms baseline models on both the easy and hard test sets, underscoring its effectiveness in jointly understanding speech and audio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09454v2">Explicit Learning and the LLM in Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      This study explores the capacity of large language models (LLMs) for explicit learning, a process involving the assimilation of metalinguistic explanations to carry out language tasks. Using constructed languages generated by cryptographic means as controlled test environments, we designed experiments to assess an LLM's ability to explicitly learn and apply grammar rules. Our results demonstrate that while LLMs possess a measurable capacity for explicit learning, this ability diminishes as the complexity of the linguistic phenomena at hand increases. Supervised fine-tuning on chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15301v1">aiXcoder-7B-v2: Training LLMs to Fully Utilize the Long Context in Repository-level Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Repository-level code completion aims to complete code based on the long contexts of the repository. Existing studies extract long contexts from the repository as inputs and leverage Large Language Models (LLMs) to generate code. However, we reveal a severe limitation of LLMs, i.e., LLMs may ignore the information within long contexts in code completion. In other words, even the contexts contain useful information (e.g., relevant APIs or similar code), LLMs may fail to utilize this information. We think this limitation is caused by an inherent bias in LLMs, i.e., relying on nearby contexts and ignoring long-range contexts. To address this, we propose a novel fine-tuning approach named CoLT. The core idea of CoLT is to provide explicit supervision signals, which emphasize that long-range contexts may hold relevant information. Specifically, CoLT proposes a reinforcement learning-based training, which explicitly encourages models to utilize the information within long contexts and punishes models for ignoring long contexts. To support CoLT, we release CoLT-132K, a large-scale dataset with 132k samples across four languages, each containing long-context inputs. We apply CoLT to a popular LLM - aiXcoder-7B and release aiXcoder-7B-v2. We conduct extensive experiments on CoLT-132K and a public benchmark - CrossCodeEval. Our experiments yield the results: 1. Effectiveness. CoLT substantially improves aiXcoder-7B. aiXcoder-7B-v2 outperforms aiXcoder-7B by up to 44% in exact match. aiXcoder-7B-v2 becomes the state-of-the-art 7B model in code completion and even surpasses larger models. 2. Generalizability. The capability learned by CoLT can generalize to new languages. Besides, CoLT is model-agnostic and effectively improves multiple LLMs. 3. Enhanced Context Utilization Capability. CoLT significantly improves the capability of LLMs in utilizing the relevant information within long contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15299v1">Inside-Out: Hidden Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) puts a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15242v1">BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity?</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes of set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their strengths and weaknesses in handling complexity requirements. In particular, token-space reasoning models are unrivaled in code generation but not in complexity understanding, hinting that they may not generalize well to tasks for which no reward was given at training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12896v2">None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15231v1">When LLMs Meet API Documentation: Can Retrieval Augmentation Aid Code Generation Just as It Helps Developers?</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has increasingly shown its power in extending large language models' (LLMs') capability beyond their pre-trained knowledge. Existing works have shown that RAG can help with software development tasks such as code generation, code update, and test generation. Yet, the effectiveness of adapting LLMs to fast-evolving or less common API libraries using RAG remains unknown. To bridge this gap, we take an initial step to study this unexplored yet practical setting - when developers code with a less common library, they often refer to its API documentation; likewise, when LLMs are allowed to look up API documentation via RAG, to what extent can LLMs be advanced? To mimic such a setting, we select four less common open-source Python libraries with a total of 1017 eligible APIs. We study the factors that affect the effectiveness of using the documentation of less common API libraries as additional knowledge for retrieval and generation. Our intensive study yields interesting findings: (1) RAG helps improve LLMs' performance by 83%-220%. (2) Example code contributes the most to advance LLMs, instead of the descriptive texts and parameter lists in the API documentation. (3) LLMs could sometimes tolerate mild noises (typos in description or incorrect parameters) by referencing their pre-trained knowledge or document context. Finally, we suggest that developers pay more attention to the quality and diversity of the code examples in the API documentation. The study sheds light on future low-code software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16457v4">Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Materials synthesis is vital for innovations such as energy storage, catalysis, electronics, and biomedical devices. Yet, the process relies heavily on empirical, trial-and-error methods guided by expert intuition. Our work aims to support the materials science community by providing a practical, data-driven resource. We have curated a comprehensive dataset of 17K expert-verified synthesis recipes from open-access literature, which forms the basis of our newly developed benchmark, AlchemyBench. AlchemyBench offers an end-to-end framework that supports research in large language models applied to synthesis prediction. It encompasses key tasks, including raw materials and equipment prediction, synthesis procedure generation, and characterization outcome forecasting. We propose an LLM-as-a-Judge framework that leverages large language models for automated evaluation, demonstrating strong statistical agreement with expert assessments. Overall, our contributions offer a supportive foundation for exploring the capabilities of LLMs in predicting and guiding materials synthesis, ultimately paving the way for more efficient experimental design and accelerated innovation in materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15117v1">Exploring Model Editing for LLM-based Aspect-Based Sentiment Classification</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 AAAI2025
    </div>
    <details class="paper-abstract">
      Model editing aims at selectively updating a small subset of a neural model's parameters with an interpretable strategy to achieve desired modifications. It can significantly reduce computational costs to adapt to large language models (LLMs). Given its ability to precisely target critical components within LLMs, model editing shows great potential for efficient fine-tuning applications. In this work, we investigate model editing to serve an efficient method for adapting LLMs to solve aspect-based sentiment classification. Through causal interventions, we trace and determine which neuron hidden states are essential for the prediction of the model. By performing interventions and restorations on each component of an LLM, we identify the importance of these components for aspect-based sentiment classification. Our findings reveal that a distinct set of mid-layer representations is essential for detecting the sentiment polarity of given aspect words. Leveraging these insights, we develop a model editing approach that focuses exclusively on these critical parts of the LLM, leading to a more efficient method for adapting LLMs. Our in-domain and out-of-domain experiments demonstrate that this approach achieves competitive results compared to the currently strongest methods with significantly fewer trainable parameters, highlighting a more efficient and interpretable fine-tuning strategy.
    </details>
</div>
