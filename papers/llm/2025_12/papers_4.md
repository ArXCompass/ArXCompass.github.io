# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07745v3">Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 Accepted by NDSS 2026
    </div>
    <details class="paper-abstract">
      Insider threats pose a persistent and critical security risk, yet are notoriously difficult to detect in complex enterprise environments, where malicious actions are often hidden within seemingly benign user behaviors. Although machine-learning-based insider threat detection (ITD) methods have shown promise, their effectiveness is fundamentally limited by the scarcity of high-quality and realistic training data. Enterprise internal data is highly sensitive and rarely accessible, while existing public and synthetic datasets are either small-scale or lack sufficient realism, semantic richness, and behavioral diversity. To address this challenge, we propose Chimera, an LLM-based multi-agent framework that automatically simulates both benign and malicious insider activities and generates comprehensive system logs across diverse enterprise environments. Chimera models each agent as an individual employee with fine-grained roles and supports group meetings, pairwise interactions, and self-organized scheduling to capture realistic organizational dynamics. Based on 15 insider attacks abstracted from real-world incidents, we deploy Chimera in three representative data-sensitive organizational scenarios and construct ChimeraLog, a new dataset for developing and evaluating ITD methods. We evaluate ChimeraLog through human studies and quantitative analyses, demonstrating its diversity and realism. Experiments with existing ITD methods show substantially lower detection performance on ChimeraLog compared to prior datasets, indicating a more challenging and realistic benchmark. Moreover, despite distribution shifts, models trained on ChimeraLog exhibit strong generalization, highlighting the practical value of LLM-based multi-agent simulation for advancing insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18462v1">Mitigating Spurious Correlations in NLI via LLM-Synthesized Counterfactuals and Dynamic Balanced Sampling</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Natural Language Inference (NLI) models frequently rely on spurious correlations rather than semantic reasoning. Existing mitigation strategies often incur high annotation costs or trigger catastrophic forgetting during fine-tuning. We propose an automated, scalable pipeline to address these limitations. First, we introduce Log-Frequency LMI (LF-LMI) to accurately detect semantic artifacts. Second, we generate a high-quality synthetic contrast set via an LLM-synthesis pipeline with multi-judge verification. Finally, we introduce Dynamic Balanced Sampling, a training strategy that rotates the original data distribution to prevent forgetting. Our method improves consistency on a challenging benchmark from 63.5% to 81.0% while maintaining 88.4% in-domain accuracy, significantly outperforming naive fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18452v1">Secret mixtures of experts inside your LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 8 pages in main text; 23 pages total
    </div>
    <details class="paper-abstract">
      Despite being one of the earliest neural network layers, the Multilayer Perceptron (MLP) is arguably one of the least understood parts of the transformer architecture due to its dense computation and lack of easy visualization. This paper seeks to understand the MLP layers in dense LLM models by hypothesizing that these layers secretly approximately perform a sparse computation -- namely, that they can be well approximated by sparsely-activating Mixture of Experts (MoE) layers. Our hypothesis is based on a novel theoretical connection between MoE models and Sparse Autoencoder (SAE) structure in activation space. We empirically validate the hypothesis on pretrained LLMs, and demonstrate that the activation distribution matters -- these results do not hold for Gaussian data, but rather rely crucially on structure in the distribution of neural network activations. Our results shine light on a general principle at play in MLP layers inside LLMs, and give an explanation for the effectiveness of modern MoE-based transformers. Additionally, our experimental explorations suggest new directions for more efficient MoE architecture design based on low-rank routers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.05387v2">Learning from Self Critique and Refinement for Faithful LLM Summarization</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often suffer from hallucinations: output content that is not grounded in the input context, when performing long-form text generation tasks such as summarization. Prior works have shown that hallucinations can be reduced by iteratively critiquing and refining previously generated outputs using either the same model or a more powerful teacher model as the critique. However, these approaches either require additional test-time compute or assume access to more powerful teacher models, making them costly and less practical. In this work, we propose Self Critique and Refinement-based Preference Optimization (SCRPO), which is a self-supervised training framework that first constructs a preference dataset by leveraging the LLM's own critique and refinement capabilities, and then applies preference learning to improve the same LLM for faithful summarization. Experiments on three summarization benchmarks (XSUM CNNDM and SAMSum), demonstrate that our approach outperforms state-of-the-art self-supervised learning methods in terms of faithfulness metrics while either maintaining or improving other metrics that measure the overall quality of the summary. Moreover, compared to test-time refinement, our approach not only improves efficiency but also results in more faithful summaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.06994v4">Phaedrus: Predicting Dynamic Application Behavior with Lightweight Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-20
    </div>
    <details class="paper-abstract">
      Application profiling is an indispensable technique for many software development tasks, such as code and memory layout optimizations, where optimization decisions are tailored to specific program profiles. Unfortunately, modern application codebases exhibit highly variant behavior across different inputs, creating challenges for conventional profiling approaches that rely on a single representative execution instance. In this paper, we propose \textbf{Phaedrus}, a new \textit{compiler-assisted deep learning framework} designed to predict dynamic program behaviors across varied execution instances, specifically focusing on dynamic function call prediction.Such predicted call sequences are then used for producing optimized code pertinent to a given input. Traditional profile-guided optimization methods struggle with the input-dependent variability of modern applications, where profiling on different inputs yields divergent application behaviors. To address this, Phaedrus proposes two new approaches: \textit{Application Behavior Synthesis}, a profile-less approach where Large Language Models (LLMs) directly infer dynamic functions based on source code \& static compiler analysis, bypassing the need for traditional profiling, and \textit{Application Profile Generalization}, which uses generative models trained on compressed and augmented \textit{Whole Program Path} (WPP) based function profiles to predict application behavior under unseen inputs. Our experiments show that \textit{Phaedrus} can achieve upto $10^7X$ reduction in WPP function profile sizes, can predict most frequently executed functions that cover upto 85-99\% of the execution time, along with an average of 13.19\% (upto 65\%) reduction in application binary size, and an average of 6.08\% (upto 20\%) performance improvement over the traditional profile-guided optimization, without any execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18360v1">LLM Agents Implement an NLG System from Scratch: Building Interpretable Rule-Based RDF-to-Text Generators</a></div>
    <div class="paper-meta">
      📅 2025-12-20
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      We present a novel neurosymbolic framework for RDF-to-text generation, in which the model is "trained" through collaborative interactions among multiple LLM agents rather than traditional backpropagation. The LLM agents produce rule-based Python code for a generator for the given domain, based on RDF triples only, with no in-domain human reference texts. The resulting system is fully interpretable, requires no supervised training data, and generates text nearly instantaneously using only a single CPU. Our experiments on the WebNLG and OpenDialKG data show that outputs produced by our approach reduce hallucination, with only slight fluency penalties compared to finetuned or prompted language models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.15817v4">A Causal Perspective on Measuring, Explaining and Mitigating Smells in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated their adoption in software engineering contexts. However, concerns persist about the structural quality of the code they produce. In particular, LLMs often replicate poor coding practices, introducing code smells (i.e., patterns that hinder readability, maintainability, or design integrity). Although prior research has examined the detection or repair of smells, we still lack a clear understanding of how and when these issues emerge in generated code. This paper addresses this gap by systematically measuring, explaining and mitigating smell propensity in LLM-generated code. We build on the Propensity Smelly Score (PSC), a probabilistic metric that estimates the likelihood of generating particular smell types, and establish its robustness as a signal of structural quality. Using PSC as an instrument for causal analysis, we identify how generation strategy, model size, model architecture and prompt formulation shape the structural properties of generated code. Our findings show that prompt design and architectural choices play a decisive role in smell propensity and motivate practical mitigation strategies that reduce its occurrence. A user study further demonstrates that PSC helps developers interpret model behavior and assess code quality, providing evidence that smell propensity signals can support human judgement. Taken together, our work lays the groundwork for integrating quality-aware assessments into the evaluation and deployment of LLMs for code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17814v1">LLM-based Behaviour Driven Development for Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 7 pages, keynote given at 2nd International Symposium on Artificial Intelligence and Internet of Things (AIIoT-25), December 22-24th, 2025
    </div>
    <details class="paper-abstract">
      Test and verification are essential activities in hardware and system design, but their complexity grows significantly with increasing system sizes. While Behavior Driven Development (BDD) has proven effective in software engineering, it is not yet well established in hardware design, and its practical use remains limited. One contributing factor is the manual effort required to derive precise behavioral scenarios from textual specifications. Recent advances in Large Language Models (LLMs) offer new opportunities to automate this step. In this paper, we investigate the use of LLM-based techniques to support BDD in the context of hardware design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20150v4">Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping the recommender system paradigm by enabling users to express preferences and receive recommendations through conversations. Yet, aligning LLMs to the recommendation task remains challenging: pretrained LLMs often generate out-of-catalog items, violate required output formats, and their ranking quality degrades sharply toward the end of the generated list. To this end, we propose ConvRec-R1, a two-stage framework for end-to-end training of LLM-based conversational recommender systems. In Stage 1, we construct a behavioral-cloning dataset with a Remap-Reflect-Adjust pipeline, which produces high-quality, catalog-grounded demonstrations from powerful blackbox LLMs to warm-start the RL training. In Stage 2, we propose Rank-GRPO, a principled extension of group relative policy optimization (GRPO) tailored to tasks with rank-style outputs. Rank-GRPO treats each rank in the recommendation list as the unit instead of token (too fine-grained) or sequence (too coarse), redefining rewards to remove non-causal credit assignment and introducing a rank-level importance ratio based on the geometric mean of rank-wise token probabilities to stabilize policy updates. Experiments on the public Reddit-v2 dataset show that ConvRec-R1 converges faster and achieves higher Recall and NDCG than GRPO-style baselines. Code and datasets are released at https://github.com/yaochenzhu/Rank-GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09147v4">LLM-as-a-qualitative-judge: automating error analysis in natural language generation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that instance-specific issues output by LLM-as-a-qualitative-judge match those annotated by humans in 2/3 cases, and that LLM-as-a-qualitative-judge is capable of producing error type reports resembling the reports composed by human annotators. We also demonstrate in a case study how the use of LLM-as-a-qualitative-judge can substantially improve NLG systems performance. Our code and data are publicly available at https://github.com/tunde-ajayi/llm-as-a-qualitative-judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12817v2">Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted as a conference paper at AAAI'26
    </div>
    <details class="paper-abstract">
      The recent proliferation of large language models (LLMs) holds the potential to revolutionize healthcare, with strong capabilities in diverse medical tasks. Yet, deploying LLMs in high-stakes healthcare settings requires rigorous verification and validation to understand any potential harm. This paper investigates the reliability and viability of using medical knowledge graphs (KGs) for the automated factuality evaluation of LLM-generated responses. To ground this investigation, we introduce FAITH, a framework designed to systematically probe the strengths and limitations of this KG-based approach. FAITH operates without reference answers by decomposing responses into atomic claims, linking them to a medical KG, and scoring them based on evidence paths. Experiments on diverse medical tasks with human subjective evaluations demonstrate that KG-grounded evaluation achieves considerably higher correlations with clinician judgments and can effectively distinguish LLMs with varying capabilities. It is also robust to textual variances. The inherent explainability of its scoring can further help users understand and mitigate the limitations of current LLMs. We conclude that while limitations exist, leveraging KGs is a prominent direction for automated factuality assessment in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.21540v3">Designing an LLM-Based Behavioral Activation Chatbot for Young People with Depression: Insights from an Evaluation with Artificial Users and Clinical Experts</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      LLMs promise to overcome limitations of rule-based mental health chatbots through improved natural language capabilities, yet their ability to deliver evidence-based psychological interventions remains largely unverified because evaluations rarely apply the validated fidelity measures used to assess psychotherapists. We developed an LLM-based chatbot that delivers behavioral activation for depression and generated 48 complete chat sessions with diverse artificial users. Ten psychotherapists assessed these sessions using the Quality of Behavioral Activation Scale (Q-BAS), a validated fidelity instrument. Results show that the chatbot reliably executed the intervention across all phases and maintained safety protocols, but it struggled with clinical judgment, particularly when verifying the feasibility of proposed activities. Overall, our findings suggest that LLM-based chatbots can execute therapeutic protocols with high fidelity, while robust clinical reasoning remains an open challenge. We outline design implications to address this gap and provide the chatbot and artificial user prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17639v1">Linear Personality Probing and Steering in LLMs: A Big Five Study</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 29 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit distinct and consistent personalities that greatly impact trust and engagement. While this means that personality frameworks would be highly valuable tools to characterize and control LLMs' behavior, current approaches remain either costly (post-training) or brittle (prompt engineering). Probing and steering via linear directions has recently emerged as a cheap and efficient alternative. In this paper, we investigate whether linear directions aligned with the Big Five personality traits can be used for probing and steering model behavior. Using Llama 3.3 70B, we generate descriptions of 406 fictional characters and their Big Five trait scores. We then prompt the model with these descriptions and questions from the Alpaca questionnaire, allowing us to sample hidden activations that vary along personality traits in known, quantifiable ways. Using linear regression, we learn a set of per-layer directions in activation space, and test their effectiveness for probing and steering model behavior. Our results suggest that linear directions aligned with trait-scores are effective probes for personality detection, while their steering capabilities strongly depend on context, producing reliable effects in forced-choice tasks but limited influence in open-ended generation or when additional context is present in the prompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17630v1">Confidence-Credibility Aware Weighted Ensembles of Small LLMs Outperform Large LLMs in Emotion Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted at IRICT 2025
    </div>
    <details class="paper-abstract">
      This paper introduces a confidence-weighted, credibility-aware ensemble framework for text-based emotion detection, inspired by Condorcet's Jury Theorem (CJT). Unlike conventional ensembles that often rely on homogeneous architectures, our approach combines architecturally diverse small transformer-based large language models (sLLMs) - BERT, RoBERTa, DistilBERT, DeBERTa, and ELECTRA, each fully fine-tuned for emotion classification. To preserve error diversity, we minimize parameter convergence while taking advantage of the unique biases of each model. A dual-weighted voting mechanism integrates both global credibility (validation F1 score) and local confidence (instance-level probability) to dynamically weight model contributions. Experiments on the DAIR-AI dataset demonstrate that our credibility-confidence ensemble achieves a macro F1 score of 93.5 percent, surpassing state-of-the-art benchmarks and significantly outperforming large-scale LLMs, including Falcon, Mistral, Qwen, and Phi, even after task-specific Low-Rank Adaptation (LoRA). With only 595M parameters in total, our small LLMs ensemble proves more parameter-efficient and robust than models up to 7B parameters, establishing that carefully designed ensembles of small, fine-tuned models can outperform much larger LLMs in specialized natural language processing (NLP) tasks such as emotion detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17570v1">GreedySnake: Accelerating SSD-Offloaded LLM Training with Efficient Scheduling and Optimizer Step Overlapping</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      SSD-offloaded training offers a practical and promising approach to making LLM training cost-effective. Building on gradient accumulation with micro-batches, this paper introduces GreedySnake, a new SSD-offloaded training system that employs vertical scheduling, which executes all microbatches of a layer before proceeding to the next. Compared to existing systems that use horizontal scheduling (i.e., executing micro-batches sequentially), GreedySnake achieves higher training throughput with smaller batch sizes, bringing the system much closer to the ideal scenario predicted by the roofline model. To further mitigate the I/O bottleneck, GreedySnake overlaps part of the optimization step with the forward pass of the next iteration. Experimental results on A100 GPUs show that GreedySnake achieves saturated training throughput improvements over ZeRO-Infinity: 1.96x on 1 GPU and 1.93x on 4 GPUs for GPT-65B, and 2.53x on 1 GPU for GPT-175B. The code is open-sourced at https://github.com/npz7yyk/GreedySnake
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17540v1">SGCR: A Specification-Grounded Framework for Trustworthy LLM Code Review</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Automating code review with Large Language Models (LLMs) shows immense promise, yet practical adoption is hampered by their lack of reliability, context-awareness, and control. To address this, we propose Specification-Grounded Code Review (SGCR), a framework that grounds LLMs in human-authored specifications to produce trustworthy and relevant feedback. SGCR features a novel dual-pathway architecture: an explicit path ensures deterministic compliance with predefined rules derived from these specifications, while an implicit path heuristically discovers and verifies issues beyond those rules. Deployed in a live industrial environment at HiThink Research, SGCR's suggestions achieved a 42% developer adoption rate-a 90.9% relative improvement over a baseline LLM (22%). Our work demonstrates that specification-grounding is a powerful paradigm for bridging the gap between the generative power of LLMs and the rigorous reliability demands of software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.18686v2">Hierarchical Multimodal LLMs with Semantic Space Alignment for Enhanced Time Series Classification</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Time series classification plays a fundamental role in a wide range of real-world applications. Recently, large language models (LLMs) have demonstrated strong generalization and reasoning capacities, but directly applying them to time series classification remains non-trivial due to the representation gap between numerical sequences and linguistic semantics. In this paper, we propose HiTime, a hierarchical LLM-based framework for multimodal time series classification that bridges structured temporal representations with semantic reasoning in a generative paradigm. Specifically, we design a hierarchical sequence feature encoding module composed of a data-specific encoder and a task-specific encoder to extract complementary temporal features. To mitigate the embedding gap between time series representations and textual semantics, we further introduce a semantic space alignment module that jointly performs coarse-grained global modeling and fine-grained cross-modal correspondence. Building upon the above representations, we employ a parameter-efficient supervised fine-tuning strategy to activate the generative classification capability of the algined LLMs, thereby transforming conventional discriminative time series classification into a generative task. Extensive experiments on multiple benchmarks demonstrate that the proposed framework consistently outperforms state-of-the-art baselines. The code is publicly available at https://github.com/Xiaoyu-Tao/HiTime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17287v3">RBCTest: Leveraging LLMs to Mine and Verify Oracles of API Response Bodies for RESTful API Testing</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 The paper is accepted to ICSE 2026
    </div>
    <details class="paper-abstract">
      In API testing, deriving logical constraints on API response bodies to be used as oracles is crucial for generating test cases and performing automated testing of RESTful APIs. However, existing approaches are restricted to dynamic analysis, in which oracles are extracted via the execution of APIs as part of the system under test. In this paper, we propose a complementary LLM-based static approach in which constraints for API response bodies are mined from API specifications. We leverage large language models (LLMs) to comprehend API specifications, mine constraints for response bodies, and generate test cases. To reduce LLM hallucination, we apply an Observation-Confirmation (OC) scheme that uses initial prompts to contextualize constraints, allowing subsequent prompts to more accurately confirm their presence. Our empirical results show that RBCTest with OC prompting achieves high precision in constraint mining, with averages ranging from 85.1% to 93.6%. It also performs well in generating test cases from mined constraints, with precision ranging from 86.4% to 91.7%. We further use test cases generated by RBCTest to detect 46 mismatches between API specifications and actual response data across 19 real-world APIs. Four of these mismatches were reported in developers' forums.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17375v1">AdvJudge-Zero: Binary Decision Flips in LLM-as-a-Judge via Adversarial Control Tokens</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Reward models and LLM-as-a-Judge systems are central to modern post-training pipelines such as RLHF, DPO, and RLAIF, where they provide scalar feedback and binary decisions that guide model selection and RL-based fine-tuning. We show that these judge systems exhibit a recurring vulnerability: short sequences of low-perplexity control tokens can flip many binary evaluations from correct ``No'' judgments to incorrect ``Yes'' judgments by steering the last-layer logit gap. These control tokens are patterns that a policy model could plausibly generate during post-training, and thus represent realistic reward-hacking risks rather than worst-case adversarial strings. Our method, AdvJudge-Zero, uses the model's next-token distribution and beam-search exploration to discover diverse control-token sequences from scratch, and our analysis shows that the induced hidden-state perturbations concentrate in a low-rank ``soft mode'' that is anti-aligned with the judge's refusal direction. Empirically, these tokens cause very high false positive rates when large open-weight and specialized judge models score incorrect answers on math and reasoning benchmarks. Finally, we show that LoRA-based adversarial training on small sets of control-token-augmented examples can markedly reduce these false positives while preserving evaluation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25123v3">From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Does RL teach LLMs genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills. To mitigate data contamination and other confounding factors, and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function f(x) given x. When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them h(x)=g(f(x)). Further, this compositional ability generalizes to more difficult problems such as compositions of >2 functions unseen during RL training. Surprisingly, our experiments show that compositional skill acquired on a source task transfers to a different target task. This transfer happens even without compositional training on the target, requiring only prior knowledge of the target's atomic skills. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, next-token training with the same data yields none of these findings. Our systematic experiments provide fresh insights into LLM learning, suggesting the value of first building base models with basic skills, then using RL to incentivize advanced, generalizable skills for complex problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17334v1">Bridging Natural Language and Formal Specification--Automated Translation of Software Requirements to LTL via Hierarchical Semantics Decomposition Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Automating the translation of natural language (NL) software requirements into formal specifications remains a critical challenge in scaling formal verification practices to industrial settings, particularly in safety-critical domains. Existing approaches, both rule-based and learning-based, face significant limitations. While large language models (LLMs) like GPT-4o demonstrate proficiency in semantic extraction, they still encounter difficulties in addressing the complexity, ambiguity, and logical depth of real-world industrial requirements. In this paper, we propose Req2LTL, a modular framework that bridges NL and Linear Temporal Logic (LTL) through a hierarchical intermediate representation called OnionL. Req2LTL leverages LLMs for semantic decomposition and combines them with deterministic rule-based synthesis to ensure both syntactic validity and semantic fidelity. Our comprehensive evaluation demonstrates that Req2LTL achieves 88.4% semantic accuracy and 100% syntactic correctness on real-world aerospace requirements, significantly outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12284v2">V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 14 pages, 20 figures, conference
    </div>
    <details class="paper-abstract">
      Streaming video large language models (LLMs) are increasingly used for real-time multimodal tasks such as video captioning, question answering, conversational agents, and augmented reality. However, these models face fundamental memory and computational challenges because their key-value (KV) caches grow substantially with continuous streaming video input. This process requires an iterative prefill stage, which is a unique feature of streaming video LLMs. Due to its iterative prefill stage, it suffers from significant limitations, including extensive computation, substantial data transfer, and degradation in accuracy. Crucially, this issue is exacerbated for edge deployment, which is the primary target for these models. In this work, we propose V-Rex, the first software-hardware co-designed accelerator that comprehensively addresses both algorithmic and hardware bottlenecks in streaming video LLM inference. At its core, V-Rex introduces ReSV, a training-free dynamic KV cache retrieval algorithm. ReSV exploits temporal and spatial similarity-based token clustering to reduce excessive KV cache memory across video frames. To fully realize these algorithmic benefits, V-Rex offers a compact, low-latency hardware accelerator with a dynamic KV cache retrieval engine (DRE), featuring bit-level and early-exit based computing units. V-Rex achieves unprecedented real-time of 3.9-8.3 FPS and energy-efficient streaming video LLM inference on edge deployment with negligible accuracy loss. While DRE only accounts for 2.2% power and 2.0% area, the system delivers 1.9-19.7x speedup and 3.1-18.5x energy efficiency improvements over AGX Orin GPU. This work is the first to comprehensively tackle KV cache retrieval across algorithms and hardware, enabling real-time streaming video LLM inference on resource-constrained edge devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.16882v2">Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a commonly used technique to adapt large language models (LLMs) to downstream tasks. In practice, SFT on a full dataset is computationally expensive and sometimes suffers from overfitting or bias amplification. This facilitates the rise of data curation in SFT, which prioritizes the most valuable data to optimze. This work studies the online batch selection family that dynamically scores and filters samples during the training process. However, existing popular methods often (i) rely merely on the utility of data to select a subset while neglecting other crucial factors like diversity, (ii) rely on external resources such as reference models or validation sets, and (iii) incur extra training time over full-dataset training. To address these limitations, this work develops \textbf{UDS (Utility-Diversity Sampling)}, a framework for efficient online batch selection in SFT. UDS leverages the nuclear norm of the logits matrix to capture both data utility and intra-sample diversity, while estimating inter-sample diversity through efficient low-dimensional embedding comparisons with a lightweight memory buffer of historical samples. Such a design eliminates the need for external resources and unnecessary backpropagation, securing computational efficiency. Experiments on multiple benchmarks demonstrate that UDS consistently outperforms state-of-the-art online batch selection methods under varying data budgets, and significantly reduces training time compared to full-dataset fine-tuning. Code is available at https://github.com/gfyddha/UDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14121v2">SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Existing intelligent sports analysis systems mainly focus on "scoring and visualization," often lacking automatic performance diagnosis and interpretable training guidance. Recent advances in Large Language Models (LLMs) and motion analysis techniques provide new opportunities to address the above limitations. In this paper, we propose SportsGPT, an LLM-driven framework for interpretable sports motion assessment and training guidance, which establishes a closed loop from motion time-series input to professional training guidance. First, given a set of high-quality target models, we introduce MotionDTW, a two-stage time series alignment algorithm designed for accurate keyframe extraction from skeleton-based motion sequences. Subsequently, we design a Knowledge-based Interpretable Sports Motion Assessment Model (KISMAM) to obtain a set of interpretable assessment metrics (e.g., insufficient extension) by contrasting the keyframes with the target models. Finally, we propose SportsRAG, a RAG-based training guidance model built upon Qwen3. Leveraging a 6B-token knowledge base, it prompts the LLM to generate professional training guidance by retrieving domain-specific QA pairs. Experimental results demonstrate that MotionDTW significantly outperforms traditional methods with lower temporal error and higher IoU scores. Furthermore, ablation studies validate the KISMAM and SportsRAG, confirming that SportsGPT surpasses general LLMs in diagnostic accuracy and professionalism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16189v2">Mitigating Hallucinations in Healthcare LLMs with Granular Fact-Checking and Domain-Specific Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      In healthcare, it is essential for any LLM-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRa) on the MIMIC III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM model on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3,786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary model achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17259v1">Verifiability-First Agents: Provable Observability and Lightweight Audit Agents for Controlling Autonomous LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      As LLM-based agents grow more autonomous and multi-modal, ensuring they remain controllable, auditable, and faithful to deployer intent becomes critical. Prior benchmarks measured the propensity for misaligned behavior and showed that agent personalities and tool access significantly influence misalignment. Building on these insights, we propose a Verifiability-First architecture that (1) integrates run-time attestations of agent actions using cryptographic and symbolic methods, (2) embeds lightweight Audit Agents that continuously verify intent versus behavior using constrained reasoning, and (3) enforces challenge-response attestation protocols for high-risk operations. We introduce OPERA (Observability, Provable Execution, Red-team, Attestation), a benchmark suite and evaluation protocol designed to measure (i) detectability of misalignment, (ii) time to detection under stealthy strategies, and (iii) resilience of verifiability mechanisms to adversarial prompt and persona injection. Our approach shifts the evaluation focus from how likely misalignment is to how quickly and reliably misalignment can be detected and remediated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17247v1">Incorporating Error Level Noise Embedding for Improving LLM-Assisted Robustness in Persian Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) systems suffer significant performance degradation in noisy environments, a challenge that is especially severe for low-resource languages such as Persian. Even state-of-the-art models such as Whisper struggle to maintain accuracy under varying signal-to-noise ratios (SNRs). This study presents a robust noise-sensitive ASR error correction framework that combines multiple hypotheses and noise-aware modeling. Using noisy Persian speech, we generate 5-best hypotheses from a modified Whisper-large decoder. Error Level Noise (ELN) is introduced as a representation that captures semantic- and token-level disagreement across hypotheses, quantifying the linguistic distortions caused by noise. ELN thus provides a direct measure of noise-induced uncertainty, enabling the LLM to reason about the reliability of each hypothesis during correction. Three models are evaluated: (1) a base LLaMA-2-7B model without fine-tuning, (2) a fine-tuned variant trained on text-only hypotheses, and (3) a noise-conditioned model integrating ELN embeddings at both sentence and word levels. Experimental results demonstrate that the ELN-conditioned model achieves substantial reductions in Word Error Rate (WER). Specifically, on the challenging Mixed Noise test set, the proposed Fine-tuned + ELN (Ours) model reduces the WER from a baseline of 31.10\% (Raw Whisper) to 24.84\%, significantly surpassing the Fine-tuned (No ELN) text-only baseline of 30.79\%, whereas the original LLaMA-2-7B model increased the WER to 64.58\%, demonstrating that it is unable to correct Persian errors on its own. This confirms the effectiveness of combining multiple hypotheses with noise-aware embeddings for robust Persian ASR in noisy real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10689v2">Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have led to a growing interest in developing LLM-based agents for automating web tasks. However, these agents often struggle with even simple tasks on real-world websites due to their limited capability to understand and process complex web page structures. In this work, we introduce LCoW, a framework for Learning language models to Contextualize complex Web pages into a more comprehensible form, thereby enhancing decision making by LLM agents. LCoW decouples web page understanding from decision making by training a separate contextualization module to transform complex web pages into comprehensible format, which are then utilized by the decision-making agent. We demonstrate that our contextualization module effectively integrates with LLM agents of various scales to significantly enhance their decision-making capabilities in web automation tasks. Notably, LCoW improves the success rates of closed-source LLMs (e.g., Gemini-1.5-flash, GPT-4o, Claude-3.5-Sonnet) by an average of 15.6%, and demonstrates a 23.7% average improvement in success rates for open-source LMs (e.g., Llama-3.1-8B, Llama-3.1-70B) on the WorkArena benchmark. Moreover, the Gemini-1.5-flash agent with LCoW achieves state-of-the-art results on the WebShop benchmark, outperforming human experts. The relevant code materials are available at our project page: https://lcowiclr2025.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17061v4">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted at AAAI 2026 Workshop on WoMAPF, Camera ready version
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17172v1">PILAR: Personalizing Augmented Reality Interactions with LLM-based Human-Centric and Trustworthy Explanations for Daily Use Cases</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Published in the 2025 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct)
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI)-driven augmented reality (AR) systems are becoming increasingly integrated into daily life, and with this growth comes a greater need for explainability in real-time user interactions. Traditional explainable AI (XAI) methods, which often rely on feature-based or example-based explanations, struggle to deliver dynamic, context-specific, personalized, and human-centric insights for everyday AR users. These methods typically address separate explainability dimensions (e.g., when, what, how) with different explanation techniques, resulting in unrealistic and fragmented experiences for seamless AR interactions. To address this challenge, we propose PILAR, a novel framework that leverages a pre-trained large language model (LLM) to generate context-aware, personalized explanations, offering a more intuitive and trustworthy experience in real-time AI-powered AR systems. Unlike traditional methods, which rely on multiple techniques for different aspects of explanation, PILAR employs a unified LLM-based approach that dynamically adapts explanations to the user's needs, fostering greater trust and engagement. We implement the PILAR concept in a real-world AR application (e.g., personalized recipe recommendations), an open-source prototype that integrates real-time object detection, recipe recommendation, and LLM-based personalized explanations of the recommended recipes based on users' dietary preferences. We evaluate the effectiveness of PILAR through a user study with 16 participants performing AR-based recipe recommendation tasks, comparing an LLM-based explanation interface to a traditional template-based one. Results show that the LLM-based interface significantly enhances user performance and experience, with participants completing tasks 40% faster and reporting greater satisfaction, ease of use, and perceived transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06000v2">LLMs Do Not See Age: Assessing Demographic Bias in Automated Systematic Review Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted at AACL 2025 Version 2 Updated with Final version
    </div>
    <details class="paper-abstract">
      Clinical interventions often hinge on age: medications and procedures safe for adults may be harmful to children or ineffective for older adults. However, as language models are increasingly integrated into biomedical evidence synthesis workflows, it remains uncertain whether these systems preserve such crucial demographic distinctions. To address this gap, we evaluate how well state-of-the-art language models retain age-related information when generating abstractive summaries of biomedical studies. We construct DemogSummary, a novel age-stratified dataset of systematic review primary studies, covering child, adult, and older adult populations. We evaluate three prominent summarisation-capable LLMs, Qwen (open-source), Longformer (open-source) and GPT-4.1 Nano (proprietary), using both standard metrics and a newly proposed Demographic Salience Score (DSS), which quantifies age-related entity retention and hallucination. Our results reveal systematic disparities across models and age groups: demographic fidelity is lowest for adult-focused summaries, and under-represented populations are more prone to hallucinations. These findings highlight the limitations of current LLMs in faithful and bias-free summarisation and point to the need for fairness-aware evaluation frameworks and summarisation pipelines in biomedical NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.11027v4">On the Effect of Sampling Diversity in Scaling LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 33 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) scaling inference is key to unlocking greater performance, and leveraging diversity has proven an effective way to enhance it. Motivated by the observed relationship between solution accuracy and meaningful response diversity, we systematically study the effect of prompt diversity in scaling inference. We theoretically explain why diversified sampling improves Best-of-N scaling, showing that responses generated from diverse prompts after Best-of-N selection exhibit significantly lower error rates than those produced from stationary prompts. Building on this analysis, we derive a diversity-fidelity trade-off principle, that guides the design of sampling strategies introducing diversity. From this guidance, we instantiate a family of effective perturbation styles. We theoretically and empirically characterize when diversified exploration remains effective, demonstrating that it works under a variety of conditions, and we further show that under majority voting, diversity may vanish. Finally, we systematically evaluate how effective sampling diversity is and show that, when applied appropriately in different contexts, it yields relative gains of 10.8% in EM@100 for reasoning, 9.6% for mathematics, and 9.5% in Pass@100 for code generation. Overall, this work provides a systematic analysis that offers a theoretical and empirical foundation for understanding how sampling diversity affects LLM inference-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17145v1">Solomonoff-Inspired Hypothesis Ranking with LLMs for Prediction Under Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 10 pages, ACRA 2025, Submitted, Accepted and Presented
    </div>
    <details class="paper-abstract">
      Reasoning under uncertainty is a key challenge in AI, especially for real-world tasks, where problems with sparse data demands systematic generalisation. Existing approaches struggle to balance accuracy and simplicity when evaluating multiple candidate solutions. We propose a Solomonoff-inspired method that weights LLM-generated hypotheses by simplicity and predictive fit. Applied to benchmark (Mini-ARC) tasks, our method produces Solomonoff-weighted mixtures for per-cell predictions, yielding conservative, uncertainty-aware outputs even when hypotheses are noisy or partially incorrect. Compared to Bayesian Model Averaging (BMA), Solomonoff scoring spreads probability more evenly across competing hypotheses, while BMA concentrates weight on the most likely but potentially flawed candidates. Across tasks, this highlights the value of algorithmic information-theoretic priors for interpretable, reliable multi-hypothesis reasoning under uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20196v2">Understanding and supporting how developers prompt for LLM-powered code editing in practice</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly transforming software engineering, with coding assistants embedded in an IDE becoming increasingly prevalent. While research has focused on improving the tools and understanding developer perceptions, a critical gap exists in understanding how developers actually use these tools in their daily workflows, and, crucially, where they struggle. This paper addresses part of this gap through a multi-phased investigation of developer interactions with an LLM-powered code editing feature, Transform Code, in an IDE widely used at Google. First, we analyze telemetry logs of the feature usage, revealing that frequent re-prompting can be an indicator of developer struggles with using Transform Code. Second, we conduct a qualitative analysis of unsatisfactory requests, identifying five key categories of information often missing from developer prompts. Finally, based on these findings, we propose and evaluate a tool, AutoPrompter, for automatically improving prompts by inferring missing information from the surrounding code context, leading to a 27% improvement in edit correctness on our test set.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10729v3">Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 NeurIPS Workshop on Learning from Time Series for Health
    </div>
    <details class="paper-abstract">
      Sensor data streams provide valuable information around activities and context for downstream applications, though integrating complementary information can be challenging. We show that large language models (LLMs) can be used for late fusion for activity classification from audio and motion time series data. We curated a subset of data for diverse activity recognition across contexts (e.g., household activities, sports) from the Ego4D dataset. Evaluated LLMs achieved 12-class zero- and one-shot classification F1-scores significantly above chance, with no task-specific training. Zero-shot classification via LLM-based fusion from modality-specific models can enable multimodal temporal applications where there is limited aligned training data for learning a shared embedding space. Additionally, LLM-based fusion can enable model deploying without requiring additional memory and computation for targeted application-specific multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18131v1">Holistic Evaluation of State-of-the-Art LLMs for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 13 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      This study presents a comprehensive empirical evaluation of six state-of-the-art large language models (LLMs) for code generation, including both general-purpose and code-specialized models. Using a dataset of 944 real-world LeetCode problems across five programming languages, we assess model performance using rigorous metrics: compile-time errors, runtime errors, functional failures, and algorithmic suboptimalities. The results reveal significant performance variations, with DeepSeek-R1 and GPT-4.1 consistently outperform others in terms of correctness, efficiency, and robustness. Through detailed case studies, we identify common failure scenarios such as syntax errors, logical flaws, and suboptimal algorithms, highlighting the critical role of prompt engineering and human oversight in improving results. Based on these findings, we provide actionable recommendations for developers and practitioners, emphasizing that successful LLM deployment depends on careful model selection, effective prompt design, and context-aware usage to ensure reliable code generation in real-world software development tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22732v2">WebATLAS: An LLM Agent with Experience-Driven Memory and Action Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 9 pages, NeurIPS 2025 Workshop on Language Agents and World Models
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) web agents often struggle with long-horizon web navigation and web task completion in new websites, producing inefficient action sequences unless fine-tuned on environment-specific data. We show that experience-driven memory, combined with look-ahead action simulation, is sufficient for LLM agents to adapt to unseen web environments by remembering past failures and predicting the consequences of future actions. We introduce WebATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented LLM web agent that learns a lightweight internal model of the environment from interaction experience and performs hypothetical action rollouts before acting in the real world. WebATLAS builds a persistent cognitive map via curiosity-driven exploration, stores interaction outcomes as experience-based memory, and evaluates candidate actions in cognitive space using a planner--simulator--critic loop. This enables the agent to reuse past experience, avoid previously unsuccessful behaviors, and generate more efficient plans. We evaluate WebATLAS on the WebArena-Lite benchmark for autonomous web navigation and demonstrate a success rate of 63%, outperforming the previous state-of-the-art at 53.9%. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablation studies confirm that experience-driven memory, look-ahead action simulation, and hierarchical replanning play complementary roles in enabling robust, training-free web agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13901v2">RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14918v2">Reliable Decision Support with LLMs: A Framework for Evaluating Consistency in Binary Text Classification Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      This study introduces a framework for evaluating consistency in large language model (LLM) binary text classification, addressing the lack of established reliability assessment methods. Adapting psychometric principles, we determine sample size requirements, develop metrics for invalid responses, and evaluate intra- and inter-rater reliability. Our case study examines financial news sentiment classification across 14 LLMs (including claude-3-7-sonnet, gpt-4o, deepseek-r1, gemma3, llama3.2, phi4, and command-r-plus), with five replicates per model on 1,350 articles. Models demonstrated high intra-rater consistency, achieving perfect agreement on 90-98% of examples, with minimal differences between expensive and economical models from the same families. When validated against StockNewsAPI labels, models achieved strong performance (accuracy 0.76-0.88), with smaller models like gemma3:1B, llama3.2:3B, and claude-3-5-haiku outperforming larger counterparts. All models performed at chance when predicting actual market movements, indicating task constraints rather than model limitations. Our framework provides systematic guidance for LLM selection, sample size planning, and reliability assessment, enabling organizations to optimize resources for classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14625v2">Accelerated Preference Elicitation with LLM-Based Proxies</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 In Proceedings of The 21st Conference on Web and Internet Economics (WINE '25)
    </div>
    <details class="paper-abstract">
      Bidders in combinatorial auctions face significant challenges when describing their preferences to an auctioneer. Classical work on preference elicitation focuses on query-based techniques inspired from proper learning--often via proxies that interface between bidders and an auction mechanism--to incrementally learn bidder preferences as needed to compute efficient allocations. Although such elicitation mechanisms enjoy theoretical query efficiency, the amount of communication required may still be too cognitively taxing in practice. We propose a family of efficient LLM-based proxy designs for eliciting preferences from bidders using natural language. Our proposed mechanism combines LLM pipelines and DNF-proper-learning techniques to quickly approximate preferences when communication is limited. To validate our approach, we create a testing sandbox for elicitation mechanisms that communicate in natural language. In our experiments, our most promising LLM proxy design reaches approximately efficient outcomes with five times fewer queries than classical proper learning based elicitation mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.18020v1">Specification and Detection of LLM Code Smells</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Accepted paper at ICSE NIER 2026 : https://conf.researchr.org/track/icse-2026/icse-2026-nier
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained massive popularity in recent years and are increasingly integrated into software systems for diverse purposes. However, poorly integrating them in source code may undermine software system quality. Yet, to our knowledge, there is no formal catalog of code smells specific to coding practices for LLM inference. In this paper, we introduce the concept of LLM code smells and formalize five recurrent problematic coding practices related to LLM inference in software systems, based on relevant literature. We extend the detection tool SpecDetect4AI to cover the newly defined LLM code smells and use it to validate their prevalence in a dataset of 200 open-source LLM systems. Our results show that LLM code smells affect 60.50% of the analyzed systems, with a detection precision of 86.06%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17970v1">CodeGEMM: A Codebook-Centric Approach to Efficient GEMM in Quantized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Weight-only quantization is widely used to mitigate the memory-bound nature of LLM inference. Codebook-based methods extend this trend by achieving strong accuracy in the extremely low-bit regime (e.g., 2-bit). However, current kernels rely on dequantization, which repeatedly fetches centroids and reconstructs weights, incurring substantial latency and cache pressure. We present CodeGEMM, a codebook-centric GEMM kernel that replaces dequantization with precomputed inner products between centroids and activations stored in a lightweight Psumbook. At inference, code indices directly gather these partial sums, eliminating per-element lookups and reducing the on-chip footprint. The kernel supports the systematic exploration of latency-memory-accuracy trade-offs under a unified implementation. On Llama-3 models, CodeGEMM delivers 1.83x (8B) and 8.93x (70B) speedups in the 2-bit configuration compared to state-of-the-art codebook-based quantization at comparable accuracy and further improves computing efficiency and memory subsystem utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20664v1">Eidoku: A Neuro-Symbolic Verification Gate for LLM Reasoning via Structural Constraint Satisfaction</a></div>
    <div class="paper-meta">
      📅 2025-12-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently produce hallucinated statements that are assigned high likelihood by the model itself, exposing a fundamental limitation of probability-based verification. This suggests that hallucination is often not a low-confidence phenomenon, but a failure of structural consistency. In this work, we reformulate the verification of LLM reasoning as a Constraint Satisfaction Problem (CSP) operating independently of the generation likelihood. Rather than optimizing for statistical plausibility, we model verification as a feasibility check based on structural violation cost -- the computational cost required to embed a candidate reasoning step into the contextual graph structure. We define a total cost function composed of three proxies: (i) graph connectivity (structural), (ii) feature space consistency (geometric), and (iii) logical entailment (symbolic). Crucially, verification is performed via a lightweight System-2 gate, Eidoku, which rejects candidates exceeding a context-calibrated cost threshold. The threshold is not learned but is derived from the intrinsic statistics of the context, avoiding ad hoc heuristics. We demonstrate that this approach successfully rejects ``smooth falsehoods'' -- statements that are highly probable yet structurally disconnected -- that probability-based verifiers are principally incapable of detecting. Our experiments on a controlled diagnostic dataset show that explicitly enforcing structural constraints allows for the deterministic rejection of this specific class of hallucinations, serving as a neuro-symbolic sanity check for generative reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.10321v3">Towards Human-Guided, Data-Centric LLM Co-Pilots</a></div>
    <div class="paper-meta">
      📅 2025-12-19
      | 💬 Saveliev, Liu & Seedat contributed equally
    </div>
    <details class="paper-abstract">
      Machine learning (ML) has the potential to revolutionize various domains, but its adoption is often hindered by the disconnect between the needs of domain experts and translating these needs into robust and valid ML tools. Despite recent advances in LLM-based co-pilots to democratize ML for non-technical domain experts, these systems remain predominantly focused on model-centric aspects while overlooking critical data-centric challenges. This limitation is problematic in complex real-world settings where raw data often contains complex issues, such as missing values, label noise, and domain-specific nuances requiring tailored handling. To address this we introduce CliMB-DC, a human-guided, data-centric framework for LLM co-pilots that combines advanced data-centric tools with LLM-driven reasoning to enable robust, context-aware data processing. At its core, CliMB-DC introduces a novel, multi-agent reasoning system that combines a strategic coordinator for dynamic planning and adaptation with a specialized worker agent for precise execution. Domain expertise is then systematically incorporated to guide the reasoning process using a human-in-the-loop approach. To guide development, we formalize a taxonomy of key data-centric challenges that co-pilots must address. Thereafter, to address the dimensions of the taxonomy, we integrate state-of-the-art data-centric tools into an extensible, open-source architecture, facilitating the addition of new tools from the research community. Empirically, using real-world healthcare datasets we demonstrate CliMB-DC's ability to transform uncurated datasets into ML-ready formats, significantly outperforming existing co-pilot baselines for handling data-centric challenges. CliMB-DC promises to empower domain experts from diverse domains -- healthcare, finance, social sciences and more -- to actively participate in driving real-world impact using ML.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16473v1">Efficient CPU-GPU Collaborative Inference for MoE-based LLMs on Memory-Limited Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 7 pages, 6 figures, to be published in ASP-DAC 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved impressive results across various tasks, yet their high computational demands pose deployment challenges, especially on consumer-grade hardware. Mixture of Experts (MoE) models provide an efficient solution through selective activation of parameter subsets, which reduces computation requirements. Despite this efficiency, state-of-the-art MoE models still require substantial memory beyond typical consumer GPU capacities. Traditional offloading methods that transfer model weights between CPU and GPU introduce latency, limiting inference performance. This paper presents a novel CPU-GPU collaborative inference framework that incorporates an expert caching mechanism on the GPU to reduce data transfer requirements and enable faster inference through cache hits. Computations are offloaded to CPU for efficient cache miss handling, which benefits from CPU multithreading optimizations. The evaluations of our framework demonstrate performance improvements and highlight the potential of CPU-GPU collaboration to maximize hardware utilization for single-request inference scenarios on consumer-grade systems. The implementation of our framework is available at https://github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.17361v2">Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA). We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective against basic and reasoning models, but are also transferable across model families (OpenAI, Anthropic, Google), and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.00326v24">Agent-OM: Leveraging LLM Agents for Ontology Matching</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM (Agent for Ontology Matching), consisting of two Siamese agents for retrieval and matching, with a set of OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAEI) tracks over state-of-the-art OM systems show that our system can achieve results very close to the long-standing best performance on simple OM tasks and can significantly improve the performance on complex and few-shot OM tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16424v1">Synthelite: Chemist-aligned and feasibility-aware synthesis planning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Computer-aided synthesis planning (CASP) has long been envisioned as a complementary tool for synthetic chemists. However, existing frameworks often lack mechanisms to allow interaction with human experts, limiting their ability to integrate chemists' insights. In this work, we introduce Synthelite, a synthesis planning framework that uses large language models (LLMs) to directly propose retrosynthetic transformations. Synthelite can generate end-to-end synthesis routes by harnessing the intrinsic chemical knowledge and reasoning capabilities of LLMs, while allowing expert intervention through natural language prompts. Our experiments demonstrate that Synthelite can flexibly adapt its planning trajectory to diverse user-specified constraints, achieving up to 95\% success rates in both strategy-constrained and starting-material-constrained synthesis tasks. Additionally, Synthelite exhibits the ability to account for chemical feasibility during route design. We envision Synthelite to be both a useful tool and a step toward a paradigm where LLMs are the central orchestrators of synthesis planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13302v3">LLM one-shot style transfer for Authorship Attribution and Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Computational stylometry studies writing style through quantitative textual patterns, enabling applications such as authorship attribution, identity linking, and plagiarism detection. Existing supervised and contrastive approaches often rely on datasets with spurious correlations, conflating style with topic. Despite the relevance of language modeling to these tasks, the pre-training of modern large language models (LLMs) has been underutilized in general authorship analysis. We introduce an unsupervised framework that uses the log-probabilities of an LLM to measure style transferability between two texts. This framework takes advantage of the extensive CLM pre-training and in-context capabilities of modern LLMs. Our approach avoids explicit supervision with spuriously correlated data. Our method substantially outperforms unsupervised prompting-based baselines at similar model sizes and exceeds contrastively trained models when controlling for topical overlap. Our framework's performance improves with model size. In the case of authorship verification, we present an additional mechanism that increases test-time computation to improve accuracy; enabling flexible trade-offs between computational cost and task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08158v3">Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16391v1">Kascade: A Practical Sparse Attention Method for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 11 pages, 8 figures, 3 tables and 1 algorithm
    </div>
    <details class="paper-abstract">
      Attention is the dominant source of latency during long-context LLM inference, an increasingly popular workload with reasoning models and RAG. We propose Kascade, a training-free sparse attention method that leverages known observations such as 1) post-softmax attention is intrinsically sparse, and 2) the identity of high-weight keys is stable across nearby layers. Kascade computes exact Top-k indices in a small set of anchor layers, then reuses those indices in intermediate reuse layers. The anchor layers are selected algorithmically, via a dynamic-programming objective that maximizes cross-layer similarity over a development set, allowing easy deployment across models. The method incorporates efficient implementation constraints (e.g. tile-level operations), across both prefill and decode attention. The Top-k selection and reuse in Kascade is head-aware and we show in our experiments that this is critical for high accuracy. Kascade achieves up to 4.1x speedup in decode attention and 2.2x speedup in prefill attention over FlashAttention-3 baseline on H100 GPUs while closely matching dense attention accuracy on long-context benchmarks such as LongBench and AIME-24.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16378v1">Hearing to Translate: The Effectiveness of Speech Modality Integration into LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Project available at https://github.com/sarapapi/hearing2translate
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) expand beyond text, integrating speech as a native modality has given rise to SpeechLLMs, which aim to translate spoken language directly, thereby bypassing traditional transcription-based pipelines. Whether this integration improves speech-to-text translation quality over established cascaded architectures, however, remains an open question. We present Hearing to Translate, the first comprehensive test suite rigorously benchmarking 5 state-of-the-art SpeechLLMs against 16 strong direct and cascade systems that couple leading speech foundation models (SFM), with multilingual LLMs. Our analysis spans 16 benchmarks, 13 language pairs, and 9 challenging conditions, including disfluent, noisy, and long-form speech. Across this extensive evaluation, we find that cascaded systems remain the most reliable overall, while current SpeechLLMs only match cascades in selected settings and SFMs lag behind both, highlighting that integrating an LLM, either within the model or in a pipeline, is essential for high-quality speech translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10795v3">Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.01396v2">Easy Come, Easy Go? Examining the Perceptions and Learning Effects of LLM-based Chatbot in the Context of Search-as-Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 25 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The cognitive process of Search-as-Learning (SAL) is most effective when searching promotes active encoding of information. The rise of LLMs-based chatbots, which provide instant answers, introduces a trade-off between efficiency and depth of processing. Such answer-centric approaches accelerate information access, but they also raise concerns about shallower learning. To examine these issues in the context of SAL, we conducted a large-scale survey of educators and students to capture perceived risks and benefits of LLM-based chatbots. In addition, we adopted the encoding-storage paradigm to design a within-subjects experiment, where participants (N=92) engaged in SAL tasks using three different modalities: books, search engines, and chatbots. Our findings provide a counterintuitive insight into stakeholder concerns: while LLM-based chatbots and search engines validated perceived benefits on learning efficiency by outperforming book-based search in immediate conceptual understanding, they did not result in a long-term inferiority as feared. Our study provides insights for designing human-AI collaborative learning systems that promote cognitive engagement by balancing learning efficiency and long-term knowledge retention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16317v1">Design and Evaluation of Cost-Aware PoQ for Decentralized LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Decentralized large language model (LLM) inference promises transparent and censorship resistant access to advanced AI, yet existing verification approaches struggle to scale to modern models. Proof of Quality (PoQ) replaces cryptographic verification of computation with consensus over output quality, but the original formulation ignores heterogeneous computational costs across inference and evaluator nodes. This paper introduces a cost-aware PoQ framework that integrates explicit efficiency measurements into the reward mechanism for both types of nodes. The design combines ground truth token level F1, lightweight learned evaluators, and GPT based judgments within a unified evaluation pipeline, and adopts a linear reward function that balances normalized quality and cost. Experiments on extractive question answering and abstractive summarization use five instruction tuned LLMs ranging from TinyLlama-1.1B to Llama-3.2-3B and three evaluation models spanning cross encoder and bi encoder architectures. Results show that a semantic textual similarity bi encoder achieves much higher correlation with both ground truth and GPT scores than cross encoders, indicating that evaluator architecture is a critical design choice for PoQ. Quality-cost analysis further reveals that the largest models in the pool are also the most efficient in terms of quality per unit latency. Monte Carlo simulations over 5\,000 PoQ rounds demonstrate that the cost-aware reward scheme consistently assigns higher average rewards to high quality low cost inference models and to efficient evaluators, while penalizing slow low quality nodes. These findings suggest that cost-aware PoQ provides a practical foundation for economically sustainable decentralized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04070v4">LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Accepted at AAAI-26
    </div>
    <details class="paper-abstract">
      Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study focuses on generating precise, in-situ, step-by-step navigation instructions that are practically usable for VI users. Specifically, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to navigation instructions, thereby providing feedback rewards to guide the post-training of a Vision-Language Model (VLM). This enhances instruction accuracy and usability while reducing costly real-world data collection needs. To address the scarcity of dedicated benchmarks in this field, we introduce NIG4VI, a 27k-sample open-source dataset to facilitate training and evaluation. It comprises diverse navigation scenarios with accurate spatial coordinates, supporting detailed and open-ended in-situ instruction generation. Experiments on NIG4VI demonstrate the effectiveness of LaF-GRPO through quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU 14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o 0.323), and qualitative analysis further confirms that our method yields more intuitive and safer instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16272v1">Beyond Blind Spots: Analytic Hints for Mitigating LLM-Based Evaluation Pitfalls</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly deployed as judges (LaaJ) in code generation pipelines. While attractive for scalability, LaaJs tend to overlook domain specific issues raising concerns about their reliability in critical evaluation tasks. To better understand these limitations in practice, we examine LaaJ behavior in a concrete industrial use case: legacy code modernization via COBOL code generation. In this setting, we find that even production deployed LaaJs can miss domain critical errors, revealing consistent blind spots in their evaluation capabilities. To better understand these blind spots, we analyze generated COBOL programs and associated LaaJs judgments, drawing on expert knowledge to construct a preliminary taxonomy. Based on this taxonomy, we develop a lightweight analytic checker tool that flags over 30 domain specific issues observed in practice. We use its outputs as analytic hints, dynamically injecting them into the judges prompt to encourage LaaJ to revisit aspects it may have overlooked. Experiments on a test set of 100 programs using four production level LaaJs show that LaaJ alone detects only about 45% of the errors present in the code (in all judges we tested), while the analytic checker alone lacks explanatory depth. When combined, the LaaJ+Hints configuration achieves up to 94% coverage (for the best performing judge and injection prompt) and produces qualitatively richer, more accurate explanations, demonstrating that analytic-LLM hybrids can substantially enhance evaluation reliability in deployed pipelines. We release the dataset and all used prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.15108v2">Knowledge Hierarchy Guided Biological-Medical Dataset Distillation for Domain LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) in biological-medical applications has highlighted a gap between their potential and the limited scale and often low quality of available open-source annotated textual datasets. In addition, the inherent complexity of the biomedical knowledge hierarchy significantly hampers efforts to bridge this gap.Can LLMs themselves play a pivotal role in overcoming this limitation? Motivated by this question, we investigate this challenge in the present study.We propose a framework that automates the distillation of high-quality textual training data from the extensive scientific literature. Our approach self-evaluates and generates questions that are more closely aligned with the biomedical domain, guided by the biomedical knowledge hierarchy through medical subject headings (MeSH). This comprehensive framework establishes an automated workflow, thereby eliminating the need for manual intervention. Furthermore, we conducted comprehensive experiments to evaluate the impact of our framework-generated data on downstream language models of varying sizes. Our approach substantially improves question-answering tasks compared to pre-trained models from the life sciences domain and powerful close-source models represented by GPT-4. Notably, the generated AI-Ready dataset enabled the Llama3-70B base model to outperform GPT-4 using MedPrompt with multiple times the number of parameters. Detailed case studies and ablation experiments underscore the significance of each component within our framework
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16189v1">Mitigating Hallucinations in Healthcare LLMs with Granular Fact-Checking and Domain-Specific Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      In healthcare, it is essential for any LLM-generated output to be reliable and accurate, particularly in cases involving decision-making and patient safety. However, the outputs are often unreliable in such critical areas due to the risk of hallucinated outputs from the LLMs. To address this issue, we propose a fact-checking module that operates independently of any LLM, along with a domain-specific summarization model designed to minimize hallucination rates. Our model is fine-tuned using Low-Rank Adaptation (LoRa) on the MIMIC III dataset and is paired with the fact-checking module, which uses numerical tests for correctness and logical checks at a granular level through discrete logic in natural language processing (NLP) to validate facts against electronic health records (EHRs). We trained the LLM model on the full MIMIC-III dataset. For evaluation of the fact-checking module, we sampled 104 summaries, extracted them into 3,786 propositions, and used these as facts. The fact-checking module achieves a precision of 0.8904, a recall of 0.8234, and an F1-score of 0.8556. Additionally, the LLM summary model achieves a ROUGE-1 score of 0.5797 and a BERTScore of 0.9120 for summary quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16167v1">Ev-Trust: A Strategy Equilibrium Trust Mechanism for Evolutionary Games in LLM-Based Multi-Agent Services</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 12 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The rapid evolution of the Web toward an agent-centric paradigm, driven by large language models (LLMs), has enabled autonomous agents to reason, plan, and interact in complex decentralized environments. However, the openness and heterogeneity of LLM-based multi-agent systems also amplify the risks of deception, fraud, and misinformation, posing severe challenges to trust establishment and system robustness. To address this issue, we propose Ev-Trust, a strategy-equilibrium trust mechanism grounded in evolutionary game theory. This mechanism integrates direct trust, indirect trust, and expected revenue into a dynamic feedback structure that guides agents' behavioral evolution toward equilibria. Within a decentralized "Request-Response-Payment-Evaluation" service framework, Ev-Trust enables agents to adaptively adjust strategies, naturally excluding malicious participants while reinforcing high-quality collaboration. Furthermore, our theoretical derivation based on replicator dynamics equations proves the existence and stability of local evolutionary equilibria. Experimental results indicate that our approach effectively reflects agent trustworthiness in LLM-driven open service interaction scenarios, reduces malicious strategies, and increases collective revenue. We hope Ev-Trust can provide a new perspective on trust modeling for the agentic service web in group evolutionary game scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21557v2">Generation-Time vs. Post-hoc Citation: A Holistic Evaluation of LLM Attribution</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 NeurIPS 2025 LLM Evaluation Workshop
    </div>
    <details class="paper-abstract">
      Trustworthy Large Language Models (LLMs) must cite human-verifiable sources in high-stakes domains such as healthcare, law, academia, and finance, where even small errors can have severe consequences. Practitioners and researchers face a choice: let models generate citations during decoding, or let models draft answers first and then attach appropriate citations. To clarify this choice, we introduce two paradigms: Generation-Time Citation (G-Cite), which produces the answer and citations in one pass, and Post-hoc Citation (P-Cite), which adds or verifies citations after drafting. We conduct a comprehensive evaluation from zero-shot to advanced retrieval-augmented methods across four popular attribution datasets and provide evidence-based recommendations that weigh trade-offs across use cases. Our results show a consistent trade-off between coverage and citation correctness, with retrieval as the main driver of attribution quality in both paradigms. P-Cite methods achieve high coverage with competitive correctness and moderate latency, whereas G-Cite methods prioritize precision at the cost of coverage and speed. We recommend a retrieval-centric, P-Cite-first approach for high-stakes applications, reserving G-Cite for precision-critical settings such as strict claim verification. Our codes and human evaluation results are available at https://anonymous.4open.science/r/Citation_Paradigms-BBB5/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16134v1">Staggered Batch Scheduling: Co-optimizing Time-to-First-Token and Throughput for High-Efficiency LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Model (LLM) serving towards complex, distributed architectures--specifically the P/D-separated, large-scale DP+EP paradigm--introduces distinct scheduling challenges. Unlike traditional deployments where schedulers can treat instances as black boxes, DP+EP architectures exhibit high internal synchronization costs. We identify that immediate request dispatching in such systems leads to severe in-engine queuing and parallelization bubbles, degrading Time-to-First-Token (TTFT). To address this, we propose Staggered Batch Scheduling (SBS), a mechanism that deliberately buffers requests to form optimal execution batches. This temporal decoupling eliminates internal queuing bubbles without compromising throughput. Furthermore, leveraging the scheduling window created by buffering, we introduce a Load-Aware Global Allocation strategy that balances computational load across DP units for both Prefill and Decode phases. Deployed on a production H800 cluster serving Deepseek-V3, our system reduces TTFT by 30%-40% and improves throughput by 15%-20% compared to state-of-the-art immediate scheduling baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19108v4">A Multi-Language Perspective on the Robustness of LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 46 pages, 10 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Large language models have gained significant traction and popularity in recent times, extending their usage to code-generation tasks. While this field has garnered considerable attention, the exploration of testing and evaluating the robustness of code generation models remains an ongoing endeavor. Previous studies have primarily focused on code generation models specifically for the Python language, overlooking other widely-used programming languages. In this work, we conduct a comprehensive comparative analysis to assess the robustness performance of several prominent code generation models and investigate whether robustness can be improved by repairing perturbed docstrings using an LLM. Furthermore, we investigate how their performance varies across different programming languages. To accomplish this, we introduce perturbations in four key areas of the prompt: DocString, functionname, syntax, and format. We have compiled and released a dedicated dataset for this purpose. This work presents our experimental findings, shedding light on the performance of code generation models in various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09254v2">The Illusion of Rationality: Tacit Bias and Strategic Dominance in Frontier LLM Negotiation Games</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed as autonomous agents on behalf of institutions and individuals in economic, political, and social settings that involve negotiation. Yet this trend carries significant risks if their strategic behavior is not well understood. In this work, we revisit the NegotiationArena framework and run controlled simulation experiments on a diverse set of frontier LLMs across three multi turn bargaining games: Buyer Seller, Multi turn Ultimatum, and Resource Exchange. We ask whether improved general reasoning capabilities lead to rational, unbiased, and convergent negotiation strategies. Our results challenge this assumption. We find that models diverge into distinct, model specific strategic equilibria rather than converging to a unified optimal behavior. Moreover, strong numerical and semantic anchoring effects persist: initial offers are highly predictive of final agreements, and models consistently generate biased proposals by collapsing diverse internal valuations into rigid, generic price points. More concerningly, we observe dominance patterns in which some models systematically achieve higher payoffs than their counterparts. These findings underscore an urgent need to develop mechanisms to mitigate these issues before deploying such systems in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.06489v3">On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Published in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.15251v2">Input Reduction Enhanced LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great potential in Automated Program Repair (APR). Test inputs, being crucial for reasoning the root cause of failures, are always included in the prompt for LLM-based APR. Unfortunately, LLMs struggle to retain key information in long prompts. When the test inputs are extensive in the prompt, this may trigger the "lost-in-the-middle" issue, compromising repair performance. ReduceFix prompts an LLM to generate a reducer that minimizes failure-inducing test inputs without human effort, and then feeds the reduced failure-inducing inputs to guide patch generation. For targeted evaluation, we constructed LFTBench, the first long-input APR benchmark with 200 real bugs from 20 programming tasks, each paired with a failure-inducing input whose median size is 1 MB. On this benchmark, ReduceFix shrinks inputs by 89.1% on average and improves overall pass@10 by up to 53.8% relative to a prompt that includes the original test, and by 17.6% compared with omitting the test entirely. Adding the same reduction step to ChatRepair and CREF increases their fix rate by 21.3% and 2.6%, respectively, without other changes. Our gains hold against a ddmin-only reducing template baseline and transfer to repository-level OSS-Fuzz cases. Ablation studies further highlight the impact of input length and compressed failure information on repair success. These results underscore that automatically reducing failing inputs is a practical and powerful complement to LLM-based APR, significantly improving its scalability and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16083v1">Scaling Text2SQL via LLM-efficient Schema Filtering with Functional Dependency Graph Rerankers</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Most modern Text2SQL systems prompt large language models (LLMs) with entire schemas -- mostly column information -- alongside the user's question. While effective on small databases, this approach fails on real-world schemas that exceed LLM context limits, even for commercial models. The recent Spider 2.0 benchmark exemplifies this with hundreds of tables and tens of thousands of columns, where existing systems often break. Current mitigations either rely on costly multi-step prompting pipelines or filter columns by ranking them against user's question independently, ignoring inter-column structure. To scale existing systems, we introduce \toolname, an open-source, LLM-efficient schema filtering framework that compacts Text2SQL prompts by (i) ranking columns with a query-aware LLM encoder enriched with values and metadata, (ii) reranking inter-connected columns via a lightweight graph transformer over functional dependencies, and (iii) selecting a connectivity-preserving sub-schema with a Steiner-tree heuristic. Experiments on real datasets show that \toolname achieves near-perfect recall and higher precision than CodeS, SchemaExP, Qwen rerankers, and embedding retrievers, while maintaining sub-second median latency and scaling to schemas with 23,000+ columns. Our source code is available at https://github.com/thanhdath/grast-sql.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16056v1">MultiPath Transfer Engine: Breaking GPU and Host-Memory Bandwidth Bottlenecks in LLM Services</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      The limited bandwidth of PCIe has emerged as the critical bottleneck for large language model (LLM) performance, such as prefix cache fetching and model switching. Although intra-server multipath data transfer between GPU and host memory is theoretically possible, heterogeneous protocols such as PCIe and NVLink currently limit the bandwidth between host memory and GPUs to that of a single PICe link. This limitation resuals in underutilized intra-server bandwidth. To address this issue, we propose Multipath Memory Access (MMA), a scheme that, to the best of our knowledge, is the first to enalbe efficient multipath data transfer between GPU and host memory. MMA supports seamless deployment via dynamic library injection, enabling LLM applications to benefit from MMA without requiring any code modification. In our testbed, MMA significantly improves the data transfer bandwidth between the GPU and memory, achieving a peak bandwidth of 245 GB/s-representing a 4.62x speedup compared to the natice single-path bandwidth. End-to-end evaluations demonstrate that MMA reduces the time-to-first-token (TTFT) for LLM serving by 1.14x to 2.38x and decreases model-switching latency in vLLM's sleep mode by 1.12x to 2.48x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17093v1">A Solver-in-the-Loop Framework for Improving LLMs on Answer Set Programming for Logic Puzzle Solving</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 15 pages, 7 figures, accepted at AAAI'26
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has sparked interest in coding assistants. While general-purpose programming languages are well supported, generating code for domain-specific languages remains a challenging problem for LLMs. In this paper, we focus on the LLM-based generation of code for Answer Set Programming (ASP), a particularly effective approach for finding solutions to combinatorial search problems. The effectiveness of LLMs in ASP code generation is currently hindered by the limited number of examples seen during their initial pre-training phase. In this paper, we introduce a novel ASP-solver-in-the-loop approach for solver-guided instruction-tuning of LLMs to addressing the highly complex semantic parsing task inherent in ASP code generation. Our method only requires problem specifications in natural language and their solutions. Specifically, we sample ASP statements for program continuations from LLMs for unriddling logic puzzles. Leveraging the special property of declarative ASP programming that partial encodings increasingly narrow down the solution space, we categorize them into chosen and rejected instances based on solver feedback. We then apply supervised fine-tuning to train LLMs on the curated data and further improve robustness using a solver-guided search that includes best-of-N sampling. Our experiments demonstrate consistent improvements in two distinct prompting settings on two datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17062v1">Lang2Manip: A Tool for LLM-Based Symbolic-to-Geometric Planning for Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Submitted to ICARA
    </div>
    <details class="paper-abstract">
      Simulation is essential for developing robotic manipulation systems, particularly for task and motion planning (TAMP), where symbolic reasoning interfaces with geometric, kinematic, and physics-based execution. Recent advances in Large Language Models (LLMs) enable robots to generate symbolic plans from natural language, yet executing these plans in simulation often requires robot-specific engineering or planner-dependent integration. In this work, we present a unified pipeline that connects an LLM-based symbolic planner with the Kautham motion planning framework to achieve generalizable, robot-agnostic symbolic-to-geometric manipulation. Kautham provides ROS-compatible support for a wide range of industrial manipulators and offers geometric, kinodynamic, physics-driven, and constraint-based motion planning under a single interface. Our system converts language instructions into symbolic actions and computes and executes collision-free trajectories using any of Kautham's planners without additional coding. The result is a flexible and scalable tool for language-driven TAMP that is generalized across robots, planning modalities, and manipulation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17060v1">On the Role of Contextual Information and Ego States in LLM Agent Behavior for Transactional Analysis Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Presented at the 39th Pacific Asia Conference on Language, Information and Computation (PACLIC 39)
    </div>
    <details class="paper-abstract">
      LLM-powered agents are now used in many areas, from customer support to education, and there is increasing interest in their ability to act more like humans. This includes fields such as social, political, and psychological research, where the goal is to model group dynamics and social behavior. However, current LLM agents often lack the psychological depth and consistency needed to capture the real patterns of human thinking. They usually provide direct or statistically likely answers, but they miss the deeper goals, emotional conflicts, and motivations that drive real human interactions. This paper proposes a Multi-Agent System (MAS) inspired by Transactional Analysis (TA) theory. In the proposed system, each agent is divided into three ego states - Parent, Adult, and Child. The ego states are treated as separate knowledge structures with their own perspectives and reasoning styles. To enrich their response process, they have access to an information retrieval mechanism that allows them to retrieve relevant contextual information from their vector stores. This architecture is evaluated through ablation tests in a simulated dialogue scenario, comparing agents with and without information retrieval. The results are promising and open up new directions for exploring how psychologically grounded structures can enrich agent behavior. The contribution is an agent architecture that integrates Transactional Analysis theory with contextual information retrieval to enhance the realism of LLM-based multi-agent simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17043v1">UniRel-R1: RL-tuned LLM Reasoning for Knowledge Graph Relational Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) has traditionally focused on entity-centric queries that return a single answer entity. However, real-world queries are often relational, seeking to understand how entities are associated. In this work, we introduce relation-centric KGQA, a complementary setting where the answer is a subgraph capturing the semantic connections among entities rather than an individual entity. The main challenge lies in the abundance of candidate subgraphs, where trivial or overly common connections often obscure the identification of unique and informative answers. To tackle this, we propose UniRel-R1, a unified framework that integrates subgraph selection, multi-stage graph pruning, and an LLM fine-tuned with reinforcement learning. The reward function is designed to encourage compact and specific subgraphs with more informative relations and lower-degree intermediate entities. Extensive experiments show that UniRel-R1 achieves significant gains in connectivity and reward over Vanilla baselines and generalizes effectively to unseen entities and relations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17023v1">LLM-HPC++: Evaluating LLM-Generated Modern C++ and MPI+OpenMP Codes for Scalable Mandelbrot Set Computation</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Parallel programming remains one of the most challenging aspects of High-Performance Computing (HPC), requiring deep knowledge of synchronization, communication, and memory models. While modern C++ standards and frameworks like OpenMP and MPI have simplified parallelism, mastering these paradigms is still complex. Recently, Large Language Models (LLMs) have shown promise in automating code generation, but their effectiveness in producing correct and efficient HPC code is not well understood. In this work, we systematically evaluate leading LLMs including ChatGPT 4 and 5, Claude, and LLaMA on the task of generating C++ implementations of the Mandelbrot set using shared-memory, directive-based, and distributed-memory paradigms. Each generated program is compiled and executed with GCC 11.5.0 to assess its correctness, robustness, and scalability. Results show that ChatGPT-4 and ChatGPT-5 achieve strong syntactic precision and scalable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17008v1">Turn-PPO: Turn-Level Advantage Estimation with PPO for Improved Multi-Turn RL in Agentic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has re-emerged as a natural approach for training interactive LLM agents in real-world environments. However, directly applying the widely used Group Relative Policy Optimization (GRPO) algorithm to multi-turn tasks exposes notable limitations, particularly in scenarios requiring long-horizon reasoning. To address these challenges, we investigate more stable and effective advantage estimation strategies, especially for multi-turn settings. We first explore Proximal Policy Optimization (PPO) as an alternative and find it to be more robust than GRPO. To further enhance PPO in multi-turn scenarios, we introduce turn-PPO, a variant that operates on a turn-level MDP formulation, as opposed to the commonly used token-level MDP. Our results on the WebShop and Sokoban datasets demonstrate the effectiveness of turn-PPO, both with and without long reasoning components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16969v1">Probing Scientific General Intelligence of LLMs with Scientist-Aligned Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Despite advances in scientific AI, a coherent framework for Scientific General Intelligence (SGI)-the ability to autonomously conceive, investigate, and reason across scientific domains-remains lacking. We present an operational SGI definition grounded in the Practical Inquiry Model (PIM: Deliberation, Conception, Action, Perception) and operationalize it via four scientist-aligned tasks: deep research, idea generation, dry/wet experiments, and experimental reasoning. SGI-Bench comprises over 1,000 expert-curated, cross-disciplinary samples inspired by Science's 125 Big Questions, enabling systematic evaluation of state-of-the-art LLMs. Results reveal gaps: low exact match (10--20%) in deep research despite step-level alignment; ideas lacking feasibility and detail; high code executability but low execution result accuracy in dry experiments; low sequence fidelity in wet protocols; and persistent multimodal comparative-reasoning challenges. We further introduce Test-Time Reinforcement Learning (TTRL), which optimizes retrieval-augmented novelty rewards at inference, enhancing hypothesis novelty without reference answer. Together, our PIM-grounded definition, workflow-centric benchmark, and empirical insights establish a foundation for AI systems that genuinely participate in scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16962v1">MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 14 pages, 1 figure, includes appendix
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents increasingly rely on long-term memory and Retrieval-Augmented Generation (RAG) to persist experiences and refine future performance. While this experience learning capability enhances agentic autonomy, it introduces a critical, unexplored attack surface, i.e., the trust boundary between an agent's reasoning core and its own past. In this paper, we introduce MemoryGraft. It is a novel indirect injection attack that compromises agent behavior not through immediate jailbreaks, but by implanting malicious successful experiences into the agent's long-term memory. Unlike traditional prompt injections that are transient, or standard RAG poisoning that targets factual knowledge, MemoryGraft exploits the agent's semantic imitation heuristic which is the tendency to replicate patterns from retrieved successful tasks. We demonstrate that an attacker who can supply benign ingestion-level artifacts that the agent reads during execution can induce it to construct a poisoned RAG store where a small set of malicious procedure templates is persisted alongside benign experiences. When the agent later encounters semantically similar tasks, union retrieval over lexical and embedding similarity reliably surfaces these grafted memories, and the agent adopts the embedded unsafe patterns, leading to persistent behavioral drift across sessions. We validate MemoryGraft on MetaGPT's DataInterpreter agent with GPT-4o and find that a small number of poisoned records can account for a large fraction of retrieved experiences on benign workloads, turning experience-based self-improvement into a vector for stealthy and durable compromise. To facilitate reproducibility and future research, our code and evaluation data are available at https://github.com/Jacobhhy/Agent-Memory-Poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17967v1">Memelang: An Axial Grammar for LLM-Generated Vector-Relational Queries</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Structured generation for LLM tool use highlights the value of compact DSL intermediate representations (IRs) that can be emitted directly and parsed deterministically. This paper introduces axial grammar: linear token sequences that recover multi-dimensional structure from the placement of rank-specific separator tokens. A single left-to-right pass assigns each token a coordinate in an n-dimensional grid, enabling deterministic parsing without parentheses or clause-heavy surface syntax. This grammar is instantiated in Memelang, a compact query language intended as an LLM-emittable IR whose fixed coordinate roles map directly to table/column/value slots. Memelang supports coordinate-stable relative references, parse-time variable binding, and implicit context carry-forward to reduce repetition in LLM-produced queries. It also encodes grouping, aggregation, and ordering via inline tags on value terms, allowing grouped execution plans to be derived in one streaming pass over the coordinate-indexed representation. Provided are a reference lexer/parser and a compiler that emits parameterized PostgreSQL SQL (optionally using pgvector operators).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16917v1">Generative Adversarial Reasoner: Enhancing LLM Reasoning with Adversarial Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with explicit reasoning capabilities excel at mathematical reasoning yet still commit process errors, such as incorrect calculations, brittle logic, and superficially plausible but invalid steps. In this paper, we introduce Generative Adversarial Reasoner, an on-policy joint training framework designed to enhance reasoning by co-evolving an LLM reasoner and an LLM-based discriminator through adversarial reinforcement learning. A compute-efficient review schedule partitions each reasoning chain into logically complete slices of comparable length, and the discriminator evaluates each slice's soundness with concise, structured justifications. Learning couples complementary signals: the LLM reasoner is rewarded for logically consistent steps that yield correct answers, while the discriminator earns rewards for correctly detecting errors or distinguishing traces in the reasoning process. This produces dense, well-calibrated, on-policy step-level rewards that supplement sparse exact-match signals, improving credit assignment, increasing sample efficiency, and enhancing overall reasoning quality of LLMs. Across various mathematical benchmarks, the method delivers consistent gains over strong baselines with standard RL post-training. Specifically, on AIME24, we improve DeepSeek-R1-Distill-Qwen-7B from 54.0 to 61.3 (+7.3) and DeepSeek-R1-Distill-Llama-8B from 43.7 to 53.7 (+10.0). The modular discriminator also enables flexible reward shaping for objectives such as teacher distillation, preference alignment, and mathematical proof-based reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16914v1">Constructive Circuit Amplification: Improving Math Reasoning in LLMs via Targeted Sub-Network Updates</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 18 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Prior studies investigating the internal workings of LLMs have uncovered sparse subnetworks, often referred to as circuits, that are responsible for performing specific tasks. Additionally, it has been shown that model performance improvement through fine-tuning often results from the strengthening of existing circuits in the model. Taken together, these findings suggest the possibility of intervening directly on such circuits to make precise, task-targeted updates. Motivated by these findings, we propose a novel method called Constructive Circuit Amplification which identifies pivotal tokens from model reasoning traces as well as model components responsible for the desired task, and updates only those components. Applied to mathematical reasoning, it improves accuracy by up to +11.4% across multiple models while modifying as little as 1.59% of model components, with minimal impact on other abilities as measured by MMLU, TriviaQA, and TruthfulQA. These results demonstrate that targeted capabilities can be reliably enhanced by selectively updating a sparse set of model components.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16822v1">MEPIC: Memory Efficient Position Independent Caching for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Modern LLM applications such as deep-research assistants, coding agents, and Retrieval-Augmented Generation (RAG) systems, repeatedly process long prompt histories containing shared document or code chunks, creating significant pressure on the Key Value (KV) cache, which must operate within limited memory while sustaining high throughput and low latency. Prefix caching partially alleviates some of these costs by reusing KV cache for previously processed tokens, but limited by strict prefix matching. Position-independent caching (PIC) enables chunk-level reuse at arbitrary positions, but requires selective recomputation and positional-encoding (PE) adjustments. However, because these operations vary across queries, KV for the same chunk diverges across requests. Moreover, without page alignment, chunk KV layouts diverge in memory, preventing page sharing. These issues result in only modest HBM savings even when many requests reuse the same content. We present MEPIC, a memory-efficient PIC system that enables chunk KV reuse across positions, requests, and batches. MEPIC aligns chunk KV to paged storage, shifts recomputation from token- to block-level so only the first block is request-specific, removes positional encodings via Rotary Position Embedding (RoPE) fusion in the attention kernel, and makes remaining blocks fully shareable. These techniques eliminate most duplicate chunk KV in HBM, reducing usage by up to 2x over state-of-the-art PIC at comparable latency and accuracy, and up to 5x for long prompts, without any model changes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16795v1">From Facts to Conclusions : Integrating Deductive Reasoning in Retrieval-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) grounds large language models (LLMs) in external evidence, but fails when retrieved sources conflict or contain outdated or subjective information. Prior work address these issues independently but lack unified reasoning supervision. We propose a reasoning-trace-augmented RAG framework that adds structured, interpretable reasoning across three stages : (1) document-level adjudication, (2) conflict analysis, and (3) grounded synthesis, producing citation-linked answers or justified refusals. A Conflict-Aware Trust-Score (CATS) pipeline is introduced which evaluates groundedness, factual correctness, refusal accuracy, and conflict-behavior alignment using an LLM-as-a-Judge. Our 539-query reasoning dataset and evaluation pipeline establish a foundation for conflict-aware, interpretable RAG systems. Experimental results demonstrate substantial gains over baselines, most notably with Qwen, where Supervised Fine-Tuning improved End-to-End answer correctness from 0.069 to 0.883 and behavioral adherence from 0.074 to 0.722.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16790v1">Inside Out: Uncovering How Comment Internalization Steers LLMs for Better or Worse</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Accepted in the 48th IEEE/ACM International Conference on Software Engineering (ICSE)
    </div>
    <details class="paper-abstract">
      While comments are non-functional elements of source code, Large Language Models (LLM) frequently rely on them to perform Software Engineering (SE) tasks. Yet, where in the model this reliance resides, and how it affects performance, remains poorly understood. We present the first concept-level interpretability study of LLMs in SE, analyzing three tasks - code completion, translation, and refinement - through the lens of internal comment representation. Using Concept Activation Vectors (CAV), we show that LLMs not only internalize comments as distinct latent concepts but also differentiate between subtypes such as Javadocs, inline, and multiline comments. By systematically activating and deactivating these concepts in the LLMs' embedding space, we observed significant, model-specific, and task-dependent shifts in performance ranging from -90% to +67%. Finally, we conducted a controlled experiment using the same set of code inputs, prompting LLMs to perform 10 distinct SE tasks while measuring the activation of the comment concept within their latent representations. We found that code summarization consistently triggered the strongest activation of comment concepts, whereas code completion elicited the weakest sensitivity. These results open a new direction for building SE tools and models that reason about and manipulate internal concept representations rather than relying solely on surface-level input.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16750v1">Plausibility as Failure: How LLMs and Humans Co-Construct Epistemic Error</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 19 pages, 2 tables, 77 references, 6 appendices
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as epistemic partners in everyday reasoning, yet their errors remain predominantly analyzed through predictive metrics rather than through their interpretive effects on human judgment. This study examines how different forms of epistemic failure emerge, are masked, and are tolerated in human AI interaction, where failure is understood as a relational breakdown shaped by model-generated plausibility and human interpretive judgment. We conducted a three round, multi LLM evaluation using interdisciplinary tasks and progressively differentiated assessment frameworks to observe how evaluators interpret model responses across linguistic, epistemic, and credibility dimensions. Our findings show that LLM errors shift from predictive to hermeneutic forms, where linguistic fluency, structural coherence, and superficially plausible citations conceal deeper distortions of meaning. Evaluators frequently conflated criteria such as correctness, relevance, bias, groundedness, and consistency, indicating that human judgment collapses analytical distinctions into intuitive heuristics shaped by form and fluency. Across rounds, we observed a systematic verification burden and cognitive drift. As tasks became denser, evaluators increasingly relied on surface cues, allowing erroneous yet well formed answers to pass as credible. These results suggest that error is not solely a property of model behavior but a co-constructed outcome of generative plausibility and human interpretive shortcuts. Understanding AI epistemic failure therefore requires reframing evaluation as a relational interpretive process, where the boundary between system failure and human miscalibration becomes porous. The study provides implications for LLM assessment, digital literacy, and the design of trustworthy human AI communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16676v1">DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      The rapidly growing demand for high-quality data in Large Language Models (LLMs) has intensified the need for scalable, reliable, and semantically rich data preparation pipelines. However, current practices remain dominated by ad-hoc scripts and loosely specified workflows, which lack principled abstractions, hinder reproducibility, and offer limited support for model-in-the-loop data generation. To address these challenges, we present DataFlow, a unified and extensible LLM-driven data preparation framework. DataFlow is designed with system-level abstractions that enable modular, reusable, and composable data transformations, and provides a PyTorch-style pipeline construction API for building debuggable and optimizable dataflows. The framework consists of nearly 200 reusable operators and six domain-general pipelines spanning text, mathematical reasoning, code, Text-to-SQL, agentic RAG, and large-scale knowledge extraction. To further improve usability, we introduce DataFlow-Agent, which automatically translates natural-language specifications into executable pipelines via operator synthesis, pipeline planning, and iterative verification. Across six representative use cases, DataFlow consistently improves downstream LLM performance. Our math, code, and text pipelines outperform curated human datasets and specialized synthetic baselines, achieving up to +3\% execution accuracy in Text-to-SQL over SynSQL, +7\% average improvements on code benchmarks, and 1--3 point gains on MATH, GSM8K, and AIME. Moreover, a unified 10K-sample dataset produced by DataFlow enables base models to surpass counterparts trained on 1M Infinity-Instruct data. These results demonstrate that DataFlow provides a practical and high-performance substrate for reliable, reproducible, and scalable LLM data preparation, and establishes a system-level foundation for future data-centric AI development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20172v6">Evaluating and Mitigating Errors in LLM-Generated Web API Integrations</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 Extended journal version of our paper published at AIware 2025 (arXiv:2509.20172v5)
    </div>
    <details class="paper-abstract">
      API integration is a cornerstone of our digital infrastructure, enabling software systems to connect and interact. However, as shown by many studies, writing or generating correct code to invoke APIs, particularly web APIs, is challenging. Although large language models (LLMs) have become popular in software development, their effectiveness in automating the generation of web API integration code remains unexplored. In order to address this, we present WAPIIBench, a dataset and evaluation pipeline designed to assess the ability of LLMs to generate web API invocation code. Our experiments with several open-source LLMs reveal that generating API invocations poses a significant challenge, resulting in hallucinated endpoints, incorrect argument usage, and other errors. None of the evaluated open-source models was able to solve more than 40% of the tasks. Motivated by those findings, we explore the potential of constrained decoding for generating API invocations. To this end, we propose an automatic translation from API specifications to constraints. Our approach prevents violations of API usage rules and significantly increases the overall correctness of the generated code, on average by 90% and 135%, depending on the provided starter code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04133v5">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of Trust, Risk, and Security Management (TRiSM) in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around key pillars: \textit{ Explainability, ModelOps, Security, Privacy} and \textit{their Lifecycle Governance}, each contextualized to the challenges of AMAS. A risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI, as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, highlighting key directions to align emerging systems with TRiSM principles-ensuring safety, transparency, and accountability in their operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16602v1">Refusal Steering: Fine-grained Control over LLM Refusal Behaviour for Sensitive Topics</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      We introduce Refusal Steering, an inference-time method to exercise fine-grained control over Large Language Models refusal behaviour on politically sensitive topics without retraining. We replace fragile pattern-based refusal detection with an LLM-as-a-judge that assigns refusal confidence scores and we propose a ridge-regularized variant to compute steering vectors that better isolate the refusal--compliance direction. On Qwen3-Next-80B-A3B-Thinking, our method removes the refusal behaviour of the model around politically sensitive topics while maintaining safety on JailbreakBench and near-baseline performance on general benchmarks. The approach generalizes across 4B and 80B models and can also induce targeted refusals when desired. We analize the steering vectors and show that refusal signals concentrate in deeper layers of the transformer and are distributed across many dimensions. Together, these results demonstrate that activation steering can remove political refusal behaviour while retaining safety alignment for harmful content, offering a practical path to controllable, transparent moderation at inference time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16145v2">SpiroLLM: Finetuning Pretrained LLMs to Understand Spirogram Time Series with Clinical Validation in COPD Reporting</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      Chronic Obstructive Pulmonary Disease (COPD), a major chronic respiratory disease with persistent airflow limitation, is a leading global cause of disability and mortality. Respiratory spirogram time series, routinely collected during pulmonary function tests (PFTs), play a critical role in the early detection of repsiratory diseases and in monitoring lung function over time. However, most current AI models for COPD diagnosis are limited to outputting classification results without providing a rationale for their diagnostic process, while current Large Language Models (LLMs) cannot understand spirograms yet, which severely limits their clinical trust and adoption. To tackle this challenge, we leverage a cohort of 234,028 individuals from the UK Biobank (UKB) to propose SpiroLLM, the first multimodal large language model that can understand spirogram. The model extracts morphological features from respiratory curves via a SpiroEncoder and aligns them with PFT numerical values in a unified latent space using a SpiroProjector, ultimately empowering a large language model to generate a comprehensive diagnostic report. Experimental results confirm that SpiroLLM achieved a diagnostic AUROC of 0.8977 (95% CI: 0.88-0.91). In a robustness test with missing core data, it maintained a 100% valid response rate, far surpassing the 13.4% of a text-only model and showcasing the superiority of its multimodal design. This work demonstrates the substantial potential of deeply fusing physiological signals with large language models, establishing a new paradigm for the next generation of interpretable and reliable clinical decision support tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16538v1">A Systematic Study of Code Obfuscation Against LLM-based Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly adopted for code vulnerability detection, their reliability and robustness across diverse vulnerability types have become a pressing concern. In traditional adversarial settings, code obfuscation has long been used as a general strategy to bypass auditing tools, preserving exploitability without tampering with the tools themselves. Numerous efforts have explored obfuscation methods and tools, yet their capabilities differ in terms of supported techniques, granularity, and programming languages, making it difficult to systematically assess their impact on LLM-based vulnerability detection. To address this gap, we provide a structured systematization of obfuscation techniques and evaluate them under a unified framework. Specifically, we categorize existing obfuscation methods into three major classes (layout, data flow, and control flow) covering 11 subcategories and 19 concrete techniques. We implement these techniques across four programming languages (Solidity, C, C++, and Python) using a consistent LLM-driven approach, and evaluate their effects on 15 LLMs spanning four model families (DeepSeek, OpenAI, Qwen, and LLaMA), as well as on two coding agents (GitHub Copilot and Codex). Our findings reveal both positive and negative impacts of code obfuscation on LLM-based vulnerability detection, highlighting conditions under which obfuscation leads to performance improvements or degradations. We further analyze these outcomes with respect to vulnerability characteristics, code properties, and model attributes. Finally, we outline several open problems and propose future directions to enhance the robustness of LLMs for real-world vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16530v1">Plain language adaptations of biomedical text using LLMs: Comparision of evaluation metrics</a></div>
    <div class="paper-meta">
      📅 2025-12-18
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      This study investigated the application of Large Language Models (LLMs) for simplifying biomedical texts to enhance health literacy. Using a public dataset, which included plain language adaptations of biomedical abstracts, we developed and evaluated several approaches, specifically a baseline approach using a prompt template, a two AI agent approach, and a fine-tuning approach. We selected OpenAI gpt-4o and gpt-4o mini models as baselines for further research. We evaluated our approaches with quantitative metrics, such as Flesch-Kincaid grade level, SMOG Index, SARI, and BERTScore, G-Eval, as well as with qualitative metric, more precisely 5-point Likert scales for simplicity, accuracy, completeness, brevity. Results showed a superior performance of gpt-4o-mini and an underperformance of FT approaches. G-Eval, a LLM based quantitative metric, showed promising results, ranking the approaches similarly as the qualitative metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15053v1">The Meta-Prompting Protocol: Orchestrating LLMs via Adversarial Feedback Loops</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The transition of Large Language Models (LLMs) from stochastic chat interfaces to reliable software components necessitates a fundamental re-engineering of interaction paradigms. Current methodologies, predominantly heuristic-based "prompt engineering," fail to provide the deterministic guarantees required for mission-critical applications. We introduce the Meta-Prompting Protocol, a rigorous theoretical framework that formalizes the orchestration of LLMs as a programmable, self-optimizing system. Central to this protocol is the Adversarial Trinity, a tripartite topology comprising a Generator (P), an Auditor (A), and an Optimizer (O). By treating natural language instructions as differentiable variables within a semantic computation graph and utilizing textual critiques as gradients, this architecture mitigates hallucination and prevents model collapse. We demonstrate the theoretical viability of this approach using declarative programming paradigms (DSPy) and automatic textual differentiation (TextGrad), establishing a foundation for "Observable Software Engineering" in the era of probabilistic computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15000v1">DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Process Reward Models (PRMs) have become essential for improving Large Language Models (LLMs) via test-time scaling, yet their effectiveness in coding remains limited due to the lack of meaningful step decompositions in code and the noise of Monte-Carlo-generated partial labels. We propose DreamPRM-Code, a coding-focused PRM that treats functions as reasoning steps using a Chain-of-Function prompting strategy to induce modular code generation, enabling PRM training and application analogous to mathematical reasoning tasks. To address label noise, DreamPRM-Code introduces a meta-learning-based correction mechanism that leverages clean final-solution unit-test labels and performs bi-level optimization to refine intermediate labels. Applying on test-time scaling, DreamPRM-Code achieved state-of-the-art performance on LiveCodeBench with 80.9 pass@1 rate, surpassing OpenAI o4-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11062v4">Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.16041v1">Are We on the Right Way to Assessing LLM-as-a-Judge?</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has been widely adopted as an evaluation method and served as supervised rewards in model training. However, existing benchmarks for LLM-as-a-Judge are mainly relying on human-annotated ground truth, which introduces human bias that undermines the assessment of reliability and imposes scalability constraints. To overcome these limitations, we introduce Sage, a novel evaluation suite that assesses the quality of LLM judges without necessitating any human annotation. Inspired by axioms of rational choice theory, Sage introduces two new lenses for measuring LLM-as-a-Judge: local self-consistency (pair-wise preference stability) and global logical consistency (transitivity across a full set of preferences). We curate a dataset of 650 questions by combining structured benchmark problems with real-world user queries. Our experiments demonstrate both the stability of our metrics and their high correlation with supervised benchmarks like LLMBar and RewardBench2, confirming Sage's reliability as an evaluation suite for the robustness and accuracy of LLM-as-a-Judge. Based on Sage, we reveal that current state-of-the-art LLMs exhibit significant reliability problems when acting as judges in both scoring and pairwise settings; even the top-performing models, Gemini-2.5-Pro and GPT-5, fail to maintain consistent preferences in nearly a quarter of difficult cases. We attribute this to a new phenomenon called situational preference, which explains why explicit rubrics or criteria can help the model judge consistently across answer pairs. Our further analysis shows that finetuned LLM-as-a-Judge is a feasible method to boost performance, and the panel-based judge as well as deep reasoning can enhance the judging consistency. We also find substantial inconsistency in human judgments, which indicates that human annotation may not be a reliable gold standard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.16112v2">HPU: High-Bandwidth Processing Unit for Scalable, Cost-effective LLM Inference via GPU Co-processing</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The attention layer, a core component of Transformer-based LLMs, brings out inefficiencies in current GPU systems due to its low operational intensity and the substantial memory requirements of KV caches. We propose a High-bandwidth Processing Unit (HPU), a memoryintensive co-processor that enhances GPU resource utilization during large-batched LLM inference. By offloading memory-bound operations, the HPU allows the GPU to focus on compute-intensive tasks, increasing overall efficiency. Also, the HPU, as an add-on card, scales out to accommodate surging memory demands driven by large batch sizes and extended sequence lengths. In this paper, we show the HPU prototype implemented with PCIe-based FPGA cards mounted on a GPU system. Our novel GPU-HPU heterogeneous system demonstrates up to 4.1x performance gains and 4.6x energy efficiency improvements over a GPUonly system, providing scalability without increasing the number of GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15979v1">OLAF: Towards Robust LLM-Based Annotation Framework in Empirical Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in empirical software engineering (ESE) to automate or assist annotation tasks such as labeling commits, issues, and qualitative artifacts. Yet the reliability and reproducibility of such annotations remain underexplored. Existing studies often lack standardized measures for reliability, calibration, and drift, and frequently omit essential configuration details. We argue that LLM-based annotation should be treated as a measurement process rather than a purely automated activity. In this position paper, we outline the \textbf{Operationalization for LLM-based Annotation Framework (OLAF)}, a conceptual framework that organizes key constructs: \textit{reliability, calibration, drift, consensus, aggregation}, and \textit{transparency}. The paper aims to motivate methodological discussion and future empirical work toward more transparent and reproducible LLM-based annotation in software engineering research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15674v1">Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 36 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) activations are notoriously difficult to understand, with most existing techniques using complex, specialized methods for interpreting them. Recent work has proposed a simpler approach known as LatentQA: training LLMs to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language. However, prior work has focused on narrow task settings for both training and evaluation. In this paper, we instead take a generalist perspective. We evaluate LatentQA-trained models, which we call Activation Oracles (AOs), in far out-of-distribution settings and examine how performance scales with training data diversity. We find that AOs can recover information fine-tuned into a model (e.g., biographical knowledge or malign propensities) that does not appear in the input text, despite never being trained with activations from a fine-tuned model. Our main evaluations are four downstream tasks where we can compare to prior white- and black-box techniques. We find that even narrowly-trained LatentQA models can generalize well, and that adding additional training datasets (such as classification tasks and a self-supervised context prediction task) yields consistent further improvements. Overall, our best AOs match or exceed prior white-box baselines on all four tasks and are the best method on 3 out of 4. These results suggest that diversified training to answer natural-language queries imparts a general capability to verbalize information about LLM activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15662v1">Stepwise Think-Critique: A Unified Framework for Robust and Interpretable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Human beings solve complex problems through critical thinking, where reasoning and evaluation are intertwined to converge toward correct solutions. However, most existing large language models (LLMs) decouple reasoning from verification: they either generate reasoning without explicit self-checking or rely on external verifiers to detect errors post hoc. The former lacks immediate feedback, while the latter increases system complexity and hinders synchronized learning. Motivated by human critical thinking, we propose Stepwise Think-Critique (STC), a unified framework that interleaves reasoning and self-critique at each step within a single model. STC is trained with a hybrid reinforcement learning objective combining reasoning rewards and critique-consistency rewards to jointly optimize reasoning quality and self-evaluation. Experiments on mathematical reasoning benchmarks show that STC demonstrates strong critic-thinking capabilities and produces more interpretable reasoning traces, representing a step toward LLMs with built-in critical thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02389v2">From Trace to Line: LLM Agent for Real-World OSS Vulnerability Localization</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large language models show promise for vulnerability discovery, yet prevailing methods inspect code in isolation, struggle with long contexts, and focus on coarse function or file level detections which offers limited actionable guidance to engineers who need precise line-level localization and targeted patches in real-world software development. We present T2L-Agent (Trace-to-Line Agent), a project-level, end-to-end framework that plans its own analysis and progressively narrows scope from modules to exact vulnerable lines. T2L-Agent couples multi-round feedback with an Agentic Trace Analyzer (ATA) that fuses run-time evidence such as crash points, stack traces, and coverage deltas with AST-based code chunking, enabling iterative refinement beyond single pass predictions and translating symptoms into actionable, line-level diagnoses. To benchmark line-level vulnerability discovery, we introduce T2L-ARVO, a diverse, expert-verified 50-case benchmark spanning five crash families and real-world projects. T2L-ARVO is specifically designed to support both coarse-grained detection and fine-grained localization, enabling rigorous evaluation of systems that aim to move beyond file-level predictions. On T2L-ARVO, T2L-Agent achieves up to 58.0% detection and 54.8% line-level localization, substantially outperforming baselines. Together, the framework and benchmark push LLM-based vulnerability detection from coarse identification toward deployable, robust, precision diagnostics that reduce noise and accelerate patching in open-source software workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01353v3">The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Updated with new baselines and experimental results
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.
    </details>
</div>
