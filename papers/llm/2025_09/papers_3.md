# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05467v2">Position: LLMs Can be Good Tutors in English Education</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 Accepted to EMNLP 2025 Main. 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      While recent efforts have begun integrating large language models (LLMs) into English education, they often rely on traditional approaches to learning tasks without fully embracing educational methodologies, thus lacking adaptability to language learning. To address this gap, we argue that LLMs have the potential to serve as effective tutors in English Education. Specifically, LLMs can play three critical roles: (1) as data enhancers, improving the creation of learning materials or serving as student simulations; (2) as task predictors, serving as learner assessment or optimizing learning pathway; and (3) as agents, enabling personalized and inclusive education. We encourage interdisciplinary research to explore these roles, fostering innovation while addressing challenges and risks, ultimately advancing English Education through the thoughtful integration of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18665v2">Membership Inference Attacks on LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05764v1">DRF: LLM-AGENT Dynamic Reputation Filtering Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 This paper has been accepted by ICONIP 2025 but not published
    </div>
    <details class="paper-abstract">
      With the evolution of generative AI, multi - agent systems leveraging large - language models(LLMs) have emerged as a powerful tool for complex tasks. However, these systems face challenges in quantifying agent performance and lack mechanisms to assess agent credibility. To address these issues, we introduce DRF, a dynamic reputation filtering framework. DRF constructs an interactive rating network to quantify agent performance, designs a reputation scoring mechanism to measure agent honesty and capability, and integrates an Upper Confidence Bound - based strategy to enhance agent selection efficiency. Experiments show that DRF significantly improves task completion quality and collaboration efficiency in logical reasoning and code - generation tasks, offering a new approach for multi - agent systems to handle large - scale tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05755v1">Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05751v1">Unleashing Hierarchical Reasoning: An LLM-Driven Framework for Training-Free Referring Video Object Segmentation</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Referring Video Object Segmentation (RVOS) aims to segment an object of interest throughout a video based on a language description. The prominent challenge lies in aligning static text with dynamic visual content, particularly when objects exhibiting similar appearances with inconsistent motion and poses. However, current methods often rely on a holistic visual-language fusion that struggles with complex, compositional descriptions. In this paper, we propose \textbf{PARSE-VOS}, a novel, training-free framework powered by Large Language Models (LLMs), for a hierarchical, coarse-to-fine reasoning across text and video domains. Our approach begins by parsing the natural language query into structured semantic commands. Next, we introduce a spatio-temporal grounding module that generates all candidate trajectories for all potential target objects, guided by the parsed semantics. Finally, a hierarchical identification module select the correct target through a two-stage reasoning process: it first performs coarse-grained motion reasoning with an LLM to narrow down candidates; if ambiguity remains, a fine-grained pose verification stage is conditionally triggered to disambiguate. The final output is an accurate segmentation mask for the target object. \textbf{PARSE-VOS} achieved state-of-the-art performance on three major benchmarks: Ref-YouTube-VOS, Ref-DAVIS17, and MeViS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05741v1">Enhancing Factual Accuracy and Citation Generation in LLMs via Multi-Stage Self-Verification</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      This research introduces VeriFact-CoT (Verified Factual Chain-of-Thought), a novel method designed to address the pervasive issues of hallucination and the absence of credible citation sources in Large Language Models (LLMs) when generating complex, fact-sensitive content. By incorporating a multi-stage mechanism of 'fact verification-reflection-citation integration,' VeriFact-CoT empowers LLMs to critically self-examine and revise their intermediate reasoning steps and final answers. This process significantly enhances the objective accuracy, trustworthiness, and traceability of the generated outputs, making LLMs more reliable for applications demanding high fidelity such as scientific research, news reporting, and legal consultation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05657v1">LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 EMNLP2025
    </div>
    <details class="paper-abstract">
      Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at https://github.com/Ashone3/LM-Searcher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00882v2">VulSolver: Vulnerability Detection via LLM-Driven Constraint Solving</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Traditional vulnerability detection methods rely heavily on predefined rule matching, which often fails to capture vulnerabilities accurately. With the rise of large language models (LLMs), leveraging their ability to understand code semantics has emerged as a promising direction for achieving more accurate and efficient vulnerability detection. However, current LLM-based approaches face significant challenges: instability in model outputs, limitations in context length, and hallucination. As a result, many existing solutions either use LLMs merely to enrich predefined rule sets, thereby keeping the detection process fundamentally rule-based, or over-rely on them, leading to poor robustness. To address these challenges, we propose a constraint-solving approach powered by LLMs named VULSOLVER. By modeling vulnerability detection as a constraint-solving problem, and by integrating static application security testing (SAST) with the semantic reasoning capabilities of LLMs, our method enables the LLM to act like a professional human security expert. We assess VULSOLVER on the OWASP Benchmark (1,023 labeled samples), achieving 96.29% accuracy, 96.55% F1-score, and 100% recall. Applied to popular GitHub repositories, VULSOLVER also identified 15 previously unknown high-severity vulnerabilities (CVSS 7.5-9.8), demonstrating its effectiveness in real-world security analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20653v2">ComplexVCoder: An LLM-Driven Framework for Systematic Generation of Complex Verilog Code</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 Withdrawn due to an error in the experimental setup that affected the results. A corrected version is in progress
    </div>
    <details class="paper-abstract">
      Recent advances have demonstrated the promising capabilities of large language models (LLMs) in generating register-transfer level (RTL) code, such as Verilog. However, existing LLM-based frameworks still face significant challenges in accurately handling the complexity of real-world RTL designs, particularly those that are large-scale and involve multi-level module instantiations. To address this issue, we present ComplexVCoder, an open-source LLM-driven framework that enhances both the generation quality and efficiency of complex Verilog code. Specifically, we introduce a two-stage generation mechanism, which leverages an intermediate representation to enable a more accurate and structured transition from natural language descriptions to intricate Verilog designs. In addition, we introduce a rule-based alignment method and a domain-specific retrieval-augmented generation (RAG) to further improve the correctness of the synthesized code by incorporating relevant design knowledge during generation. To evaluate our approach, we construct a comprehensive dataset comprising 55 complex Verilog designs derived from real-world implementations. We also release an open-source benchmark suite for systematically assessing the quality of auto-generated RTL code together with the ComplexVCoder framework. Experimental results show that ComplexVCoder outperforms SOTA frameworks such as CodeV and RTLCoder by 14.6% and 22.2%, respectively, in terms of function correctness on complex Verilog benchmarks. Furthermore, ComplexVcoder achieves comparable generation performances in terms of functionality correctness using a lightweight 32B model (Qwen2.5), rivaling larger-scale models such as GPT-3.5 and DeepSeek-V3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05608v1">Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      The widespread deployment of LLMs across enterprise services has created a critical security blind spot. Organizations operate multiple LLM services handling billions of queries daily, yet regulatory compliance boundaries prevent these services from sharing threat intelligence about prompt injection attacks, the top security risk for LLMs. When an attack is detected in one service, the same threat may persist undetected in others for months, as privacy regulations prohibit sharing user prompts across compliance boundaries. We present BinaryShield, the first privacy-preserving threat intelligence system that enables secure sharing of attack fingerprints across compliance boundaries. BinaryShield transforms suspicious prompts through a unique pipeline combining PII redaction, semantic embedding, binary quantization, and randomized response mechanism to potentially generate non-invertible fingerprints that preserve attack patterns while providing privacy. Our evaluations demonstrate that BinaryShield achieves an F1-score of 0.94, significantly outperforming SimHash (0.77), the privacy-preserving baseline, while achieving 64x storage reduction and 38x faster similarity search compared to dense embeddings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01386v2">ThinkEval: Practical Evaluation of Knowledge Leakage in LLM Editing using Thought-based Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-09-06
    </div>
    <details class="paper-abstract">
      Robust model-editing techniques are essential for deploying large language models (LLMs) in practical applications, to enable cost-effective ways to deal with challenges such as privacy breaches, bias mitigation and misinformation spread. For example, an LLM-based healthcare assistance may need to update out-dated or incorrect knowledge to prevent harmful recommendations. However, many editing techniques focus on isolated facts, which critically fail to prevent indirect knowledge leakage -- the unintended reconstruction of edited-out information through persistent causal links and contextual relationships. To assist users in selecting the right editing technique, we develop and present ThinkEval, a framework to systematically quantify indirect knowledge leakage and ripple effects in model-editing. ThinkEval builds and employs specialized knowledge graphs to analyze the causal structure of facts before and after editing. To support this approach, we present KnowGIC, a benchmark dataset comprising multi-step reasoning paths that precisely measure these complex knowledge transformation effects. We evaluate five editing techniques: AlphaEdit, RECT, ROME, MEMIT, and PRUNE across multiple LLMs. Our results show that these techniques struggle to balance indirect fact suppression with the preservation of related knowledge, compromising the contextual integrity of a model's knowledge. Our dataset is available at: https://anonymous.4open.science/r/KnowGIC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08000v1">AntiDote: Bi-level Adversarial Training for Tamper-Resistant LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-06
      | 💬 19 pages
    </div>
    <details class="paper-abstract">
      The release of open-weight large language models (LLMs) creates a tension between advancing accessible research and preventing misuse, such as malicious fine-tuning to elicit harmful content. Current safety measures struggle to preserve the general capabilities of the LLM while resisting a determined adversary with full access to the model's weights and architecture, who can use full-parameter fine-tuning to erase existing safeguards. To address this, we introduce AntiDote, a bi-level optimization procedure for training LLMs to be resistant to such tampering. AntiDote involves an auxiliary adversary hypernetwork that learns to generate malicious Low-Rank Adaptation (LoRA) weights conditioned on the defender model's internal activations. The defender LLM is then trained with an objective to nullify the effect of these adversarial weight additions, forcing it to maintain its safety alignment. We validate this approach against a diverse suite of 52 red-teaming attacks, including jailbreak prompting, latent space manipulation, and direct weight-space attacks. AntiDote is upto 27.4\% more robust against adversarial attacks compared to both tamper-resistance and unlearning baselines. Crucially, this robustness is achieved with a minimal trade-off in utility, incurring a performance degradation of upto less than 0.5\% across capability benchmarks including MMLU, HellaSwag, and GSM8K. Our work offers a practical and compute efficient methodology for building open-weight models where safety is a more integral and resilient property.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04770v1">Research on Multi-hop Inference Optimization of LLM Based on MQUAKE Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Accurately answering complex questions has consistently been a significant challenge for Large Language Models (LLMs). To address this, this paper proposes a multi-hop question decomposition method for complex questions, building upon research within the MQUAKE framework. Utilizing the LLAMA3 model, we systematically investigate the impact of multi-hop question decomposition within knowledge graphs on model comprehension and reasoning accuracy, both before and after model training. In our experiments, we systematically partitioned and converted the MQUAKE-T dataset into two distinct formats: a single-hop dataset designed for directly answering complex questions, and a multi-hop dataset constructed using the multi-hop question decomposition method. We then fine-tuned the LLAMA3 model on these datasets and conducted inference tests. Our results demonstrate that, without fine-tuning the LLM, the prediction performance based on the multi-hop question decomposition method significantly outperforms the method of directly answering complex questions. After fine-tuning using the LoRA (Low-Rank Adaptation) method, the performance of both approaches improved compared to the untrained baseline. Crucially, the method utilizing multi-hop decomposition consistently maintained its superiority. These findings validate the effectiveness of the multi-hop decomposition method both before and after training, demonstrating its capability to effectively enhance the LLM's ability to answer complex questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03730v2">The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 We make public all code and source data at https://github.com/psychology-of-AI/Personality-Illusion for full reproducibility
    </div>
    <details class="paper-abstract">
      Personality traits have long been studied as predictors of human behavior. Recent advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation. Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation. In this study, we systematically characterize LLM personality across three dimensions: (1) the dynamic emergence and evolution of trait profiles throughout training stages; (2) the predictive validity of self-reported traits in behavioral tasks; and (3) the impact of targeted interventions, such as persona injection, on both self-reports and behavior. Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data. However, these self-reported traits do not reliably predict behavior, and observed associations often diverge from human patterns. While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09061v4">CRANE: Reasoning with constrained LLM generation</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Accepted at ICML 2025, Code at: https://github.com/uiuc-focal-lab/CRANE
    </div>
    <details class="paper-abstract">
      Code generation, symbolic math reasoning, and other tasks require LLMs to produce outputs that are both syntactically and semantically correct. Constrained LLM generation is a promising direction to enforce adherence to formal grammar, but prior works have empirically observed that strict enforcement of formal constraints often diminishes the reasoning capabilities of LLMs. In this work, we first provide a theoretical explanation for why constraining LLM outputs to very restrictive grammars that only allow syntactically valid final answers reduces the reasoning capabilities of the model. Second, we demonstrate that by augmenting the output grammar with carefully designed additional rules, it is always possible to preserve the reasoning capabilities of the LLM while ensuring syntactic and semantic correctness in its outputs. Building on these theoretical insights, we propose a reasoning-augmented constrained decoding algorithm, CRANE, which effectively balances the correctness of constrained generation with the flexibility of unconstrained generation. Experiments on multiple open-source LLMs and benchmarks show that CRANE significantly outperforms both state-of-the-art constrained decoding strategies and standard unconstrained decoding, showing up to 10% points accuracy improvement over baselines on challenging symbolic reasoning benchmarks GSM-symbolic and FOLIO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05946v3">InstructMPC: A Human-LLM-in-the-Loop Framework for Context-Aware Control</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Model Predictive Control (MPC) is a powerful control strategy widely utilized in domains like energy management, building control, and autonomous systems. However, its effectiveness in real-world settings is challenged by the need to incorporate context-specific predictions and expert instructions, which traditional MPC often neglects. We propose InstructMPC, a novel framework that addresses this gap by integrating real-time human instructions through a Large Language Model (LLM) to produce context-aware predictions for MPC. Our method employs a Language-to-Distribution (L2D) module to translate contextual information into predictive disturbance trajectories, which are then incorporated into the MPC optimization. Unlike existing context-aware and language-based MPC models, InstructMPC enables dynamic human-LLM interaction and fine-tunes the L2D module in a closed loop with theoretical performance guarantees, achieving a regret bound of $O(\sqrt{T\log T})$ for linear dynamics when optimized via advanced fine-tuning methods such as Direct Preference Optimization (DPO) using a tailored loss function.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04714v1">ThumbnailTruth: A Multi-Modal LLM Approach for Detecting Misleading YouTube Thumbnails Across Diverse Cultural Settings</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Misleading video thumbnails on platforms like YouTube are a pervasive problem, undermining user trust and platform integrity. This paper proposes a novel multi-modal detection pipeline that uses Large Language Models (LLMs) to flag misleading thumbnails. We first construct a comprehensive dataset of 2,843 videos from eight countries, including 1,359 misleading thumbnail videos that collectively amassed over 7.6 billion views -- providing a unique cross-cultural perspective on this global issue. Our detection pipeline integrates video-to-text descriptions, thumbnail images, and subtitle transcripts to holistically analyze content and flag misleading thumbnails. Through extensive experimentation and prompt engineering, we evaluate the performance of state-of-the-art LLMs, including GPT-4o, GPT-4o Mini, Claude 3.5 Sonnet, and Gemini-1.5 Flash. Our findings show the effectiveness of LLMs in identifying misleading thumbnails, with Claude 3.5 Sonnet consistently showing strong performance, achieving an accuracy of 93.8\%, precision over 92\%, and recall exceeding 94\% in certain scenarios. We discuss the implications of our findings for content moderation, user experience, and the ethical considerations of deploying such systems at scale. Our findings pave the way for more transparent, trustworthy video platforms and stronger content integrity for audiences worldwide.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05540v1">Combining TSL and LLM to Automate REST API Testing: A Comparative Study</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 10 pages, article computer science, software engineering, software testing, ia, llm
    </div>
    <details class="paper-abstract">
      The effective execution of tests for REST APIs remains a considerable challenge for development teams, driven by the inherent complexity of distributed systems, the multitude of possible scenarios, and the limited time available for test design. Exhaustive testing of all input combinations is impractical, often resulting in undetected failures, high manual effort, and limited test coverage. To address these issues, we introduce RestTSLLM, an approach that uses Test Specification Language (TSL) in conjunction with Large Language Models (LLMs) to automate the generation of test cases for REST APIs. The approach targets two core challenges: the creation of test scenarios and the definition of appropriate input data. The proposed solution integrates prompt engineering techniques with an automated pipeline to evaluate various LLMs on their ability to generate tests from OpenAPI specifications. The evaluation focused on metrics such as success rate, test coverage, and mutation score, enabling a systematic comparison of model performance. The results indicate that the best-performing LLMs - Claude 3.5 Sonnet (Anthropic), Deepseek R1 (Deepseek), Qwen 2.5 32b (Alibaba), and Sabia 3 (Maritaca) - consistently produced robust and contextually coherent REST API tests. Among them, Claude 3.5 Sonnet outperformed all other models across every metric, emerging in this study as the most suitable model for this task. These findings highlight the potential of LLMs to automate the generation of tests based on API specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05484v1">From Staff Messages to Actionable Insights: A Multi-Stage LLM Classification Framework for Healthcare Analytics</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Hospital call centers serve as the primary contact point for patients within a hospital system. They also generate substantial volumes of staff messages as navigators process patient requests and communicate with the hospital offices following the established protocol restrictions and guidelines. This continuously accumulated large amount of text data can be mined and processed to retrieve insights; however, traditional supervised learning approaches require annotated data, extensive training, and model tuning. Large Language Models (LLMs) offer a paradigm shift toward more computationally efficient methodologies for healthcare analytics. This paper presents a multi-stage LLM-based framework that identifies staff message topics and classifies messages by their reasons in a multi-class fashion. In the process, multiple LLM types, including reasoning, general-purpose, and lightweight models, were evaluated. The best-performing model was o3, achieving 78.4% weighted F1-score and 79.2% accuracy, followed closely by gpt-5 (75.3% Weighted F1-score and 76.2% accuracy). The proposed methodology incorporates data security measures and HIPAA compliance requirements essential for healthcare environments. The processed LLM outputs are integrated into a visualization decision support tool that transforms the staff messages into actionable insights accessible to healthcare professionals. This approach enables more efficient utilization of the collected staff messaging data, identifies navigator training opportunities, and supports improved patient experience and care quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05449v1">Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Membership inference attacks (MIAs) reveal whether specific data was used to train machine learning models, serving as important tools for privacy auditing and compliance assessment. Recent studies have reported that MIAs perform only marginally better than random guessing against large language models, suggesting that modern pre-training approaches with massive datasets may be free from privacy leakage risks. Our work offers a complementary perspective to these findings by exploring how examining LLMs' internal representations, rather than just their outputs, may provide additional insights into potential membership inference signals. Our framework, \emph{memTrace}, follows what we call \enquote{neural breadcrumbs} extracting informative signals from transformer hidden states and attention patterns as they process candidate sequences. By analyzing layer-wise representation dynamics, attention distribution characteristics, and cross-layer transition patterns, we detect potential memorization fingerprints that traditional loss-based approaches may not capture. This approach yields strong membership detection across several model families achieving average AUC scores of 0.85 on popular MIA benchmarks. Our findings suggest that internal model behaviors can reveal aspects of training data exposure even when output-based signals appear protected, highlighting the need for further research into membership privacy and the development of more robust privacy-preserving training techniques for large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04439v2">ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. We evaluate on ARC-AGI, a benchmark that stresses compositional generalization and abstract reasoning, making it a natural fit for concept memory. Our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, dynamically updating memory during test-time outperforms fixed settings, supporting the hypothesis that accumulating and abstracting patterns enables further solutions in a form of self-improvement. Code is available at https://github.com/matt-seb-ho/arc_memo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20401v2">Revealing Potential Biases in LLM-Based Recommender Systems in the Cold Start Setting</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 In Proceedings of 2nd Workshop on Evaluating and Applying Recommendation Systems with Large Language Models (EARL) at RecSys 2025 (EARL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for recommendation tasks due to their general-purpose capabilities. While LLMs perform well in rich-context settings, their behavior in cold-start scenarios, where only limited signals such as age, gender, or language are available, raises fairness concerns because they may rely on societal biases encoded during pretraining. We introduce a benchmark specifically designed to evaluate fairness in zero-context recommendation. Our modular pipeline supports configurable recommendation domains and sensitive attributes, enabling systematic and flexible audits of any open-source LLM. Through evaluations of state-of-the-art models (Gemma 3 and Llama 3.2), we uncover consistent biases across recommendation domains (music, movies, and colleges) including gendered and cultural stereotypes. We also reveal a non-linear relationship between model size and fairness, highlighting the need for nuanced analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05291v1">Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) learn non-trivial abstractions during pretraining, like detecting irregular plural noun subjects. However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07236v2">Simple Yet Effective: An Information-Theoretic Approach to Multi-LLM Uncertainty Quantification</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often behave inconsistently across inputs, indicating uncertainty and motivating the need for its quantification in high-stakes settings. Prior work on calibration and uncertainty quantification often focuses on individual models, overlooking the potential of model diversity. We hypothesize that LLMs make complementary predictions due to differences in training and the Zipfian nature of language, and that aggregating their outputs leads to more reliable uncertainty estimates. To leverage this, we propose MUSE (Multi-LLM Uncertainty via Subset Ensembles), a simple information-theoretic method that uses Jensen-Shannon Divergence to identify and aggregate well-calibrated subsets of LLMs. Experiments on binary prediction tasks demonstrate improved calibration and predictive performance compared to single-model and na\"ive ensemble baselines. In addition, we explore using MUSE as guided signals with chain-of-thought distillation to fine-tune LLMs for calibration. MUSE is available at:https://github.com/LARK-NLP-Lab/MUSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05528v2">Conversational Education at Scale: A Multi-LLM Agent Workflow for Procedural Learning and Pedagogic Quality Assessment</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 14 pages, accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced virtual educators and learners, bridging NLP with AI4Education. Existing work often lacks scalability and fails to leverage diverse, large-scale course content, with limited frameworks for assessing pedagogic quality. To this end, we propose WikiHowAgent, a multi-agent workflow leveraging LLMs to simulate interactive teaching-learning conversations. It integrates teacher and learner agents, an interaction manager, and an evaluator to facilitate procedural learning and assess pedagogic quality. We introduce a dataset of 114,296 teacher-learner conversations grounded in 14,287 tutorials across 17 domains and 727 topics. Our evaluation protocol combines computational and rubric-based metrics with human judgment alignment. Results demonstrate the workflow's effectiveness in diverse setups, offering insights into LLM capabilities across domains. Our datasets and implementations are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06460v2">Can LLMs Simulate Personas with Reversed Performance? A Benchmark for Counterfactual Instruction Following</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now increasingly widely used to simulate personas in virtual environments, leveraging their instruction-following capability. However, we discovered that even state-of-the-art LLMs cannot simulate personas with reversed performance (e.g., student personas with low proficiency in educational settings), which impairs the simulation diversity and limits the practical applications of the simulated environments. In this work, using mathematical reasoning as a representative scenario, we propose the first benchmark dataset for evaluating LLMs on simulating personas with reversed performance, a capability that we dub "counterfactual instruction following". We evaluate both open-weight and closed-source LLMs on this task and find that LLMs, including the OpenAI o1 reasoning model, all struggle to follow counterfactual instructions for simulating reversedly performing personas. Intersectionally simulating both the performance level and the race population of a persona worsens the effect even further. These results highlight the challenges of counterfactual instruction following and the need for further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22809v2">First Steps Towards Overhearing LLM Agents: A Case Study With Dungeons & Dragons Gameplay</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 9 pages, 5 figures. COLM 2025 Workshop on AI Agents
    </div>
    <details class="paper-abstract">
      Much work has been done on conversational LLM agents which directly assist human users with tasks. We present an alternative paradigm for interacting with LLM agents, which we call "overhearing agents". These overhearing agents do not actively participate in conversation -- instead, they "listen in" on human-to-human conversations and perform background tasks or provide suggestions to assist the user. In this work, we explore the overhearing agents paradigm through the lens of Dungeons & Dragons gameplay. We present an in-depth study using large multimodal audio-language models as overhearing agents to assist a Dungeon Master. We perform a human evaluation to examine the helpfulness of such agents and find that some large audio-language models have the emergent ability to perform overhearing agent tasks using implicit audio cues. Finally, we release Python libraries and our project code to support further research into the overhearing agents paradigm at https://github.com/zhudotexe/overhearing_agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.18416v5">PersonaGym: Evaluating Persona Agents and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      Persona agents, which are LLM agents conditioned to act according to an assigned persona, enable contextually rich and user aligned interactions across domains like education and healthcare. However, evaluating how faithfully these agents adhere to their personas remains a significant challenge, particularly in free-form settings that demand consistency across diverse, persona-relevant environments. We introduce PersonaGym, the first dynamic evaluation framework for persona agents, and PersonaScore, a human-aligned automatic metric grounded in decision theory that enables comprehensive large-scale evaluation. Our evaluation of 10 leading LLMs across 200 personas and 10,000 questions reveals significant advancement opportunities. For example, GPT-4.1 had the exact same PersonaScore as LLaMA-3-8b despite being a more recent and advanced closed source model. Importantly, increased model size and complexity do not necessarily enhance persona agent capabilities, underscoring the need for algorithmic and architectural innovation toward faithful, performant persona agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20938v2">ParEval-Repo: A Benchmark Suite for Evaluating LLMs with Repository-level HPC Translation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      GPGPU architectures have become significantly more diverse in recent years, which has led to an emergence of a variety of specialized programming models and software stacks to support them. Portable programming models exist, but they require significant developer effort to port to and optimize for different hardware architectures. Large language models (LLMs) may help to reduce this programmer burden. In this paper, we present a novel benchmark and testing framework, ParEval-Repo, which can be used to evaluate the efficacy of LLM-based approaches in automatically translating entire codebases across GPGPU execution models. ParEval-Repo includes several scientific computing and AI mini-applications in a range of programming models and levels of repository complexity. We use ParEval-Repo to evaluate a range of state-of-the-art open-source and commercial LLMs, with both a non-agentic and a top-down agentic approach. We assess code generated by the LLMs and approaches in terms of compilability, functional correctness, categories of build errors, and the cost of translation in terms of the number of inference tokens. Our results demonstrate that LLM translation of scientific applications is feasible for small programs but difficulty with generating functional build systems and cross-file dependencies pose challenges in scaling to larger codebases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05199v1">Triadic Fusion of Cognitive, Functional, and Causal Dimensions for Explainable LLMs: The TAXAL Framework</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 27 pages, 9 tables and 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed in high-risk domains where opacity, bias, and instability undermine trust and accountability. Traditional explainability methods, focused on surface outputs, do not capture the reasoning pathways, planning logic, and systemic impacts of agentic LLMs. We introduce TAXAL (Triadic Alignment for eXplainability in Agentic LLMs), a triadic fusion framework that unites three complementary dimensions: cognitive (user understanding), functional (practical utility), and causal (faithful reasoning). TAXAL provides a unified, role-sensitive foundation for designing, evaluating, and deploying explanations in diverse sociotechnical settings. Our analysis synthesizes existing methods, ranging from post-hoc attribution and dialogic interfaces to explanation-aware prompting, and situates them within the TAXAL triadic fusion model. We further demonstrate its applicability through case studies in law, education, healthcare, and public services, showing how explanation strategies adapt to institutional constraints and stakeholder roles. By combining conceptual clarity with design patterns and deployment pathways, TAXAL advances explainability as a technical and sociotechnical practice, supporting trustworthy and context-sensitive LLM applications in the era of agentic AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00461v2">TECP: Token-Entropy Conformal Prediction for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Uncertainty quantification (UQ) for open-ended language generation remains a critical yet underexplored challenge, especially under black-box constraints where internal model signals are inaccessible. In this paper, we introduce Token-Entropy Conformal Prediction (TECP), a novel framework that leverages token-level entropy as a logit-free, reference-free uncertainty measure and integrates it into a split conformal prediction (CP) pipeline to construct prediction sets with formal coverage guarantees. Unlike existing approaches that rely on semantic consistency heuristics or white-box features, TECP directly estimates epistemic uncertainty from the token entropy structure of sampled generations and calibrates uncertainty thresholds via CP quantiles to ensure provable error control. Empirical evaluations across six large language models and two benchmarks (CoQA and TriviaQA) demonstrate that TECP consistently achieves reliable coverage and compact prediction sets, outperforming prior self-consistency-based UQ methods. Our method provides a principled and efficient solution for trustworthy generation in black-box LLM settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05042v1">Shared Autonomy through LLMs and Reinforcement Learning for Applications to Ship Hull Inspections</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Shared autonomy is a promising paradigm in robotic systems, particularly within the maritime domain, where complex, high-risk, and uncertain environments necessitate effective human-robot collaboration. This paper investigates the interaction of three complementary approaches to advance shared autonomy in heterogeneous marine robotic fleets: (i) the integration of Large Language Models (LLMs) to facilitate intuitive high-level task specification and support hull inspection missions, (ii) the implementation of human-in-the-loop interaction frameworks in multi-agent settings to enable adaptive and intent-aware coordination, and (iii) the development of a modular Mission Manager based on Behavior Trees to provide interpretable and flexible mission control. Preliminary results from simulation and real-world lake-like environments demonstrate the potential of this multi-layered architecture to reduce operator cognitive load, enhance transparency, and improve adaptive behaviour alignment with human intent. Ongoing work focuses on fully integrating these components, refining coordination mechanisms, and validating the system in operational port scenarios. This study contributes to establishing a modular and scalable foundation for trustworthy, human-collaborative autonomy in safety-critical maritime robotics applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14387v2">Don't Make It Up: Preserving Ignorance Awareness in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Existing work on mitigating catastrophic forgetting during large language models (LLMs) fine-tuning for new knowledge instances has primarily focused on preserving performance on previously seen data, while critically overlooking the collapse of essential capabilities instilled through alignment, most notably the model's ability to faithfully express epistemic uncertainty (a property we term 'Ignorance Awareness'). In this work, we formalize the notion of Ignorance Awareness and illustrate that conventional fine-tuning methods can result in substantial activation displacement. This displacement undermines the critical capability of ignorance awareness, leading to undesirable behaviors such as hallucinations. To address this challenge, we introduce SEAT, a simple and principled fine-tuning approach that not only enables the model to effectively acquire new knowledge instances but also preserves its aligned ignorance awareness. SEAT integrates two key components: (1) sparse tuning that constrains activation drift, and (2) a novel entity perturbation method designed to counter knowledge entanglement. Experimental results demonstrate that, across both real-world and synthetic datasets, SEAT significantly outperforms baselines in preserving ignorance awareness while retaining optimal fine-tuning performance, offering a more robust solution for LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14506v3">InteLiPlan: An Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      We introduce an interactive LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting embodied intelligence. Our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline that embodies an LLM. Our framework, InteLiPlan, ensures that the LLM's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention when user instruction is required. We evaluate our method in both simulation and on the real Toyota Human Support Robot and Anymal D-Unitree Z1 platforms. Our method achieves a 95% success rate in the 'fetch me' task completion with failure recovery, highlighting its capability in both failure reasoning and task planning. InteLiPlan achieves comparable performance to state-of-the-art large-scale LLM-based robotics planners, while using only real-time onboard computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04365v3">AutoPDL: Automatic Prompt Optimization for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) depends on how they are prompted, with choices spanning both the high-level prompting pattern (e.g., Zero-Shot, CoT, ReAct, ReWOO) and the specific prompt content (instructions and few-shot demonstrations). Manually tuning this combination is tedious, error-prone, and specific to a given LLM and task. Therefore, this paper proposes AutoPDL, an automated approach to discovering good LLM agent configurations. Our approach frames this as a structured AutoML problem over a combinatorial space of agentic and non-agentic prompting patterns and demonstrations, using successive halving to efficiently navigate this space. We introduce a library implementing common prompting patterns using the PDL prompt programming language. AutoPDL solutions are human-readable, editable, and executable PDL programs that use this library. This approach also enables source-to-source optimization, allowing human-in-the-loop refinement and reuse. Evaluations across three tasks and seven LLMs (ranging from 3B to 70B parameters) show consistent accuracy gains ($9.21\pm15.46$ percentage points), up to 67.5pp, and reveal that selected prompting strategies vary across models and tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04993v1">LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 This paper has been accepted by IEEE Communications Magazine
    </div>
    <details class="paper-abstract">
      The ubiquitous computing resources in 6G networks provide ideal environments for the fusion of large language models (LLMs) and intelligent services through the agent framework. With auxiliary modules and planning cores, LLM-enabled agents can autonomously plan and take actions to deal with diverse environment semantics and user intentions. However, the limited resources of individual network devices significantly hinder the efficient operation of LLM-enabled agents with complex tool calls, highlighting the urgent need for efficient multi-level device collaborations. To this end, the framework and method of the LLM-enabled multi-agent system with dual-loop terminal-edge collaborations are proposed in 6G networks. Firstly, the outer loop consists of the iterative collaborations between the global agent and multiple sub-agents deployed on edge servers and terminals, where the planning capability is enhanced through task decomposition and parallel sub-task distribution. Secondly, the inner loop utilizes sub-agents with dedicated roles to circularly reason, execute, and replan the sub-task, and the parallel tool calling generation with offloading strategies is incorporated to improve efficiency. The improved task planning capability and task execution efficiency are validated through the conducted case study in 6G-supported urban safety governance. Finally, the open challenges and future directions are thoroughly analyzed in 6G networks, accelerating the advent of the 6G era.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17450v2">Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 To appear at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11987v3">FutureX: An Advanced Live Benchmark for LLM Agents in Future Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Technical report, 51 pages. Update the results
    </div>
    <details class="paper-abstract">
      Future prediction is a complex task for LLM agents, requiring a high level of analytical thinking, information gathering, contextual understanding, and decision-making under uncertainty. Agents must not only gather and interpret vast amounts of dynamic information but also integrate diverse data sources, weigh uncertainties, and adapt predictions based on emerging trends, just as human experts do in fields like politics, economics, and finance. Despite its importance, no large-scale benchmark exists for evaluating agents on future prediction, largely due to challenges in handling real-time updates and retrieving timely, accurate answers. To address this, we introduce $\textbf{FutureX}$, a dynamic and live evaluation benchmark specifically designed for LLM agents performing future prediction tasks. FutureX is the largest and most diverse live benchmark for future prediction, supporting real-time daily updates and eliminating data contamination through an automated pipeline for question gathering and answer collection. We evaluate 25 LLM/agent models, including those with reasoning, search capabilities, and integration of external tools such as the open-source Deep Research Agent and closed-source Deep Research models. This comprehensive evaluation assesses agents' adaptive reasoning and performance in dynamic environments. Additionally, we provide in-depth analyses of agents' failure modes and performance pitfalls in future-oriented tasks, including the vulnerability to fake web pages and the temporal validity. Our goal is to establish a dynamic, contamination-free evaluation standard that drives the development of LLM agents capable of performing at the level of professional human analysts in complex reasoning and predictive thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20541v4">MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      This paper introduces MeLA, a Metacognitive LLM-Driven Architecture that presents a new paradigm for Automatic Heuristic Design (AHD). Traditional evolutionary methods operate directly on heuristic code; in contrast, MeLA evolves the instructional prompts used to guide a Large Language Model (LLM) in generating these heuristics. This process of "prompt evolution" is driven by a novel metacognitive framework where the system analyzes performance feedback to systematically refine its generative strategy. MeLA's architecture integrates a problem analyzer to construct an initial strategic prompt, an error diagnosis system to repair faulty code, and a metacognitive search engine that iteratively optimizes the prompt based on heuristic effectiveness. In comprehensive experiments across both benchmark and real-world problems, MeLA consistently generates more effective and robust heuristics, significantly outperforming state-of-the-art methods. Ultimately, this research demonstrates the profound potential of using cognitive science as a blueprint for AI architecture, revealing that by enabling an LLM to metacognitively regulate its problem-solving process, we unlock a more robust and interpretable path to AHD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.12226v4">AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 28 pages, 16 figures, under review, work in progress
    </div>
    <details class="paper-abstract">
      We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitating any-to-any multimodal conversation while achieving performance comparable to specialized models across all modalities, proving that discrete representations can effectively and conveniently unify multiple modalities within a language model. Demos are shown in https://junzhan2000.github.io/AnyGPT.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04876v1">OSC: Cognitive Orchestration through Dynamic Knowledge Alignment in Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 Accepted at EMNLP 2025 (Long Paper)
    </div>
    <details class="paper-abstract">
      This paper introduces OSC (Orchestrating Cognitive Synergy), a knowledge-aware adaptive collaboration framework designed to enhance cognitive synergy in multi-agent systems with large language models. While prior work has advanced agent selection and result aggregation, efficient linguistic interactions for deep collaboration among expert agents remain a critical bottleneck. OSC addresses this gap as a pivotal intermediate layer between selection and aggregation, introducing Collaborator Knowledge Models (CKM) to enable each agent to dynamically perceive its collaborators' cognitive states. Through real-time cognitive gap analysis, agents adaptively adjust communication behaviors, including content focus, detail level, and expression style, using learned strategies. Experiments on complex reasoning and problem-solving benchmarks demonstrate that OSC significantly improves task performance and communication efficiency, transforming "parallel-working individuals'' into a "deeply collaborative cognitive team.'' This framework not only optimizes multi-agent collaboration but also offers new insights into LLM agent interaction behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04868v1">Using LLMs for Multilingual Clinical Entity Linking to ICD-10</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 7 pages, 2 Figures, to be published in Proceedings of the 15th International Conference on Recent Advances in Natural Language Processing, RANLP 2025
    </div>
    <details class="paper-abstract">
      The linking of clinical entities is a crucial part of extracting structured information from clinical texts. It is the process of assigning a code from a medical ontology or classification to a phrase in the text. The International Classification of Diseases - 10th revision (ICD-10) is an international standard for classifying diseases for statistical and insurance purposes. Automatically assigning the correct ICD-10 code to terms in discharge summaries will simplify the work of healthcare professionals and ensure consistent coding in hospitals. Our paper proposes an approach for linking clinical terms to ICD-10 codes in different languages using Large Language Models (LLMs). The approach consists of a multistage pipeline that uses clinical dictionaries to match unambiguous terms in the text and then applies in-context learning with GPT-4.1 to predict the ICD-10 code for the terms that do not match the dictionary. Our system shows promising results in predicting ICD-10 codes on different benchmark datasets in Spanish - 0.89 F1 for categories and 0.78 F1 on subcategories on CodiEsp, and Greek - 0.85 F1 on ElCardioCC.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21433v2">The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 v2: Fixed typos and formatting issues
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents solve complex tasks through iterative reasoning, exploration, and tool-use, a process that can result in long, expensive context histories. While state-of-the-art Software Engineering ( SE) agents like OpenHands or Cursor use LLM-based summarization to tackle this issue, it is unclear whether the increased complexity offers tangible performance benefits compared to simply omitting older observations. We present a systematic comparison of these strategies within SWE-agent on SWE-bench Verified across five diverse model configurations. We find that a simple observation-masking strategy halves cost relative to a raw agent while matching, and sometimes slightly exceeding, the solve rate of LLM summarization. For example, with Qwen3-Coder 480B, masking improves solve rate from 53.8% (raw agent) to 54.8%, while remaining competitive with summarization at a lower cost. These results suggest that, at least within SWE-agent on SWE-bench Verified, the most effective and efficient context management can be the simplest. We release code and data for reproducibility
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04827v1">VoltanaLLM: Feedback-Driven Frequency Control and State-Space Routing for Energy-Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Modern Large Language Model (LLM) serving systems increasingly support interactive applications, like real-time chat assistants, code generation tools, and agentic workflows. However, the soaring energy cost of LLM inference presents a growing challenge for sustainable and cost-effective deployment. This paper introduces VoltanaLLM, a system for SLO-aware, energy-efficient LLM serving, built from a control theory perspective. VoltanaLLM co-designs frequency scaling and request routing in emerging prefill/decode disaggregated architectures, leveraging their decoupled execution to enable fine-grained phase-specific control. It consists of a feedback-driven frequency controller that dynamically adapts GPU frequency for prefill and decode phases, and a state-space router that explores routing decisions across frequency-scaled instances to minimize energy under latency constraints. We implement VoltanaLLM in SGLang and evaluate its performance over multiple state-of-the-art LLMs and real-world datasets. The results demonstrate that VoltanaLLM achieves up to 36.3% energy savings while maintaining near-perfect SLO attainment rate, paving the way for sustainable and intelligent LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04794v1">Personality as a Probe for LLM Evaluation: Method Trade-offs and Downstream Effects</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Personality manipulation in large language models (LLMs) is increasingly applied in customer service and agentic scenarios, yet its mechanisms and trade-offs remain unclear. We present a systematic study of personality control using the Big Five traits, comparing in-context learning (ICL), parameter-efficient fine-tuning (PEFT), and mechanistic steering (MS). Our contributions are fourfold. First, we construct a contrastive dataset with balanced high/low trait responses, enabling effective steering vector computation and fair cross-method evaluation. Second, we introduce a unified evaluation framework based on within-run $\Delta$ analysis that disentangles, reasoning capability, agent performance, and demographic bias across MMLU, GAIA, and BBQ benchmarks. Third, we develop trait purification techniques to separate openness from conscientiousness, addressing representational overlap in trait encoding. Fourth, we propose a three-level stability framework that quantifies method-, trait-, and combination-level robustness, offering practical guidance under deployment constraints. Experiments on Gemma-2-2B-IT and LLaMA-3-8B-Instruct reveal clear trade-offs: ICL achieves strong alignment with minimal capability loss, PEFT delivers the highest alignment at the cost of degraded task performance, and MS provides lightweight runtime control with competitive effectiveness. Trait-level analysis shows openness as uniquely challenging, agreeableness as most resistant to ICL, and personality encoding consolidating around intermediate layers. Taken together, these results establish personality manipulation as a multi-level probe into behavioral representation, linking surface conditioning, parameter encoding, and activation-level steering, and positioning mechanistic steering as a lightweight alternative to fine-tuning for both deployment and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17422v3">Multimodal LLM Guided Exploration and Active Mapping using Fisher Information</a></div>
    <div class="paper-meta">
      📅 2025-09-05
      | 💬 ICCV 2025
    </div>
    <details class="paper-abstract">
      We present an active mapping system that plans for both long-horizon exploration goals and short-term actions using a 3D Gaussian Splatting (3DGS) representation. Existing methods either do not take advantage of recent developments in multimodal Large Language Models (LLM) or do not consider challenges in localization uncertainty, which is critical in embodied agents. We propose employing multimodal LLMs for long-horizon planning in conjunction with detailed motion planning using our information-based objective. By leveraging high-quality view synthesis from our 3DGS representation, our method employs a multimodal LLM as a zero-shot planner for long-horizon exploration goals from the semantic perspective. We also introduce an uncertainty-aware path proposal and selection algorithm that balances the dual objectives of maximizing the information gain for the environment while minimizing the cost of localization errors. Experiments conducted on the Gibson and Habitat-Matterport 3D datasets demonstrate state-of-the-art results of the proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04781v1">The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      When given the option, will LLMs choose to leave the conversation (bail)? We investigate this question by giving models the option to bail out of interactions using three different bail methods: a bail tool the model can call, a bail string the model can output, and a bail prompt that asks the model if it wants to leave. On continuations of real world data (Wildchat and ShareGPT), all three of these bail methods find models will bail around 0.28-32\% of the time (depending on the model and bail method). However, we find that bail rates can depend heavily on the model used for the transcript, which means we may be overestimating real world bail rates by up to 4x. If we also take into account false positives on bail prompt (22\%), we estimate real world bail rates range from 0.06-7\%, depending on the model and bail method. We use observations from our continuations of real world data to construct a non-exhaustive taxonomy of bail cases, and use this taxonomy to construct BailBench: a representative synthetic dataset of situations where some models bail. We test many models on this dataset, and observe some bail behavior occurring for most of them. Bail rates vary substantially between models, bail methods, and prompt wordings. Finally, we study the relationship between refusals and bails. We find: 1) 0-13\% of continuations of real world conversations resulted in a bail without a corresponding refusal 2) Jailbreaks tend to decrease refusal rates, but increase bail rates 3) Refusal abliteration increases no-refuse bail rates, but only for some bail methods 4) Refusal rate on BailBench does not appear to predict bail rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20474v2">MountainLion: A Multi-Modal LLM-Based Agent System for Interpretable and Adaptive Financial Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-05
    </div>
    <details class="paper-abstract">
      Cryptocurrency trading is a challenging task requiring the integration of heterogeneous data from multiple modalities. Traditional deep learning and reinforcement learning approaches typically demand large training datasets and encode diverse inputs into numerical representations, often at the cost of interpretability. Recent progress in large language model (LLM)-based agents has demonstrated the capacity to process multi-modal data and support complex investment decision-making. Building on these advances, we present \textbf{MountainLion}, a multi-modal, multi-agent system for financial trading that coordinates specialized LLM-based agents to interpret financial data and generate investment strategies. MountainLion processes textual news, candlestick charts, and trading signal charts to produce high-quality financial reports, while also enabling modification of reports and investment recommendations through data-driven user interaction and question answering. A central reflection module analyzes historical trading signals and outcomes to continuously refine decision processes, and the system is capable of real-time report analysis, summarization, and dynamic adjustment of investment strategies. Empirical results confirm that MountainLion systematically enriches technical price triggers with contextual macroeconomic and capital flow signals, providing a more interpretable, robust, and actionable investment framework that improves returns and strengthens investor confidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12736v2">ACING: Actor-Critic for Instruction Learning in Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      The effectiveness of Large Language Models (LLMs) in solving tasks depends significantly on the quality of their instructions, which often require substantial human effort to craft. This underscores the need for automated instruction optimization. However, optimizing instructions is particularly challenging when working with black-box LLMs, where model parameters and gradients are inaccessible. We introduce ACING, an actor-critic reinforcement learning framework that formulates instruction optimization as a stateless, continuous-action problem, enabling exploration of infinite instruction spaces using only black-box feedback. ACING automatically discovers prompts that outperform human-written prompts in 76% of instruction-induction tasks, with gains of up to 33 points and a 10-point median improvement over the best automatic baseline in 33 tasks spanning instruction-induction, summarization, and chain-of-thought reasoning. Extensive ablations highlight its robustness and efficiency. An implementation of ACING is available at https://github.com/salmakh1/ACING.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04439v1">ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. On the challenging ARC-AGI benchmark, our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, we confirm that dynamically updating memory during test-time outperforms an otherwise identical fixed memory setting with additional attempts, supporting the hypothesis that solving more problems and abstracting more patterns to memory enables further solutions in a form of self-improvement. Code available at https://github.com/matt-seb-ho/arc_memo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03312v2">AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agentic systems, often comprising multiple models, complex tool invocations, and orchestration protocols, substantially outperform monolithic agents. Yet this very sophistication amplifies their fragility, making them more prone to system failure. Pinpointing the specific agent or step responsible for an error within long execution traces defines the task of agentic system failure attribution. Current state-of-the-art reasoning LLMs, however, remain strikingly inadequate for this challenge, with accuracy generally below 10%. To address this gap, we propose AgenTracer, the first automated framework for annotating failed multi-agent trajectories via counterfactual replay and programmed fault injection, producing the curated dataset TracerTraj. Leveraging this resource, we develop AgenTracer-8B, a lightweight failure tracer trained with multi-granular reinforcement learning, capable of efficiently diagnosing errors in verbose multi-agent interactions. On the Who&When benchmark, AgenTracer-8B outperforms giant proprietary LLMs like Gemini-2.5-Pro and Claude-4-Sonnet by up to 18.18%, setting a new standard in LLM agentic failure attribution. More importantly, AgenTracer-8B delivers actionable feedback to off-the-shelf multi-agent systems like MetaGPT and MaAS with 4.8-14.2% performance gains, empowering self-correcting and self-evolving agentic AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04392v1">Denoising GER: A Noise-Robust Generative Error Correction with LLM for Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLM) have made significant progress in the task of generation error correction (GER) for automatic speech recognition (ASR) post-processing. However, in complex noisy environments, they still face challenges such as poor adaptability and low information utilization, resulting in limited effectiveness of GER. To address these issues, this paper proposes a noise-robust multi-modal GER framework (Denoising GER). The framework enhances the model's adaptability to different noisy scenarios through a noise-adaptive acoustic encoder and optimizes the integration of multi-modal information via a heterogeneous feature compensation dynamic fusion (HFCDF) mechanism, improving the LLM's utilization of multi-modal information. Additionally, reinforcement learning (RL) training strategies are introduced to enhance the model's predictive capabilities. Experimental results demonstrate that Denoising GER significantly improves accuracy and robustness in noisy environments and exhibits good generalization abilities in unseen noise scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04373v1">Measuring Bias or Measuring the Task: Understanding the Brittle Nature of LLM Gender Biases</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly applied in socially impactful settings, concerns about gender bias have prompted growing efforts both to measure and mitigate such bias. These efforts often rely on evaluation tasks that differ from natural language distributions, as they typically involve carefully constructed task prompts that overtly or covertly signal the presence of gender bias-related content. In this paper, we examine how signaling the evaluative purpose of a task impacts measured gender bias in LLMs. Concretely, we test models under prompt conditions that (1) make the testing context salient, and (2) make gender-focused content salient. We then assess prompt sensitivity across four task formats with both token-probability and discrete-choice metrics. We find that even minor prompt changes can substantially alter bias outcomes, sometimes reversing their direction entirely. Discrete-choice metrics further tend to amplify bias relative to probabilistic measures. These findings do not only highlight the brittleness of LLM gender bias evaluations but open a new puzzle for the NLP benchmarking and development community: To what extent can well-controlled testing designs trigger LLM ``testing mode'' performance, and what does this mean for the ecological validity of future benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07900v2">MiniCPM4: Ultra-Efficient LLMs on End Devices</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 MiniCPM4 Technical Report
    </div>
    <details class="paper-abstract">
      This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose CPM.cu that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Furthermore, we construct a hybrid reasoning model, MiniCPM4.1, which can be used in both deep reasoning mode and non-reasoning mode. Evaluation results demonstrate that MiniCPM4 and MiniCPM4.1 outperform similar-sized open-source models across benchmarks, with the 8B variants showing significant speed improvements on long sequence understanding and generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04340v1">Write on Paper, Wrong in Practice: Why LLMs Still Struggle with Writing Clinical Notes</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are often proposed as tools to streamline clinical documentation, a task viewed as both high-volume and low-risk. However, even seemingly straightforward applications of LLMs raise complex sociotechnical considerations to translate into practice. This case study, conducted at KidsAbility, a pediatric rehabilitation facility in Ontario, Canada examined the use of LLMs to support occupational therapists in reducing documentation burden.We conducted a qualitative study involving 20 clinicians who participated in pilot programs using two AI technologies: a general-purpose proprietary LLM and a bespoke model fine-tuned on proprietary historical documentation. Our findings reveal that documentation challenges are sociotechnical in nature, shaped by clinical workflows, organizational policies, and system constraints. Four key themes emerged: (1) the heterogeneity of workflows, (2) the documentation burden is systemic and not directly linked to the creation of any single type of documentation, (3) the need for flexible tools and clinician autonomy, and (4) effective implementation requires mutual learning between clinicians and AI systems. While LLMs show promise in easing documentation tasks, their success will depend on flexible, adaptive integration that supports clinician autonomy. Beyond technical performance, sustained adoption will require training programs and implementation strategies that reflect the complexity of clinical environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04316v2">Small Changes, Large Consequences: Analyzing the Allocational Fairness of LLMs in Hiring Contexts</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed in high-stakes applications like hiring, yet their potential for unfair decision-making remains understudied in generative and retrieval settings. In this work, we examine the allocational fairness of LLM-based hiring systems through two tasks that reflect actual HR usage: resume summarization and applicant ranking. By constructing a synthetic resume dataset with controlled perturbations and curating job postings, we investigate whether model behavior differs across demographic groups. Our findings reveal that generated summaries exhibit meaningful differences more frequently for race than for gender perturbations. Models also display non-uniform retrieval selection patterns across demographic groups and exhibit high ranking sensitivity to both gender and race perturbations. Surprisingly, retrieval models can show comparable sensitivity to both demographic and non-demographic changes, suggesting that fairness issues may stem from broader model brittleness. Overall, our results indicate that LLM-based hiring systems, especially in the retrieval stage, can exhibit notable biases that lead to discriminatory outcomes in real-world contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02761v2">Plan Verification for LLM-Based Embodied Task Completion Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based task plans and corresponding human demonstrations for embodied AI may be noisy, with unnecessary actions, redundant navigation, and logical errors that reduce policy quality. We propose an iterative verification framework in which a Judge LLM critiques action sequences and a Planner LLM applies the revisions, yielding progressively cleaner and more spatially coherent trajectories. Unlike rule-based approaches, our method relies on natural language prompting, enabling broad generalization across error types including irrelevant actions, contradictions, and missing steps. On a set of manually annotated actions from the TEACh embodied AI dataset, our framework achieves up to 90% recall and 100% precision across four state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout). The refinement loop converges quickly, with 96.5% of sequences requiring at most three iterations, while improving both temporal efficiency and spatial action organization. Crucially, the method preserves human error-recovery patterns rather than collapsing them, supporting future work on robust corrective behavior. By establishing plan verification as a reliable LLM capability for spatial planning and action refinement, we provide a scalable path to higher-quality training data for imitation learning in embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04310v1">EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21080v4">EQ-Knight: A Memory-Augmented LLM Agent for Strategic Affective Gaming in Debt Recovery</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language model-based chatbots have enhanced engagement in financial negotiations, but their overreliance on passive empathy introduces critical risks in credit collection. While empathy-driven approaches preserve client satisfaction in benign cases, they fail catastrophically against dishonest debtors--individuals who exploit conciliatory tactics to manipulate terms or evade repayment. Blindly prioritizing "customer experience" in such scenarios leads to creditor vulnerabilities: revenue leakage, moral hazard, and systemic exploitation. To address this, we propose EQ-Knight, an LLM agent that dynamically optimizes emotional strategy to defend creditor interests. Unlike naive empathy-centric bots, EQ-Knight integrates emotion memory and game-theoretic reasoning, powered by a Hidden Markov Model (HMM) to track and predict debtor emotional states. By analyzing both real-time and historical emotional cues, EQ-Knight strategically counters negative emotions (e.g., aggression, feigned distress) while preserving productive debtor relationships. Experiments demonstrate EQ-Knight's superiority over conventional LLM negotiators: it achieves a 32\% reduction in concession losses without compromising recovery rates, particularly in adversarial cases where debtors weaponize negative emotions (e.g., intimidation, guilt-tripping) to coerce concessions. For credit agencies, EQ-Knight transforms LLMs from high-risk "people-pleasers" into strategic emotion-defenders--balancing emotional intelligence with tactical rigor to enforce accountability and deter exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04292v1">Inverse IFEval: Can LLMs Unlearn Stubborn Training Conventions to Follow Real Instructions?</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong performance on diverse tasks but often exhibit cognitive inertia, struggling to follow instructions that conflict with the standardized patterns learned during supervised fine-tuning (SFT). To evaluate this limitation, we propose Inverse IFEval, a benchmark that measures models Counter-intuitive Abilitytheir capacity to override training-induced biases and comply with adversarial instructions. Inverse IFEval introduces eight types of such challenges, including Question Correction, Intentional Textual Flaws, Code without Comments, and Counterfactual Answering. Using a human-in-the-loop pipeline, we construct a dataset of 1012 high-quality Chinese and English questions across 23 domains, evaluated under an optimized LLM-as-a-Judge framework. Experiments on existing leading LLMs demonstrate the necessity of our proposed Inverse IFEval benchmark. Our findings emphasize that future alignment efforts should not only pursue fluency and factual correctness but also account for adaptability under unconventional contexts. We hope that Inverse IFEval serves as both a diagnostic tool and a foundation for developing methods that mitigate cognitive inertia, reduce overfitting to narrow patterns, and ultimately enhance the instruction-following reliability of LLMs in diverse and unpredictable real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04250v1">How many patients could we save with LLM priors?</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Imagine a world where clinical trials need far fewer patients to achieve the same statistical power, thanks to the knowledge encoded in large language models (LLMs). We present a novel framework for hierarchical Bayesian modeling of adverse events in multi-center clinical trials, leveraging LLM-informed prior distributions. Unlike data augmentation approaches that generate synthetic data points, our methodology directly obtains parametric priors from the model. Our approach systematically elicits informative priors for hyperparameters in hierarchical Bayesian models using a pre-trained LLM, enabling the incorporation of external clinical expertise directly into Bayesian safety modeling. Through comprehensive temperature sensitivity analysis and rigorous cross-validation on real-world clinical trial data, we demonstrate that LLM-derived priors consistently improve predictive performance compared to traditional meta-analytical approaches. This methodology paves the way for more efficient and expert-informed clinical trial design, enabling substantial reductions in the number of patients required to achieve robust safety assessment and with the potential to transform drug safety monitoring and regulatory decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11701v2">DMN-Guided Prompting: A Framework for Controlling LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Large Language Models, Decision Model and Notation, Automated Feedback, Prompt Engineering
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown considerable potential in automating decision logic within knowledge-intensive processes. However, their effectiveness largely depends on the strategy and quality of prompting. Since decision logic is typically embedded in prompts, it becomes challenging for end users to modify or refine it. Decision Model and Notation (DMN) offers a standardized graphical approach for defining decision logic in a structured, user-friendly manner. This paper introduces a DMN-guided prompting framework that breaks down complex decision logic into smaller, manageable components, guiding LLMs through structured decision pathways. We implemented the framework in a graduate-level course where students submitted assignments. The assignments and DMN models representing feedback instructions served as inputs to our framework. The instructor evaluated the generated feedback and labeled it for performance assessment. Our approach demonstrated promising results, outperforming chain-of-thought (CoT) prompting in our case study. Students also responded positively to the generated feedback, reporting high levels of perceived usefulness in a survey based on the Technology Acceptance Model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13287v6">PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware Selection of Generative Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      Selecting a sample generation scheme from multiple prompt-based generative models, including large language models (LLMs) and prompt-guided image and video generation models, is typically addressed by choosing the model that maximizes an averaged evaluation score. However, this score-based selection overlooks the possibility that different models achieve the best generation performance for different types of text prompts. An online identification of the best generation model for various input prompts can reduce the costs associated with querying sub-optimal models. In this work, we explore the possibility of varying rankings of text-based generative models for different text prompts and propose an online learning framework to predict the best data generation model for a given input prompt. The proposed PAK-UCB algorithm addresses a contextual bandit (CB) setting with shared context variables across the arms, utilizing the generated data to update kernel-based functions that predict the score of each model available for unseen text prompts. Additionally, we leverage random Fourier features (RFF) to accelerate the online learning process of PAK-UCB. Our numerical experiments on real and simulated text-to-image and image-to-text generative models show that RFF-UCB performs successfully in identifying the best generation model across different sample types. The code is available at: github.com/yannxiaoyanhu/dgm-online-select.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04198v1">Are LLM Agents the New RPA? A Comparative Study with RPA Across Enterprise Workflows</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has introduced a new paradigm in automation: LLM agents or Agentic Automation with Computer Use (AACU). Unlike traditional Robotic Process Automation (RPA), which relies on rule-based workflows and scripting, AACU enables intelligent agents to perform tasks through natural language instructions and autonomous interaction with user interfaces. This study investigates whether AACU can serve as a viable alternative to RPA in enterprise workflow automation. We conducted controlled experiments across three standard RPA challenges data entry, monitoring, and document extraction comparing RPA (via UiPath) and AACU (via Anthropic's Computer Use Agent) in terms of speed, reliability, and development effort. Results indicate that RPA outperforms AACU in execution speed and reliability, particularly in repetitive, stable environments. However, AACU significantly reduces development time and adapts more flexibly to dynamic interfaces. While current AACU implementations are not yet production-ready, their promise in rapid prototyping and lightweight automation is evident. Future research should explore multi-agent orchestration, hybrid RPA-AACU architectures, and more robust evaluation across industries and platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08523v2">InferLog: Accelerating LLM Inference for Online Log Parsing via ICL-oriented Prefix Caching</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Modern software systems generate massive volumes of runtime logs, necessitating efficient and accurate log parsing to enable critical downstream tasks such as anomaly detection and root cause analysis. Recently, large language models (LLMs) have achieved advanced accuracy on log parsing, but their deployment in production environments faces two major limitations: (1) the privacy risks associated with commercial LLMs, driving the adoption of local deployment, and (2) the stringent latency and throughput requirements imposed by high-volume log streams, which existing LLM-based parsers fail to meet. Although recent efforts have reduced the number of LLM queries, they overlook the high latency of the LLM invocations, where concurrent log parsing requests can cause serve performance degradation of LLM inference system. In this study, we present InferLog, the first LLM inference optimization method for online log parsing. Our key insight is that the inference efficiency emerges as the vital bottleneck in LLM-based online log parsing, rather than parsing accuracy. InferLog accelerates inference by designing (1) A Prefix-aware ICL Refinement policy to refine the examples and permutation of in-context learning to improve the prefix caching efficiency. (2) A rapid and task-specific configuration tuning pipeline based on meta-learning to find the optimal LLM scheduling-related configuration for dynamic log parsing workloads. The experimental results based on Loghub dataset and vLLM demonstrate that InferLog significantly outperforms existing inference optimization methods and markedly accelerates the state-of-the-art LLM-based log parser without compromising parsing accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04191v1">KubeGuard: LLM-Assisted Kubernetes Hardening via Configuration Files and Runtime Logs Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      The widespread adoption of Kubernetes (K8s) for orchestrating cloud-native applications has introduced significant security challenges, such as misconfigured resources and overly permissive configurations. Failing to address these issues can result in unauthorized access, privilege escalation, and lateral movement within clusters. Most existing K8s security solutions focus on detecting misconfigurations, typically through static analysis or anomaly detection. In contrast, this paper presents KubeGuard, a novel runtime log-driven recommender framework aimed at mitigating risks by addressing overly permissive configurations. KubeGuard is designed to harden K8s environments through two complementary tasks: Resource Creation and Resource Refinement. It leverages large language models (LLMs) to analyze manifests and runtime logs reflecting actual system behavior, using modular prompt-chaining workflows. This approach enables KubeGuard to create least-privilege configurations for new resources and refine existing manifests to reduce the attack surface. KubeGuard's output manifests are presented as recommendations that users (e.g., developers and operators) can review and adopt to enhance cluster security. Our evaluation demonstrates that KubeGuard effectively generates and refines K8s manifests for Roles, NetworkPolicies, and Deployments, leveraging both proprietary and open-source LLMs. The high precision, recall, and F1-scores affirm KubeGuard's practicality as a framework that translates runtime observability into actionable, least-privilege configuration guidance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12065v3">Autoformalization in the Wild: Assessing LLMs on Real-World Mathematical Definitions</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 EMNLP 2025 Camera-Ready Version
    </div>
    <details class="paper-abstract">
      Thanks to their linguistic capabilities, LLMs offer an opportunity to bridge the gap between informal mathematics and formal languages through autoformalization. However, it is still unclear how well LLMs generalize to sophisticated and naturally occurring mathematical statements. To address this gap, we investigate the task of autoformalizing real-world mathematical definitions: a critical component of mathematical discourse. Specifically, we introduce two novel resources for autoformalization, collecting definitions from Wikipedia (Def_Wiki) and arXiv papers (Def_ArXiv). We then systematically evaluate a range of LLMs, analyzing their ability to formalize definitions into Isabelle/HOL. Furthermore, we investigate strategies to enhance LLMs' performance including refinement through external feedback from Proof Assistants, and formal definition grounding, where we augment LLMs' formalizations through relevant contextual elements from formal mathematical libraries. Our findings reveal that definitions present a greater challenge compared to existing benchmarks, such as miniF2F. In particular, we found that LLMs still struggle with self-correction, and aligning with relevant mathematical libraries. At the same time, structured refinement methods and definition grounding strategies yield notable improvements of up to 16% on self-correction capabilities and 43% on the reduction of undefined errors, highlighting promising directions for enhancing LLM-based autoformalization in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04152v1">TAGAL: Tabular Data Generation using Agentic LLM Methods</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      The generation of data is a common approach to improve the performance of machine learning tasks, among which is the training of models for classification. In this paper, we present TAGAL, a collection of methods able to generate synthetic tabular data using an agentic workflow. The methods leverage Large Language Models (LLMs) for an automatic and iterative process that uses feedback to improve the generated data without any further LLM training. The use of LLMs also allows for the addition of external knowledge in the generation process. We evaluate TAGAL across diverse datasets and different aspects of quality for the generated data. We look at the utility of downstream ML models, both by training classifiers on synthetic data only and by combining real and synthetic data. Moreover, we compare the similarities between the real and the generated data. We show that TAGAL is able to perform on par with state-of-the-art approaches that require LLM training and generally outperforms other training-free approaches. These findings highlight the potential of agentic workflow and open new directions for LLM-based data generation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09454v4">Explicit Learning and the LLM in Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      This study explores an LLM's ability to learn new languages using explanations found in a grammar book, a process we term "explicit learning." To rigorously assess this ability, we design controlled translation experiments between English and constructed languages generated, through specific cryptographic means, from Latin or French. Contrary to previous studies, our results demonstrate that LLMs do possess a measurable capacity for explicit learning. This ability, however, diminishes as the complexity of the linguistic phenomena to be learned increases. Supervised fine-tuning on ad hoc chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs, benefiting low-resource languages typically described in grammar books but lacking extensive corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04083v1">Intermediate Languages Matter: Formal Languages and LLMs affect Neurosymbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 To appear in the proceedings of The Second Workshop on Knowledge Graphs and Neurosymbolic AI (KG-NeSy) Co-located with SEMANTiCS 2025 Conference, Vienna, Austria - September 3rd, 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve astonishing results on a wide range of tasks. However, their formal reasoning ability still lags behind. A promising approach is Neurosymbolic LLM reasoning. It works by using LLMs as translators from natural to formal languages and symbolic solvers for deriving correct results. Still, the contributing factors to the success of Neurosymbolic LLM reasoning remain unclear. This paper demonstrates that one previously overlooked factor is the choice of the formal language. We introduce the intermediate language challenge: selecting a suitable formal language for neurosymbolic reasoning. By comparing four formal languages across three datasets and seven LLMs, we show that the choice of formal language affects both syntactic and semantic reasoning capabilities. We also discuss the varying effects across different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20038v3">Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19740v2">Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Reducing the key-value (KV) cache burden in Large Language Models (LLMs) significantly accelerates inference. Dynamically selecting critical KV caches during decoding helps maintain performance. Existing methods use random linear hashing to identify important tokens, but this approach is inefficient due to the orthogonal distribution of queries and keys within two narrow cones in LLMs. We introduce Spotlight Attention, a novel method that employs non-linear hashing functions to optimize the embedding distribution of queries and keys, enhancing coding efficiency and robustness. We also developed a lightweight, stable training framework using a Bradley-Terry ranking-based loss, enabling optimization of the non-linear hashing module on GPUs with 16GB memory in 8 hours. Experimental results show that Spotlight Attention drastically improves retrieval precision while shortening the length of the hash code at least 5$\times$ compared to traditional linear hashing. Finally, we exploit the computational advantages of bitwise operations by implementing specialized CUDA kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04013v1">On Robustness and Reliability of Benchmark-Based Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Accepted at ECAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) effectiveness is usually evaluated by means of benchmarks such as MMLU, ARC-C, or HellaSwag, where questions are presented in their original wording, thus in a fixed, standardized format. However, real-world applications involve linguistic variability, requiring models to maintain their effectiveness across diverse rewordings of the same question or query. In this study, we systematically assess the robustness of LLMs to paraphrased benchmark questions and investigate whether benchmark-based evaluations provide a reliable measure of model capabilities. We systematically generate various paraphrases of all the questions across six different common benchmarks, and measure the resulting variations in effectiveness of 34 state-of-the-art LLMs, of different size and effectiveness. Our findings reveal that while LLM rankings remain relatively stable across paraphrased inputs, absolute effectiveness scores change, and decline significantly. This suggests that LLMs struggle with linguistic variability, raising concerns about their generalization abilities and evaluation methodologies. Furthermore, the observed performance drop challenges the reliability of benchmark-based evaluations, indicating that high benchmark scores may not fully capture a model's robustness to real-world input variations. We discuss the implications of these findings for LLM evaluation methodologies, emphasizing the need for robustness-aware benchmarks that better reflect practical deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04007v1">AutoPBO: LLM-powered Optimization for Local Search PBO Solvers</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Pseudo-Boolean Optimization (PBO) provides a powerful framework for modeling combinatorial problems through pseudo-Boolean (PB) constraints. Local search solvers have shown excellent performance in PBO solving, and their efficiency is highly dependent on their internal heuristics to guide the search. Still, their design often requires significant expert effort and manual tuning in practice. While Large Language Models (LLMs) have demonstrated potential in automating algorithm design, their application to optimizing PBO solvers remains unexplored. In this work, we introduce AutoPBO, a novel LLM-powered framework to automatically enhance PBO local search solvers. We conduct experiments on a broad range of four public benchmarks, including one real-world benchmark, a benchmark from PB competition, an integer linear programming optimization benchmark, and a crafted combinatorial benchmark, to evaluate the performance improvement achieved by AutoPBO and compare it with six state-of-the-art competitors, including two local search PBO solvers NuPBO and OraSLS, two complete PB solvers PBO-IHS and RoundingSat, and two mixed integer programming (MIP) solvers Gurobi and SCIP. AutoPBO demonstrates significant improvements over previous local search approaches, while maintaining competitive performance compared to state-of-the-art competitors. The results suggest that AutoPBO offers a promising approach to automating local search solver design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03990v1">Meta-Policy Reflexion: Reusable Reflective Memory and Rule Admissibility for Resource-Efficient LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents achieve impressive single-task performance but commonly exhibit repeated failures, inefficient exploration, and limited cross-task adaptability. Existing reflective strategies (e.g., Reflexion, ReAct) improve per-episode behavior but typically produce ephemeral, task-specific traces that are not reused across tasks. Reinforcement-learning based alternatives can produce transferable policies but require substantial parameter updates and compute. In this work we introduce Meta-Policy Reflexion (MPR): a hybrid framework that consolidates LLM-generated reflections into a structured, predicate-like Meta-Policy Memory (MPM) and applies that memory at inference time through two complementary mechanisms soft memory-guided decoding and hard rule admissibility checks(HAC). MPR (i) externalizes reusable corrective knowledge without model weight updates, (ii) enforces domain constraints to reduce unsafe or invalid actions, and (iii) retains the adaptability of language-based reflection. We formalize the MPM representation, present algorithms for update and decoding, and validate the approach in a text-based agent environment following the experimental protocol described in the provided implementation (AlfWorld-based). Empirical results reported in the supplied material indicate consistent gains in execution accuracy and robustness when compared to Reflexion baselines; rule admissibility further improves stability. We analyze mechanisms that explain these gains, discuss scalability and failure modes, and outline future directions for multimodal and multi?agent extensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03020v2">Training LLMs to be Better Text Embedders through Bidirectional Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have increasingly been explored as powerful text embedders. Existing LLM-based text embedding approaches often leverage the embedding of the final token, typically a reserved special token such as [EOS]. However, these tokens have not been intentionally trained to capture the semantics of the whole context, limiting their capacity as text embeddings, especially for retrieval and re-ranking tasks. We propose to add a new training stage before contrastive learning to enrich the semantics of the final token embedding. This stage employs bidirectional generative reconstruction tasks, namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based Document-to-Query), which interleave to anchor the [EOS] embedding and reconstruct either side of Query-Document pairs. Experimental results demonstrate that our additional training stage significantly improves LLM performance on the Massive Text Embedding Benchmark (MTEB), achieving new state-of-the-art results across different LLM base models and scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08715v3">MultiGen: Child-Friendly Multilingual Speech Generator with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      Generative speech models have demonstrated significant potential in improving human-machine interactions, offering valuable real-world applications such as language learning for children. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiGen, a multilingual speech generation model with child-friendly interaction, leveraging LLM architecture for speech generation tailored for low-resource languages. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, which can be used to facilitate young children's communication with AI systems through culturally relevant context in three low-resource languages: Singaporean accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiGen compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03957v1">CANDY: Benchmarking LLMs' Limitations and Assistive Potential in Chinese Misinformation Fact-Checking</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      The effectiveness of large language models (LLMs) to fact-check misinformation remains uncertain, despite their growing use. To this end, we present CANDY, a benchmark designed to systematically evaluate the capabilities and limitations of LLMs in fact-checking Chinese misinformation. Specifically, we curate a carefully annotated dataset of ~20k instances. Our analysis shows that current LLMs exhibit limitations in generating accurate fact-checking conclusions, even when enhanced with chain-of-thought reasoning and few-shot prompting. To understand these limitations, we develop a taxonomy to categorize flawed LLM-generated explanations for their conclusions and identify factual fabrication as the most common failure mode. Although LLMs alone are unreliable for fact-checking, our findings indicate their considerable potential to augment human performance when deployed as assistive tools in scenarios. Our dataset and code can be accessed at https://github.com/SCUNLP/CANDY
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03659v2">Specification-Guided Repair of Arithmetic Errors in Dafny Programs using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Debugging and repairing faults when programs fail to formally verify can be complex and time-consuming. Automated Program Repair (APR) can ease this burden by automatically identifying and fixing faults. However, traditional APR techniques often rely on test suites for validation, but these may not capture all possible scenarios. In contrast, formal specifications provide strong correctness criteria, enabling more effective automated repair. In this paper, we present an APR tool for Dafny, a verification-aware programming language that uses formal specifications - including pre-conditions, post-conditions, and invariants - as oracles for fault localization and repair. Assuming the correctness of the specifications and focusing on arithmetic bugs, we localize faults through a series of steps, which include using Hoare logic to determine the state of each statement within the program, and applying Large Language Models (LLMs) to synthesize candidate fixes. The models considered are GPT-4o mini, Llama 3, Mistral 7B, and Llemma 7B. We evaluate our approach using DafnyBench, a benchmark of real-world Dafny programs. Our tool achieves 89.6% fault localization coverage and GPT-4o mini yields the highest repair success rate of 74.18%. These results highlight the potential of combining formal reasoning with LLM-based program synthesis for automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03890v1">FaMA: LLM-Empowered Agentic Assistant for Consumer-to-Consumer Marketplace</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      The emergence of agentic AI, powered by Large Language Models (LLMs), marks a paradigm shift from reactive generative systems to proactive, goal-oriented autonomous agents capable of sophisticated planning, memory, and tool use. This evolution presents a novel opportunity to address long-standing challenges in complex digital environments. Core tasks on Consumer-to-Consumer (C2C) e-commerce platforms often require users to navigate complex Graphical User Interfaces (GUIs), making the experience time-consuming for both buyers and sellers. This paper introduces a novel approach to simplify these interactions through an LLM-powered agentic assistant. This agent functions as a new, conversational entry point to the marketplace, shifting the primary interaction model from a complex GUI to an intuitive AI agent. By interpreting natural language commands, the agent automates key high-friction workflows. For sellers, this includes simplified updating and renewal of listings, and the ability to send bulk messages. For buyers, the agent facilitates a more efficient product discovery process through conversational search. We present the architecture for Facebook Marketplace Assistant (FaMA), arguing that this agentic, conversational paradigm provides a lightweight and more accessible alternative to traditional app interfaces, allowing users to manage their marketplace activities with greater efficiency. Experiments show FaMA achieves a 98% task success rate on solving complex tasks on the marketplace and enables up to a 2x speedup on interaction time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03867v1">Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Accepted for oral presentation at the EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We introduce Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth", utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive. While such expressions may resemble surface-level nonsense, they encode implicit meaning requiring contextual inference, moral reasoning, or emotional interpretation. We find that current large language models (LLMs), despite excelling at many natural language processing (NLP) tasks, consistently fail to grasp the layered semantics of Drivelological text. To investigate this, we construct a small but diverse benchmark dataset of over 1,200 meticulously curated examples, with select instances in English, Mandarin, Spanish, French, Japanese, and Korean. Annotation was especially challenging: each of the examples required careful expert review to verify that it truly reflected Drivelological characteristics. The process involved multiple rounds of discussion and adjudication to address disagreements, highlighting the subtle and subjective nature of the Drivelology. We evaluate a range of LLMs on classification, generation, and reasoning tasks. Our results reveal clear limitations of LLMs: models often confuse Drivelology with shallow nonsense, produce incoherent justifications, or miss the implied rhetorical function altogether. These findings highlight a deeper representational gap in LLMs' pragmatic understanding and challenge the assumption that statistical fluency implies cognitive comprehension. We release our dataset and code to facilitate further research in modelling linguistic depth beyond surface-level coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03827v1">What Would an LLM Do? Evaluating Policymaking Capabilities of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being adopted in high-stakes domains. Their capacity to process vast amounts of unstructured data, explore flexible scenarios, and handle a diversity of contextual factors can make them uniquely suited to provide new insights for the complexity of social policymaking. This article evaluates whether LLMs' are aligned with domain experts (and among themselves) to inform social policymaking on the subject of homelessness alleviation - a challenge affecting over 150 million people worldwide. We develop a novel benchmark comprised of decision scenarios with policy choices across four geographies (South Bend, USA; Barcelona, Spain; Johannesburg, South Africa; Macau SAR, China). The policies in scope are grounded in the conceptual framework of the Capability Approach for human development. We also present an automated pipeline that connects the benchmarked policies to an agent-based model, and we explore the social impact of the recommended policies through simulated social scenarios. The paper results reveal promising potential to leverage LLMs for social policy making. If responsible guardrails and contextual calibrations are introduced in collaboration with local domain experts, LLMs can provide humans with valuable insights, in the form of alternative policies at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01085v5">Explaining Length Bias in LLM-Based Preference Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) as judges, particularly in preference comparisons, has become widespread, but this reveals a notable bias towards longer responses, undermining the reliability of such evaluations. To better understand such bias, we propose to decompose the preference evaluation metric, specifically the win rate, into two key components: desirability and information mass, where the former is length-independent and related to trustworthiness such as correctness, toxicity, and consistency, and the latter is length-dependent and represents the amount of information in the response. We empirically demonstrated the decomposition through controlled experiments and found that response length impacts evaluations by influencing information mass. To derive a reliable evaluation metric that assesses content quality without being confounded by response length, we propose AdapAlpaca, a simple yet effective adjustment to win rate measurement. Specifically, AdapAlpaca ensures a fair comparison of response quality by aligning the lengths of reference and test model responses under equivalent length intervals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03817v1">Learning to Deliberate: Meta-policy Collaboration for Agentic LLMs with Multi-agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Multi-agent systems of large language models (LLMs) show promise for complex reasoning, but their effectiveness is often limited by fixed collaboration protocols. These frameworks typically focus on macro-level orchestration while overlooking agents' internal deliberative capabilities. This critical meta-cognitive blindspot treats agents as passive executors unable to adapt their strategy based on internal cognitive states like uncertainty or confidence. We introduce the Meta-Policy Deliberation Framework (MPDF), where agents learn a decentralized policy over a set of high-level meta-cognitive actions: Persist, Refine, and Concede. To overcome the instability of traditional policy gradients in this setting, we develop SoftRankPO, a novel reinforcement learning algorithm. SoftRankPO stabilizes training by shaping advantages based on the rank of rewards mapped through smooth normal quantiles, making the learning process robust to reward variance. Experiments show that MPDF with SoftRankPO achieves a a 4-5% absolute gain in average accuracy across five mathematical and general reasoning benchmarks compared to six state-of-the-art heuristic and learning-based multi-agent reasoning algorithms. Our work presents a paradigm for learning adaptive, meta-cognitive policies for multi-agent LLM systems, shifting the focus from designing fixed protocols to learning dynamic, deliberative strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03811v1">Leveraging LLM-Based Agents for Intelligent Supply Chain Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      In supply chain management, planning is a critical concept. The movement of physical products across different categories, from suppliers to warehouse management, to sales, and logistics transporting them to customers, entails the involvement of many entities. It covers various aspects such as demand forecasting, inventory management, sales operations, and replenishment. How to collect relevant data from an e-commerce platform's perspective, formulate long-term plans, and dynamically adjust them based on environmental changes, while ensuring interpretability, efficiency, and reliability, is a practical and challenging problem. In recent years, the development of AI technologies, especially the rapid progress of large language models, has provided new tools to address real-world issues. In this work, we construct a Supply Chain Planning Agent (SCPA) framework that can understand domain knowledge, comprehend the operator's needs, decompose tasks, leverage or create new tools, and return evidence-based planning reports. We deploy this framework in JD.com's real-world scenario, demonstrating the feasibility of LLM-agent applications in the supply chain. It effectively reduced labor and improved accuracy, stock availability, and other key metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17882v2">Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Scientific research is inherently global. However, the vast majority of academic journals are published exclusively in English, creating barriers for non-native-English-speaking researchers. In this study, we leverage large language models (LLMs) to translate published scientific articles while preserving their native JATS XML formatting, thereby developing a practical, automated approach for implementation by academic journals. Using our approach, we translate articles across multiple scientific disciplines into 28 languages. To evaluate translation accuracy, we introduce a novel question-and-answer (QA) benchmarking method, in which an LLM generates comprehension-based questions from the original text and then answers them based on the translated text. Our benchmark results show an average performance of 95.9%, showing that the key scientific details are accurately conveyed. In a user study, we translate the scientific papers of 15 researchers into their native languages, finding that the authors consistently found the translations to accurately capture the original information in their articles. Interestingly, a third of the authors found many technical terms "overtranslated," expressing a preference to keep terminology more familiar in English untranslated. Finally, we demonstrate how in-context learning techniques can be used to align translations with domain-specific preferences such as mitigating overtranslation, highlighting the adaptability and utility of LLM-driven scientific translation. The code and translated articles are available at https://hankleid.github.io/ProjectMundo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18102v5">How Can I Publish My LLM Benchmark Without Giving the True Answers Away?</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
    </div>
    <details class="paper-abstract">
      Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14240v3">HuggingGraph: Understanding the Supply Chain of LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) leverage deep learning architectures to process and predict sequences of words, enabling them to perform a wide range of natural language processing tasks, such as translation, summarization, question answering, and content generation. As existing LLMs are often built from base models or other pre-trained models and use external datasets, they can inevitably inherit vulnerabilities, biases, or malicious components that exist in previous models or datasets. Therefore, it is critical to understand these components' origin and development process to detect potential risks, improve model fairness, and ensure compliance with regulatory frameworks. Motivated by that, this project aims to study such relationships between models and datasets, which are the central parts of the LLM supply chain. First, we design a methodology to systematically collect LLMs' supply chain information. Then, we design a new graph to model the relationships between models and datasets, which is a directed heterogeneous graph, having 402,654 nodes and 462,524 edges. Lastly, we perform different types of analysis and make multiple interesting findings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04696v1">ODKE+: Ontology-Guided Open-Domain Knowledge Extraction with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Knowledge graphs (KGs) are foundational to many AI applications, but maintaining their freshness and completeness remains costly. We present ODKE+, a production-grade system that automatically extracts and ingests millions of open-domain facts from web sources with high precision. ODKE+ combines modular components into a scalable pipeline: (1) the Extraction Initiator detects missing or stale facts, (2) the Evidence Retriever collects supporting documents, (3) hybrid Knowledge Extractors apply both pattern-based rules and ontology-guided prompting for large language models (LLMs), (4) a lightweight Grounder validates extracted facts using a second LLM, and (5) the Corroborator ranks and normalizes candidate facts for ingestion. ODKE+ dynamically generates ontology snippets tailored to each entity type to align extractions with schema constraints, enabling scalable, type-consistent fact extraction across 195 predicates. The system supports batch and streaming modes, processing over 9 million Wikipedia pages and ingesting 19 million high-confidence facts with 98.8% precision. ODKE+ significantly improves coverage over traditional methods, achieving up to 48% overlap with third-party KGs and reducing update lag by 50 days on average. Our deployment demonstrates that LLM-based extraction, grounded in ontological structure and verification workflows, can deliver trustworthiness, production-scale knowledge ingestion with broad real-world applicability. A recording of the system demonstration is included with the submission and is also available at https://youtu.be/UcnE3_GsTWs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04656v1">AraHalluEval: A Fine-grained Hallucination Evaluation Framework for Arabic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Recently, extensive research on the hallucination of the large language models (LLMs) has mainly focused on the English language. Despite the growing number of multilingual and Arabic-specific LLMs, evaluating LLMs' hallucination in the Arabic context remains relatively underexplored. The knowledge gap is particularly pressing given Arabic's widespread use across many regions and its importance in global communication and media. This paper presents the first comprehensive hallucination evaluation of Arabic and multilingual LLMs on two critical Arabic natural language generation tasks: generative question answering (GQA) and summarization. This study evaluates a total of 12 LLMs, including 4 Arabic pre-trained models, 4 multilingual models, and 4 reasoning-based models. To assess the factual consistency and faithfulness of LLMs' outputs, we developed a fine-grained hallucination evaluation framework consisting of 12 fine-grained hallucination indicators that represent the varying characteristics of each task. The results reveal that factual hallucinations are more prevalent than faithfulness errors across all models and tasks. Notably, the Arabic pre-trained model Allam consistently demonstrates lower hallucination rates than multilingual models and a comparative performance with reasoning-based models. The code is available at: \href{https://github.com/aishaalansari57/AraHalluEval}{Github link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04655v1">Polysemantic Dropout: Conformal OOD Detection for Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
      | 💬 Accepted to EMNLP 2025 main conference
    </div>
    <details class="paper-abstract">
      We propose a novel inference-time out-of-domain (OOD) detection algorithm for specialized large language models (LLMs). Despite achieving state-of-the-art performance on in-domain tasks through fine-tuning, specialized LLMs remain vulnerable to incorrect or unreliable outputs when presented with OOD inputs, posing risks in critical applications. Our method leverages the Inductive Conformal Anomaly Detection (ICAD) framework, using a new non-conformity measure based on the model's dropout tolerance. Motivated by recent findings on polysemanticity and redundancy in LLMs, we hypothesize that in-domain inputs exhibit higher dropout tolerance than OOD inputs. We aggregate dropout tolerance across multiple layers via a valid ensemble approach, improving detection while maintaining theoretical false alarm bounds from ICAD. Experiments with medical-specialized LLMs show that our approach detects OOD inputs better than baseline methods, with AUROC improvements of $2\%$ to $37\%$ when treating OOD datapoints as positives and in-domain test datapoints as negatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04633v1">Scaling Environments for Organoid Intelligence with LLM-Automated Design and Plasticity-Based Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      As the complexity of artificial agents increases, the design of environments that can effectively shape their behavior and capabilities has become a critical research frontier. We propose a framework that extends this principle to a novel class of agents: biological neural networks in the form of neural organoids. This paper introduces three scalable, closed-loop virtual environments designed to train organoid-based biological agents and probe the underlying mechanisms of learning, such as long-term potentiation (LTP) and long-term depression (LTD). We detail the design of three distinct task environments with increasing complexity: (1) a conditional avoidance task, (2) a one-dimensional predator-prey scenario, and (3) a replication of the classic Pong game. For each environment, we formalize the state and action spaces, the sensory encoding and motor decoding mechanisms, and the feedback protocols based on predictable (reward) and unpredictable (punishment) stimulation. Furthermore, we propose a novel meta-learning approach where a Large Language Model (LLM) is used to automate the generation and optimization of experimental protocols, scaling the process of environment and curriculum design. Finally, we outline a multi-modal approach for evaluating learning by measuring synaptic plasticity at electrophysiological, cellular, and molecular levels. This work bridges the gap between computational neuroscience and agent-based AI, offering a unique platform for studying embodiment, learning, and intelligence in a controlled biological substrate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21934v5">Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Recent math benchmarks for large language models (LLMs) such as MathArena indicate that state-of-the-art reasoning models achieve impressive performance on mathematical competitions like AIME, with the leading model, Gemini-2.5-Pro, achieving scores comparable to top human competitors. However, these benchmarks evaluate models solely based on final numerical answers, neglecting rigorous reasoning and proof generation which are essential for real-world mathematical tasks. To address this, we introduce a comprehensive evaluation of full-solution reasoning for challenging mathematical problems. Using expert human annotators, we evaluated several state-of-the-art reasoning models on the six problems from the 2025 USAMO within hours of their release. Our results reveal that all tested models struggled significantly: only Gemini-2.5-Pro achieves a non-trivial score of 25%, while all other models achieve less than 5%. Through detailed analysis of reasoning traces, we identify the most common failure modes and find several unwanted artifacts arising from the optimization strategies employed during model training. Overall, our results suggest that current LLMs are inadequate for rigorous mathematical reasoning tasks, highlighting the need for substantial improvements in reasoning and proof generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04615v1">Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21445v2">Text2Cypher Across Languages: Evaluating and Finetuning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled natural language interfaces that translate user questions into database queries, such as Text2SQL, Text2SPARQL, and Text2Cypher. While these interfaces enhance database accessibility, most research today focuses on English, with limited evaluation in other languages. This paper investigates the performance of both foundational and finetuned LLMs on the Text2Cypher task across multiple languages. We create and release a multilingual dataset by translating English questions into Spanish and Turkish while preserving the original Cypher queries, enabling fair cross-lingual comparison. Using standardized prompts and metrics, we evaluate several foundational models and observe a consistent performance pattern: highest on English, followed by Spanish, and lowest on Turkish. We attribute this to differences in training data availability and linguistic features. We also examine the impact of translating task prompts into Spanish and Turkish. Results show little to no change in evaluation metrics, suggesting prompt translation has minor impact. Furthermore, we finetune a foundational model on two datasets: one in English only, and one multilingual. Finetuning on English improves overall accuracy but widens the performance gap between languages. In contrast, multilingual finetuning narrows the gap, resulting in more balanced performance. Our findings highlight the importance for multilingual evaluation and training to build more inclusive and robust query generation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04537v1">Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60\% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05367v1">Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19992v2">HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      The explosive growth of complex datasets across various modalities necessitates advanced analytical tools that not only group data effectively but also provide human-understandable insights into the discovered structures. We introduce HERCULES (Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization), a novel algorithm and Python package designed for hierarchical k-means clustering of diverse data types, including text, images, and numeric data (processed one modality per run). HERCULES constructs a cluster hierarchy by recursively applying k-means clustering, starting from individual data points at level 0. A key innovation is its deep integration of Large Language Models (LLMs) to generate semantically rich titles and descriptions for clusters at each level of the hierarchy, significantly enhancing interpretability. The algorithm supports two main representation modes: `direct' mode, which clusters based on original data embeddings or scaled numeric features, and `description' mode, which clusters based on embeddings derived from LLM-generated summaries. Users can provide a `topic\_seed' to guide LLM-generated summaries towards specific themes. An interactive visualization tool facilitates thorough analysis and understanding of the clustering results. We demonstrate HERCULES's capabilities and discuss its potential for extracting meaningful, hierarchical knowledge from complex datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03020v1">Training LLMs to be Better Text Embedders through Bidirectional Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-03
      | 💬 accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have increasingly been explored as powerful text embedders. Existing LLM-based text embedding approaches often leverage the embedding of the final token, typically a reserved special token such as [EOS]. However, these tokens have not been intentionally trained to capture the semantics of the whole context, limiting their capacity as text embeddings, especially for retrieval and re-ranking tasks. We propose to add a new training stage before contrastive learning to enrich the semantics of the final token embedding. This stage employs bidirectional generative reconstruction tasks, namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based Document-to-Query), which interleave to anchor the [EOS] embedding and reconstruct either side of Query-Document pairs. Experimental results demonstrate that our additional training stage significantly improves LLM performance on the Massive Text Embedding Benchmark (MTEB), achieving new state-of-the-art results across different LLM base models and scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03018v1">Mycroft: Tracing Dependencies in Collective Communication Towards Reliable LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-09-03
    </div>
    <details class="paper-abstract">
      Reliability is essential for ensuring efficiency in LLM training. However, many real-world reliability issues remain difficult to resolve, resulting in wasted resources and degraded model performance. Unfortunately, today's collective communication libraries operate as black boxes, hiding critical information needed for effective root cause analysis. We propose Mycroft, a lightweight distributed tracing and root cause analysis system designed to address previously hidden reliability issues in collective communication. Mycroft's key idea is to trace collective communication states and leverage internal control and data dependencies to resolve reliability problems in LLM training. Mycroft has been deployed at ByteDance for over six months to debug collective communication related issues at runtime. It detected anomalies within 15 seconds in 90% of cases and identified the root cause within 20 seconds in 60% of cases. We also conducted extensive fault injection experiments to demonstrate Mycroft's capability and efficiency.
    </details>
</div>
