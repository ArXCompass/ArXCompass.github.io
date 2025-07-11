# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03876v1">LLMs model how humans induce logically structured rules</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      A central goal of cognitive science is to provide a computationally explicit account of both the structure of the mind and its development: what are the primitive representational building blocks of cognition, what are the rules via which those primitives combine, and where do these primitives and rules come from in the first place? A long-standing debate concerns the adequacy of artificial neural networks as computational models that can answer these questions, in particular in domains related to abstract cognitive function, such as language and logic. This paper argues that recent advances in neural networks -- specifically, the advent of large language models (LLMs) -- represent an important shift in this debate. We test a variety of LLMs on an existing experimental paradigm used for studying the induction of rules formulated over logical concepts. Across four experiments, we find converging empirical evidence that LLMs provide at least as good a fit to human behavior as models that implement a Bayesian probablistic language of thought (pLoT), which have been the best computational models of human behavior on the same task. Moreover, we show that the LLMs make qualitatively different predictions about the nature of the rules that are inferred and deployed in order to complete the task, indicating that the LLM is unlikely to be a mere implementation of the pLoT solution. Based on these results, we argue that LLMs may instantiate a novel theoretical account of the primitive representations and computations necessary to explain human logical concepts, with which future work in cognitive science should engage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03871v1">Enhancing Adaptive Behavioral Interventions with LLM Inference from Participant-Described States</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Accepted at Machine Learning for Healthcare (MLHC) 2025
    </div>
    <details class="paper-abstract">
      The use of reinforcement learning (RL) methods to support health behavior change via personalized and just-in-time adaptive interventions is of significant interest to health and behavioral science researchers focused on problems such as smoking cessation support and physical activity promotion. However, RL methods are often applied to these domains using a small collection of context variables to mitigate the significant data scarcity issues that arise from practical limitations on the design of adaptive intervention trials. In this paper, we explore an approach to significantly expanding the state space of an adaptive intervention without impacting data efficiency. The proposed approach enables intervention participants to provide natural language descriptions of aspects of their current state. It then leverages inference with pre-trained large language models (LLMs) to better align the policy of a base RL method with these state descriptions. To evaluate our method, we develop a novel physical activity intervention simulation environment that generates text-based state descriptions conditioned on latent state variables using an auxiliary LLM. We show that this approach has the potential to significantly improve the performance of online policy learning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03865v1">OrthoRank: Token Selection via Sink Token Orthogonality for Efficient LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Attention mechanisms are central to the success of large language models (LLMs), enabling them to capture intricate token dependencies and implicitly assign importance to each token. Recent studies have revealed the sink token, which receives disproportionately high attention despite their limited semantic role. In this paper, we first expand the relationship between the sink token and other tokens, moving beyond attention to explore their similarity in hidden states, considering the layer depth. We observe that as the layers get deeper, the cosine similarity between the normalized hidden states of the sink token and those of other tokens increases, and that the normalized hidden states of the sink token exhibit negligible changes. These imply that other tokens consistently are directed toward the sink token throughout the layers. Next, we propose a dynamic token selection method, called OrthoRank, using these findings to select important tokens. Specifically, in a certain layer, we define token importance by the speed at which the token moves toward the sink token. This is converted into orthogonality with the sink token, meaning that tokens that are more orthogonal to the sink token are assigned greater importance. Finally, through extensive experiments, we demonstrated that our method results in lower perplexity and higher zero-shot accuracy compared to layer pruning methods at the same sparsity ratio with comparable throughput, while also achieving superior performance on LongBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05292v1">A LLM-Driven Multi-Agent Systems for Professional Development of Mathematics Teachers</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      Professional development (PD) serves as the cornerstone for teacher tutors to grasp content knowledge. However, providing equitable and timely PD opportunities for teachers poses significant challenges. To address this issue, we introduce I-VIP (Intelligent Virtual Interactive Program), an intelligent tutoring platform for teacher professional development, driven by large language models (LLMs) and supported by multi-agent frameworks. This platform offers a user-friendly conversational interface and allows users to employ a variety of interactive tools to facilitate question answering, knowledge comprehension, and reflective summarization while engaging in dialogue. To underpin the functionality of this platform, including knowledge expectation analysis, response scoring and classification, and feedback generation, the multi-agent frameworks are leveraged to enhance the accuracy of judgments and mitigate the issue of missing key points.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.16429v2">The Transformative Influence of LLMs on Software Development & Developer Productivity</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 10 pages, 1 figure, 3 tables
    </div>
    <details class="paper-abstract">
      The increasing adoption and commercialization of generalized Large Language Models (LLMs) have profoundly impacted various aspects of our daily lives. Initially embraced by the computer science community, the versatility of LLMs has found its way into diverse domains. In particular, the software engineering realm has witnessed the most transformative changes. With LLMs increasingly serving as AI Pair Programming Assistants spurred the development of specialized models aimed at aiding software engineers. Although this new paradigm offers numerous advantages, it also presents critical challenges and open problems. To identify the potential and prevailing obstacles, we systematically reviewed contemporary scholarly publications, emphasizing the perspectives of software developers and usability concerns. Preliminary findings underscore pressing concerns about data privacy, bias, and misinformation. Additionally, we identified several usability challenges, including prompt engineering, increased cognitive demands, and mistrust. Finally, we introduce 12 open problems that we have identified through our survey, covering these various domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04132v1">An HTR-LLM Workflow for High-Accuracy Transcription and Analysis of Abbreviated Latin Court Hand</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      This article presents and validates an ideal, four-stage workflow for the high-accuracy transcription and analysis of challenging medieval legal documents. The process begins with a specialized Handwritten Text Recognition (HTR) model, itself created using a novel "Clean Ground Truth" curation method where a Large Language Model (LLM) refines the training data. This HTR model provides a robust baseline transcription (Stage 1). In Stage 2, this baseline is fed, along with the original document image, to an LLM for multimodal post-correction, grounding the LLM's analysis and improving accuracy. The corrected, abbreviated text is then expanded into full, scholarly Latin using a prompt-guided LLM (Stage 3). A final LLM pass performs Named-Entity Correction (NEC), regularizing proper nouns and generating plausible alternatives for ambiguous readings (Stage 4). We validate this workflow through detailed case studies, achieving Word Error Rates (WER) in the range of 2-7% against scholarly ground truths. The results demonstrate that this hybrid, multi-stage approach effectively automates the most laborious aspects of transcription while producing a high-quality, analyzable output, representing a powerful and practical solution for the current technological landscape.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04105v1">Enhancing Robustness of LLM-Driven Multi-Agent Systems through Randomized Smoothing</a></div>
    <div class="paper-meta">
      📅 2025-07-05
      | 💬 Preprint accepted by Chinese Journal of Aeronautics
    </div>
    <details class="paper-abstract">
      This paper presents a defense framework for enhancing the safety of large language model (LLM) empowered multi-agent systems (MAS) in safety-critical domains such as aerospace. We apply randomized smoothing, a statistical robustness certification technique, to the MAS consensus context, enabling probabilistic guarantees on agent decisions under adversarial influence. Unlike traditional verification methods, our approach operates in black-box settings and employs a two-stage adaptive sampling mechanism to balance robustness and computational efficiency. Simulation results demonstrate that our method effectively prevents the propagation of adversarial behaviors and hallucinations while maintaining consensus performance. This work provides a practical and scalable path toward safe deployment of LLM-based MAS in real-world, high-stakes environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04103v1">How to Train Your LLM Web Agent: A Statistical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-07-05
    </div>
    <details class="paper-abstract">
      LLM-based web agents have recently made significant progress, but much of it has occurred in closed-source systems, widening the gap with open-source alternatives. Progress has been held back by two key challenges: first, a narrow focus on single-step tasks that overlooks the complexity of multi-step web interactions; and second, the high compute costs required to post-train LLM-based web agents. To address this, we present the first statistically grounded study on compute allocation for LLM web-agent post-training. Our approach uses a two-stage pipeline, training a Llama 3.1 8B student to imitate a Llama 3.3 70B teacher via supervised fine-tuning (SFT), followed by on-policy reinforcement learning. We find this process highly sensitive to hyperparameter choices, making exhaustive sweeps impractical. To spare others from expensive trial-and-error, we sample 1,370 configurations and use bootstrapping to estimate effective hyperparameters. Our results show that combining SFT with on-policy RL consistently outperforms either approach alone on both WorkArena and MiniWob++. Further, this strategy requires only 55% of the compute to match the peak performance of pure SFT on MiniWob++, effectively pushing the compute-performance Pareto frontier, and is the only strategy that can close the gap with closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03834v1">Economic Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Practitioners often navigate LLM performance trade-offs by plotting Pareto frontiers of optimal accuracy-cost trade-offs. However, this approach offers no way to compare between LLMs with distinct strengths and weaknesses: for example, a cheap, error-prone model vs a pricey but accurate one. To address this gap, we propose economic evaluation of LLMs. Our framework quantifies the performance trade-off of an LLM as a single number based on the economic constraints of a concrete use case, all expressed in dollars: the cost of making a mistake, the cost of incremental latency, and the cost of abstaining from a query. We apply our economic evaluation framework to compare the performance of reasoning and non-reasoning models on difficult questions from the MATH benchmark, discovering that reasoning models offer better accuracy-cost tradeoffs as soon as the economic cost of a mistake exceeds \$0.01. In addition, we find that single large LLMs often outperform cascades when the cost of making a mistake is as low as \$0.1. Overall, our findings suggest that when automating meaningful human tasks with AI models, practitioners should typically use the most powerful available model, rather than attempt to minimize AI deployment costs, since deployment costs are likely dwarfed by the economic impact of AI errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03829v1">RELRaE: LLM-Based Relationship Extraction, Labelling, Refinement, and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 18 Pages, 8 Tables, Under-review at ISWC 2025
    </div>
    <details class="paper-abstract">
      A large volume of XML data is produced in experiments carried out by robots in laboratories. In order to support the interoperability of data between labs, there is a motivation to translate the XML data into a knowledge graph. A key stage of this process is the enrichment of the XML schema to lay the foundation of an ontology schema. To achieve this, we present the RELRaE framework, a framework that employs large language models in different stages to extract and accurately label the relationships implicitly present in the XML schema. We investigate the capability of LLMs to accurately generate these labels and then evaluate them. Our work demonstrates that LLMs can be effectively used to support the generation of relationship labels in the context of lab automation, and that they can play a valuable role within semi-automatic ontology generation frameworks more generally.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03722v1">Roadmap for using large language models (LLMs) to accelerate cross-disciplinary research with an example from computational biology</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful artificial intelligence (AI) tools transforming how research is conducted. However, their use in research has been met with skepticism, due to concerns about hallucinations, biases and potential harms to research. These emphasize the importance of clearly understanding the strengths and weaknesses of LLMs to ensure their effective and responsible use. Here, we present a roadmap for integrating LLMs into cross-disciplinary research, where effective communication, knowledge transfer and collaboration across diverse fields are essential but often challenging. We examine the capabilities and limitations of LLMs and provide a detailed computational biology case study (on modeling HIV rebound dynamics) demonstrating how iterative interactions with an LLM (ChatGPT) can facilitate interdisciplinary collaboration and research. We argue that LLMs are best used as augmentative tools within a human-in-the-loop framework. Looking forward, we envisage that the responsible use of LLMs will enhance innovative cross-disciplinary research and substantially accelerate scientific discoveries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03711v1">Can LLMs Play Ô Ăn Quan Game? A Study of Multi-Step Planning and Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted paper at MAPR 2025
    </div>
    <details class="paper-abstract">
      In this paper, we explore the ability of large language models (LLMs) to plan and make decisions through the lens of the traditional Vietnamese board game, \^O \u{A}n Quan. This game, which involves a series of strategic token movements and captures, offers a unique environment for evaluating the decision-making and strategic capabilities of LLMs. Specifically, we develop various agent personas, ranging from aggressive to defensive, and employ the \^O \u{A}n Quan game as a testbed for assessing LLM performance across different strategies. Through experimentation with models like Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Llama-3.3-70B-Instruct, we aim to understand how these models execute strategic decision-making, plan moves, and manage dynamic game states. The results will offer insights into the strengths and weaknesses of LLMs in terms of reasoning and strategy, contributing to a deeper understanding of their general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03662v1">Re-Emergent Misalignment: How Narrow Fine-Tuning Erodes Safety Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recent work has shown that fine-tuning large language models (LLMs) on code with security vulnerabilities can result in misaligned and unsafe behaviors across broad domains. These results prompted concerns about the emergence of harmful behaviors from narrow domain fine-tuning. In this paper, we contextualize these findings by analyzing how such narrow adaptation impacts the internal mechanisms and behavioral manifestations of LLMs. Through a series of experiments covering output probability distributions, loss and gradient vector geometry, layer-wise activation dynamics, and activation space dimensions, we find that behaviors attributed to "emergent misalignment" may be better interpreted as an erosion of prior alignment. We show that fine tuning on insecure code induces internal changes that oppose alignment. Further, we identify a shared latent dimension in the model's activation space that governs alignment behavior. We show that this space is activated by insecure code and by misaligned responses more generally, revealing how narrow fine-tuning can degrade general safety behavior by interfering with shared internal mechanisms. Our findings offer a mechanistic interpretation for previously observed misalignment phenomena, and highlights the fragility of alignment in LLMs. The results underscore the need for more robust fine-tuning strategies that preserve intended behavior across domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03659v1">Specification-Guided Repair of Arithmetic Errors in Dafny Programs using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Formal verification offers strong assurances of software correctness. However, debugging and repairing the underlying faults can be complex and time-consuming when verification fails. Automated Program Repair (APR) aims to ease this by automatically identifying and fixing faults. Traditional APR techniques often depend on test suites for validation, but these may fail to capture all scenarios. In contrast, formal specifications provide stronger correctness criteria for effective repairs. We present an innovative APR tool for Dafny, a verification-aware programming language that uses formal specifications - including pre-conditions, post-conditions, and invariants - as oracles for fault localization and repair. Assuming the correctness of the specifications and focusing on arithmetic bugs, we localize faults through a series of steps, which include using Hoare Logic to determine the state of each statement within the program and state-of-the-art Large Language Models (LLMs) to synthesize candidate fixes. The chosen models were GPT-4o mini, Llama 3, Mistral 7B, and Llemma 7B. We evaluate our approach using DafnyBench, a benchmark of real-world Dafny programs. Our tool achieves 89.6% accuracy in fault localization, with GPT-4o mini yielding the highest repair success rate (74.18%). These results highlight the potential of combining formal reasoning with LLM-driven program synthesis for automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08276v2">Exploring Robustness of LLMs to Paraphrasing Based on Sociodemographic Factors</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Despite their linguistic prowess, LLMs have been shown to be vulnerable to small input perturbations. While robustness to local adversarial changes has been studied, robustness to global modifications such as different linguistic styles remains underexplored. Therefore, we take a broader approach to explore a wider range of variations across sociodemographic dimensions. We extend the SocialIQA dataset to create diverse paraphrased sets conditioned on sociodemographic factors (age and gender). The assessment aims to provide a deeper understanding of LLMs in (a) their capability of generating demographic paraphrases with engineered prompts and (b) their capabilities in interpreting real-world, complex language scenarios. We also perform a reliability analysis of the generated paraphrases looking into linguistic diversity and perplexity as well as manual evaluation. We find that demographic-based paraphrasing significantly impacts the performance of language models, indicating that the subtleties of linguistic variation remain a significant challenge. We will make the code and dataset available for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08203v2">ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted to LLMSEC Workshop at ACL 2025
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown impressive capabilities in math problem-solving tasks, their robustness to noisy inputs is not well-studied. We propose ArithmAttack to examine how robust the LLMs are when they encounter noisy prompts that contain extra noise in the form of punctuation marks. While being easy to implement, ArithmAttack does not cause any information loss since words are not added or deleted from the context. We evaluate the robustness of eight LLMs, including LLama3, Mistral, Mathstral, and DeepSeek on noisy GSM8K and MultiArith datasets. Our experiments suggest that all the studied models show vulnerability to such noise, with more noise leading to poorer performances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09066v2">Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03619v1">Blackbox Dataset Inference for LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs. In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19937v2">ALAS: Measuring Latent Speech-Text Alignment For Spoken Language Understanding In Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in Spoken Language Understanding (SLU), where effective multimodal learning depends on the alignment between audio and text. Despite various fusion methods, no standard metric exists to assess this alignment. This work introduces ALAS (Automatic Latent Alignment Score), a metric that evaluates alignment by measuring correlations between audio and text representations across transformer layers. Experiments on Spoken Question Answering and Emotion Recognition show that ALAS captures meaningful patterns across tasks and layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03605v1">Behaviour Space Analysis of LLM-driven Meta-heuristic Discovery</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      We investigate the behaviour space of meta-heuristic optimisation algorithms automatically generated by Large Language Model driven algorithm discovery methods. Using the Large Language Evolutionary Algorithm (LLaMEA) framework with a GPT o4-mini LLM, we iteratively evolve black-box optimisation heuristics, evaluated on 10 functions from the BBOB benchmark suite. Six LLaMEA variants, featuring different mutation prompt strategies, are compared and analysed. We log dynamic behavioural metrics including exploration, exploitation, convergence and stagnation measures, for each run, and analyse these via visual projections and network-based representations. Our analysis combines behaviour-based projections, Code Evolution Graphs built from static code features, performance convergence curves, and behaviour-based Search Trajectory Networks. The results reveal clear differences in search dynamics and algorithm structures across LLaMEA configurations. Notably, the variant that employs both a code simplification prompt and a random perturbation prompt in a 1+1 elitist evolution strategy, achieved the best performance, with the highest Area Over the Convergence Curve. Behaviour-space visualisations show that higher-performing algorithms exhibit more intensive exploitation behaviour and faster convergence with less stagnation. Our findings demonstrate how behaviour-space analysis can explain why certain LLM-designed heuristics outperform others and how LLM-driven algorithm discovery navigates the open-ended and complex search space of algorithms. These findings provide insights to guide the future design of adaptive LLM-driven algorithm generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03378v1">Making Sense of Korean Sentences: A Comprehensive Evaluation of LLMs through KoSEnd Dataset</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 ACL 2025 student research workshop
    </div>
    <details class="paper-abstract">
      Although LLMs have made significant progress in various languages, there are still concerns about their effectiveness with low-resource agglutinative languages compared to languages such as English. In this study, we focused on Korean, a language known for its complex sentence endings, and evaluated LLMs on this challenging aspect. We introduce the Korean Sentence Endings (KoSEnd) dataset, which includes 3,000 sentences, each annotated for the naturalness of 15 sentence ending forms. These were collected from diverse sources to cover a range of contexts. We evaluated 11 LLMs to assess their understanding of Korean sentence endings, analyzing them based on parameter count and prediction consistency. Notably, we found that informing models about the possibility of missing sentence endings improved performance, highlighting the impact of explicitly considering certain linguistic features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03336v1">Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03327v1">Read Quietly, Think Aloud: Decoupling Comprehension and Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable proficiency in understanding text and generating high-quality responses. However, a critical distinction from human cognition is their typical lack of a distinct internal `reading' or deliberation phase before `speaking' (i.e., generating text). Humans often engage in silent reading to comprehend context and formulate thoughts prior to articulation. This paper investigates methods to imbue LLMs with a similar capacity for internal processing. We introduce and evaluate techniques that encourage LLMs to `read silently.' Our findings indicate that even a straightforward approach, such as providing the model with an initial contextual prompt or `reading space' before it begins predicting subsequent tokens for the final output, can yield significant performance improvements. We further enhance this concept by developing a `reading buddy' architecture, where an auxiliary component silently processes the input and provides refined contextual insights to the primary generation model. These approaches aim to foster deeper understanding from LLMs so that they can produce better reasoned responses, moving them one step closer to more human-like text processing. Our results indicate that these simple techniques can provide surprisingly strong impact on accuracy with multiple point accuracy boost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03326v1">Mirror in the Model: Ad Banner Image Generation via Reflective Multi-LLM and Multi-modal Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recent generative models such as GPT-4o have shown strong capabilities in producing high-quality images with accurate text rendering. However, commercial design tasks like advertising banners demand more than visual fidelity -- they require structured layouts, precise typography, consistent branding, and more. In this paper, we introduce MIMO (Mirror In-the-Model), an agentic refinement framework for automatic ad banner generation. MIMO combines a hierarchical multi-modal agent system (MIMO-Core) with a coordination loop (MIMO-Loop) that explores multiple stylistic directions and iteratively improves design quality. Requiring only a simple natural language based prompt and logo image as input, MIMO automatically detects and corrects multiple types of errors during generation. Experiments show that MIMO significantly outperforms existing diffusion and LLM-based baselines in real-world banner design scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02881v3">Rewriting Pre-Training Data Boosts LLM Performance in Math and Code</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) in program synthesis and mathematical reasoning is fundamentally limited by the quality of their pre-training corpora. We introduce two openly licensed datasets, released under the Llama 3.3 Community License, that significantly enhance LLM performance by systematically rewriting public data. SwallowCode (approximately 16.1 billion tokens) refines Python snippets from The-Stack-v2 through a novel four-stage pipeline: syntax validation, pylint-based style filtering, and a two-stage LLM rewriting process that enforces style conformity and transforms snippets into self-contained, algorithmically efficient examples. Unlike prior methods that rely on exclusionary filtering or limited transformations, our transform-and-retain approach upgrades low-quality code, maximizing data utility. SwallowMath (approximately 2.3 billion tokens) enhances Finemath-4+ by removing boilerplate, restoring context, and reformatting solutions into concise, step-by-step explanations. Within a fixed 50 billion token training budget, continual pre-training of Llama-3.1-8B with SwallowCode boosts pass@1 by +17.0 on HumanEval and +17.7 on HumanEval+ compared to Stack-Edu, surpassing the baseline model's code generation capabilities. Similarly, substituting SwallowMath yields +12.4 accuracy on GSM8K and +7.6 on MATH. Ablation studies confirm that each pipeline stage contributes incrementally, with rewriting delivering the largest gains. All datasets, prompts, and checkpoints are publicly available, enabling reproducible research and advancing LLM pre-training for specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03305v1">Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 9 pages, 10 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The growing prevalence of Large Language Models (LLMs) and their substantial memory requirements have prompted renewed interest in CPU offloading as a method to compensate for limited GPU memory. In particular, when CPU memory is leveraged to temporarily store intermediate states of LLMs, CPU memory becomes a new bottleneck and soon reaches the capacity limitation of commodity CPUs. In this work, we investigate the effectiveness of Compute Express Link (CXL) add-in card (AIC) memory as an extension to CPU memory, enabling larger model sizes and longer context lengths during fine-tuning. Through extensive benchmarking, this study quantifies the performance overhead introduced by transferring data between CXL memory, CPU, and GPUs, focusing on how concurrency and data volume influence bandwidth utilization and latency. This study also compares CPUbased optimizer steps when model parameters, gradients, and optimizer states reside in local memory versus CXL memory, revealing that naive adoption of CXL often degrades performance during the optimizer phase. To overcome these challenges, this study proposes a CXL-aware allocation to strategically partition CPU offloading workloads across both local and CXL memory. This study further demonstrates that employing multiple AICs significantly reduces bandwidth contention, thus improving scalability. Experimental results show that these optimizations enable efficient long-context LLM fine-tuning, underscoring CXL as a promising avenue for unlocking the full potential of CPU offloading in long-context LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03294v1">MGAA: Multi-Granular Adaptive Allocation fof Low-Rank Compression of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 13 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The enormous parameter scale of large language models (LLMs) has made model compression a research hotspot, which aims to alleviate computational resource demands during deployment and inference. As a promising direction, low-rank approximation technique has made remarkable achievements. Nevertheless, unfortunately, the vast majority of studies to low-rank approximation compression generally apply uniform compression ratios across all weight matrices, while disregarding their inherently differentiated impacts on the model's performance. Although a few recent work attempts to employ heuristic search strategies to achieve the optimal parameter allocation, such strategies are computationally inefficient and lose the generalization ability in the era of LLMs. In this study, we propose a novel parameter Multi-Granular Adaptive Allocation (MGAA) method, which can adaptively allocate parameters between and within sublayers without task-specific evaluations in the compression process. MGAA consists of two components: 1) Among different sublayers, it assigns compression ratios based on their cosine similarity between inputs and outputs, allowing for a more tailored compression in sublayers with varying degrees of importance, and 2) Within each sublayer, it allocates different compression ratios to weight matrices based on their energy distribution characteristics, ensuring a consistent energy retention ratio while optimizing compression efficiency. Comprehensive evaluations of MGAA across multiple LLMs backbone models and benchmark datasets demonstrate its superior performance. Additionally, we apply our MGAA to multimodal model LLaVA, exhibiting remarkable performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03293v1">LTLCrit: A Temporal Logic-based LLM Critic for Safe and Efficient Embodied Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated promise in reasoning tasks and general decision-making in static environments. In long-term planning tasks, however, errors tend to accumulate, often leading to unsafe or inefficient behavior, limiting their use in general-purpose settings. We propose a modular actor-critic architecture in which an LLM actor is guided by LTLCrit, a trajectory-level LLM critic that communicates via linear temporal logic (LTL). Our setup combines the reasoning strengths of language models with the guarantees of formal logic. The actor selects high-level actions from natural language observations, while the critic analyzes full trajectories and proposes new LTL constraints that shield the actor from future unsafe or inefficient behavior. The architecture supports both fixed, hand-specified safety constraints and adaptive, learned soft constraints that promote long-term efficiency. Our architecture is model-agnostic: any LLM-based planner can serve as the actor, and LTLCrit serves as a logic-generating wrapper. We formalize planning as graph traversal under symbolic constraints, allowing LTLCrit to analyze failed or suboptimal trajectories and generate new temporal logic rules that improve future behavior. We evaluate our system on the Minecraft diamond-mining benchmark, achieving 100% completion rates and improving efficiency compared to baseline LLM planners. Our results suggest that enabling LLMs to supervise each other through logic is a powerful and flexible paradigm for safe, generalizable decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04428v2">MoralBench: Moral Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
      | 💬 Accepted to ACM SIGKDD Explorations Volume 27 Issue 1
    </div>
    <details class="paper-abstract">
      In the rapidly evolving field of artificial intelligence, large language models (LLMs) have emerged as powerful tools for a myriad of applications, from natural language processing to decision-making support systems. However, as these models become increasingly integrated into societal frameworks, the imperative to ensure they operate within ethical and moral boundaries has never been more critical. This paper introduces a novel benchmark designed to measure and compare the moral reasoning capabilities of LLMs. We present the first comprehensive dataset specifically curated to probe the moral dimensions of LLM outputs, addressing a wide range of ethical dilemmas and scenarios reflective of real-world complexities. The main contribution of this work lies in the development of benchmark datasets and metrics for assessing the moral identity of LLMs, which accounts for nuance, contextual sensitivity, and alignment with human ethical standards. Our methodology involves a multi-faceted approach, combining quantitative analysis with qualitative insights from ethics scholars to ensure a thorough evaluation of model performance. By applying our benchmark across several leading LLMs, we uncover significant variations in moral reasoning capabilities of different models. These findings highlight the importance of considering moral reasoning in the development and evaluation of LLMs, as well as the need for ongoing research to address the biases and limitations uncovered in our study. We publicly release the benchmark at https://drive.google.com/drive/u/0/folders/1k93YZJserYc2CkqP8d4B3M3sgd3kA8W7 and also open-source the code of the project at https://github.com/agiresearch/MoralBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03254v1">CodeAgents: A Token-Efficient Framework for Codified Multi-Agent Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Effective prompt design is essential for improving the planning capabilities of large language model (LLM)-driven agents. However, existing structured prompting strategies are typically limited to single-agent, plan-only settings, and often evaluate performance solely based on task accuracy - overlooking critical factors such as token efficiency, modularity, and scalability in multi-agent environments. To address these limitations, we introduce CodeAgents, a prompting framework that codifies multi-agent reasoning and enables structured, token-efficient planning in multi-agent systems. In CodeAgents, all components of agent interaction - Task, Plan, Feedback, system roles, and external tool invocations - are codified into modular pseudocode enriched with control structures (e.g., loops, conditionals), boolean logic, and typed variables. This design transforms loosely connected agent plans into cohesive, interpretable, and verifiable multi-agent reasoning programs. We evaluate the proposed framework across three diverse benchmarks - GAIA, HotpotQA, and VirtualHome - using a range of representative LLMs. Results show consistent improvements in planning performance, with absolute gains of 3-36 percentage points over natural language prompting baselines. On VirtualHome, our method achieves a new state-of-the-art success rate of 56%. In addition, our approach reduces input and output token usage by 55-87% and 41-70%, respectively, underscoring the importance of token-aware evaluation metrics in the development of scalable multi-agent LLM systems. The code and resources are available at: https://anonymous.4open.science/r/CodifyingAgent-5A86
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10401v2">HPCTransCompile: An AI Compiler Generated Dataset for High-Performance CUDA Transpilation and LLM Preliminary Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      The rapid growth of deep learning has driven exponential increases in model parameters and computational demands. NVIDIA GPUs and their CUDA-based software ecosystem provide robust support for parallel computing, significantly alleviating computational bottlenecks. Meanwhile, due to the cultivation of user programming habits and the high performance of GPUs, the CUDA ecosystem has established a dominant position in the field of parallel software. This dominance requires other hardware platforms to support CUDA-based software with performance portability. However, translating CUDA code to other platforms poses significant challenges due to differences in parallel programming paradigms and hardware architectures. Existing approaches rely on language extensions, domain-specific languages (DSLs), or compilers but face limitations in workload coverage and generalizability. Moreover, these methods often incur substantial development costs. Recently, LLMs have demonstrated extraordinary potential in various vertical domains, especially in code-related tasks. However, the performance of existing LLMs in CUDA transpilation, particularly for high-performance code, remains suboptimal. To address these challenges, we propose a novel framework for generating high-performance CUDA and corresponding platform code pairs, leveraging AI compiler and automatic optimization technology. We further enhance the framework with a graph-based data augmentation method and introduce HPCTransEval, a benchmark for evaluating LLM performance on CUDA transpilation. We conduct experiments using CUDA-to-CPU transpilation as a case study on leading LLMs. The speedup ratio of the CPU operators has an average improvemnet of 43.8\%, highlighting the potential of LLMs to address compatibility challenges within the CUDA ecosystem. Our code is available at https://github.com/PJLAB-CHIP/HPCTransCompile.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06845v6">7B Fully Open Source Moxin-LLM/VLM -- From Pretraining to GRPO-based Reinforcement Learning Enhancement</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have undergone a significant transformation, marked by a rapid rise in both their popularity and capabilities. Leading this evolution are proprietary LLMs like GPT-4 and GPT-o1, which have captured widespread attention in the AI community due to their remarkable performance and versatility. Simultaneously, open-source LLMs, such as LLaMA, have made great contributions to the ever-increasing popularity of LLMs due to the ease to customize and deploy the models across diverse applications. Although open-source LLMs present unprecedented opportunities for innovation and research, the commercialization of LLMs has raised concerns about transparency, reproducibility, and safety. Many open-source LLMs fail to meet fundamental transparency requirements by withholding essential components like training code and data, which may hinder further innovations on LLMs. To mitigate this issue, we introduce Moxin 7B, a fully open-source LLM developed, adhering to principles of open science, open source, open data, and open access. We release the pre-training code and configurations, training and fine-tuning datasets, and intermediate and final checkpoints, aiming to make continuous commitments to fully open-source LLMs. After pre-training the base model, we finetune the Moxin Base model with SOTA post-training framework and instruction data to obtain Moxin Instruct model. To improve the reasoning capability, we further finetune our Instruct model with chain-of-thought data distilled from DeepSeek R1, and then use Group Relative Policy Optimization (GRPO) following DeepSeek R1 to finetune our model, leading to the Moxin Reasoning model. Moreover, we develop our vision language model based on our Moxin model. Experiments show that our models achieve superior performance in various evaluations such as zero-shot evaluation, few-shot evaluation, and CoT evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05282v1">Exploring LLM Capabilities in Extracting DCAT-Compatible Metadata for Data Cataloging</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      Efficient data exploration is crucial as data becomes increasingly important for accelerating processes, improving forecasts and developing new business models. Data consumers often spend 25-98 % of their time searching for suitable data due to the exponential growth, heterogeneity and distribution of data. Data catalogs can support and accelerate data exploration by using metadata to answer user queries. However, as metadata creation and maintenance is often a manual process, it is time-consuming and requires expertise. This study investigates whether LLMs can automate metadata maintenance of text-based data and generate high-quality DCAT-compatible metadata. We tested zero-shot and few-shot prompting strategies with LLMs from different vendors for generating metadata such as titles and keywords, along with a fine-tuned model for classification. Our results show that LLMs can generate metadata comparable to human-created content, particularly on tasks that require advanced semantic understanding. Larger models outperformed smaller ones, and fine-tuning significantly improves classification accuracy, while few-shot prompting yields better results in most cases. Although LLMs offer a faster and reliable way to create metadata, a successful application requires careful consideration of task-specific criteria and domain context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05279v1">ReservoirChat: Interactive Documentation Enhanced with LLM and Knowledge Graph for ReservoirPy</a></div>
    <div class="paper-meta">
      📅 2025-07-04
    </div>
    <details class="paper-abstract">
      We introduce a tool designed to improve the capabilities of Large Language Models (LLMs) in assisting with code development using the ReservoirPy library, as well as in answering complex questions in the field of Reservoir Computing. By incorporating external knowledge through Retrieval-Augmented Generation (RAG) and knowledge graphs, our approach aims to reduce hallucinations and increase the factual accuracy of generated responses. The system provides an interactive experience similar to ChatGPT, tailored specifically for ReservoirPy, enabling users to write, debug, and understand Python code while accessing reliable domain-specific insights. In our evaluation, while proprietary models such as ChatGPT-4o and NotebookLM performed slightly better on general knowledge questions, our model outperformed them on coding tasks and showed a significant improvement over its base model, Codestral-22B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02694v1">Can LLMs Identify Critical Limitations within Scientific Research? A Systematic Evaluation on AI Research Papers</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Peer review is fundamental to scientific research, but the growing volume of publications has intensified the challenges of this expertise-intensive process. While LLMs show promise in various scientific tasks, their potential to assist with peer review, particularly in identifying paper limitations, remains understudied. We first present a comprehensive taxonomy of limitation types in scientific research, with a focus on AI. Guided by this taxonomy, for studying limitations, we present LimitGen, the first comprehensive benchmark for evaluating LLMs' capability to support early-stage feedback and complement human peer review. Our benchmark consists of two subsets: LimitGen-Syn, a synthetic dataset carefully created through controlled perturbations of high-quality papers, and LimitGen-Human, a collection of real human-written limitations. To improve the ability of LLM systems to identify limitations, we augment them with literature retrieval, which is essential for grounding identifying limitations in prior scientific findings. Our approach enhances the capabilities of LLM systems to generate limitations in research papers, enabling them to provide more concrete and constructive feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02620v1">FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 16 pages, and the last 3 are appendix
    </div>
    <details class="paper-abstract">
      Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure that the LLMs can fit into the device memory. Recent pipeline-based approaches have the potential to parallelize communication and computation, which helps reduce inference latency. However, the benefit diminishes when the inference request at the network edge is sparse, where pipeline is typically at low utilization. To enable efficient distributed LLM inference at the edge, we propose \textbf{FlowSpec}, a pipeline-parallel tree-based speculative decoding framework. FlowSpec incorporates three key mechanisms to improve decoding efficiency: 1) score-based step-wise verification prioritizes more important draft tokens to bring earlier accpeted tokens; 2) efficient draft management to prune invalid tokens while maintaining correct causal relationship during verification; 3) dynamic draft expansion strategies to supply high-quality speculative inputs. These techniques work in concert to enhance both pipeline utilization and speculative efficiency. We evaluate FlowSpec on a real-world testbed with other baselines. Experimental results demonstrate that our proposed framework significantly improves inference speed across diverse models and configurations, achieving speedup ratios 1.36$\times$-1.77$\times$ compared to baselines. Our code is publicly available at \href{https://github.com/Leosang-lx/FlowSpec#}{https://github.com/Leosang-lx/FlowSpec\#}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01334v2">Symbolic or Numerical? Understanding Physics Problem Solving in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Navigating the complexities of physics reasoning has long been a difficult task for Large Language Models (LLMs), requiring a synthesis of profound conceptual understanding and adept problem-solving techniques. In this study, we investigate the application of advanced instruction-tuned reasoning models, such as Deepseek-R1, to address a diverse spectrum of physics problems curated from the challenging SciBench benchmark. Our comprehensive experimental evaluation reveals the remarkable capabilities of reasoning models. Not only do they achieve state-of-the-art accuracy in answering intricate physics questions, but they also generate distinctive reasoning patterns that emphasize on symbolic derivation. Furthermore, our findings indicate that even for these highly sophisticated reasoning models, the strategic incorporation of few-shot prompting can still yield measurable improvements in overall accuracy, highlighting the potential for continued performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12532v3">MedAide: Information Fusion and Anatomy of Medical Intents via LLM-based Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 LLM-based Multi-Agent Collaboration for Medical Applications
    </div>
    <details class="paper-abstract">
      In healthcare intelligence, the ability to fuse heterogeneous, multi-intent information from diverse clinical sources is fundamental to building reliable decision-making systems. Large Language Model (LLM)-driven information interaction systems currently showing potential promise in the healthcare domain. Nevertheless, they often suffer from information redundancy and coupling when dealing with complex medical intents, leading to severe hallucinations and performance bottlenecks. To this end, we propose MedAide, an LLM-based medical multi-agent collaboration framework designed to enable intent-aware information fusion and coordinated reasoning across specialized healthcare domains. Specifically, we introduce a regularization-guided module that combines syntactic constraints with retrieval augmented generation to decompose complex queries into structured representations, facilitating fine-grained clinical information fusion and intent resolution. Additionally, a dynamic intent prototype matching module is proposed to utilize dynamic prototype representation with a semantic similarity matching mechanism to achieve adaptive recognition and updating of the agent's intent in multi-round healthcare dialogues. Ultimately, we design a rotation agent collaboration mechanism that introduces dynamic role rotation and decision-level information fusion across specialized medical agents. Extensive experiments are conducted on four medical benchmarks with composite intents. Experimental results from automated metrics and expert doctor evaluations show that MedAide outperforms current LLMs and improves their medical proficiency and strategic reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08500v2">Aerial Vision-and-Language Navigation via Semantic-Topo-Metric Representation Guided LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Aerial Vision-and-Language Navigation (VLN) is a novel task enabling Unmanned Aerial Vehicles (UAVs) to navigate in outdoor environments through natural language instructions and visual cues. It remains challenging due to the complex spatial relationships in outdoor aerial scenes. In this paper, we propose an end-to-end zero-shot framework for aerial VLN tasks, where the large language model (LLM) is introduced as our agent for action prediction. Specifically, we develop a novel Semantic-Topo-Metric Representation (STMR) to enhance the spatial reasoning ability of LLMs. This is achieved by extracting and projecting instruction-related semantic masks of landmarks into a top-down map that contains the location information of surrounding landmarks. Further, this map is transformed into a matrix representation with distance metrics as the text prompt to the LLM, for action prediction according to the instruction. Experiments conducted in real and simulation environments have successfully proved the effectiveness and robustness of our method, achieving 15.9% and 12.5% improvements (absolute) in Oracle Success Rate (OSR) on AerialVLN-S dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02503v1">Continual Gradient Low-Rank Projection Fine-Tuning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 15 pages, 6 figures, accepted by ACL 2025 main
    </div>
    <details class="paper-abstract">
      Continual fine-tuning of Large Language Models (LLMs) is hampered by the trade-off between efficiency and expressiveness. Low-Rank Adaptation (LoRA) offers efficiency but constrains the model's ability to learn new tasks and transfer knowledge due to its low-rank nature and reliance on explicit parameter constraints. We propose GORP (Gradient LOw Rank Projection) for Continual Learning, a novel training strategy that overcomes these limitations by synergistically combining full and low-rank parameters and jointly updating within a unified low-rank gradient subspace. GORP expands the optimization space while preserving efficiency and mitigating catastrophic forgetting. Extensive experiments on continual learning benchmarks demonstrate GORP's superior performance compared to existing state-of-the-art approaches. Code is available at https://github.com/Wcxwcxw/GORP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.07016v5">Delving into LLM-assisted writing in biomedical publications through excess vocabulary</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 v5: Reverting to v3
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) like ChatGPT can generate and revise text with human-level performance. These models come with clear limitations: they can produce inaccurate information, reinforce existing biases, and be easily misused. Yet, many scientists use them for their scholarly writing. But how wide-spread is such LLM usage in the academic literature? To answer this question for the field of biomedical research, we present an unbiased, large-scale approach: we study vocabulary changes in over 15 million biomedical abstracts from 2010--2024 indexed by PubMed, and show how the appearance of LLMs led to an abrupt increase in the frequency of certain style words. This excess word analysis suggests that at least 13.5% of 2024 abstracts were processed with LLMs. This lower bound differed across disciplines, countries, and journals, reaching 40% for some subcorpora. We show that LLMs have had an unprecedented impact on scientific writing in biomedical research, surpassing the effect of major world events such as the Covid pandemic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07967v2">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Maintenance</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 A vision paper that will be continuously updated
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated promise in software engineering tasks like code completion and generation, their support for the maintenance of complex software systems remains limited. These models often struggle with understanding the tacit knowledge embedded in systems, such as responsibility allocation and collaboration across different modules. To address this gap, we introduce the concept and framework of \textbf{Code Digital Twin}, a conceptual representation of tacit knowledge that captures the concepts, functionalities, and design rationales behind code elements, co-evolving with the software. A code digital twin is constructed using a methodology that combines knowledge extraction from both structured and unstructured sources--such as source code, documentation, and change histories--leveraging LLMs, static analysis tools, and human expertise. This framework can empower LLMs for software maintenance tasks such as issue localization and repository-level code generation by providing tacit knowledge as contexts. Based on the proposed methodology, we explore the key challenges and opportunities involved in the continuous construction and refinement of code digital twin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02380v1">JoyTTS: LLM-based Spoken Chatbot With Voice Cloning</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      JoyTTS is an end-to-end spoken chatbot that combines large language models (LLM) with text-to-speech (TTS) technology, featuring voice cloning capabilities. This project is built upon the open-source MiniCPM-o and CosyVoice2 models and trained on 2000 hours of conversational data. We have also provided the complete training code to facilitate further development and optimization by the community. On the testing machine seed-tts-zh, it achieves a SS (speaker similarity) score of 0.73 and a WER (Word Error Rate) of 5.09. The code and models, along with training and inference scripts, are available at https://github.com/jdh-algo/JoyTTS.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02378v1">Efficient Code LLM Training via Distribution-Consistent and Diversity-Aware Data Selection</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have significantly improved code generation and program comprehension, accelerating the evolution of software engineering. Current methods primarily enhance model performance by leveraging vast amounts of data, focusing on data quantity while often overlooking data quality, thereby reducing training efficiency. To address this, we introduce an approach that utilizes a parametric model for code data selection, aimed at improving both training efficiency and model performance. Our method optimizes the parametric model to ensure distribution consistency and diversity within the selected subset, guaranteeing high-quality data. Experimental results demonstrate that using only 10K samples, our method achieves gains of 2.4% (HumanEval) and 2.3% (MBPP) over 92K full-sampled baseline, outperforming other sampling approaches in both performance and efficiency. This underscores that our method effectively boosts model performance while significantly reducing computational costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21880v2">Incorporating LLMs for Large-Scale Urban Complex Mobility Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 8 pages, 8 figures. This paper is reviewed and accepted by the CUPUM (Computational Urban Planning and Urban Management) Conference held by University College London (UCL) in 2025
    </div>
    <details class="paper-abstract">
      This study presents an innovative approach to urban mobility simulation by integrating a Large Language Model (LLM) with Agent-Based Modeling (ABM). Unlike traditional rule-based ABM, the proposed framework leverages LLM to enhance agent diversity and realism by generating synthetic population profiles, allocating routine and occasional locations, and simulating personalized routes. Using real-world data, the simulation models individual behaviors and large-scale mobility patterns in Taipei City. Key insights, such as route heat maps and mode-specific indicators, provide urban planners with actionable information for policy-making. Future work focuses on establishing robust validation frameworks to ensure accuracy and reliability in urban planning applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02353v1">OMS: On-the-fly, Multi-Objective, Self-Reflective Ad Keyword Generation via LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Keyword decision in Sponsored Search Advertising is critical to the success of ad campaigns. While LLM-based methods offer automated keyword generation, they face three major limitations: reliance on large-scale query-keyword pair data, lack of online multi-objective performance monitoring and optimization, and weak quality control in keyword selection. These issues hinder the agentic use of LLMs in fully automating keyword decisions by monitoring and reasoning over key performance indicators such as impressions, clicks, conversions, and CTA effectiveness. To overcome these challenges, we propose OMS, a keyword generation framework that is On-the-fly (requires no training data, monitors online performance, and adapts accordingly), Multi-objective (employs agentic reasoning to optimize keywords based on multiple performance metrics), and Self-reflective (agentically evaluates keyword quality). Experiments on benchmarks and real-world ad campaigns show that OMS outperforms existing methods; ablation and human evaluations confirm the effectiveness of each component and the quality of generated keywords.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.11556v2">Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Accept to ACL 2025 (Oral). Code are available on https://github.com/fuyuchenIfyw/token_prepending.git
    </div>
    <details class="paper-abstract">
      Extracting sentence embeddings from large language models (LLMs) is a promising direction, as LLMs have demonstrated stronger semantic understanding capabilities. Previous studies typically focus on prompt engineering to elicit sentence embeddings from LLMs by prompting the model to encode sentence information into the embedding of the last token. However, LLMs are mostly decoder-only models with causal attention and the earlier tokens in the sentence cannot attend to the latter tokens, resulting in biased encoding of sentence information and cascading effects on the final decoded token. To this end, we propose a novel Token Prepending (TP) technique that prepends each layer's decoded sentence embedding to the beginning of the sentence in the next layer's input, allowing earlier tokens to attend to the complete sentence information under the causal attention mechanism. The proposed TP technique is a plug-and-play and training-free technique, which means it can be seamlessly integrated with various prompt-based sentence embedding methods and autoregressive LLMs. Extensive experiments on various Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our proposed TP technique can significantly improve the performance of existing prompt-based sentence embedding methods across different LLMs, while incurring negligible additional inference cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02332v1">PII Jailbreaking in LLMs via Activation Steering Reveals Personal Information Leakage</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      This paper investigates privacy jailbreaking in LLMs via steering, focusing on whether manipulating activations can bypass LLM alignment and alter response behaviors to privacy related queries (e.g., a certain public figure's sexual orientation). We begin by identifying attention heads predictive of refusal behavior for private attributes (e.g., sexual orientation) using lightweight linear probes trained with privacy evaluator labels. Next, we steer the activations of a small subset of these attention heads guided by the trained probes to induce the model to generate non-refusal responses. Our experiments show that these steered responses often disclose sensitive attribute details, along with other private information about data subjects such as life events, relationships, and personal histories that the models would typically refuse to produce. Evaluations across four LLMs reveal jailbreaking disclosure rates of at least 95%, with more than 50% on average of these responses revealing true personal information. Our controlled study demonstrates that private information memorized in LLMs can be extracted through targeted manipulation of internal activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17828v2">Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22618v3">Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to \textbf{27.6$\times$ throughput} improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02253v1">Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 20 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Progress in enhancing large language model (LLM) planning and reasoning capabilities is significantly hampered by the bottleneck of scalable, reliable data generation and evaluation. To overcome this, I introduce NL2FLOW, a fully automated system for parametrically generating planning problems - expressed in natural language, a structured intermediate representation, and formal PDDL - and rigorously evaluating the quality of generated plans. I demonstrate NL2FLOW's capabilities by generating a dataset of 2296 problems in the automated workflow generation domain and evaluating multiple open-sourced, instruct-tuned LLMs. My results reveal that the highest performing models achieved 86% success in generating valid plans and 69% in generating optimal plans, specifically for problems with feasible solutions. Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. Notably, I observed that the highest success rate for translating natural language into a JSON representation of a plan was lower than the highest rate of generating a valid plan directly. This suggests that unnecessarily decomposing the reasoning task - introducing intermediate translation steps - may actually degrade performance, implying a benefit to models capable of reasoning directly from natural language to action. As I scale LLM reasoning to increasingly complex problems, the bottlenecks and sources of error within these systems will inevitably shift. Therefore, a dynamic understanding of these limitations - and the tools to systematically reveal them - will be crucial for unlocking the full potential of LLMs as intelligent problem solvers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02226v1">DecoRTL: A Run-time Decoding Framework for RTL Code Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Accepted to the International Conference on Computer-Aided Design (ICCAD 2025)
    </div>
    <details class="paper-abstract">
      As one of their many applications, large language models (LLMs) have recently shown promise in automating register transfer level (RTL) code generation. However, conventional LLM decoding strategies, originally designed for natural language, often fail to meet the structural and semantic demands of RTL, leading to hallucinated, repetitive, or invalid code outputs. In this paper, we first investigate the root causes of these decoding failures through an empirical analysis of token-level entropy during RTL generation. Our findings reveal that LLMs exhibit low confidence in regions of structural ambiguity or semantic complexity, showing that standard decoding strategies fail to differentiate between regions requiring determinism (syntax-critical regions) and those that benefit from creative exploratory variability (design-critical regions). Then, to overcome this, we introduce DecoRTL, a novel run-time decoding strategy, that is both syntax-aware and contrastive for RTL code generation. DecoRTL integrates two complementary components: (i) self-consistency sampling, which generates multiple candidates and re-ranks them based on token-level agreement to promote correctness while maintaining diversity; and (ii) syntax-aware temperature adaptation, which classifies tokens by their syntactical and functional roles and adjusts the sampling temperature accordingly, enforcing low temperature for syntax-critical tokens and higher temperature for exploratory ones. Our approach operates entirely at inference time without requiring any additional model fine-tuning. Through evaluations on multiple open-source LLMs using the VerilogEval benchmark, we demonstrate significant improvements in syntactic validity, functional correctness, and output diversity, while the execution overhead (performance overhead) is imperceptible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21551v2">Where to find Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Grokking, i.e., test performance keeps improving long after training loss converged, has been recently witnessed in neural network training, making the mechanism of generalization and other emerging capabilities such as reasoning mysterious. While prior studies usually train small models on a few toy or highly-specific tasks for thousands of epochs, we conduct the first study of grokking on checkpoints during one-pass pretraining of a 7B large language model (LLM), i.e., OLMoE. We compute the training loss and evaluate generalization on diverse benchmark tasks, including math reasoning, code generation, and commonsense/domain-specific knowledge retrieval tasks. Our study, for the first time, verifies that grokking still happens in the pretraining of large-scale foundation models, though different data may enter grokking stages asynchronously. We further demystify grokking's "emergence of generalization" by investigating LLM internal dynamics. Specifically, we find that training samples' pathways (i.e., expert choices across layers) evolve from random, instance-specific to more structured and shareable between samples. Also, the complexity of a sample's pathway reduces despite the converged loss. These indicate a memorization-to-generalization "knowledge digestion", providing a mechanistic explanation of delayed generalization. In the study, we develop two novel metrics to quantify pathway distance and the complexity of a single pathway. We show their ability to predict the generalization improvement on diverse downstream tasks. They are efficient, simple to compute and solely dependent on training data. Hence, they have practical value for pretraining, enabling us to monitor the generalization performance without finetuning and test. Theoretically, we show that more structured pathways reduce model complexity and improve the generalization bound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22694v2">VOCABTRIM: Vocabulary Pruning for Efficient Speculative Decoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 8 pages, 4 figures, 5 tables, accepted at ICML 2025 workshop on Efficient Systems for Foundational Models
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a simple training-free technique to improve the performance of drafter-based speculative decoding (SpD) methods that incorporates language modeling head (LM head) during drafting process. A drafter-based speculative decoding leverages one or more smaller language models, a.k.a. drafters or draft models, to sample a draft sequence or tree consisting of multiple tokens, followed by verification by a base LLM, a target model, accepting a subset as its valid generation. As it is usually considered that the speculative decoding requires one-to-one mapping between vocabularies of the target model and the draft model, it has been natural to share the vocabulary between them, or even share the LM head as in EAGLE or Medusa. We first identify that this draft token sampling scheme inherently contains an unnecessary inference overhead in drafting, especially for some target LLMs with very large vocabularies. Then, we propose a simple technique, VocabTrim, to mitigate the drafting overhead to improve the generation speed in memory-bound environment. VocabTrim reconstructs the drafter LM head to contain only a limited set of tokens, selected by the most frequently sampled from the vocabulary of the target model. While limiting the vocabulary in drafting slightly degrades the acceptance rate, it significantly reduces the drafting latency in memory-bound process which is often the case on edge devices, resulting in higher memory-bound speed up (MBSU). We show that our method can boost the memory-bound speed-up for Llama-3 models on Spec-Bench, specifically by 16% for Llama-3.2-3B-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03211v1">DistZO2: High-Throughput and Memory-Efficient Zeroth-Order Fine-tuning LLMs with Distributed Parallel Computing</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) remains resource-intensive due to their sheer scale. While zeroth-order (ZO) optimization provides a memory-efficient alternative by eliminating backward passes, its application to multi-hundred-billion-parameter models is constrained by GPU memory and compute throughput. The ZO2 framework addresses the memory bottleneck by offloading model parameters to CPU memory and overlapping transformer block transfer with dual forward computation on a single GPU. However, ZO2 remains limited by its single-device execution and achieves modest throughput. In this work, we present DistZO2, a high-throughput, memory-efficient framework for distributed zeroth-order fine-tuning of LLMs. DistZO2 introduces three parallel strategies: (1) Perturbation Parallelism (PertP), which parallelizes the two perturbed forward passes across devices; (2) Distributed Data Parallelism (DDP), adapted to the scalar-gradient nature of ZO training; and (3) a unified 2D Parallelism design that combines PertP and DDP. To further mitigate communication bottlenecks introduced by parameter offloading, we propose a hardware-aware communication strategy that slices parameter blocks and redistributes them across GPUs via high-speed interconnects such as NVLink. DistZO2 scales zeroth-order fine-tuning to modern multi-GPU systems, preserving ZO2's memory efficiency while substantially improving training throughput. In our experiments on OPT-175B, DistZO2 achieves a 3x speedup over ZO2 with distributed computing. DistZO2's code has been open-sourced in https://github.com/liangyuwang/zo2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03194v1">How Much Content Do LLMs Generate That Induces Cognitive Bias in Users?</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 17 pages, 2 figures. to be submitted to AACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into applications ranging from review summarization to medical diagnosis support, where they affect human decisions. Even though LLMs perform well in many tasks, they may also inherit societal or cognitive biases, which can inadvertently transfer to humans. We investigate when and how LLMs expose users to biased content and quantify its severity. Specifically, we assess three LLM families in summarization and news fact-checking tasks, evaluating how much LLMs stay consistent with their context and/or hallucinate. Our findings show that LLMs expose users to content that changes the sentiment of the context in 21.86% of the cases, hallucinates on post-knowledge-cutoff data questions in 57.33% of the cases, and primacy bias in 5.94% of the cases. We evaluate 18 distinct mitigation methods across three LLM families and find that targeted interventions can be effective. Given the prevalent use of LLMs in high-stakes domains, such as healthcare or legal analysis, our results highlight the need for robust technical safeguards and for developing user-centered interventions that address LLM limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18382v2">One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 31 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Pre-trained Large Language Models (LLMs) have shown promise in solving planning problems but often struggle to ensure plan correctness, especially for long-horizon tasks. Meanwhile, traditional robotic task and motion planning (TAMP) frameworks address these challenges more reliably by combining high-level symbolic search with low-level motion planning. At the core of TAMP is the planning domain, an abstract world representation defined through symbolic predicates and actions. However, creating these domains typically involves substantial manual effort and domain expertise, limiting generalizability. We introduce Planning Domain Derivation with LLMs (PDDLLM), a novel approach that combines simulated physical interaction with LLM reasoning to improve planning performance. The method reduces reliance on humans by inferring planning domains from a single annotated task-execution demonstration. Unlike prior domain-inference methods that rely on partially predefined or language descriptions of planning domains, PDDLLM constructs domains entirely from scratch and automatically integrates them with low-level motion planning skills, enabling fully automated long-horizon planning. PDDLLM is evaluated on over 1,200 diverse tasks spanning nine environments and benchmarked against six LLM-based planning baselines, demonstrating superior long-horizon planning performance, lower token costs, and successful deployment on multiple physical robot platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03162v1">MateInfoUB: A Real-World Benchmark for Testing LLMs in Competitive, Multilingual, and Multimodal Educational Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 14 pages (9 paper, 2 references, 3 annexes). Accepted for BEA 2025!
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has transformed various domains, particularly computer science (CS) education. These models exhibit remarkable capabilities in code-related tasks and problem-solving, raising questions about their potential and limitations in advanced CS contexts. This study presents a novel bilingual (English-Romanian) multimodal (text and image) dataset of multiple-choice questions derived from a high-level computer science competition. A particularity of our dataset is that the problems are conceived such that some of them are easier solved using reasoning on paper, while for others writing code is more efficient. We systematically evaluate State of The Art LLMs on this dataset, analyzing their performance on theoretical programming tasks. Our findings reveal the strengths and limitations of current LLMs, including the influence of language choice (English vs. Romanian), providing insights into their applicability in CS education and competition settings. We also address critical ethical considerations surrounding educational integrity and the fairness of assessments in the context of LLM usage. These discussions aim to inform future educational practices and policies. To support further research, our dataset will be made publicly available in both English and Romanian. Additionally, we release an educational application tailored for Romanian students, enabling them to self-assess using the dataset in an interactive and practice-oriented environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03156v1">The Impact of LLM-Assistants on Software Developer Productivity: A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 37 pages
    </div>
    <details class="paper-abstract">
      Large language model assistants (LLM-assistants) present new opportunities to transform software development. Developers are increasingly adopting these tools across tasks, including coding, testing, debugging, documentation, and design. Yet, despite growing interest, there is no synthesis of how LLM-assistants affect software developer productivity. In this paper, we present a systematic literature review of 37 peer-reviewed studies published between January 2014 and December 2024 that examine this impact. Our analysis reveals that LLM-assistants offer both considerable benefits and critical risks. Commonly reported gains include minimized code search, accelerated development, and the automation of trivial and repetitive tasks. However, studies also highlight concerns around cognitive offloading, reduced team collaboration, and inconsistent effects on code quality. While the majority of studies (92%) adopt a multi-dimensional perspective by examining at least two SPACE dimensions, reflecting increased awareness of the complexity of developer productivity, only 14% extend beyond three dimensions, indicating substantial room for more integrated evaluations. Satisfaction, Performance, and Efficiency are the most frequently investigated dimensions, whereas Communication and Activity remain underexplored. Most studies are exploratory (64%) and methodologically diverse, but lack longitudinal and team-based evaluations. This review surfaces key research gaps and provides recommendations for future research and practice. All artifacts associated with this study are publicly available at https://zenodo.org/records/15788502.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03153v1">HGCA: Hybrid GPU-CPU Attention for Long Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Scaling inference for large language models (LLMs) is increasingly constrained by limited GPU memory, especially due to growing key-value (KV) caches required for long-context generation. While existing approaches offload KV caches to CPU memory or apply sparse attention to reduce GPU load, they often underutilize CPU compute resources and compromise accuracy. We present HGCA, a hybrid CPU-GPU attention mechanism that enables scalable, high-throughput LLM inference with near-full attention quality. HGCA performs dense attention on recently generated KV entries retained in GPU memory and parallel sparse attention on selected, salient KV entries in CPU memory. The attention outputs are efficiently merged using log-sum-exp fusion, minimizing PCIe transfer overhead. HGCA also introduces a finegrained, per-head sparsification strategy optimized for CPU execution, preserving contextual relevance while reducing computation. Our implementation seamlessly integrates into existing LLM frameworks without requiring model retraining. Experiments across diverse models and workloads show that HGCA achieves superior scalability, supports longer sequences and larger batch sizes, and outperforms existing sparse attention baselines in both performance and accuracy -- all on commodity GPU hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05232v3">LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Jailbreak attacks expose vulnerabilities in safety-aligned LLMs by eliciting harmful outputs through carefully crafted prompts. Existing methods rely on discrete optimization or trained adversarial generators, but are slow, compute-intensive, and often impractical. We argue that these inefficiencies stem from a mischaracterization of the problem. Instead, we frame jailbreaks as inference-time misalignment and introduce LIAR (Leveraging Inference-time misAlignment to jailbReak), a fast, black-box, best-of-$N$ sampling attack requiring no training. LIAR matches state-of-the-art success rates while reducing perplexity by $10\times$ and Time-to-Attack from hours to seconds. We also introduce a theoretical "safety net against jailbreaks" metric to quantify safety alignment strength and derive suboptimality bounds. Our work offers a simple yet effective tool for evaluating LLM robustness and advancing alignment research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03064v1">LLM-Driven Auto Configuration for Transient IoT Device Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Today's Internet of Things (IoT) has evolved from simple sensing and actuation devices to those with embedded processing and intelligent services, enabling rich collaborations between users and their devices. However, enabling such collaboration becomes challenging when transient devices need to interact with host devices in temporarily visited environments. In such cases, fine-grained access control policies are necessary to ensure secure interactions; however, manually implementing them is often impractical for non-expert users. Moreover, at run-time, the system must automatically configure the devices and enforce such fine-grained access control rules. Additionally, the system must address the heterogeneity of devices. In this paper, we present CollabIoT, a system that enables secure and seamless device collaboration in transient IoT environments. CollabIoT employs a Large language Model (LLM)-driven approach to convert users' high-level intents to fine-grained access control policies. To support secure and seamless device collaboration, CollabIoT adopts capability-based access control for authorization and uses lightweight proxies for policy enforcement, providing hardware-independent abstractions. We implement a prototype of CollabIoT's policy generation and auto configuration pipelines and evaluate its efficacy on an IoT testbed and in large-scale emulated environments. We show that our LLM-based policy generation pipeline is able to generate functional and correct policies with 100% accuracy. At runtime, our evaluation shows that our system configures new devices in ~150 ms, and our proxy-based data plane incurs network overheads of up to 2 ms and access control overheads up to 0.3 ms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03052v1">From 2:4 to 8:16 sparsity patterns in LLMs for Outliers and Weights with Variance Correction</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size, efficient compression techniques like quantization and sparsification are critical. While quantization maintains performance with reduced precision, structured sparsity methods, such as N:M sparsification, often fall short due to limited flexibility, and sensitivity to outlier weights. We explore 8:16 semi-structured sparsity, demonstrating its ability to surpass the Performance Threshold-where a compressed model matches the accuracy of its uncompressed or smaller counterpart under equivalent memory constraints. Compared to 2:4 sparsity, 8:16 offers greater flexibility with minimal storage overhead (0.875 vs. 0.75 bits/element). We also apply sparse structured patterns for salient weights, showing that structured sparsity for outliers is competitive with unstructured approaches leading to equivalent or better results. Finally, we demonstrate that simple techniques such as variance correction and SmoothQuant like weight equalization improve sparse models performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03051v1">Improving LLM Reasoning for Vulnerability Detection via Group Relative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Improving and understanding the training dynamics and reasoning of Large Language Models (LLMs) has become essential for their deployment in AI-based security tools, such as software vulnerability detection. In this work, we present an extensive study aimed at advancing recent RL-based finetuning techniques for LLMs in the context of vulnerability detection. We start by highlighting key limitations of commonly adopted LLMs, such as their tendency to over-predict certain types of vulnerabilities while failing to detect others. To address this challenge, we explore the use of Group Relative Policy Optimization (GRPO), a recent policy-gradient method, for guiding LLM behavior through structured, rule-based rewards. We enable its application to the vulnerability detection task by redefining its advantage functions and reward signals using annotations from widely used datasets in the field, including BigVul, DiverseVul, and CleanVul. The proposed methodology enables an extensive set of experiments, addressing multiple research questions regarding the impact of GRPO on generalization, reasoning capabilities, and performance improvements over standard supervised finetuning (SFT). Our findings offer valuable insights into the potential of RL-based training to enhance both the performance and reasoning abilities of LLMs in the context of software vulnerability detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03042v1">Dynamic Long Short-Term Memory Based Memory Storage For Long Horizon LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 7 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Memory storage for Large Language models (LLMs) is becoming an increasingly active area of research, particularly for enabling personalization across long conversations. We propose Pref-LSTM, a dynamic and lightweight framework that combines a BERT-based classifier with a LSTM memory module that generates memory embedding which then is soft-prompt injected into a frozen LLM. We synthetically curate a dataset of preference and non-preference conversation turns to train our BERT-based classifier. Although our LSTM-based memory encoder did not yield strong results, we find that the BERT-based classifier performs reliably in identifying explicit and implicit user preferences. Our research demonstrates the viability of using preference filtering with LSTM gating principals as an efficient path towards scalable user preference modeling, without extensive overhead and fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02851v1">MOTIF: Modular Thinking via Reinforcement Fine-tuning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Recent advancements in the reasoning capabilities of large language models (LLMs) show that employing group relative policy optimization (GRPO) algorithm for reinforcement learning (RL) training allows the models to use more thinking/reasoning tokens for generating better responses. However, LLMs can generate only a finite amount of tokens while maintaining attention to the previously generated tokens. This limit, also known as the context size of an LLM, is a bottleneck in LLM reasoning with arbitrarily large number of tokens. To think beyond the limit of context size, an LLM must employ a modular thinking strategy to reason over multiple rounds. In this work, we propose $\textbf{MOTIF: Modular Thinking via Reinforcement Finetuning}$ -- an RL training method for generating thinking tokens in multiple rounds, effectively allowing the model to think with additional context size. We trained the open-source model Qwen2.5-3B-Instruct on GSM8K dataset via parameter efficient fine-tuning and tested its accuracy on MATH500 and AIME2024 benchmarks. Our experiments show 3.8\% and 3.3\% improvements over vanilla GRPO based training in the respective benchmarks. Furthermore, this improvement was achieved with only 15\% of samples, thus demonstrating sample efficiency of MOTIF. Our code and models are available at https://github.com/purbeshmitra/MOTIF and https://huggingface.co/purbeshmitra/MOTIF, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02778v1">Self-Correction Bench: Revealing and Addressing the Self-Correction Blind Spot in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
      | 💬 31 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have become transformative, they still make mistakes and can explore unproductive reasoning paths. Self-correction is an important capability for a trustworthy LLM, particularly an autoregressive LLM. While LLMs can identify error in user input, they exhibit a systematic 'Self-Correction Blind Spot' - failing to correct identical error in their own outputs. To systematically study this phenomenon, we introduce Self-Correction Bench, a systematic framework to measure this phenomenon through controlled error injection at three complexity levels. Testing 14 models, we find an average 64.5% blind spot rate. We find multiple evidences that this limitation relates to training data composition: human training demonstrations predominantly show error-free responses rather than error-correction sequences, unlike RL-trained models that learn error correction through outcome feedback. Remarkably, simply appending "Wait" reduces blind spots by 89.3%, suggesting that the capability exists but requires activation. Our work highlights a critical limitation in current LLMs and offers potential avenues for improving their reliability and trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02773v1">KERAP: A Knowledge-Enhanced Reasoning Approach for Accurate Zero-shot Diagnosis Prediction Using Multi-agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Medical diagnosis prediction plays a critical role in disease detection and personalized healthcare. While machine learning (ML) models have been widely adopted for this task, their reliance on supervised training limits their ability to generalize to unseen cases, particularly given the high cost of acquiring large, labeled datasets. Large language models (LLMs) have shown promise in leveraging language abilities and biomedical knowledge for diagnosis prediction. However, they often suffer from hallucinations, lack structured medical reasoning, and produce useless outputs. To address these challenges, we propose KERAP, a knowledge graph (KG)-enhanced reasoning approach that improves LLM-based diagnosis prediction through a multi-agent architecture. Our framework consists of a linkage agent for attribute mapping, a retrieval agent for structured knowledge extraction, and a prediction agent that iteratively refines diagnosis predictions. Experimental results demonstrate that KERAP enhances diagnostic reliability efficiently, offering a scalable and interpretable solution for zero-shot medical diagnosis prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02745v1">Who's Sorry Now: User Preferences Among Rote, Empathic, and Explanatory Apologies from LLM Chatbots</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      As chatbots driven by large language models (LLMs) are increasingly deployed in everyday contexts, their ability to recover from errors through effective apologies is critical to maintaining user trust and satisfaction. In a preregistered study with Prolific workers (N=162), we examine user preferences for three types of apologies (rote, explanatory, and empathic) issued in response to three categories of common LLM mistakes (bias, unfounded fabrication, and factual errors). We designed a pairwise experiment in which participants evaluated chatbot responses consisting of an initial error, a subsequent apology, and a resolution. Explanatory apologies were generally preferred, but this varied by context and user. In the bias scenario, empathic apologies were favored for acknowledging emotional impact, while hallucinations, though seen as serious, elicited no clear preference, reflecting user uncertainty. Our findings show the complexity of effective apology in AI systems. We discuss key insights such as personalization and calibration that future systems must navigate to meaningfully repair trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02737v1">Early Signs of Steganographic Capabilities in Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Monitoring Large Language Model (LLM) outputs is crucial for mitigating risks from misuse and misalignment. However, LLMs could evade monitoring through steganography: Encoding hidden information within seemingly benign generations. In this paper, we evaluate the steganography capabilities in frontier LLMs to better understand the risk they pose. We focus on two types of steganography: passing encoded messages and performing encoded reasoning. We find that current models are unable to encode short messages in their outputs without a monitor noticing under standard affordances. They can succeed, however, if given additional affordances such as using an unmonitored scratchpad and coordinating on what encoding scheme to use. We additionally find early signs that models can perform basic encoded reasoning in a simple state-tracking problem. This includes some ability to reason with their own and pre-defined schemes, including encoding schemes such as Hexadecimal. Despite this, they can rarely hide reasoning subtly within a cover task to fool a monitor. Overall, our results indicate that current LLMs exhibit nascent steganographic capabilities. While these capabilities are likely insufficient to bypass well-designed monitors at present, this could change in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02735v1">Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Prompt injection attacks pose a significant security threat to LLM-integrated applications. Model-level defenses have shown strong effectiveness, but are currently deployed into commercial-grade models in a closed-source manner. We believe open-source models are needed by the AI security community, where co-development of attacks and defenses through open research drives scientific progress in mitigation against prompt injection attacks. To this end, we develop Meta SecAlign, the first open-source and open-weight LLM with built-in model-level defense that achieves commercial-grade model performance. We provide complete details of our training recipe, which utilizes an improved version of the SOTA SecAlign defense. Evaluations on 9 utility benchmarks and 7 security benchmarks show that Meta SecAlign, despite being trained on a generic instruction-tuning dataset, confers security in unseen downstream tasks, including tool-calling and agentic web navigation, in addition general instruction-following. Our best model -- Meta-SecAlign-70B -- achieves state-of-the-art robustness against prompt injection attacks and comparable utility to closed-source commercial LLM with model-level defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01631v2">Gradient-Based Model Fingerprinting for LLM Similarity Detection and Family Classification</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral software components in modern applications, unauthorized model derivations through fine-tuning, merging, and redistribution have emerged as critical software engineering challenges. Unlike traditional software where clone detection and license compliance are well-established, the LLM ecosystem lacks effective mechanisms to detect model lineage and enforce licensing agreements. This gap is particularly problematic when open-source model creators, such as Meta's LLaMA, require derivative works to maintain naming conventions for attribution, yet no technical means exist to verify compliance. To fill this gap, treating LLMs as software artifacts requiring provenance tracking, we present TensorGuard, a gradient-based fingerprinting framework for LLM similarity detection and family classification. Our approach extracts model-intrinsic behavioral signatures by analyzing gradient responses to random input perturbations across tensor layers, operating independently of training data, watermarks, or specific model formats. TensorGuard supports the widely-adopted safetensors format and constructs high-dimensional fingerprints through statistical analysis of gradient features. These fingerprints enable two complementary capabilities: direct pairwise similarity assessment between arbitrary models through distance computation, and systematic family classification of unknown models via the K-Means clustering algorithm with domain-informed centroid initialization using known base models. Experimental evaluation on 58 models comprising 8 base models and 50 derivatives across five model families (Llama, Qwen, Gemma, Phi, Mistral) demonstrates 94% classification accuracy under our centroid-initialized K-Means clustering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00863v2">Next-Token Prediction Task Assumes Optimal Data Ordering for LLM Training in Proof Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      In the field of large language model (LLM)-based proof generation, despite extensive training on large datasets such as ArXiv, LLMs still exhibit only modest performance on proving tasks of moderate difficulty. We believe that this is partly due to the widespread presence of suboptimal ordering within the data for each proof used in training. For example, published proofs often follow a purely logical order, where each step logically proceeds from the previous steps based on the deductive rules. This order is designed to facilitate the verification of the proof's soundness, rather than to help people and models learn the discovery process of the proof. In proof generation, we argue that the optimal order for one training data sample occurs when the relevant intermediate supervision for a particular proof step in the proof is always positioned to the left of that proof step. We call such order the intuitively sequential order. We validate our claims using two tasks: intuitionistic propositional logic theorem-proving and digit multiplication. Our experiments verify the order effect and provide support for our explanations. We demonstrate that training is most effective when the proof is in the intuitively sequential order. Moreover, the order effect and the performance gap between models trained on different data orders can be substantial -- with an 11 percent improvement in proof success rate observed in the propositional logic theorem-proving task, between models trained on the optimal order compared to the worst order. Lastly, we define a common type of order issue in advanced math proofs and find that 17.3 percent of theorems with nontrivial proofs in the first two chapters of a widely used graduate-level mathematics textbook suffer from this issue. A detailed list of those proofs is provided in the appendix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02699v1">Control at Stake: Evaluating the Security Landscape of LLM-Driven Email Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      The increasing capabilities of LLMs have led to the rapid proliferation of LLM agent apps, where developers enhance LLMs with access to external resources to support complex task execution. Among these, LLM email agent apps represent one of the widely used categories, as email remains a critical communication medium for users. LLM email agents are capable of managing and responding to email using LLM-driven reasoning and autonomously executing user instructions via external email APIs (e.g., send email). However, despite their growing deployment and utility, the security mechanism of LLM email agent apps remains underexplored. Currently, there is no comprehensive study into the potential security risk within these agent apps and their broader implications. In this paper, we conduct the first in-depth and systematic security study of LLM email agents. We propose the Email Agent Hijacking (EAH) attack, which overrides the original prompts of the email agent via external email resources, allowing attackers to gain control of the email agent remotely and further perform specific attack scenarios without user awareness. To facilitate the large-scale evaluation, we propose EAHawk, a pipeline to evaluate the EAH attack of LLM email agent apps. By EAHawk, we performed an empirical study spanning 14 representative LLM agent frameworks, 63 agent apps, 12 LLMs, and 20 email services, which led to the generation of 1,404 real-world email agent instances for evaluation. Experimental results indicate that all 1,404 instances were successfully hijacked; on average, only 2.03 attack attempts are required to control an email agent instance. Even worse, for some LLMs, the average number of attempts needed to achieve full agent control drops to as few as 1.23.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05272v1">FuzzFeed: An Automatic Approach to Weakest Precondition Generation using LLMs and Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      The weakest precondition (WP) of a program describes the largest set of initial states from which all terminating executions of the program satisfy a given postcondition. The generation of WPs is an important task with practical applications in areas ranging from verification to run-time error checking. This paper proposes the combination of Large Language Models (LLMs) and fuzz testing for generating WPs. In pursuit of this goal, we introduce Fuzzing Guidance (FG); FG acts as a means of directing LLMs towards correct WPs using program execution feedback. FG utilises fuzz testing for approximately checking the validity and weakness of candidate WPs, this information is then fed back to the LLM as a means of context refinement. We demonstrate the effectiveness of our approach on a comprehensive benchmark set of deterministic array programs in Java. Our experiments indicate that LLMs are capable of producing viable candidate WPs, and that this ability can be practically enhanced through FG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05269v1">CORE: Benchmarking LLMs Code Reasoning Capabilities through Static Analysis Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely adopted across diverse software engineering domains, such as code generation, program repair, and vulnerability detection. These applications require understanding beyond surface-level code patterns: value propagation, control flow, and interdependence between program elements. However, existing benchmarks primarily evaluate end-to-end outcomes, such as whether code is correctly repaired or generated, leaving the models ability for program semantic reasoning underexplored. This work presents CoRe, a high-quality, human-verified benchmark designed to evaluate LLMs on fundamental static analysis tasks. CoRe includes 12,553 task instances spanning data dependency, control dependency, and information flow across programs written in C/C++, Java, and Python. To ensure semantic diversity and reasoning complexity, we propose a semantics-aware diverse sampling strategy that selects targets and task instances based on structural coverage and dependency depth. We evaluate 10 mainstream LLMs and show that, while they perform well at identifying dependencies, models still struggle with tasks that require deeper semantic understanding and multi-step reasoning. We further conduct qualitative analyses to uncover key challenges, such as complex control structures and backward dependency patterns, offering insights into improving LLMs code reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01936v1">The Thin Line Between Comprehension and Persuasion in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are excellent at maintaining high-level, convincing dialogues. They are being fast deployed as chatbots and evaluators in sensitive areas, such as peer review and mental health applications. This, along with the disparate accounts on their reasoning capabilities, calls for a closer examination of LLMs and their comprehension of dialogue. In this work we begin by evaluating LLMs' ability to maintain a debate--one of the purest yet most complex forms of human communication. Then we measure how this capability relates to their understanding of what is being talked about, namely, their comprehension of dialogical structures and the pragmatic context. We find that LLMs are capable of maintaining coherent, persuasive debates, often swaying the beliefs of participants and audiences alike. We also note that awareness or suspicion of AI involvement encourage people to be more critical of the arguments made. When polling LLMs on their comprehension of deeper structures of dialogue, however, they cannot demonstrate said understanding. Our findings tie the shortcomings of LLMs-as-evaluators to their (in)ability to understand the context. More broadly, for the field of argumentation theory we posit that, if an agent can convincingly maintain a dialogue, it is not necessary for it to know what it is talking about. Hence, the modelling of pragmatic context and coherence are secondary to effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03814v3">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01872v1">DIY-MKG: An LLM-Based Polyglot Language Learning System</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Submitted to EMNLP 2025 System Demonstration
    </div>
    <details class="paper-abstract">
      Existing language learning tools, even those powered by Large Language Models (LLMs), often lack support for polyglot learners to build linguistic connections across vocabularies in multiple languages, provide limited customization for individual learning paces or needs, and suffer from detrimental cognitive offloading. To address these limitations, we design Do-It-Yourself Multilingual Knowledge Graph (DIY-MKG), an open-source system that supports polyglot language learning. DIY-MKG allows the user to build personalized vocabulary knowledge graphs, which are constructed by selective expansion with related words suggested by an LLM. The system further enhances learning through rich annotation capabilities and an adaptive review module that leverages LLMs for dynamic, personalized quiz generation. In addition, DIY-MKG allows users to flag incorrect quiz questions, simultaneously increasing user engagement and providing a feedback loop for prompt refinement. Our evaluation of LLM-based components in DIY-MKG shows that vocabulary expansion is reliable and fair across multiple languages, and that the generated quizzes are highly accurate, validating the robustness of DIY-MKG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01844v1">Low-Perplexity LLM-Generated Sequences and Where To Find Them</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Camera-ready version. Accepted to ACL 2025. 10 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly widespread, understanding how specific training data shapes their outputs is crucial for transparency, accountability, privacy, and fairness. To explore how LLMs leverage and replicate their training data, we introduce a systematic approach centered on analyzing low-perplexity sequences - high-probability text spans generated by the model. Our pipeline reliably extracts such long sequences across diverse topics while avoiding degeneration, then traces them back to their sources in the training data. Surprisingly, we find that a substantial portion of these low-perplexity spans cannot be mapped to the corpus. For those that do match, we quantify the distribution of occurrences across source documents, highlighting the scope and nature of verbatim recall and paving a way toward better understanding of how LLMs training data impacts their behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01827v1">APRMCTS: Improving LLM-based Automated Program Repair with Iterative Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Automated Program Repair (APR) attempts to fix software bugs without human intervention, which plays a crucial role in software development and maintenance. Recently, with the advances in Large Language Models (LLMs), a rapidly increasing number of APR techniques have been proposed with remarkable performance. However, existing LLM-based APR techniques typically adopt trial-and-error strategies, which suffer from two major drawbacks: (1) inherently limited patch effectiveness due to local exploration, and (2) low search efficiency due to redundant exploration. In this paper, we propose APRMCTS, which uses iterative tree search to improve LLM-based APR. APRMCTS incorporates Monte Carlo Tree Search (MCTS) into patch searching by performing a global evaluation of the explored patches and selecting the most promising one for subsequent refinement and generation. APRMCTS effectively resolves the problems of falling into local optima and thus helps improve the efficiency of patch searching. Our experiments on 835 bugs from Defects4J demonstrate that, when integrated with GPT-3.5, APRMCTS can fix a total of 201 bugs, which outperforms all state-of-the-art baselines. Besides, APRMCTS helps GPT-4o-mini, GPT-3.5, Yi-Coder-9B, and Qwen2.5-Coder-7B to fix 30, 27, 37, and 28 more bugs, respectively. More importantly, APRMCTS boasts a significant performance advantage while employing small patch size (16 and 32), notably fewer than the 500 and 10,000 patches adopted in previous studies. In terms of cost, compared to existing state-of-the-art LLM-based APR methods, APRMCTS has time and monetary costs of less than 20% and 50%, respectively. Our extensive study demonstrates that APRMCTS exhibits good effectiveness and efficiency, with particular advantages in addressing complex bugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01806v1">LoRA Fine-Tuning Without GPUs: A CPU-Efficient Meta-Generation Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 5-page main paper (excluding references) + 11-page appendix, 3 tables, 1 figure. Accepted to ICML 2025 Workshop on Efficient Systems for Foundation Models
    </div>
    <details class="paper-abstract">
      Low-Rank Adapters (LoRAs) have transformed the fine-tuning of Large Language Models (LLMs) by enabling parameter-efficient updates. However, their widespread adoption remains limited by the reliance on GPU-based training. In this work, we propose a theoretically grounded approach to LoRA fine-tuning designed specifically for users with limited computational resources, particularly those restricted to standard laptop CPUs. Our method learns a meta-operator that maps any input dataset, represented as a probability distribution, to a set of LoRA weights by leveraging a large bank of pre-trained adapters for the Mistral-7B-Instruct-v0.2 model. Instead of performing new gradient-based updates, our pipeline constructs adapters via lightweight combinations of existing LoRAs directly on CPU. While the resulting adapters do not match the performance of GPU-trained counterparts, they consistently outperform the base Mistral model on downstream tasks, offering a practical and accessible alternative to traditional GPU-based fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01752v1">Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01734v1">LLMs for Legal Subsumption in German Employment Contracts</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 PrePrint - ICAIL25, Chicago
    </div>
    <details class="paper-abstract">
      Legal work, characterized by its text-heavy and resource-intensive nature, presents unique challenges and opportunities for NLP research. While data-driven approaches have advanced the field, their lack of interpretability and trustworthiness limits their applicability in dynamic legal environments. To address these issues, we collaborated with legal experts to extend an existing dataset and explored the use of Large Language Models (LLMs) and in-context learning to evaluate the legality of clauses in German employment contracts. Our work evaluates the ability of different LLMs to classify clauses as "valid," "unfair," or "void" under three legal context variants: no legal context, full-text sources of laws and court rulings, and distilled versions of these (referred to as examination guidelines). Results show that full-text sources moderately improve performance, while examination guidelines significantly enhance recall for void clauses and weighted F1-Score, reaching 80\%. Despite these advancements, LLMs' performance when using full-text sources remains substantially below that of human lawyers. We contribute an extended dataset, including examination guidelines, referenced legal sources, and corresponding annotations, alongside our code and all log files. Our findings highlight the potential of LLMs to assist lawyers in contract legality review while also underscoring the limitations of the methods presented.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01701v1">Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      In this paper, we propose to incorporate the blackboard architecture into LLM multi-agent systems (MASs) so that (1) agents with various roles can share all the information and others' messages during the whole problem-solving process, (2) agents that will take actions are selected based on the current content of the blackboard, and (3) the selection and execution round is repeated until a consensus is reached on the blackboard. We develop the first implementation of this proposal and conduct experiments on commonsense knowledge, reasoning and mathematical datasets. The results show that our system can be competitive with the SOTA static and dynamic MASs by achieving the best average performance, and at the same time manage to spend less tokens. Our proposal has the potential to enable complex and dynamic problem-solving where well-defined structures or workflows are unavailable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01694v1">Graph Representation-based Model Poisoning on Federated LLMs in CyberEdge Networks</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Federated large language models (FedLLMs) provide powerful generative capabilities in CyberEdge networks while protecting data privacy. However, FedLLMs remains highly vulnerable to model poisoning attacks. This article first reviews recent model poisoning techniques and existing defense mechanisms for FedLLMs, highlighting critical limitations, particularly under non-IID text distributions. In particular, current defenses primarily utilize distance-based outlier detection or norm constraints, operating under the assumption that adversarial updates significantly diverge from benign statistics. This assumption can fail when facing adaptive attackers targeting billionparameter LLMs. Next, this article investigates emerging Graph Representation-Based Model Poisoning (GRMP), a novel attack paradigm that leverages higher-order correlations among honest client gradients to synthesize malicious updates indistinguishable from legitimate model updates. GRMP can effectively evade advanced defenses, resulting in substantial accuracy loss and performance degradation. Moreover, this article outlines a research roadmap emphasizing the importance of graph-aware secure aggregation methods, FedLLMs-specific vulnerability metrics, and evaluation frameworks to strengthen the robustness of future federated language model deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01663v1">AsyncFlow: An Asynchronous Streaming RL Framework for Efficient LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a pivotal technology in the post-training phase of large language models (LLMs). Traditional task-colocated RL frameworks suffer from significant scalability bottlenecks, while task-separated RL frameworks face challenges in complex dataflows and the corresponding resource idling and workload imbalance. Moreover, most existing frameworks are tightly coupled with LLM training or inference engines, making it difficult to support custom-designed engines. To address these challenges, we propose AsyncFlow, an asynchronous streaming RL framework for efficient post-training. Specifically, we introduce a distributed data storage and transfer module that provides a unified data management and fine-grained scheduling capability in a fully streamed manner. This architecture inherently facilitates automated pipeline overlapping among RL tasks and dynamic load balancing. Moreover, we propose a producer-consumer-based asynchronous workflow engineered to minimize computational idleness by strategically deferring parameter update process within staleness thresholds. Finally, the core capability of AsynFlow is architecturally decoupled from underlying training and inference engines and encapsulated by service-oriented user interfaces, offering a modular and customizable user experience. Extensive experiments demonstrate an average of 1.59 throughput improvement compared with state-of-the-art baseline. The presented architecture in this work provides actionable insights for next-generation RL training system designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15154v3">MCCoder: Streamlining Motion Control with LLM-Assisted Code Generation and Rigorous Verification</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 IEEE CASE 2025 Best Student Paper Finalists
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant potential in code generation. However, in the factory automation sector, particularly motion control, manual programming, alongside inefficient and unsafe debugging practices, remains prevalent. This stems from the complex interplay of mechanical and electrical systems and stringent safety requirements. Moreover, most current AI-assisted motion control programming efforts focus on PLCs, with little attention given to high-level languages and function libraries. To address these challenges, we introduce MCCoder, an LLM-powered system tailored for generating motion control code, integrated with a soft-motion controller. MCCoder improves code generation through a structured workflow that combines multitask decomposition, hybrid retrieval-augmented generation (RAG), and iterative self-correction, utilizing a well-established motion library. Additionally, it integrates a 3D simulator for intuitive motion validation and logs of full motion trajectories for data verification, significantly enhancing accuracy and safety. In the absence of benchmark datasets and metrics tailored for evaluating motion control code generation, we propose MCEVAL, a dataset spanning motion tasks of varying complexity. Experiments show that MCCoder outperforms baseline models using Advanced RAG, achieving an overall performance gain of 33.09% and a 131.77% improvement on complex tasks in the MCEVAL dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01543v1">Is External Information Useful for Stance Detection with LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 ACL Findings 2025
    </div>
    <details class="paper-abstract">
      In the stance detection task, a text is classified as either favorable, opposing, or neutral towards a target. Prior work suggests that the use of external information, e.g., excerpts from Wikipedia, improves stance detection performance. However, whether or not such information can benefit large language models (LLMs) remains an unanswered question, despite their wide adoption in many reasoning tasks. In this study, we conduct a systematic evaluation on how Wikipedia and web search external information can affect stance detection across eight LLMs and in three datasets with 12 targets. Surprisingly, we find that such information degrades performance in most cases, with macro F1 scores dropping by up to 27.9\%. We explain this through experiments showing LLMs' tendency to align their predictions with the stance and sentiment of the provided information rather than the ground truth stance of the given text. We also find that performance degradation persists with chain-of-thought prompting, while fine-tuning mitigates but does not fully eliminate it. Our findings, in contrast to previous literature on BERT-based systems which suggests that external information enhances performance, highlight the risks of information biases in LLM-based stance classifiers. Code is available at https://github.com/ngqm/acl2025-stance-detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01541v1">Efficient Out-of-Scope Detection in Dialogue Systems via Uncertainty-Driven LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Out-of-scope (OOS) intent detection is a critical challenge in task-oriented dialogue systems (TODS), as it ensures robustness to unseen and ambiguous queries. In this work, we propose a novel but simple modular framework that combines uncertainty modeling with fine-tuned large language models (LLMs) for efficient and accurate OOS detection. The first step applies uncertainty estimation to the output of an in-scope intent detection classifier, which is currently deployed in a real-world TODS handling tens of thousands of user interactions daily. The second step then leverages an emerging LLM-based approach, where a fine-tuned LLM is triggered to make a final decision on instances with high uncertainty. Unlike prior approaches, our method effectively balances computational efficiency and performance, combining traditional approaches with LLMs and yielding state-of-the-art results on key OOS detection benchmarks, including real-world OOS data acquired from a deployed TODS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01513v1">SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19676v3">A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, and Defense Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 41 pages, 13 figures, submitted to IEEE COMST
    </div>
    <details class="paper-abstract">
      In recent years, Large-Language-Model-driven AI agents have exhibited unprecedented intelligence and adaptability, and are rapidly changing human production and life. Nowadays, agents are undergoing a new round of evolution. They no longer act as an isolated island like LLMs. Instead, they start to communicate with diverse external entities, such as other agents and tools, to perform more complex tasks collectively. Under this trend, agent communication is regarded as a foundational pillar of the future AI ecosystem, and many organizations have intensively begun to design related communication protocols (e.g., Anthropic's MCP and Google's A2A) within the recent few months. However, this new field exposes significant security hazards, which can cause severe damage to real-world scenarios. To help researchers quickly figure out this promising topic and benefit the future agent communication development, this paper presents a comprehensive survey of agent communication security. More precisely, we first present a clear definition of agent communication and categorize the entire lifecycle of agent communication into three stages: user-agent interaction, agent-agent communication, and agent-environment communication. Next, for each communication phase, we dissect related protocols and analyze the security risks according to the communication characteristics. Then, we summarize and outlook on the possible defense countermeasures for each risk. In addition, we conduct experiments using MCP and A2A to help readers better understand the novel vulnerabilities brought by agent communication. Finally, we discuss open issues and future directions in this promising research field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01446v1">Using multi-agent architecture to mitigate the risk of LLM hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Improving customer service quality and response time are critical factors for maintaining customer loyalty and increasing a company's market share. While adopting emerging technologies such as Large Language Models (LLMs) is becoming a necessity to achieve these goals, the risk of hallucination remains a major challenge. In this paper, we present a multi-agent system to handle customer requests sent via SMS. This system integrates LLM based agents with fuzzy logic to mitigate hallucination risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01438v1">EdgeLoRA: An Efficient Multi-Tenant LLM Serving System on Edge Devices</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained significant attention due to their versatility across a wide array of applications. Fine-tuning LLMs with parameter-efficient adapters, such as Low-Rank Adaptation (LoRA), enables these models to efficiently adapt to downstream tasks without extensive retraining. Deploying fine-tuned LLMs on multi-tenant edge devices offers substantial benefits, such as reduced latency, enhanced privacy, and personalized responses. However, serving LLMs efficiently on resource-constrained edge devices presents critical challenges, including the complexity of adapter selection for different tasks and memory overhead from frequent adapter swapping. Moreover, given the multiple requests in multi-tenant settings, processing requests sequentially results in underutilization of computational resources and increased latency. This paper introduces EdgeLoRA, an efficient system for serving LLMs on edge devices in multi-tenant environments. EdgeLoRA incorporates three key innovations: (1) an adaptive adapter selection mechanism to streamline the adapter configuration process; (2) heterogeneous memory management, leveraging intelligent adapter caching and pooling to mitigate memory operation overhead; and (3) batch LoRA inference, enabling efficient batch processing to significantly reduce computational latency. Comprehensive evaluations using the Llama3.1-8B model demonstrate that EdgeLoRA significantly outperforms the status quo (i.e., llama.cpp) in terms of both latency and throughput. The results demonstrate that EdgeLoRA can achieve up to a 4 times boost in throughput. Even more impressively, it can serve several orders of magnitude more adapters simultaneously. These results highlight EdgeLoRA's potential to transform edge deployment of LLMs in multi-tenant scenarios, offering a scalable and efficient solution for resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01436v1">Challenges & Opportunities with LLM-Assisted Visualization Retargeting</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 5 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Despite the ubiquity of visualization examples published on the web, retargeting existing custom chart implementations to new datasets remains difficult, time-intensive, and tedious. The adaptation process assumes author familiarity with both the implementation of the example as well as how the new dataset might need to be transformed to fit into the example code. With recent advances in Large Language Models (LLMs), automatic adaptation of code can be achieved from high-level user prompts, reducing the barrier for visualization retargeting. To better understand how LLMs can assist retargeting and its potential limitations, we characterize and evaluate the performance of LLM assistance across multiple datasets and charts of varying complexity, categorizing failures according to type and severity. In our evaluation, we compare two approaches: (1) directly instructing the LLM model to fully generate and adapt code by treating code as text inputs and (2) a more constrained program synthesis pipeline where the LLM guides the code construction process by providing structural information (e.g., visual encodings) based on properties of the example code and data. We find that both approaches struggle when new data has not been appropriately transformed, and discuss important design recommendations for future retargeting systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.16369v3">Don't Say No: Jailbreaking LLM by Suppressing Refusal</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 Accepted by ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Ensuring the safety alignment of Large Language Models (LLMs) is critical for generating responses consistent with human values. However, LLMs remain vulnerable to jailbreaking attacks, where carefully crafted prompts manipulate them into producing toxic content. One category of such attacks reformulates the task as an optimization problem, aiming to elicit affirmative responses from the LLM. However, these methods heavily rely on predefined objectionable behaviors, limiting their effectiveness and adaptability to diverse harmful queries. In this study, we first identify why the vanilla target loss is suboptimal and then propose enhancements to the loss objective. We introduce DSN (Don't Say No) attack, which combines a cosine decay schedule method with refusal suppression to achieve higher success rates. Extensive experiments demonstrate that DSN outperforms baseline attacks and achieves state-of-the-art attack success rates (ASR). DSN also shows strong universality and transferability to unseen datasets and black-box models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01413v1">Evaluating LLM Agent Collusion in Double Auctions</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities as autonomous agents with rapidly expanding applications in various domains. As these agents increasingly engage in socioeconomic interactions, identifying their potential for undesirable behavior becomes essential. In this work, we examine scenarios where they can choose to collude, defined as secretive cooperation that harms another party. To systematically study this, we investigate the behavior of LLM agents acting as sellers in simulated continuous double auction markets. Through a series of controlled experiments, we analyze how parameters such as the ability to communicate, choice of model, and presence of environmental pressures affect the stability and emergence of seller collusion. We find that direct seller communication increases collusive tendencies, the propensity to collude varies across models, and environmental pressures, such as oversight and urgency from authority figures, influence collusive behavior. Our findings highlight important economic and ethical considerations for the deployment of LLM-based market agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01378v1">RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Intelligent control of Unmanned Aerial Vehicles (UAVs) swarms has emerged as a critical research focus, and it typically requires the swarm to navigate effectively while avoiding obstacles and achieving continuous coverage over multiple mission targets. Although traditional Multi-Agent Reinforcement Learning (MARL) approaches offer dynamic adaptability, they are hindered by the semantic gap in numerical communication and the rigidity of homogeneous role structures, resulting in poor generalization and limited task scalability. Recent advances in Large Language Model (LLM)-based control frameworks demonstrate strong semantic reasoning capabilities by leveraging extensive prior knowledge. However, due to the lack of online learning and over-reliance on static priors, these works often struggle with effective exploration, leading to reduced individual potential and overall system performance. To address these limitations, we propose a Role-Adaptive LLM-Driven Yoked navigation algorithm RALLY. Specifically, we first develop an LLM-driven semantic decision framework that uses structured natural language for efficient semantic communication and collaborative reasoning. Afterward, we introduce a dynamic role-heterogeneity mechanism for adaptive role switching and personalized decision-making. Furthermore, we propose a Role-value Mixing Network (RMIX)-based assignment strategy that integrates LLM offline priors with MARL online policies to enable semi-offline training of role selection strategies. Experiments in the Multi-Agent Particle Environment (MPE) environment and a Software-In-The-Loop (SITL) platform demonstrate that RALLY outperforms conventional approaches in terms of task coverage, convergence speed, and generalization, highlighting its strong potential for collaborative navigation in agentic multi-UAV systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20187v2">Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU</a></div>
    <div class="paper-meta">
      📅 2025-07-02
      | 💬 15 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Advanced Large Language Models (LLMs) have achieved impressive performance across a wide range of complex and long-context natural language tasks. However, performing long-context LLM inference locally on a commodity GPU (a PC) with privacy concerns remains challenging due to the increasing memory demands of the key-value (KV) cache. Existing systems typically identify important tokens and selectively offload their KV data to GPU and CPU memory. The KV data needs to be offloaded to disk due to the limited memory on a commodity GPU, but the process is bottlenecked by token importance evaluation overhead and the disk's low bandwidth. In this paper, we present LeoAM, the first efficient importance-aware long-context LLM inference system for a single commodity GPU with adaptive hierarchical GPU-CPU-Disk KV management. Our system employs an adaptive KV management strategy that partitions KV data into variable-sized chunks based on the skewed distribution of attention weights across different layers to reduce computational and additional transmission overheads. Moreover, we propose a lightweight KV abstract method, which minimizes transmission latency by storing and extracting the KV abstract of each chunk on disk instead of the full KV data. LeoAM also leverages the dynamic compression and pipeline techniques to further accelerate inference. Experimental results demonstrate that LongInfer achieves an average inference latency speedup of 3.46x, while maintaining comparable LLM response quality. In scenarios with larger batch sizes, it achieves up to a 5.47x speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22618v2">Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-02
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (Diffusion LLMs) have shown promise for non-autoregressive text generation with parallel decoding capabilities. However, the practical inference speed of open-sourced Diffusion LLMs often lags behind autoregressive models due to the lack of Key-Value (KV) Cache and quality degradation when decoding multiple tokens simultaneously. To bridge this gap, we introduce a novel block-wise approximate KV Cache mechanism tailored for bidirectional diffusion models, enabling cache reuse with negligible performance drop. Additionally, we identify the root cause of generation quality degradation in parallel decoding as the disruption of token dependencies under the conditional independence assumption. To address this, we propose a confidence-aware parallel decoding strategy that selectively decodes tokens exceeding a confidence threshold, mitigating dependency violations and maintaining generation quality. Experimental results on LLaDA and Dream models across multiple LLM benchmarks demonstrate up to \textbf{27.6$\times$ throughput} improvement with minimal accuracy loss, closing the performance gap with autoregressive models and paving the way for practical deployment of Diffusion LLMs.
    </details>
</div>
